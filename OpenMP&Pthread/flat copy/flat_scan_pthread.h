#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <pthread.h>
#include <queue>
#include <utility>
#include <vector>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include "../simd/flat_scan_simd.h"
#define ANN_FLAT_HAS_NEON 1
#elif defined(__AVX2__)
#include "../simd/flat_scan_avx2.h"
#define ANN_FLAT_HAS_AVX2 1
#else
#include "flat_scan.h"
#endif

#include "thread_pool.h"

namespace ann_flat_pthread {

using FlatHeap = std::priority_queue<std::pair<float, uint32_t>>;

static inline int NormalizeThreads(int nthreads) {
    return nthreads < 1 ? 1 : nthreads;
}

static inline float Distance(const float* base, const float* query, size_t d) {
#if defined(ANN_FLAT_HAS_NEON)
    return ip_distance_simd(base, query, static_cast<int>(d));
#elif defined(ANN_FLAT_HAS_AVX2)
    return ann_avx2::ip_distance_avx2(base, query, static_cast<int>(d));
#else
    float dot = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        dot += base[i] * query[i];
    }
    return 1.0f - dot;
#endif
}

static inline void PushTopK(FlatHeap& heap, float dist, uint32_t id, size_t k) {
    if (k == 0) {
        return;
    }
    if (heap.size() < k) {
        heap.push(std::make_pair(dist, id));
    } else if (dist < heap.top().first) {
        heap.push(std::make_pair(dist, id));
        heap.pop();
    }
}

static inline FlatHeap MergeHeaps(std::vector<FlatHeap>& heaps, size_t k) {
    FlatHeap result;
    for (size_t i = 0; i < heaps.size(); ++i) {
        FlatHeap& heap = heaps[i];
        while (!heap.empty()) {
            const std::pair<float, uint32_t> item = heap.top();
            heap.pop();
            PushTopK(result, item.first, item.second, k);
        }
    }
    return result;
}

struct InterStaticArg {
    float* base;
    float* queries;
    size_t base_n;
    size_t d;
    size_t k;
    size_t q_start;
    size_t q_end;
    std::vector<FlatHeap>* results;
};

static inline void* InterStaticWorker(void* arg) {
    InterStaticArg* a = static_cast<InterStaticArg*>(arg);
    for (size_t i = a->q_start; i < a->q_end; ++i) {
        (*a->results)[i] =
            flat_search(a->base, a->queries + i * a->d, a->base_n, a->d, a->k);
    }
    return nullptr;
}

struct IntraStaticArg {
    float* base;
    float* query;
    size_t d;
    size_t k;
    size_t b_start;
    size_t b_end;
    FlatHeap local_heap;
};

static inline void* IntraStaticWorker(void* arg) {
    IntraStaticArg* a = static_cast<IntraStaticArg*>(arg);
    for (size_t i = a->b_start; i < a->b_end; ++i) {
        const float dist = Distance(a->base + i * a->d, a->query, a->d);
        PushTopK(a->local_heap, dist, static_cast<uint32_t>(i), a->k);
    }
    return nullptr;
}

struct InterDynamicArg {
    float* base;
    float* queries;
    size_t base_n;
    size_t query_n;
    size_t d;
    size_t k;
    std::atomic<size_t>* next_query;
    std::vector<FlatHeap>* results;
};

static inline void* InterDynamicWorker(void* arg) {
    InterDynamicArg* a = static_cast<InterDynamicArg*>(arg);
    while (true) {
        const size_t i = a->next_query->fetch_add(1, std::memory_order_relaxed);
        if (i >= a->query_n) {
            break;
        }
        (*a->results)[i] =
            flat_search(a->base, a->queries + i * a->d, a->base_n, a->d, a->k);
    }
    return nullptr;
}

struct IntraDynamicArg {
    float* base;
    float* query;
    size_t base_n;
    size_t d;
    size_t k;
    std::atomic<size_t>* next_base;
};

static inline void* IntraDynamicWorker(void* arg) {
    IntraDynamicArg* a = static_cast<IntraDynamicArg*>(arg);
    FlatHeap* heap = new FlatHeap();
    while (true) {
        const size_t i = a->next_base->fetch_add(1, std::memory_order_relaxed);
        if (i >= a->base_n) {
            break;
        }
        const float dist = Distance(a->base + i * a->d, a->query, a->d);
        PushTopK(*heap, dist, static_cast<uint32_t>(i), a->k);
    }
    return heap;
}

}  // namespace ann_flat_pthread

static inline void flat_search_inter_static(
    float* base, float* queries, size_t base_n, size_t query_n,
    size_t d, size_t k, int nthreads,
    std::vector<ann_flat_pthread::FlatHeap>& results) {
    using namespace ann_flat_pthread;
    nthreads = NormalizeThreads(nthreads);
    results.clear();
    results.resize(query_n);

    std::vector<pthread_t> threads(static_cast<size_t>(nthreads));
    std::vector<InterStaticArg> args(static_cast<size_t>(nthreads));
    const size_t chunk = (query_n + static_cast<size_t>(nthreads) - 1) /
                         static_cast<size_t>(nthreads);

    for (int t = 0; t < nthreads; ++t) {
        const size_t start = static_cast<size_t>(t) * chunk;
        const size_t end = std::min(start + chunk, query_n);
        args[static_cast<size_t>(t)] = {base, queries, base_n, d, k, start, end, &results};
        pthread_create(&threads[static_cast<size_t>(t)], nullptr,
                       &InterStaticWorker, &args[static_cast<size_t>(t)]);
    }
    for (int t = 0; t < nthreads; ++t) {
        pthread_join(threads[static_cast<size_t>(t)], nullptr);
    }
}

static inline ann_flat_pthread::FlatHeap flat_search_intra_static(
    float* base, float* query, size_t base_n, size_t d, size_t k, int nthreads) {
    using namespace ann_flat_pthread;
    nthreads = NormalizeThreads(nthreads);
    std::vector<pthread_t> threads(static_cast<size_t>(nthreads));
    std::vector<IntraStaticArg> args(static_cast<size_t>(nthreads));
    const size_t chunk = (base_n + static_cast<size_t>(nthreads) - 1) /
                         static_cast<size_t>(nthreads);

    for (int t = 0; t < nthreads; ++t) {
        const size_t start = static_cast<size_t>(t) * chunk;
        const size_t end = std::min(start + chunk, base_n);
        args[static_cast<size_t>(t)] = {base, query, d, k, start, end, FlatHeap()};
        pthread_create(&threads[static_cast<size_t>(t)], nullptr,
                       &IntraStaticWorker, &args[static_cast<size_t>(t)]);
    }
    for (int t = 0; t < nthreads; ++t) {
        pthread_join(threads[static_cast<size_t>(t)], nullptr);
    }

    std::vector<FlatHeap> heaps;
    heaps.reserve(static_cast<size_t>(nthreads));
    for (int t = 0; t < nthreads; ++t) {
        heaps.push_back(std::move(args[static_cast<size_t>(t)].local_heap));
    }
    return MergeHeaps(heaps, k);
}

static inline void flat_search_inter_dynamic(
    float* base, float* queries, size_t base_n, size_t query_n,
    size_t d, size_t k, int nthreads,
    std::vector<ann_flat_pthread::FlatHeap>& results) {
    using namespace ann_flat_pthread;
    nthreads = NormalizeThreads(nthreads);
    results.clear();
    results.resize(query_n);

    std::atomic<size_t> next_query(0);
    std::vector<pthread_t> threads(static_cast<size_t>(nthreads));
    InterDynamicArg arg = {base, queries, base_n, query_n, d, k, &next_query, &results};

    for (int t = 0; t < nthreads; ++t) {
        pthread_create(&threads[static_cast<size_t>(t)], nullptr,
                       &InterDynamicWorker, &arg);
    }
    for (int t = 0; t < nthreads; ++t) {
        pthread_join(threads[static_cast<size_t>(t)], nullptr);
    }
}

static inline ann_flat_pthread::FlatHeap flat_search_intra_dynamic(
    float* base, float* query, size_t base_n, size_t d, size_t k, int nthreads) {
    using namespace ann_flat_pthread;
    nthreads = NormalizeThreads(nthreads);
    std::atomic<size_t> next_base(0);
    std::vector<pthread_t> threads(static_cast<size_t>(nthreads));
    IntraDynamicArg arg = {base, query, base_n, d, k, &next_base};

    for (int t = 0; t < nthreads; ++t) {
        pthread_create(&threads[static_cast<size_t>(t)], nullptr,
                       &IntraDynamicWorker, &arg);
    }

    std::vector<FlatHeap> heaps;
    heaps.reserve(static_cast<size_t>(nthreads));
    for (int t = 0; t < nthreads; ++t) {
        void* ret = nullptr;
        pthread_join(threads[static_cast<size_t>(t)], &ret);
        FlatHeap* heap = static_cast<FlatHeap*>(ret);
        heaps.push_back(std::move(*heap));
        delete heap;
    }
    return MergeHeaps(heaps, k);
}

static inline void flat_search_inter_pool(
    float* base, float* queries, size_t base_n, size_t query_n,
    size_t d, size_t k, int nthreads,
    std::vector<ann_flat_pthread::FlatHeap>& results) {
    using namespace ann_flat_pthread;
    nthreads = NormalizeThreads(nthreads);
    results.clear();
    results.resize(query_n);

    ThreadPool pool(nthreads);
    for (size_t i = 0; i < query_n; ++i) {
        pool.Enqueue({i, i + 1, [&, i](size_t, size_t) {
            results[i] = flat_search(base, queries + i * d, base_n, d, k);
        }});
    }
    pool.WaitAll();
}

static inline ann_flat_pthread::FlatHeap flat_search_intra_pool(
    float* base, float* query, size_t base_n, size_t d, size_t k, int nthreads) {
    using namespace ann_flat_pthread;
    nthreads = NormalizeThreads(nthreads);
    std::vector<FlatHeap> heaps(static_cast<size_t>(nthreads));
    const size_t chunk = (base_n + static_cast<size_t>(nthreads) - 1) /
                         static_cast<size_t>(nthreads);

    ThreadPool pool(nthreads);
    for (int t = 0; t < nthreads; ++t) {
        const size_t tid = static_cast<size_t>(t);
        const size_t start = tid * chunk;
        const size_t end = std::min(start + chunk, base_n);
        pool.Enqueue({start, end, [&, tid](size_t begin, size_t finish) {
            for (size_t i = begin; i < finish; ++i) {
                const float dist = Distance(base + i * d, query, d);
                PushTopK(heaps[tid], dist, static_cast<uint32_t>(i), k);
            }
        }});
    }
    pool.WaitAll();
    return MergeHeaps(heaps, k);
}

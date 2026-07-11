#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <thread>
#include <queue>
#include <utility>
#include <vector>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include "../simd/flat_scan_simd.h"
#elif defined(__AVX2__)
#include "../simd/flat_scan_avx2.h"
#else
#include "flat_scan.h"
#endif

namespace ann_stdthread {

using FlatHeap = std::priority_queue<std::pair<float, uint32_t>>;

static inline float Distance(const float* base, const float* query, size_t d) {
#if defined(__aarch64__) || defined(__ARM_NEON)
    return ip_distance_simd(base, query, static_cast<int>(d));
#elif defined(__AVX2__)
    return ann_avx2::ip_distance_avx2(base, query, static_cast<int>(d));
#else
    float dot = 0.0f;
    for (size_t i = 0; i < d; ++i) dot += base[i] * query[i];
    return 1.0f - dot;
#endif
}

static inline void PushTopK(FlatHeap& heap, float dist, uint32_t id, size_t k) {
    if (k == 0) return;
    if (heap.size() < k) {
        heap.push({dist, id});
    } else if (dist < heap.top().first) {
        heap.pop();
        heap.push({dist, id});
    }
}

struct InterDynamicArg {
    const float* base;
    const float* queries;
    size_t base_n;
    size_t d;
    size_t k;
    size_t query_n;
    std::atomic<size_t>* next_query;
    std::vector<FlatHeap>* results;
};

static inline void InterDynamicWorker(InterDynamicArg* a) {
    while (true) {
        const size_t i = a->next_query->fetch_add(1, std::memory_order_relaxed);
        if (i >= a->query_n) break;
        FlatHeap heap;
        for (size_t j = 0; j < a->base_n; ++j) {
            const float dist = Distance(a->base + j * a->d,
                                        a->queries + i * a->d, a->d);
            PushTopK(heap, dist, static_cast<uint32_t>(j), a->k);
        }
        (*a->results)[i] = std::move(heap);
    }
}

inline void search_inter_dynamic(
    const float* base, const float* queries,
    size_t base_n, size_t query_n, size_t d, size_t k,
    int nthreads,
    std::vector<FlatHeap>& results)
{
    if (nthreads < 1) nthreads = 1;
    results.resize(query_n);
    std::atomic<size_t> next_query(0);
    InterDynamicArg arg{base, queries, base_n, d, k, query_n, &next_query, &results};

    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(nthreads));
    for (int t = 0; t < nthreads; ++t) {
        threads.emplace_back(InterDynamicWorker, &arg);
    }
    for (auto& th : threads) th.join();
}

}  // namespace ann_stdthread

#ifndef FLAT_PTHREAD_H
#define FLAT_PTHREAD_H
// ===================================================================
// Flat-SIMD Pthread 并行 — 参考 flat_scan_pthread.h 架构
//
// 四种模式:
//   IntraStatic  — 底库静态均分, 每线程局部堆(p), 最后 MergeHeaps(k)
//   IntraDynamic — 底库原子竞争, 每线程动态抢占向量, 负载更均衡
//   InterStatic  — 查询静态均分
//   InterDynamic — 查询原子分发
//
// 关键参数 p: 每线程局部堆大小.
//   p=k → 堆操作最少, 延迟最低 (精确距离场景)
//   p>k → 保留更多候选, 提升召回 (近似距离场景如 PQ/ADC)
//   p 越大 → latency 越高, recall 在近似场景中越好 (trade-off)
// ===================================================================

#include <algorithm>
#include <atomic>
#include <pthread.h>
#include <queue>
#include <vector>

#include "hnswlib/hnswlib/simd_utils.h"

namespace flat_mt {

using FlatHeap = std::priority_queue<std::pair<float, int>>;

// ---- distance (NEON inner-product) ----
inline float distance(const float* a, const float* b, size_t d) {
    return InnerProductSIMDNeon(a, b, d);
}

// ---- push into bounded max-heap (top = worst among best) ----
inline void heap_push(FlatHeap& h, float dist, int id, size_t limit) {
    if (limit == 0) return;
    if (h.size() < limit) {
        h.push({dist, id});
    } else if (dist < h.top().first) {
        h.pop();
        h.push({dist, id});
    }
}

// ---- merge N local heaps into one global top-k ----
inline FlatHeap heap_merge(std::vector<FlatHeap>& heaps, size_t k) {
    FlatHeap result;
    for (auto& h : heaps) {
        while (!h.empty()) {
            auto p = h.top(); h.pop();
            heap_push(result, p.first, p.second, k);
        }
    }
    return result;
}

// ---- serial baseline ----
inline FlatHeap search_serial(
    const float* base, const float* query, size_t base_n, size_t d, size_t k)
{
    FlatHeap h;
    for (size_t i = 0; i < base_n; ++i) {
        float dist = distance(base + i * d, query, d);
        heap_push(h, dist, static_cast<int>(i), k);
    }
    return h;
}

// ===================================================================
// IntraStatic: base 静态均分, 每线程扫固定区间
// ===================================================================
struct IntraStaticArg {
    const float* base; const float* query; size_t d; size_t k; size_t p;
    size_t b_start; size_t b_end;
    FlatHeap local_heap;
};

static inline void* intra_static_worker(void* arg) {
    IntraStaticArg* a = static_cast<IntraStaticArg*>(arg);
    for (size_t i = a->b_start; i < a->b_end; ++i) {
        float dist = distance(a->base + i * a->d, a->query, a->d);
        heap_push(a->local_heap, dist, static_cast<int>(i), a->p);
    }
    return nullptr;
}

inline FlatHeap search_intra_static(
    const float* base, const float* query, size_t base_n,
    size_t d, size_t k, size_t p, int nthreads)
{
    if (nthreads < 1) nthreads = 1;
    if (p < k) p = k;
    std::vector<pthread_t> threads(nthreads);
    std::vector<IntraStaticArg> args(nthreads);
    size_t chunk = (base_n + nthreads - 1) / nthreads;
    int actual = 0;
    for (int t = 0; t < nthreads; ++t) {
        size_t s = t * chunk;
        size_t e = std::min(s + chunk, base_n);
        if (s >= e) break;
        args[t].base = base; args[t].query = query;
        args[t].d = d; args[t].k = k; args[t].p = p;
        args[t].b_start = s; args[t].b_end = e;
        pthread_create(&threads[t], nullptr, intra_static_worker, &args[t]);
        actual = t + 1;
    }
    for (int t = 0; t < actual; ++t) pthread_join(threads[t], nullptr);

    std::vector<FlatHeap> heaps(actual);
    for (int t = 0; t < actual; ++t) heaps[t] = std::move(args[t].local_heap);
    return heap_merge(heaps, k);
}

// ===================================================================
// IntraDynamic: base 原子竞争, 每线程动态抢占向量索引
//   优势: 当各向量距离计算耗时不同时, 负载更均衡
// ===================================================================
struct IntraDynamicArg {
    const float* base; const float* query;
    size_t base_n; size_t d; size_t k; size_t p;
    std::atomic<size_t>* next_base;
};

static inline void* intra_dynamic_worker(void* arg) {
    IntraDynamicArg* a = static_cast<IntraDynamicArg*>(arg);
    FlatHeap* h = new FlatHeap();
    while (true) {
        size_t i = a->next_base->fetch_add(1, std::memory_order_relaxed);
        if (i >= a->base_n) break;
        float dist = distance(a->base + i * a->d, a->query, a->d);
        heap_push(*h, dist, static_cast<int>(i), a->p);
    }
    return h;
}

inline FlatHeap search_intra_dynamic(
    const float* base, const float* query, size_t base_n,
    size_t d, size_t k, size_t p, int nthreads)
{
    if (nthreads < 1) nthreads = 1;
    if (p < k) p = k;
    std::atomic<size_t> next_base(0);
    std::vector<pthread_t> threads(nthreads);
    IntraDynamicArg arg;
    arg.base = base; arg.query = query;
    arg.base_n = base_n; arg.d = d; arg.k = k; arg.p = p;
    arg.next_base = &next_base;
    for (int t = 0; t < nthreads; ++t)
        pthread_create(&threads[t], nullptr, intra_dynamic_worker, &arg);

    std::vector<FlatHeap> heaps;
    for (int t = 0; t < nthreads; ++t) {
        void* ret = nullptr;
        pthread_join(threads[t], &ret);
        FlatHeap* h = static_cast<FlatHeap*>(ret);
        heaps.push_back(std::move(*h));
        delete h;
    }
    return heap_merge(heaps, k);
}

// ===================================================================
// InterDynamic: query 原子分发, 每线程独立串行搜索
// ===================================================================
struct InterDynamicArg {
    const float* base; const float* queries;
    size_t base_n; size_t d; size_t k;
    size_t query_n;
    std::atomic<size_t>* next_query;
    std::vector<FlatHeap>* results;
};

static inline void* inter_dynamic_worker(void* arg) {
    InterDynamicArg* a = static_cast<InterDynamicArg*>(arg);
    while (true) {
        size_t i = a->next_query->fetch_add(1, std::memory_order_relaxed);
        if (i >= a->query_n) break;
        (*a->results)[i] = search_serial(a->base, a->queries + i * a->d,
                                         a->base_n, a->d, a->k);
    }
    return nullptr;
}

inline void search_inter_dynamic(
    const float* base, const float* queries,
    size_t base_n, size_t query_n, size_t d, size_t k, int nthreads,
    std::vector<FlatHeap>& results)
{
    if (nthreads < 1) nthreads = 1;
    results.resize(query_n);
    std::atomic<size_t> next_query(0);
    std::vector<pthread_t> threads(nthreads);
    InterDynamicArg arg;
    arg.base = base; arg.queries = queries;
    arg.base_n = base_n; arg.d = d; arg.k = k;
    arg.query_n = query_n;
    arg.next_query = &next_query; arg.results = &results;
    for (int t = 0; t < nthreads; ++t)
        pthread_create(&threads[t], nullptr, inter_dynamic_worker, &arg);
    for (int t = 0; t < nthreads; ++t)
        pthread_join(threads[t], nullptr);
}

}  // namespace flat_mt
#endif

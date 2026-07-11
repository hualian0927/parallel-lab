#ifndef HNSW_PTHREAD_H
#define HNSW_PTHREAD_H
// ===================================================================
// HNSW 图搜索 Pthread 并行 — 基于 hnswlib 真实框架
//
// 策略: inter-query batch, atomic fetch_add 分发 query
//   searchKnn 是 const 方法 → 多线程并发读安全
// ===================================================================

#include <atomic>
#include <pthread.h>
#include <sys/time.h>
#include <set>
#include <vector>
#include "hnswlib/hnswlib/hnswlib.h"

struct HNSWBatchArg {
    hnswlib::HierarchicalNSW<float>* index;
    const float* queries;
    size_t query_n, dim, k;
    std::atomic<size_t>* next;
    std::vector<float>* lats;
    std::vector<float>* recs;
    const int* gt; size_t gt_dim;
};

static inline void* hnsw_batch_worker(void* arg) {
    HNSWBatchArg* a = static_cast<HNSWBatchArg*>(arg);
    const unsigned long C = 1000 * 1000;
    while (true) {
        size_t i = a->next->fetch_add(1, std::memory_order_relaxed);
        if (i >= a->query_n) break;

        struct timeval tv; gettimeofday(&tv, nullptr);
        auto ret = a->index->searchKnn(a->queries + i * a->dim, a->k);
        struct timeval tv2; gettimeofday(&tv2, nullptr);
        (*a->lats)[i] = (float)((tv2.tv_sec * C + tv2.tv_usec) - (tv.tv_sec * C + tv.tv_usec));

        std::set<uint32_t> gs;
        for (size_t j = 0; j < a->k; ++j) gs.insert((uint32_t)a->gt[j + i * a->gt_dim]);
        size_t hits = 0;
        while (!ret.empty()) { if (gs.count((uint32_t)ret.top().second)) ++hits; ret.pop(); }
        (*a->recs)[i] = (float)hits / a->k;
    }
    return nullptr;
}

inline void hnsw_batch_pthread(
    hnswlib::HierarchicalNSW<float>* index,
    const float* queries, size_t query_n, size_t dim, size_t k,
    int nthreads,
    std::vector<float>& lats, std::vector<float>& recs,
    const int* gt, size_t gt_dim)
{
    lats.resize(query_n); recs.resize(query_n);
    std::atomic<size_t> next(0);
    std::vector<pthread_t> th(nthreads);
    HNSWBatchArg arg;
    arg.index = index; arg.queries = queries; arg.query_n = query_n;
    arg.dim = dim; arg.k = k; arg.next = &next;
    arg.lats = &lats; arg.recs = &recs; arg.gt = gt; arg.gt_dim = gt_dim;
    for (int t = 0; t < nthreads; ++t)
        pthread_create(&th[t], nullptr, hnsw_batch_worker, &arg);
    for (int t = 0; t < nthreads; ++t)
        pthread_join(th[t], nullptr);
}
#endif

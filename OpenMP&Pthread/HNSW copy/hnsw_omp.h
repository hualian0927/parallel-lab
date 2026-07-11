#ifndef HNSW_OMP_H
#define HNSW_OMP_H
// ===================================================================
// HNSW 图搜索 OpenMP 并行 — 基于 hnswlib 真实框架
//
// 策略 A (intra-query): 不实现。hnswlib 内部图结构不可访问，
//   指导书明确指出 intra-query 并行"非常困难"，通常负优化。
// 策略 B (inter-query batch): #pragma omp parallel for schedule(dynamic)
// ===================================================================

#include <omp.h>
#include <sys/time.h>
#include <set>
#include <vector>
#include "hnswlib/hnswlib/hnswlib.h"

inline void hnsw_batch_omp(
    hnswlib::HierarchicalNSW<float>* index,
    const float* queries, size_t query_n, size_t dim, size_t k,
    int nthreads,
    std::vector<float>& latencies,
    std::vector<float>& recalls,
    const int* gt, size_t gt_dim)
{
    latencies.resize(query_n);
    recalls.resize(query_n);
    omp_set_num_threads(nthreads);
    const unsigned long C = 1000 * 1000;

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < query_n; ++i) {
        struct timeval tv; gettimeofday(&tv, nullptr);

        auto ret = index->searchKnn(queries + i * dim, k);

        struct timeval tv2; gettimeofday(&tv2, nullptr);
        latencies[i] = (float)((tv2.tv_sec * C + tv2.tv_usec) - (tv.tv_sec * C + tv.tv_usec));

        std::set<uint32_t> gtset;
        for (size_t j = 0; j < k; ++j) gtset.insert((uint32_t)gt[j + i * gt_dim]);
        size_t hits = 0;
        while (!ret.empty()) {
            if (gtset.count((uint32_t)ret.top().second)) ++hits;
            ret.pop();
        }
        recalls[i] = (float)hits / k;
    }
}
#endif

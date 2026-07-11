#ifndef NSW_GRAPH_H
#define NSW_GRAPH_H
// ===================================================================
// NSW 图 (HNSW Layer 0 等价) — 指导书 §2.3 底层图搜索
//
// 图结构: 每个节点有 M 个邻居 (双向, 贪心构建)
// 搜索算法: 指导书图 2.5 — pq(候选队列) + H(结果集, size=ef)
//   1. 从入口点出发
//   2. 每次从 pq 弹出距离 query 最近的点 vc
//   3. 探索 vc 的所有邻居, 更新 H 和 pq
//   4. 当 pq 中最近的点也比 H 中最差的更远时停止
//
// 并行策略:
//   A. intra-query 多入口点: N 线程各从不同随机入口出发, 最后合并 H
//   B. inter-query 批量: N 线程各处理不同 query
// ===================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <set>
#include <vector>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace nsw {

// ---- distance ----
inline float ip_dist(const float* a, const float* b, size_t d) {
#if defined(__aarch64__) || defined(__ARM_NEON)
    float32x4_t sum4 = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= d; i += 4) {
        sum4 = vmlaq_f32(sum4, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    float tmp[4]; vst1q_f32(tmp, sum4);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < d; ++i) sum += a[i] * b[i];
    return 1.0f - sum;
#else
    float sum = 0.0f;
    for (size_t i = 0; i < d; ++i) sum += a[i] * b[i];
    return 1.0f - sum;
#endif
}

// ---- NSW 图 ----
struct Graph {
    size_t n = 0;        // 节点数
    size_t d = 0;        // 向量维度
    int M = 16;          // 每节点最大邻居数
    std::vector<std::vector<int>> neighbors; // adjacency[n]
    const float* base = nullptr;             // 原始向量数据

    // ----- 构建: 贪心插入 + 随机采样近似近邻 -----
    void build(const float* data, size_t n_, size_t d_, int M_ = 16) {
        n = n_; d = d_; M = M_;
        base = data;
        neighbors.resize(n);

        std::mt19937 rng(42);
        size_t sample = std::min<size_t>(200, n - 1);

        for (size_t i = 0; i < n; ++i) {
            // 随机采样候选邻居
            std::vector<size_t> candidates;
            candidates.reserve(sample);
            std::uniform_int_distribution<size_t> pick(0, i > 0 ? i - 1 : 0);
            std::set<size_t> seen;
            while (candidates.size() < sample && candidates.size() < i) {
                size_t j = pick(rng);
                if (j != i && seen.insert(j).second)
                    candidates.push_back(j);
            }

            // 找最近的 M 个
            std::vector<std::pair<float, int>> dists;
            for (size_t j : candidates) {
                dists.push_back({ip_dist(data + i * d, data + j * d, d), (int)j});
            }
            size_t keep = std::min<size_t>(M, dists.size());
            std::partial_sort(dists.begin(), dists.begin() + keep, dists.end());

            for (size_t k = 0; k < keep; ++k) {
                int nb = dists[k].second;
                neighbors[i].push_back(nb);
                // 双向连接 (限制度数)
                if (neighbors[nb].size() < (size_t)M * 2) {
                    neighbors[nb].push_back((int)i);
                }
            }
        }
        std::cerr << "[NSW] Graph built: " << n << " nodes, M=" << M << std::endl;
    }

    // ----- 串行搜索 (指导书 图 2.5 算法) -----
    // entry_points: 搜索起点 (随机选 r 个)
    // ef: 结果集大小 (越大召回越高, 越慢)
    std::priority_queue<std::pair<float, int>>
    search(const float* query, size_t k, size_t ef,
           const std::vector<int>& entry_points) const
    {
        // visited: 节点是否已在 H/pq 中 (避免重复探索)
        std::vector<bool> visited(n, false);

        // H: 结果集 (max-heap, top = worst among ef best)
        std::priority_queue<std::pair<float, int>> H;

        // pq: 候选集 (min-heap, top = closest to query)
        // 用 greater 实现 min-heap
        using MinHeap = std::priority_queue<std::pair<float, int>,
            std::vector<std::pair<float, int>>,
            std::greater<std::pair<float, int>>>;
        MinHeap pq;

        // 从所有入口点出发
        for (int ep : entry_points) {
            if (ep < 0 || ep >= (int)n) continue;
            float d = ip_dist(query, base + ep * this->d, this->d);
            pq.push({d, ep});
            H.push({d, ep});
            if (H.size() > ef) H.pop();
            visited[ep] = true;
        }

        while (!pq.empty()) {
            auto [dist_c, vc] = pq.top();
            // 停止条件: 最近候选也比 H 中最差的更远
            if (dist_c > H.top().first) break;
            pq.pop();

            for (int nb : neighbors[vc]) {
                if (visited[nb]) continue;
                visited[nb] = true;

                float d = ip_dist(query, base + nb * this->d, this->d);
                float worst_in_H = H.top().first;

                if (d < worst_in_H || H.size() < ef) {
                    pq.push({d, nb});
                    H.push({d, nb});
                    if (H.size() > ef) H.pop();
                }
            }
        }

        // 裁剪到 k
        while (H.size() > k) H.pop();
        return H;
    }

    // ---- serialization ----
    bool save(std::ostream& os) const {
        auto w = [&](const void* p, size_t sz) {
            os.write((const char*)p, sz); return !!os; };
        if (!w(&n, sizeof(n)) || !w(&d, sizeof(d)) || !w(&M, sizeof(M))) return false;
        for (size_t i = 0; i < n; ++i) {
            size_t sz = neighbors[i].size();
            if (!w(&sz, sizeof(sz))) return false;
            if (sz && !w(neighbors[i].data(), sz * sizeof(int))) return false;
        }
        return true;
    }
    bool load(std::istream& is) {
        auto r = [&](void* p, size_t sz) {
            is.read((char*)p, sz); return !!is; };
        if (!r(&n, sizeof(n)) || !r(&d, sizeof(d)) || !r(&M, sizeof(M))) return false;
        neighbors.resize(n);
        for (size_t i = 0; i < n; ++i) {
            size_t sz;
            if (!r(&sz, sizeof(sz))) return false;
            neighbors[i].resize(sz);
            if (sz && !r(neighbors[i].data(), sz * sizeof(int))) return false;
        }
        return true;
    }
};

// ---- 随机入口点 ----
inline std::vector<int> random_entries(size_t n, size_t r, int seed = 42) {
    static std::mt19937 rng(seed);
    std::vector<int> entries;
    entries.reserve(r);
    std::uniform_int_distribution<size_t> pick(0, n - 1);
    for (size_t i = 0; i < r; ++i)
        entries.push_back((int)pick(rng));
    return entries;
}

}  // namespace nsw

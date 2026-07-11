/*
 * s2_ivf: GPU IVF (Inverted File) for ANNS
 * Query-batched IVF with grouping strategy to minimize wasted GEMM computation.
 *
 * Algorithm:
 * 1. CPU: k-means training + build inverted lists
 * 2. GPU: Upload index (centroids, reordered base, list offsets)
 * 3. Per batch:
 *    a. Coarse: GEMM Q[m*d] x C^T[d*nlist] -> m x nlist
 *    b. Top-nprobe per query (kernel)
 *    c. Group (q,c) pairs by cluster -> batch GEMM per cluster
 *    d. Top-k reduction per query
 *
 * Target: RTX 4070 Laptop 8GB, recall >= 0.95
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <set>
#include <string>
#include <vector>

// ========== Macros ==========
#define CUDA_CHECK(call) do { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t s = call; \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)s); \
        exit(1); \
    } \
} while(0)

// ========== Data Loader ==========
template<typename T>
T* load_fbin(const std::string& path, size_t& n, size_t& d) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "[ERROR] open: %s\n", path.c_str()); exit(1); }
    int32_t h[2];
    f.read((char*)h, 8);
    n = h[0]; d = h[1];
    T* data = new T[n * d];
    f.read((char*)data, n * d * sizeof(T));
    fprintf(stderr, "[LOAD] %s n=%zu d=%zu\n", path.c_str(), n, d);
    return data;
}

// ========== CPU IVF Training ==========
struct CPUIvfIndex {
    int nlist, n, d;
    std::vector<float> centroids;
    std::vector<int> list_offsets;      // [nlist+1]
    std::vector<float> reordered_base;  // vectors grouped by cluster
    std::vector<int> reordered_ids;     // original IDs

    void build(const float* base, int n_, int d_, int nlist_, int niter=15) {
        nlist = nlist_; n = n_; d = d_;
        centroids.resize(nlist * d);

        // k-means init
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> pick(0, n-1);
        for (int c = 0; c < nlist; c++)
            memcpy(&centroids[c*d], base + pick(rng)*d, d*sizeof(float));

        std::vector<int> assign(n);
        std::vector<float> sum_c(nlist * d);
        std::vector<int> cnt(nlist);

        for (int it = 0; it < niter; it++) {
            // Assign
            for (int i = 0; i < n; i++) {
                float best = std::numeric_limits<float>::max();
                int best_c = 0;
                for (int c = 0; c < nlist; c++) {
                    float dist = 0;
                    for (int j = 0; j < d; j++) {
                        float diff = base[i*d+j] - centroids[c*d+j];
                        dist += diff * diff;
                    }
                    if (dist < best) { best = dist; best_c = c; }
                }
                assign[i] = best_c;
            }
            // Update
            std::fill(sum_c.begin(), sum_c.end(), 0.0f);
            std::fill(cnt.begin(), cnt.end(), 0);
            for (int i = 0; i < n; i++) {
                int c = assign[i];
                cnt[c]++;
                for (int j = 0; j < d; j++) sum_c[c*d+j] += base[i*d+j];
            }
            for (int c = 0; c < nlist; c++) {
                if (cnt[c] == 0) continue;
                float inv = 1.0f / cnt[c];
                for (int j = 0; j < d; j++) centroids[c*d+j] = sum_c[c*d+j] * inv;
            }
        }

        // Build inverted lists
        list_offsets.resize(nlist + 1, 0);
        for (int i = 0; i < n; i++) list_offsets[assign[i] + 1]++;
        for (int c = 0; c < nlist; c++) list_offsets[c+1] += list_offsets[c];

        reordered_base.resize(n * d);
        reordered_ids.resize(n);
        std::vector<int> cur = list_offsets;
        for (int i = 0; i < n; i++) {
            int c = assign[i];
            int pos = cur[c]++;
            reordered_ids[pos] = i;
            memcpy(&reordered_base[pos*d], base + i*d, d*sizeof(float));
        }
        fprintf(stderr, "[IVF] Built nlist=%d, total vectors=%d\n", nlist, n);
    }
};

// ========== GPU IVF kernels ==========

// Kernel: per-query top-nprobe selection from centroid inner products
// Input:  scores[m * nlist] (col-major from GEMM: scores[ci*m + qi] = dot(centroid_ci, query_qi))
// Output: probes[m * nprobe] = centroid indices (largest inner products = closest)
// Note: GEMM C[nlist x m] col-major = C(ci, qi) at offset ci + qi * nlist
//       This equals the row-major access scores[qi * nlist + ci]
__global__ void select_probes_kernel(
    const float* scores, int* probes,
    int m, int nlist, int nprobe)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= m) return;

    const float* col_start = scores + qid * nlist;
    int* out = probes + (size_t)qid * nprobe;

    // Select LARGEST nprobe inner products (= closest centroids)
    // Maintain min-heap: heap[0] is the smallest among selected
    struct Pair { float ip; int id; };
    Pair heap[128];
    int hsize = 0;

    for (int c = 0; c < nlist; c++) {
        float ip = col_start[c];  // inner product
        if (hsize < nprobe) {
            // Insert maintaining ascending order (heap[0] is min)
            int pos = hsize;
            while (pos > 0 && heap[pos-1].ip > ip) {
                heap[pos] = heap[pos-1]; pos--;
            }
            heap[pos].ip = ip; heap[pos].id = c;
            hsize++;
        } else if (ip > heap[0].ip) {
            // Replace min if new IP is larger
            int pos = 0;
            while (pos < nprobe - 1 && heap[pos+1].ip < ip) {
                heap[pos] = heap[pos+1]; pos++;
            }
            while (pos > 0 && heap[pos-1].ip > ip) {
                heap[pos] = heap[pos-1]; pos--;
            }
            heap[pos].ip = ip; heap[pos].id = c;
        }
    }
    for (int i = 0; i < nprobe; i++)
        out[i] = heap[i].id;
}

// Kernel: build cluster->query mapping from probes
// Input:  probes[m * nprobe] = centroid ids per query
// Output: cluster_queries[nlist * max_q_per_c] (padded), cluster_counts[nlist]
//          query_to_bucket[q * nprobe] maps (probe_idx -> bucket_pos)
// This kernel counts and distributes
__global__ void build_query_cluster_map_kernel(
    const int* probes,
    int* cluster_counts,      // [nlist]
    int* query_positions,     // [m * nprobe] position in cluster_queries
    int m, int nlist, int nprobe)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * nprobe;
    if (idx >= total) return;

    int qid = idx / nprobe;
    int pid = idx % nprobe;
    int cid = probes[(size_t)qid * nprobe + pid];

    // Atomic add to get position in cluster's query list
    int pos = atomicAdd(&cluster_counts[cid], 1);
    query_positions[(size_t)qid * nprobe + pid] = pos;
}

// Kernel: scatter queries into cluster buffers
__global__ void scatter_queries_kernel(
    const int* probes,
    const int* query_positions,
    const float* queries,       // [m * d]
    int* cluster_query_ids,     // [nlist * max_qpc] query indices
    float* cluster_queries,     // [nlist * max_qpc * d]
    int m, int nlist, int nprobe, int d, int max_qpc)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * nprobe;
    if (idx >= total) return;

    int qid = idx / nprobe;
    int pid = idx % nprobe;
    int cid = probes[(size_t)qid * nprobe + pid];
    int pos = query_positions[(size_t)qid * nprobe + pid];

    if (pos < max_qpc) {
        cluster_query_ids[(size_t)cid * max_qpc + pos] = qid;
        // Copy query vector
        const float* src = queries + (size_t)qid * d;
        float* dst = cluster_queries + ((size_t)cid * max_qpc + pos) * d;
        for (int j = 0; j < d; j++) dst[j] = src[j];
    }
}

// ========== Helper: CPU-side top-k with inner product distance ==========
inline float ip_dist_cpu(const float* a, const float* b, int d) {
    float sum = 0;
    for (int i = 0; i < d; i++) sum += a[i] * b[i];
    return 1.0f - sum;
}

// ========== Timing ==========
double get_time_ms() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(now.time_since_epoch()).count();
}

// ========== Main Search Function ==========
struct IvfSearchResult {
    std::vector<int> topk_ids;     // [m * k]
    float recall;
    float avg_latency_us;
};

IvfSearchResult ivf_gpu_search_batch(
    cublasHandle_t cublas_handle,
    const CPUIvfIndex& index,
    const float* h_queries, const int* h_gt, int gt_dim,
    int m, int k, int nprobe,
    // GPU buffers (pre-allocated)
    float* d_centroids, float* d_reordered_base, int* d_reordered_ids,
    int* d_list_offsets,
    float* d_queries,
    float* d_coarse_scores, int* d_probes,
    int* d_cluster_counts, int* d_query_positions,
    int* d_cluster_query_ids, float* d_cluster_queries,
    float* d_cluster_scores,
    int max_qpc)
{
    IvfSearchResult result;
    result.topk_ids.resize(m * k);
    int nlist = index.nlist;
    int d = index.d;
    int n = index.n;

    float alpha = 1.0f, beta = 0.0f;
    double total_ms = 0;

    // Step 1: Copy queries to GPU
    CUDA_CHECK(cudaMemcpy(d_queries, h_queries, (size_t)m * d * sizeof(float), cudaMemcpyHostToDevice));

    double t0 = get_time_ms();

    // Step 2: Coarse search - GEMM Q[m*d] x C^T[d*nlist] -> [m x nlist]
    // d_centroids: row-major [nlist x d] = col-major C_col[d x nlist]
    // d_queries: row-major [m x d] = col-major Q_col[d x m]
    // Want: Scores[m x nlist] = Q[m x d] x C^T[d x nlist] for inner product
    // In cuBLAS col-major: C(m,n) = op(A) * op(B)
    // Q_col is d x m, C_col is d x nlist
    // We want: Scores_col(nlist x m) = C_col^T * Q_col
    // C(m,n) where m=nlist, n=m, k=d
    // transa=CUBLAS_OP_T (C^T), transb=CUBLAS_OP_N (Q as is)
    CUBLAS_CHECK(cublasSgemm(cublas_handle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              nlist, m, d,
                              &alpha,
                              d_centroids, d,      // C_col: d x nlist
                              d_queries, d,         // Q_col: d x m
                              &beta,
                              d_coarse_scores, nlist)); // [nlist x m] col-major

    // Step 3: Select top-nprobe per query
    {
        int block = 256;
        int grid = (m + block - 1) / block;
        select_probes_kernel<<<grid, block>>>(
            d_coarse_scores, d_probes, m, nlist, nprobe);
    }

    // Step 4: Build cluster->query mapping
    CUDA_CHECK(cudaMemset(d_cluster_counts, 0, (size_t)nlist * sizeof(int)));
    {
        int total = m * nprobe;
        int block = 256;
        int grid = (total + block - 1) / block;
        build_query_cluster_map_kernel<<<grid, block>>>(
            d_probes, d_cluster_counts, d_query_positions, m, nlist, nprobe);
    }

    // Step 5: Scatter queries into cluster buffers
    CUDA_CHECK(cudaMemset(d_cluster_queries, 0, (size_t)nlist * max_qpc * d * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_cluster_query_ids, 0, (size_t)nlist * max_qpc * sizeof(int)));
    {
        int total = m * nprobe;
        int block = 256;
        int grid = (total + block - 1) / block;
        scatter_queries_kernel<<<grid, block>>>(
            d_probes, d_query_positions, d_queries,
            d_cluster_query_ids, d_cluster_queries,
            m, nlist, nprobe, d, max_qpc);
    }

    // Copy cluster counts back to CPU to decide which clusters to process
    std::vector<int> h_cluster_counts(nlist);
    CUDA_CHECK(cudaMemcpy(h_cluster_counts.data(), d_cluster_counts,
                          (size_t)nlist * sizeof(int), cudaMemcpyDeviceToHost));

    // Step 6: For each cluster with queries, do GEMM and aggregate results
    // Per-query results on CPU
    std::vector<std::priority_queue<std::pair<float, int>>> per_query_heap(m);
    int total_candidates = 0;

    for (int c = 0; c < nlist; c++) {
        int nq = h_cluster_counts[c];
        if (nq == 0) continue;

        int list_size = index.list_offsets[c+1] - index.list_offsets[c];
        if (list_size == 0) continue;

        int nq_rounded = nq > max_qpc ? max_qpc : nq;

        // GEMM: cluster_queries[nq x d] x cluster_vecs^T[d x list_size] -> [nq x list_size]
        // d_cluster_queries + c*max_qpc*d: [nq x d] row-major -> col-major [d x nq]
        // d_reordered_base + offset*d: [list_size x d] row-major -> col-major [d x list_size]
        // Want scores: list_size x nq (col-major)
        int base_offset = index.list_offsets[c];
        CUBLAS_CHECK(cublasSgemm(cublas_handle,
                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                  list_size, nq_rounded, d,
                                  &alpha,
                                  d_reordered_base + (size_t)base_offset * d, d,
                                  d_cluster_queries + (size_t)c * max_qpc * d, d,
                                  &beta,
                                  d_cluster_scores, list_size));

        // Copy scores back to CPU
        std::vector<float> h_scores(nq_rounded * list_size);
        CUDA_CHECK(cudaMemcpy(h_scores.data(), d_cluster_scores,
                              (size_t)nq_rounded * list_size * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // Copy query IDs for this cluster
        std::vector<int> h_query_ids(nq_rounded);
        CUDA_CHECK(cudaMemcpy(h_query_ids.data(),
                              d_cluster_query_ids + (size_t)c * max_qpc,
                              nq_rounded * sizeof(int), cudaMemcpyDeviceToHost));

        // Copy vector IDs for this cluster
        std::vector<int> h_vec_ids(list_size);
        CUDA_CHECK(cudaMemcpy(h_vec_ids.data(),
                              d_reordered_ids + (size_t)base_offset,
                              list_size * sizeof(int), cudaMemcpyDeviceToHost));

        // Update per-query heaps (inner product: smaller = closer)
        for (int qi = 0; qi < nq_rounded; qi++) {
            int gqid = h_query_ids[qi];
            auto& heap = per_query_heap[gqid];
            for (int vi = 0; vi < list_size; vi++) {
                float score = h_scores[(size_t)qi * list_size + vi];
                float dist = 1.0f - score;  // convert to distance
                if ((int)heap.size() < k) {
                    heap.push({dist, h_vec_ids[vi]});
                } else if (dist < heap.top().first) {
                    heap.pop();
                    heap.push({dist, h_vec_ids[vi]});
                }
            }
        }
        total_candidates += nq_rounded * list_size;
    }

    double t1 = get_time_ms();
    total_ms = t1 - t0;

    // Extract top-k IDs
    for (int qi = 0; qi < m; qi++) {
        auto& heap = per_query_heap[qi];
        // heap is max-heap, we want ascending order of distance
        std::vector<std::pair<float, int>> tmp;
        while (!heap.empty()) { tmp.push_back(heap.top()); heap.pop(); }
        // tmp is descending, reverse
        std::reverse(tmp.begin(), tmp.end());
        for (int j = 0; j < k && j < (int)tmp.size(); j++)
            result.topk_ids[qi * k + j] = tmp[j].second;
        // Pad if fewer than k
        for (int j = tmp.size(); j < k; j++)
            result.topk_ids[qi * k + j] = 0;
    }

    // Compute recall
    size_t total_hits = 0;
    for (int i = 0; i < m; i++) {
        std::set<int> gtset(h_gt + i * gt_dim, h_gt + i * gt_dim + k);
        for (int j = 0; j < k; j++) {
            if (gtset.count(result.topk_ids[i * k + j])) total_hits++;
        }
    }
    result.recall = (float)total_hits / (m * k);
    result.avg_latency_us = (float)(total_ms * 1000.0 / m);

    return result;
}

// ========== Main ==========
int main(int argc, char* argv[]) {
    const char* data_dir = (argc > 1) ? argv[1] : "../../ANN_DATA/";
    int batch_size = (argc > 2) ? atoi(argv[2]) : 500;
    int nprobe = (argc > 3) ? atoi(argv[3]) : 50;
    int nlist = (argc > 4) ? atoi(argv[4]) : 1000;
    const int k = 10;

    // Load data
    size_t base_n, base_d, query_n, query_d, gt_n, gt_d;
    float* h_base = load_fbin<float>(std::string(data_dir) + "DEEP100K.base.100k.fbin", base_n, base_d);
    float* h_query = load_fbin<float>(std::string(data_dir) + "DEEP100K.query.fbin", query_n, query_d);
    int*   h_gt   = load_fbin<int>(std::string(data_dir) + "DEEP100K.gt.query.100k.top100.bin", gt_n, gt_d);

    int m_total = (int)query_n;
    int d = (int)base_d;
    int n = (int)base_n;

    // Train IVF index on CPU
    fprintf(stderr, "\n=== GPU IVF ===\n");
    fprintf(stderr, "Training IVF index (nlist=%d, niter=15)...\n", nlist);
    double t_train = get_time_ms();
    CPUIvfIndex ivf_index;
    ivf_index.build(h_base, n, d, nlist, 15);
    fprintf(stderr, "Training done in %.0f ms\n", get_time_ms() - t_train);

    // Allocate GPU memory
    float *d_centroids, *d_reordered_base;
    int *d_reordered_ids, *d_list_offsets;
    CUDA_CHECK(cudaMalloc(&d_centroids, (size_t)nlist * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_reordered_base, (size_t)n * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_reordered_ids, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_list_offsets, (size_t)(nlist + 1) * sizeof(int)));

    // Copy index to GPU
    CUDA_CHECK(cudaMemcpy(d_centroids, ivf_index.centroids.data(),
                          (size_t)nlist * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_reordered_base, ivf_index.reordered_base.data(),
                          (size_t)n * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_reordered_ids, ivf_index.reordered_ids.data(),
                          (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_list_offsets, ivf_index.list_offsets.data(),
                          (size_t)(nlist + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // Batch GPU buffers
    float *d_queries, *d_coarse_scores;
    int *d_probes, *d_cluster_counts, *d_query_positions;
    int *d_cluster_query_ids;
    float *d_cluster_queries, *d_cluster_scores;

    // Estimate max queries per cluster: worst case all queries pick same cluster
    int max_qpc = std::min(batch_size, batch_size * nprobe / 10 + 100);  // generous estimate
    max_qpc = std::min(max_qpc, batch_size);  // can't exceed batch size

    CUDA_CHECK(cudaMalloc(&d_queries, (size_t)batch_size * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_coarse_scores, (size_t)nlist * batch_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_probes, (size_t)batch_size * nprobe * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cluster_counts, (size_t)nlist * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_query_positions, (size_t)batch_size * nprobe * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cluster_query_ids, (size_t)nlist * max_qpc * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cluster_queries, (size_t)nlist * max_qpc * d * sizeof(float)));

    // Max list size for score buffer
    int max_list_size = 0;
    for (int c = 0; c < nlist; c++)
        max_list_size = std::max(max_list_size, ivf_index.list_offsets[c+1] - ivf_index.list_offsets[c]);
    int max_score_size = max_qpc * max_list_size;
    CUDA_CHECK(cudaMalloc(&d_cluster_scores, (size_t)max_score_size * sizeof(float)));

    // cuBLAS
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    fprintf(stderr, "\n=== Search (nprobe=%d, batch=%d) ===\n", nprobe, batch_size);

    // Process all queries in batches
    std::vector<int> all_ids(m_total * k);
    float total_recall = 0;
    float total_latency_us = 0;
    int num_batches = (m_total + batch_size - 1) / batch_size;

    for (int b = 0; b < num_batches; b++) {
        int m_cur = std::min(batch_size, m_total - b * batch_size);
        int q_offset = b * batch_size;

        auto res = ivf_gpu_search_batch(
            cublas_handle, ivf_index,
            h_query + (size_t)q_offset * d,
            h_gt + (size_t)q_offset * gt_d, (int)gt_d,
            m_cur, k, nprobe,
            d_centroids, d_reordered_base, d_reordered_ids, d_list_offsets,
            d_queries, d_coarse_scores, d_probes,
            d_cluster_counts, d_query_positions,
            d_cluster_query_ids, d_cluster_queries,
            d_cluster_scores, max_qpc);

        memcpy(all_ids.data() + (size_t)q_offset * k, res.topk_ids.data(),
               (size_t)m_cur * k * sizeof(int));
        total_recall += res.recall * m_cur;
        total_latency_us += res.avg_latency_us * m_cur;

        if (b % 10 == 0)
            fprintf(stderr, "  Batch %d/%d: recall=%.4f latency=%.0f us/q\n",
                    b+1, num_batches, res.recall, res.avg_latency_us);
    }

    total_recall /= m_total;
    total_latency_us /= m_total;

    printf("\n| %-30s | %10s | %12s |\n", "Method", "Recall@10", "Latency(us/q)");
    printf("|%-32s|%12.5f|%14.1f|\n", "GPU-IVF", total_recall, total_latency_us);
    fprintf(stderr, "\nGPU-IVF: recall=%.5f avg_latency=%.1f us/q (%.2f ms/q)\n",
            total_recall, total_latency_us, total_latency_us / 1000.0f);

    // Write result file
    std::ofstream res("result_tradeoff.md");
    res << "# GPU IVF Results\n\n";
    res << "## Configuration\n";
    res << "- GPU: RTX 4070 Laptop 8GB\n";
    res << "- Dataset: DEEP100K (n=" << n << ", d=" << d << ", queries=10K)\n";
    res << "- Distance: Inner Product (1.0 - dot)\n";
    res << "- nlist=" << nlist << ", nprobe=" << nprobe << "\n";
    res << "- Batch size: " << batch_size << "\n\n";
    res << "## Results\n\n";
    res << "| Method | nprobe | Batch | Recall@10 | Avg Latency (us/q) | Avg Latency (ms/q) |\n";
    res << "|--------|--------|-------|-----------|--------------------|--------------------|\n";
    res << "| GPU-IVF | " << nprobe << " | " << batch_size << " | "
        << std::fixed << std::setprecision(5) << total_recall << " | "
        << std::setprecision(1) << total_latency_us << " | "
        << std::setprecision(3) << total_latency_us / 1000.0f << " |\n";
    res << "\n## Analysis\n";
    res << "- Coarse search: GEMM on GPU\n";
    res << "- Fine search: Grouped GEMM per cluster with query batching\n";
    res << "- Grouping minimizes wasted computation when different queries select different clusters\n";
    res << "- Recall depends on nprobe (more probes = higher recall, higher latency)\n";
    res.close();

    // Cleanup
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUDA_CHECK(cudaFree(d_centroids)); CUDA_CHECK(cudaFree(d_reordered_base));
    CUDA_CHECK(cudaFree(d_reordered_ids)); CUDA_CHECK(cudaFree(d_list_offsets));
    CUDA_CHECK(cudaFree(d_queries)); CUDA_CHECK(cudaFree(d_coarse_scores));
    CUDA_CHECK(cudaFree(d_probes)); CUDA_CHECK(cudaFree(d_cluster_counts));
    CUDA_CHECK(cudaFree(d_query_positions)); CUDA_CHECK(cudaFree(d_cluster_query_ids));
    CUDA_CHECK(cudaFree(d_cluster_queries)); CUDA_CHECK(cudaFree(d_cluster_scores));
    delete[] h_base; delete[] h_query; delete[] h_gt;

    return 0;
}

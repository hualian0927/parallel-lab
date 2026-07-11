/*
 * s3_ivfpq: GPU IVF-PQ (Inverted File with Product Quantization)
 * Query-batched IVF with PQ-accelerated distance computation.
 *
 * Algorithm:
 * 1. CPU: Train IVF + PQ codebooks
 * 2. CPU: Encode base vectors as PQ codes (M bytes each)
 * 3. GPU: Upload IVF index + PQ codes + codebooks
 * 4. Per batch:
 *    a. Coarse: GEMM Q[m*d] x C^T -> select top-nprobe clusters
 *    b. Build per-query PQ LUT on GPU
 *    c. Grouped ADC: per cluster, compute approximate distances via LUT
 *    d. Select top rerank_p candidates
 *    e. Rerank with exact distances on GPU
 *    f. Top-k selection
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

// ========== CPU IVF-PQ Index ==========
struct CPUIvfPqIndex {
    int nlist, n, d;
    int M;           // PQ subspaces
    int K_pq;        // PQ centroids per subspace (256)
    int sub_dim;     // d / M

    // IVF
    std::vector<float> centroids;
    std::vector<int> list_offsets;
    std::vector<float> reordered_base;
    std::vector<int> reordered_ids;

    // Global PQ
    std::vector<float> codebooks;   // SoA: [M * K_pq * sub_dim]
    std::vector<uint8_t> pq_codes;  // [n * M]

    void build(const float* base, int n_, int d_, int nlist_, int M_, int K_pq_,
               int ivf_iter=15, int pq_iter=10) {
        nlist = nlist_; n = n_; d = d_; M = M_; K_pq = K_pq_;
        sub_dim = d / M;

        // --- IVF clustering (same as before) ---
        centroids.resize(nlist * d);
        {
            std::mt19937 rng(42);
            std::uniform_int_distribution<int> pick(0, n-1);
            for (int c = 0; c < nlist; c++)
                memcpy(&centroids[c*d], base + pick(rng)*d, d*sizeof(float));
        }
        std::vector<int> assign(n);
        std::vector<float> sum_c(nlist * d);
        std::vector<int> cnt(nlist);

        for (int it = 0; it < ivf_iter; it++) {
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

        // --- Global PQ training ---
        fprintf(stderr, "[PQ] Training global PQ M=%d K=%d sub_dim=%d...\n", M, K_pq, sub_dim);
        codebooks.resize(M * K_pq * sub_dim);

        for (int m = 0; m < M; m++) {
            // Extract sub-vectors
            std::vector<float> sub_vectors(n * sub_dim);
            for (int i = 0; i < n; i++)
                memcpy(&sub_vectors[i * sub_dim], base + i*d + m*sub_dim, sub_dim*sizeof(float));

            // k-means for this subspace
            float* cb_m = codebooks.data() + m * K_pq * sub_dim;
            std::mt19937 rng_m(42 + m * 1000);
            for (int k = 0; k < K_pq; k++) {
                int ri = rng_m() % n;
                memcpy(&cb_m[k * sub_dim], &sub_vectors[ri * sub_dim], sub_dim*sizeof(float));
            }

            std::vector<int> sub_assign(n);
            std::vector<float> sum_cb(K_pq * sub_dim, 0.0f);
            std::vector<int> cnt_cb(K_pq, 0);

            for (int it = 0; it < pq_iter; it++) {
                for (int i = 0; i < n; i++) {
                    float best = std::numeric_limits<float>::max();
                    int best_k = 0;
                    for (int k = 0; k < K_pq; k++) {
                        float dist = 0;
                        for (int j = 0; j < sub_dim; j++) {
                            float diff = sub_vectors[i*sub_dim+j] - cb_m[k*sub_dim+j];
                            dist += diff * diff;
                        }
                        if (dist < best) { best = dist; best_k = k; }
                    }
                    sub_assign[i] = best_k;
                }
                std::fill(sum_cb.begin(), sum_cb.end(), 0.0f);
                std::fill(cnt_cb.begin(), cnt_cb.end(), 0);
                for (int i = 0; i < n; i++) {
                    int k = sub_assign[i];
                    cnt_cb[k]++;
                    for (int j = 0; j < sub_dim; j++)
                        sum_cb[k*sub_dim+j] += sub_vectors[i*sub_dim+j];
                }
                for (int k = 0; k < K_pq; k++) {
                    if (cnt_cb[k] == 0) continue;
                    float inv = 1.0f / cnt_cb[k];
                    for (int j = 0; j < sub_dim; j++)
                        cb_m[k*sub_dim+j] = sum_cb[k*sub_dim+j] * inv;
                }
            }
        }

        // Encode all vectors
        pq_codes.resize(n * M);
        for (int i = 0; i < n; i++) {
            for (int m = 0; m < M; m++) {
                const float* sub_v = base + i*d + m*sub_dim;
                const float* cb_m = codebooks.data() + m * K_pq * sub_dim;
                float best = std::numeric_limits<float>::max();
                uint8_t best_k = 0;
                for (int k = 0; k < K_pq; k++) {
                    float dist = 0;
                    for (int j = 0; j < sub_dim; j++) {
                        float diff = sub_v[j] - cb_m[k*sub_dim+j];
                        dist += diff * diff;
                    }
                    if (dist < best) { best = dist; best_k = (uint8_t)k; }
                }
                pq_codes[i * M + m] = best_k;
            }
        }
        fprintf(stderr, "[PQ] Encoding done.\n");
    }
};

// ========== GPU Kernels ==========

// Select top-nprobe clusters per query (largest inner product = closest)
// Input: scores from GEMM arranged as scores[qi * nlist + ci] = dot(centroid_ci, query_qi)
__global__ void select_probes_kernel(
    const float* scores, int* probes,
    int m, int nlist, int nprobe)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= m) return;

    const float* col_start = scores + (size_t)qid * nlist;
    int* out = probes + (size_t)qid * nprobe;

    // Select LARGEST nprobe inner products
    struct Pair { float ip; int id; };
    Pair heap[128];
    int hsize = 0;

    for (int c = 0; c < nlist; c++) {
        float ip = col_start[c];
        if (hsize < nprobe) {
            int pos = hsize;
            while (pos > 0 && heap[pos-1].ip > ip) { heap[pos] = heap[pos-1]; pos--; }
            heap[pos].ip = ip; heap[pos].id = c;
            hsize++;
        } else if (ip > heap[0].ip) {
            int pos = 0;
            while (pos < nprobe - 1 && heap[pos+1].ip < ip) { heap[pos] = heap[pos+1]; pos++; }
            while (pos > 0 && heap[pos-1].ip > ip) { heap[pos] = heap[pos-1]; pos--; }
            heap[pos].ip = ip; heap[pos].id = c;
        }
    }
    for (int i = 0; i < nprobe; i++) out[i] = heap[i].id;
}

// Build per-query LUT: LUT[q][m][k] = distance from q's m-th subvector to k-th centroid
// LUT layout: [m * batch_q * K_pq] row-major, SoA by subspace m
// For each m, we compute Q_sub[m*batch*d] x CB_sub^T[K_pq*sub_dim] -> [batch * K_pq]
// This is done via GEMM, not a kernel

// Build cluster->query map (same as IVF)
__global__ void build_query_cluster_map_kernel(
    const int* probes, int* cluster_counts, int* query_positions,
    int m, int nlist, int nprobe)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * nprobe;
    if (idx >= total) return;
    int qid = idx / nprobe;
    int pid = idx % nprobe;
    int cid = probes[(size_t)qid * nprobe + pid];
    int pos = atomicAdd(&cluster_counts[cid], 1);
    query_positions[(size_t)qid * nprobe + pid] = pos;
}

__global__ void scatter_query_ids_kernel(
    const int* probes, const int* query_positions,
    int* cluster_query_ids, int m, int nlist, int nprobe, int max_qpc)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * nprobe;
    if (idx >= total) return;
    int qid = idx / nprobe;
    int pid = idx % nprobe;
    int cid = probes[(size_t)qid * nprobe + pid];
    int pos = query_positions[(size_t)qid * nprobe + pid];
    if (pos < max_qpc)
        cluster_query_ids[(size_t)cid * max_qpc + pos] = qid;
}

// ADC kernel: for a batch of queries and a cluster, compute approximate distances via LUT
// grid = (num_queries, 1), block = (min(256, list_size), 1)
// Each block handles one query, each thread handles one vector
__global__ void adc_lookup_kernel(
    const float* lut,           // [batch_q * M * K_pq]
    const uint8_t* pq_codes,    // [n * M] indexed by ORIGINAL vector ID
    int list_offset, int list_size,
    const int* reordered_ids,   // maps reordered pos -> original ID
    int M, int K_pq,
    // Output
    float* out_scores,          // [num_q * list_size]
    int pitch)                   // list_size
{
    int q_local = blockIdx.x;
    int vi = threadIdx.x;
    if (vi >= list_size) return;

    const float* my_lut = lut + (size_t)q_local * M * K_pq;
    int original_id = reordered_ids[list_offset + vi];
    const uint8_t* code = pq_codes + (size_t)original_id * M;

    float dot = 0;
    for (int m = 0; m < M; m++) {
        dot += my_lut[m * K_pq + code[m]];
    }
    // Distance = 1.0 - dot for inner product
    out_scores[(size_t)q_local * pitch + vi] = 1.0f - dot;
}

// Exact rerank kernel: compute exact distances for top candidates
__global__ void rerank_kernel(
    const float* queries,       // [batch_q * d]
    const float* base,          // [n * d]
    const int* candidate_ids,   // [batch_q * max_rerank]
    int d, int max_rerank,
    const int* actual_counts,   // [batch_q] actual number of candidates per query
    float* out_dists)            // [batch_q * max_rerank]
{
    int qid = blockIdx.x;
    int actual = actual_counts[qid];

    // Initialize all output slots with sentinel
    for (int cid = threadIdx.x; cid < max_rerank; cid += blockDim.x) {
        out_dists[(size_t)qid * max_rerank + cid] = 1e30f;
    }
    __syncthreads();

    // Each thread processes a strided subset of candidates
    for (int cid = threadIdx.x; cid < actual; cid += blockDim.x) {
        int vec_id = candidate_ids[(size_t)qid * max_rerank + cid];
        const float* q = queries + (size_t)qid * d;
        const float* v = base + (size_t)vec_id * d;

        float dot = 0;
        for (int i = 0; i < d; i++) dot += q[i] * v[i];
        out_dists[(size_t)qid * max_rerank + cid] = 1.0f - dot;
    }
}

// ========== Timing ==========
double get_time_ms() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(now.time_since_epoch()).count();
}

// ========== Main ==========
int main(int argc, char* argv[]) {
    const char* data_dir = (argc > 1) ? argv[1] : "../../ANN_DATA/";
    int batch_size = (argc > 2) ? atoi(argv[2]) : 500;
    int nprobe = (argc > 3) ? atoi(argv[3]) : 50;
    int nlist = (argc > 4) ? atoi(argv[4]) : 1000;
    int M = (argc > 5) ? atoi(argv[5]) : 4;
    int K_pq = (argc > 6) ? atoi(argv[6]) : 256;
    int rerank_p = (argc > 7) ? atoi(argv[7]) : 200;
    const int k = 10;

    // Load data
    size_t base_n, base_d, query_n, query_d, gt_n, gt_d;
    float* h_base = load_fbin<float>(std::string(data_dir) + "DEEP100K.base.100k.fbin", base_n, base_d);
    float* h_query = load_fbin<float>(std::string(data_dir) + "DEEP100K.query.fbin", query_n, query_d);
    int*   h_gt   = load_fbin<int>(std::string(data_dir) + "DEEP100K.gt.query.100k.top100.bin", gt_n, gt_d);

    int m_total = (int)query_n;
    int d = (int)base_d;
    int n = (int)base_n;

    // Build IVFPQ index
    fprintf(stderr, "\n=== GPU IVF-PQ ===\n");
    fprintf(stderr, "Building index (nlist=%d, M=%d, K_pq=%d)...\n", nlist, M, K_pq);
    double t_build = get_time_ms();
    CPUIvfPqIndex index;
    index.build(h_base, n, d, nlist, M, K_pq, 15, 10);
    fprintf(stderr, "Build done in %.0f ms\n", get_time_ms() - t_build);

    // Upload index to GPU
    float *d_centroids, *d_reordered_base, *d_codebooks, *d_base_orig;
    int *d_reordered_ids, *d_list_offsets;
    uint8_t *d_pq_codes;

    CUDA_CHECK(cudaMalloc(&d_centroids, (size_t)nlist * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_reordered_base, (size_t)n * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_base_orig, (size_t)n * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_reordered_ids, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_list_offsets, (size_t)(nlist + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_codebooks, index.codebooks.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pq_codes, index.pq_codes.size() * sizeof(uint8_t)));

    // Upload IVF index
    CUDA_CHECK(cudaMemcpy(d_centroids, index.centroids.data(),
                          (size_t)nlist * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_reordered_base, index.reordered_base.data(),
                          (size_t)n * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_base_orig, h_base,
                          (size_t)n * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_reordered_ids, index.reordered_ids.data(),
                          (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_list_offsets, index.list_offsets.data(),
                          (size_t)(nlist + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_codebooks, index.codebooks.data(),
                          index.codebooks.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pq_codes, index.pq_codes.data(),
                          index.pq_codes.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));

    // Batch GPU buffers
    int sub_dim = d / M;
    float *d_queries, *d_coarse_scores, *d_lut, *d_adc_scores, *d_rerank_dists;
    int *d_probes, *d_cluster_counts, *d_query_positions, *d_cluster_query_ids;
    int *d_candidate_ids;

    int max_qpc = std::min(batch_size, batch_size * nprobe / 5 + 50);
    int max_list_size = 0;
    for (int c = 0; c < nlist; c++)
        max_list_size = std::max(max_list_size, index.list_offsets[c+1] - index.list_offsets[c]);

    CUDA_CHECK(cudaMalloc(&d_queries, (size_t)batch_size * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_coarse_scores, (size_t)nlist * batch_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lut, (size_t)batch_size * M * K_pq * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_adc_scores, (size_t)max_qpc * max_list_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rerank_dists, (size_t)batch_size * rerank_p * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_probes, (size_t)batch_size * nprobe * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cluster_counts, (size_t)nlist * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_query_positions, (size_t)batch_size * nprobe * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cluster_query_ids, (size_t)nlist * max_qpc * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_candidate_ids, (size_t)batch_size * rerank_p * sizeof(int)));
    int *d_actual_counts;
    CUDA_CHECK(cudaMalloc(&d_actual_counts, (size_t)batch_size * sizeof(int)));

    // cuBLAS
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    float alpha = 1.0f, beta = 0.0f;

    fprintf(stderr, "\n=== Search (nprobe=%d, M=%d, batch=%d, rerank_p=%d) ===\n",
            nprobe, M, batch_size, rerank_p);

    // Process all queries
    std::vector<int> all_ids(m_total * k);
    float total_recall = 0;
    float total_latency_us = 0;
    int num_batches = (m_total + batch_size - 1) / batch_size;

    for (int b = 0; b < num_batches; b++) {
        int m_cur = std::min(batch_size, m_total - b * batch_size);
        int q_offset = b * batch_size;

        CUDA_CHECK(cudaMemcpy(d_queries, h_query + (size_t)q_offset * d,
                              (size_t)m_cur * d * sizeof(float), cudaMemcpyHostToDevice));

        double t0 = get_time_ms();

        // Step 1: Coarse search via GEMM
        CUBLAS_CHECK(cublasSgemm(cublas_handle,
                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                  nlist, m_cur, d,
                                  &alpha, d_centroids, d, d_queries, d,
                                  &beta, d_coarse_scores, nlist));

        // Step 2: Select top-nprobe
        {
            int block = 256;
            int grid = (m_cur + block - 1) / block;
            select_probes_kernel<<<grid, block>>>(d_coarse_scores, d_probes, m_cur, nlist, nprobe);
        }

        // Step 3: Build PQ LUT on CPU (fast for batch <= 1000, avoids layout complexity)
        std::vector<float> h_lut(m_cur * M * K_pq);
        for (int qi = 0; qi < m_cur; qi++) {
            const float* q = h_query + (size_t)(q_offset + qi) * d;
            for (int m = 0; m < M; m++) {
                const float* sub_q = q + m * sub_dim;
                float* lut_m = h_lut.data() + ((size_t)qi * M + m) * K_pq;
                const float* cb_m = index.codebooks.data() + (size_t)m * K_pq * sub_dim;
                for (int k = 0; k < K_pq; k++) {
                    float dot = 0;
                    for (int j = 0; j < sub_dim; j++)
                        dot += sub_q[j] * cb_m[k * sub_dim + j];
                    lut_m[k] = dot;  // store inner product
                }
            }
        }
        CUDA_CHECK(cudaMemcpy(d_lut, h_lut.data(),
                              (size_t)m_cur * M * K_pq * sizeof(float),
                              cudaMemcpyHostToDevice));

        // Step 4: Build cluster->query map
        CUDA_CHECK(cudaMemset(d_cluster_counts, 0, (size_t)nlist * sizeof(int)));
        {
            int total = m_cur * nprobe;
            int block = 256;
            int grid = (total + block - 1) / block;
            build_query_cluster_map_kernel<<<grid, block>>>(
                d_probes, d_cluster_counts, d_query_positions, m_cur, nlist, nprobe);
            scatter_query_ids_kernel<<<grid, block>>>(
                d_probes, d_query_positions, d_cluster_query_ids, m_cur, nlist, nprobe, max_qpc);
        }

        std::vector<int> h_cluster_counts(nlist);
        CUDA_CHECK(cudaMemcpy(h_cluster_counts.data(), d_cluster_counts,
                              (size_t)nlist * sizeof(int), cudaMemcpyDeviceToHost));

        // Step 5: ADC lookup per cluster, then top-rerank_p selection per query (on CPU)
        // Per-query: maintain heap of (pq_dist, vec_id)
        std::vector<std::priority_queue<std::pair<float, int>>> per_query_pq(m_cur);

        for (int c = 0; c < nlist; c++) {
            int nq = h_cluster_counts[c];
            if (nq == 0) continue;
            int list_size = index.list_offsets[c+1] - index.list_offsets[c];
            if (list_size == 0) continue;

            int nq_clamped = std::min(nq, max_qpc);

            // ADC kernel for this cluster
            int block_dim = std::min(256, list_size);
            adc_lookup_kernel<<<nq_clamped, block_dim>>>(
                d_lut, d_pq_codes,
                index.list_offsets[c], list_size,
                d_reordered_ids, M, K_pq,
                d_adc_scores, list_size);

            // Copy scores back (only the needed portion)
            std::vector<float> h_adc(nq_clamped * list_size);
            CUDA_CHECK(cudaMemcpy(h_adc.data(), d_adc_scores,
                                  (size_t)nq_clamped * list_size * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            // Copy query IDs and vector IDs
            std::vector<int> h_qids(nq_clamped);
            CUDA_CHECK(cudaMemcpy(h_qids.data(),
                                  d_cluster_query_ids + (size_t)c * max_qpc,
                                  nq_clamped * sizeof(int), cudaMemcpyDeviceToHost));

            std::vector<int> h_vids(list_size);
            CUDA_CHECK(cudaMemcpy(h_vids.data(),
                                  d_reordered_ids + (size_t)index.list_offsets[c],
                                  list_size * sizeof(int), cudaMemcpyDeviceToHost));

            // Update per-query PQ heaps
            for (int qi = 0; qi < nq_clamped; qi++) {
                int gqid = h_qids[qi];
                auto& heap = per_query_pq[gqid];
                for (int vi = 0; vi < list_size; vi++) {
                    float dist = h_adc[(size_t)qi * list_size + vi];
                    if ((int)heap.size() < rerank_p) {
                        heap.push({dist, h_vids[vi]});
                    } else if (dist < heap.top().first) {
                        heap.pop();
                        heap.push({dist, h_vids[vi]});
                    }
                }
            }
        }

        // Step 6: Rerank - extract top candidates, compute exact distances
        std::vector<int> h_candidates(m_cur * rerank_p);
        std::vector<int> h_actual_cnt(m_cur);
        for (int qi = 0; qi < m_cur; qi++) {
            auto& heap = per_query_pq[qi];
            std::vector<std::pair<float, int>> tmp;
            while (!heap.empty()) { tmp.push_back(heap.top()); heap.pop(); }
            std::reverse(tmp.begin(), tmp.end());
            int cnt = std::min(rerank_p, (int)tmp.size());
            h_actual_cnt[qi] = cnt;
            for (int j = 0; j < cnt; j++)
                h_candidates[(size_t)qi * rerank_p + j] = tmp[j].second;
            for (int j = cnt; j < rerank_p; j++)
                h_candidates[(size_t)qi * rerank_p + j] = 0;
        }

        CUDA_CHECK(cudaMemcpy(d_candidate_ids, h_candidates.data(),
                              (size_t)m_cur * rerank_p * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_actual_counts, h_actual_cnt.data(),
                              (size_t)m_cur * sizeof(int),
                              cudaMemcpyHostToDevice));

        // Rerank kernel (uses original-order base for exact distance)
        {
            int block_dim = std::min(256, rerank_p);
            rerank_kernel<<<m_cur, block_dim>>>(
                d_queries, d_base_orig,
                d_candidate_ids, d, rerank_p, d_actual_counts, d_rerank_dists);
        }

        std::vector<float> h_rerank_dists(m_cur * rerank_p);
        CUDA_CHECK(cudaMemcpy(h_rerank_dists.data(), d_rerank_dists,
                              (size_t)m_cur * rerank_p * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // Step 7: Final top-k per query
        for (int qi = 0; qi < m_cur; qi++) {
            std::priority_queue<std::pair<float, int>> final_heap;
            int actual = h_actual_cnt[qi];
            for (int j = 0; j < actual; j++) {
                float dist = h_rerank_dists[(size_t)qi * rerank_p + j];
                int vid = h_candidates[(size_t)qi * rerank_p + j];
                if ((int)final_heap.size() < k) {
                    final_heap.push({dist, vid});
                } else if (dist < final_heap.top().first) {
                    final_heap.pop();
                    final_heap.push({dist, vid});
                }
            }
            std::vector<std::pair<float, int>> tmp;
            while (!final_heap.empty()) { tmp.push_back(final_heap.top()); final_heap.pop(); }
            std::reverse(tmp.begin(), tmp.end());
            for (int j = 0; j < k && j < (int)tmp.size(); j++)
                all_ids[(size_t)(q_offset + qi) * k + j] = tmp[j].second;
            for (int j = (int)tmp.size(); j < k; j++)
                all_ids[(size_t)(q_offset + qi) * k + j] = 0;
        }

        double t1 = get_time_ms();
        float batch_latency = (float)((t1 - t0) * 1000.0 / m_cur); // us/q
        total_latency_us += batch_latency * m_cur;

        // Compute batch recall
        size_t hits_batch = 0;
        for (int qi = 0; qi < m_cur; qi++) {
            std::set<int> gtset(h_gt + (q_offset + qi) * gt_d,
                                h_gt + (q_offset + qi) * gt_d + k);
            for (int j = 0; j < k; j++) {
                if (gtset.count(all_ids[(q_offset + qi) * k + j])) hits_batch++;
            }
        }
        total_recall += (float)hits_batch / (m_cur * k) * m_cur;

        if (b % 10 == 0)
            fprintf(stderr, "  Batch %d/%d: recall=%.4f latency=%.0f us/q\n",
                    b+1, num_batches, (float)hits_batch/(m_cur*k), batch_latency);
    }

    total_recall /= m_total;
    total_latency_us /= m_total;

    printf("\n| %-30s | %10s | %12s |\n", "Method", "Recall@10", "Latency(us/q)");
    printf("|%-32s|%12.5f|%14.1f|\n", "GPU-IVFPQ", total_recall, total_latency_us);
    fprintf(stderr, "\nGPU-IVFPQ: recall=%.5f avg_latency=%.1f us/q (%.2f ms/q)\n",
            total_recall, total_latency_us, total_latency_us / 1000.0f);

    // Write results
    std::ofstream res("result_tradeoff.md");
    res << "# GPU IVF-PQ Results\n\n";
    res << "## Configuration\n";
    res << "- GPU: RTX 4070 Laptop 8GB\n";
    res << "- Dataset: DEEP100K (n=" << n << ", d=" << d << ", queries=10K)\n";
    res << "- Distance: Inner Product (1.0 - dot)\n";
    res << "- nlist=" << nlist << ", nprobe=" << nprobe << "\n";
    res << "- PQ: M=" << M << ", K=" << K_pq << ", sub_dim=" << sub_dim << "\n";
    res << "- Batch size: " << batch_size << ", rerank_p: " << rerank_p << "\n\n";
    res << "## Results\n\n";
    res << "| Method | nprobe | PQ | rerank_p | Recall@10 | Avg Latency (us/q) | Avg Latency (ms/q) |\n";
    res << "|--------|--------|-----|----------|-----------|--------------------|--------------------|\n";
    res << "| GPU-IVFPQ | " << nprobe << " | M" << M << "K" << K_pq
        << " | " << rerank_p << " | " << std::fixed << std::setprecision(5) << total_recall
        << " | " << std::setprecision(1) << total_latency_us << " | "
        << std::setprecision(3) << total_latency_us / 1000.0f << " |\n";
    res << "\n## Analysis\n";
    res << "- PQ enables fast approximate distance via LUT lookup\n";
    res << "- Reranking with exact distances ensures recall quality\n";
    res << "- PQ reduces memory bandwidth requirements vs exact distances\n";
    res.close();

    // Cleanup
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUDA_CHECK(cudaFree(d_centroids)); CUDA_CHECK(cudaFree(d_reordered_base));
    CUDA_CHECK(cudaFree(d_base_orig));
    CUDA_CHECK(cudaFree(d_reordered_ids)); CUDA_CHECK(cudaFree(d_list_offsets));
    CUDA_CHECK(cudaFree(d_codebooks)); CUDA_CHECK(cudaFree(d_pq_codes));
    CUDA_CHECK(cudaFree(d_queries)); CUDA_CHECK(cudaFree(d_coarse_scores));
    CUDA_CHECK(cudaFree(d_lut)); CUDA_CHECK(cudaFree(d_adc_scores));
    CUDA_CHECK(cudaFree(d_rerank_dists)); CUDA_CHECK(cudaFree(d_probes));
    CUDA_CHECK(cudaFree(d_cluster_counts)); CUDA_CHECK(cudaFree(d_query_positions));
    CUDA_CHECK(cudaFree(d_cluster_query_ids)); CUDA_CHECK(cudaFree(d_candidate_ids));
    CUDA_CHECK(cudaFree(d_actual_counts));
    delete[] h_base; delete[] h_query; delete[] h_gt;

    return 0;
}

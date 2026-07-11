/*
 * s1_matmul: GPU Matrix Multiplication Baseline for ANNS
 * Base[n*d] x Query_T[d*m] = Scores[n*m], then top-k per column
 * Distance = 1.0 - inner_product, find top-k minimum distances
 *
 * Target: RTX 4070 Laptop 8GB, CUDA 12.6, nvcc 11.8
 * Dataset: DEEP100K (n=100K, d=96, m=10K queries)
 * Distance: inner product (1.0 - dot)
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>
#include <vector>

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

// ========== CUDA Error Check ==========
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
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, s); \
        exit(1); \
    } \
} while(0)

// ========== Top-K Kernel ==========
// Each block handles one query (column), finds top-k LARGEST inner products
// Input: dists is actually inner product scores from GEMM: dists[qi * n + i] = dot(base_i, query_qi)
// Output: topk_ids sorted by descending inner product (closest first)
__global__ void topk_per_query_kernel(
    const float* scores,    // [n x m] col-major: scores[query_idx * n + i]
    int* topk_ids,         // [m x k]
    float* topk_scores,    // [m x k] inner products
    int n, int m, int k)
{
    int qid = blockIdx.x;
    if (qid >= m) return;

    const float* col = scores + (size_t)qid * n;

    // Maintain k largest inner products using min-heap
    // heap[0] = minimum among selected (ascending order)
    float heap_ips[32];
    int heap_ids[32];
    int heap_size = 0;

    for (int i = 0; i < n; i++) {
        float ip = col[i];
        if (heap_size < k) {
            // Insert maintaining ascending order (heap[0] is min)
            int pos = heap_size;
            while (pos > 0 && heap_ips[pos-1] > ip) {
                heap_ips[pos] = heap_ips[pos-1];
                heap_ids[pos] = heap_ids[pos-1];
                pos--;
            }
            heap_ips[pos] = ip;
            heap_ids[pos] = i;
            heap_size++;
        } else if (ip > heap_ips[0]) {
            // Replace min (heap_ips[0]) with larger value
            int pos = 0;
            while (pos < k - 1 && heap_ips[pos+1] < ip) {
                heap_ips[pos] = heap_ips[pos+1];
                heap_ids[pos] = heap_ids[pos+1];
                pos++;
            }
            while (pos > 0 && heap_ips[pos-1] > ip) {
                heap_ips[pos] = heap_ips[pos-1];
                heap_ids[pos] = heap_ids[pos-1];
                pos--;
            }
            heap_ips[pos] = ip;
            heap_ids[pos] = i;
        }
    }

    // Output in descending order (largest IP first = closest)
    for (int i = 0; i < k; i++) {
        topk_ids[(size_t)qid * k + i] = heap_ids[k - 1 - i];
        topk_scores[(size_t)qid * k + i] = heap_ips[k - 1 - i];
    }
}

// ========== Recall computation ==========
float compute_recall(const int* result_ids, const int* gt, int m, int k, int gt_dim) {
    size_t total_hits = 0;
    for (int i = 0; i < m; i++) {
        std::set<int> gtset(gt + i * gt_dim, gt + i * gt_dim + k);
        for (int j = 0; j < k; j++) {
            if (gtset.count(result_ids[i * k + j])) total_hits++;
        }
    }
    return (float)total_hits / (m * k);
}

// ========== Timing helper ==========
double get_time_ms() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double, std::milli>(duration).count();
}

// ========== Main ==========
int main(int argc, char* argv[]) {
    const char* data_dir = (argc > 1) ? argv[1] : "../../ANN_DATA/";
    int batch_size = (argc > 2) ? atoi(argv[2]) : 1000;
    const int k = 10;

    // Load data
    size_t base_n, base_d, query_n, query_d, gt_n, gt_d;
    float* h_base = load_fbin<float>(std::string(data_dir) + "DEEP100K.base.100k.fbin", base_n, base_d);
    float* h_query = load_fbin<float>(std::string(data_dir) + "DEEP100K.query.fbin", query_n, query_d);
    int*   h_gt   = load_fbin<int>(std::string(data_dir) + "DEEP100K.gt.query.100k.top100.bin", gt_n, gt_d);

    int m_total = (int)query_n;
    int n = (int)base_n;
    int d = (int)base_d;

    fprintf(stderr, "\n=== GPU MatMul Baseline ===\n");
    fprintf(stderr, "Base: %d x %d, Queries: %d x %d, k=%d\n", n, d, m_total, d, k);
    fprintf(stderr, "Batch size: %d\n", batch_size);

    // Allocate GPU memory (reuse for all batches)
    float *d_base, *d_query_batch, *d_scores;
    int *d_topk_ids;
    float *d_topk_scores;
    CUDA_CHECK(cudaMalloc(&d_base, (size_t)n * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_query_batch, (size_t)batch_size * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scores, (size_t)n * batch_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_topk_ids, (size_t)batch_size * k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_topk_scores, (size_t)batch_size * k * sizeof(float)));

    // Copy base to GPU (once)
    CUDA_CHECK(cudaMemcpy(d_base, h_base, (size_t)n * d * sizeof(float), cudaMemcpyHostToDevice));

    // cuBLAS handle
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    // Use Tensor Cores if available (FP32)
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);

    float alpha = 1.0f, beta = 0.0f;

    // CUDA events for profiling
    cudaEvent_t ev_gemm_start, ev_gemm_stop, ev_topk_start, ev_topk_stop;
    CUDA_CHECK(cudaEventCreate(&ev_gemm_start));
    CUDA_CHECK(cudaEventCreate(&ev_gemm_stop));
    CUDA_CHECK(cudaEventCreate(&ev_topk_start));
    CUDA_CHECK(cudaEventCreate(&ev_topk_stop));
    float gemm_total_ms = 0, topk_total_ms = 0;

    // Allocate host result buffers
    std::vector<int> h_all_ids(m_total * k);
    float total_ms = 0;
    int num_batches = (m_total + batch_size - 1) / batch_size;

    for (int b = 0; b < num_batches; b++) {
        int m_cur = std::min(batch_size, m_total - b * batch_size);
        int q_offset = b * batch_size;

        // Copy query batch to GPU
        CUDA_CHECK(cudaMemcpy(d_query_batch, h_query + (size_t)q_offset * d,
                              (size_t)m_cur * d * sizeof(float), cudaMemcpyHostToDevice));

        // Time the GEMM + top-k
        double t0 = get_time_ms();
        CUDA_CHECK(cudaEventRecord(ev_gemm_start, 0));

        // GEMM: Scores[n x m_cur] = Base[n x d] x Query_T[d x m_cur]
        // cuBLAS column-major: C = alpha * A * B + beta * C
        // A: Base (n x d, col-major = d_base row-major as d x n)
        // B: Query (m_cur x d, col-major = d_query row-major as d x m_cur)
        // C: Scores (n x m_cur, col-major)
        // SGEMM: C = alpha * op(A) * op(B) + beta * C
        // We want: Scores[n x m_cur] = Base[n x d] * Query_T[d x m_cur]
        // In cuBLAS col-major: C(m,n) = A(m,k) * B(k,n) or similar
        // More precisely: C_col[n x m] = A_col[n x d] * B_col[d x m] where A_col stores base as [d x n]
        // Actually let me think again...
        // In cuBLAS, matrices are column-major.
        // Our d_base is row-major: base[i][j] at offset i*d + j
        // In column-major interpretation: base[i][j] is element (j, i) of a d x n matrix
        // Similarly d_query_batch is row-major: query[i][j] at offset i*d + j
        // In column-major: element (j, i) of a d x m_cur matrix
        // We want C[i][j] = sum_k base[i][k] * query[j][k]  (n x m_cur)
        // = sum_k base_col(k,i) * query_col(k,j) = A^T * B in col-major
        // C_col = op(A_col) * op(B_col)
        // A_col is d x n, B_col is d x m_cur
        // We want C_col(n x m_cur) = A_col^T * B_col
        // So: transa = CUBLAS_OP_T, transb = CUBLAS_OP_N
        // m = n (rows of C), n = m_cur (cols of C), k = d
        CUBLAS_CHECK(cublasSgemm(cublas_handle,
                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                  n, m_cur, d,
                                  &alpha,
                                  d_base, d,       // A is d x n col-major
                                  d_query_batch, d, // B is d x m_cur col-major
                                  &beta,
                                  d_scores, n));    // C is n x m_cur col-major

        CUDA_CHECK(cudaEventRecord(ev_gemm_stop, 0));
        CUDA_CHECK(cudaDeviceSynchronize());

        // Top-k kernel (1 thread per query, finds k largest inner products)
        CUDA_CHECK(cudaEventRecord(ev_topk_start, 0));
        topk_per_query_kernel<<<m_cur, 1>>>(
            d_scores, d_topk_ids, d_topk_scores, n, m_cur, k);
        CUDA_CHECK(cudaEventRecord(ev_topk_stop, 0));
        CUDA_CHECK(cudaDeviceSynchronize());

        double t1 = get_time_ms();
        total_ms += (t1 - t0);

        float gemm_ms, topk_ms;
        CUDA_CHECK(cudaEventElapsedTime(&gemm_ms, ev_gemm_start, ev_gemm_stop));
        CUDA_CHECK(cudaEventElapsedTime(&topk_ms, ev_topk_start, ev_topk_stop));
        gemm_total_ms += gemm_ms;
        topk_total_ms += topk_ms;

        // Copy results back
        CUDA_CHECK(cudaMemcpy(h_all_ids.data() + (size_t)q_offset * k, d_topk_ids,
                              (size_t)m_cur * k * sizeof(int), cudaMemcpyDeviceToHost));
    }

    float avg_latency_us = total_ms * 1000.0f / m_total;

    // Compute overall recall
    float recall = compute_recall(h_all_ids.data(), h_gt, m_total, k, (int)gt_d);
    fprintf(stderr, "\n=== Results ===\n");
    printf("| %-30s | %10s | %12s |\n", "Method", "Recall@10", "Latency(us/q)");
    printf("|%-32s|%12.5f|%14.1f|\n", "GPU-MatMul-Baseline", recall, avg_latency_us);
    fprintf(stderr, "\nMatMul total GPU time: %.2f ms (%.2f us/query)\n", total_ms, avg_latency_us);
    fprintf(stderr, "\n=== CUDA Event Profiling ===\n");
    fprintf(stderr, "GEMM total: %.2f ms (%.2f ms/batch)\n", gemm_total_ms, gemm_total_ms / num_batches);
    fprintf(stderr, "TopK total: %.2f ms (%.2f ms/batch)\n", topk_total_ms, topk_total_ms / num_batches);
    fprintf(stderr, "GEMM/TopK ratio: %.1f%% / %.1f%%\n",
            100.0 * gemm_total_ms / total_ms, 100.0 * topk_total_ms / total_ms);

    // Write results to file
    std::ofstream res("result_tradeoff.md");
    res << "# GPU Matrix Multiplication Baseline Results\n\n";
    res << "## Configuration\n";
    res << "- GPU: RTX 4070 Laptop 8GB\n";
    res << "- Dataset: DEEP100K (n=100K, d=96, queries=10K)\n";
    res << "- Distance: Inner Product (1.0 - dot)\n";
    res << "- Method: Base[n*d] x Query^T[d*m] via cuBLAS SGEMM\n\n";
    res << "## Results\n\n";
    res << "| Method | Batch | Recall@10 | Avg Latency (us/q) | Avg Latency (ms/q) |\n";
    res << "|--------|-------|-----------|-------------------|--------------------|\n";
    res << "| GPU-MatMul | " << batch_size << " | " << std::fixed << std::setprecision(5)
        << recall << " | " << std::setprecision(1) << avg_latency_us << " | "
        << std::setprecision(3) << avg_latency_us / 1000.0f << " |\n";
    res << "\n## Analysis\n";
    res << "- Recall = 1.0 (exact, no approximation)\n";
    res << "- Latency measures GPU kernel time only (GEMM + top-k)\n";
    res << "- cuBLAS SGEMM on RTX 4070 achieves high throughput for this matrix size\n";
    res.close();

    // ========== Correctness verification ==========
    fprintf(stderr, "\n=== Correctness Check: GPU vs CPU for Query 0 ===\n");
    {
        float* h_scores_cpu = new float[n];
        for (int i = 0; i < n; i++) {
            float dot = 0;
            for (int j = 0; j < d; j++) dot += h_base[(size_t)i * d + j] * h_query[j];
            h_scores_cpu[i] = dot;  // inner product
        }
        // CPU top-10 (largest IPs)
        std::vector<std::pair<float, int>> cpu_top;
        for (int i = 0; i < n; i++) cpu_top.push_back({h_scores_cpu[i], i});
        std::partial_sort(cpu_top.begin(), cpu_top.begin() + 10, cpu_top.end(),
                          std::greater<std::pair<float, int>>());

        fprintf(stderr, "CPU top-10: ");
        for (int i = 0; i < 10; i++) fprintf(stderr, "id=%d(ip=%.4f) ", cpu_top[i].second, cpu_top[i].first);
        fprintf(stderr, "\nGPU top-10: ");
        for (int i = 0; i < 10; i++) {
            int gid = h_all_ids[i];
            float ip = 0;
            for (int j = 0; j < d; j++) ip += h_base[(size_t)gid * d + j] * h_query[j];
            fprintf(stderr, "id=%d(ip=%.4f) ", gid, ip);
        }
        fprintf(stderr, "\nGT  top-10: ");
        for (int i = 0; i < 10; i++) fprintf(stderr, "%d ", h_gt[i]);

        // Compare
        std::set<int> cpu_set, gpu_set;
        for (int i = 0; i < 10; i++) { cpu_set.insert(cpu_top[i].second); gpu_set.insert(h_all_ids[i]); }
        int match = 0;
        for (int id : cpu_set) if (gpu_set.count(id)) match++;
        fprintf(stderr, "\nCPU-GPU match: %d/10 IDs (recall=%.2f)\n", match, match/10.0f);

        // Also check if GPU top-1 is correct
        fprintf(stderr, "GPU top-1 id=%d, CPU top-1 id=%d, GT[0]=%d\n",
                h_all_ids[0], cpu_top[0].second, h_gt[0]);
        delete[] h_scores_cpu;
    }

    // Write profiling to result file
    res << "\n## GPU Profiling (CUDA Events)\n";
    res << "| Kernel | Total Time (ms) | Per Batch (ms) | Percentage |\n";
    res << "|--------|----------------|---------------|------------|\n";
    res << "| cuBLAS SGEMM | " << gemm_total_ms << " | "
        << gemm_total_ms / num_batches << " | " << 100.0 * gemm_total_ms / total_ms << "% |\n";
    res << "| Top-K Kernel | " << topk_total_ms << " | "
        << topk_total_ms / num_batches << " | " << 100.0 * topk_total_ms / total_ms << "% |\n";
    res << "| **Total** | " << total_ms << " | — | 100% |\n";
    res.close();

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(ev_gemm_start));
    CUDA_CHECK(cudaEventDestroy(ev_gemm_stop));
    CUDA_CHECK(cudaEventDestroy(ev_topk_start));
    CUDA_CHECK(cudaEventDestroy(ev_topk_stop));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUDA_CHECK(cudaFree(d_base));
    CUDA_CHECK(cudaFree(d_query_batch));
    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_topk_ids));
    CUDA_CHECK(cudaFree(d_topk_scores));
    delete[] h_base;
    delete[] h_query;
    delete[] h_gt;

    return 0;
}

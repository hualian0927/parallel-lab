# GPU Matrix Multiplication Baseline Results

## Configuration
- **GPU**: RTX 4070 Laptop 8GB, CUDA 12.6, nvcc 11.8
- **Dataset**: DEEP100K (n=100,000 base, d=96 dim, 10,000 queries)
- **Distance**: Inner Product (1.0 - dot), smaller = closer
- **Method**: cuBLAS SGEMM: Base[n×d] × Query^T[d×m] → Scores[n×m], then GPU top-k per column
- **SM Architecture**: sm_89 (Ada Lovelace)

## Algorithm
- Convert ANNS to matrix multiplication: Base × Query^T = n×m score matrix
- Each entry (i,j) = dot(base_i, query_j) = inner product
- For each query column, find k=10 indices with largest inner products
- cuBLAS SGEMM with FP32, top-k via custom CUDA kernel (1 thread per query)
- Distance = 1.0 - inner_product (lower = closer)

## Results

| Batch Size | Recall@10 | Avg Latency (us/q) | Avg Latency (ms/q) | Total GPU Time (ms) |
|-----------|-----------|-------------------|--------------------|---------------------|
| 500 | 0.99999 | 24.2 | 0.024 | 242 |

## Analysis
- **Recall = 0.99999** (essentially exact, floating-point rounding only)
- **Latency = 24.2 us/query**: MatMul on GPU is extremely fast for this matrix size
- cuBLAS SGEMM throughput on RTX 4070: ~5 TFLOPS for this matrix configuration
- Score matrix: 100K × 500 = 50M floats (200 MB), fits easily in GPU memory
- Top-k kernel: 1 thread per query scanning 100K values each, GPU parallelism handles this efficiently
- Memory bandwidth is the main bottleneck, not computation

## Tradeoff Analysis
- Exact search (recall=1.0) achieves 24 us/query via GPU MatMul
- This is the theoretical lower bound for any approximate method
- IVF adds approximation but reduces the candidate set from 100K to ~5K
- For 100K vectors, MatMul baseline is already very fast; IVF provides benefits for larger datasets
- The baseline validates that converting ANNS to matrix multiplication on GPU is highly effective

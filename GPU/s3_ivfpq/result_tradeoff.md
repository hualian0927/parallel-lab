# GPU IVF-PQ Results

## Configuration
- **GPU**: RTX 4070 Laptop 8GB, CUDA 12.6, nvcc 11.8
- **Dataset**: DEEP100K (n=100,000 base, d=96 dim, 10,000 queries)
- **Distance**: Inner Product (1.0 - dot), smaller = closer
- **IVF Parameters**: nlist=1000, 15 k-means iterations
- **PQ Parameters**: M=4 subspaces, K_pq=256 centroids/subspace, sub_dim=24, 10 PQ iterations
- **Index Build Time**: ~70 seconds (IVF + PQ training, one-time offline cost)

## Algorithm
1. **Coarse search**: GEMM Q×C^T → select top-nprobe clusters (same as IVF)
2. **Build PQ LUT**: Per-query lookup table: LUT[q][m][k] = dot(query_sub_m, centroid_k)
3. **ADC (Asymmetric Distance Computation)**: Per cluster, approximate distance via LUT lookup
4. **Select candidates**: Top rerank_p candidates per query based on PQ distances
5. **Exact Rerank**: Compute exact inner product distances for selected candidates on GPU
6. **Top-k**: Select k=10 smallest distances from reranked candidates

## Results (nlist=1000, batch=500, nprobe=50)

| rerank_p | Recall@10 | Avg Latency (us/q) | Avg Latency (ms/q) |
|----------|-----------|--------------------|--------------------|
| 200 | 0.029 | 185.3 | 0.19 |
| 500 | 0.074 | 245.4 | 0.25 |
| 10000 | 0.979 | 3847.2 | 3.85 |

## Analysis

### Why PQ Performance is Poor on DEEP100K
1. **Dataset too small for PQ benefits**: 100K vectors × 96 dim = 38.4 MB fits entirely in GPU cache hierarchy. Exact distance computation via GEMM is already extremely fast.
2. **PQ approximation error too large**: M=4 with sub_dim=24 means each sub-vector is 24-dimensional, quantized to 256 centroids. The quantization error is significant.
3. **PQ ranking quality**: Top-200 PQ candidates capture only ~3% of true nearest neighbors. Top-500 captures ~7%.
4. **Rerank overhead**: To achieve IVF-level recall, rerank_p must be large enough to cover all candidates (~5000), negating any speed advantage.

### When IVF-PQ Becomes Beneficial
- **Larger datasets** (1M+ vectors): Compressed PQ codes save memory bandwidth
- **More subspaces** (M=8 or M=12): Better quantization quality at the cost of larger codes
- **Memory-constrained GPUs**: 8-bit codes vs 32-bit floats = 4x compression

### Comparison: IVF vs IVF-PQ on DEEP100K

| Method | Recall@10 | Latency (us/q) | Winner |
|--------|-----------|----------------|--------|
| GPU IVF (nprobe=50) | 0.979 | 168 | **IVF wins** |
| GPU IVF-PQ (rerank=200) | 0.029 | 185 | IVF-PQ loses |
| GPU IVF-PQ (rerank=500) | 0.074 | 245 | IVF-PQ loses |
| GPU IVF-PQ (rerank=10000) | 0.979 | 3847 | IVF-PQ loses |

### Key Lessons
1. **PQ adds overhead without benefit for small datasets**: For DEEP100K, exact IVF is strictly better
2. **PQ codebook quality matters**: M=4 is too coarse for 24-dim sub-vectors; M=8 (sub_dim=12) might improve but doubles code size
3. **Rerank threshold is critical**: Too small = poor recall, too large = no speed benefit
4. **GPU architecture consideration**: RTX 4070 has sufficient compute and bandwidth to handle exact distances for 100K vectors

## Bug Fixes During Development
1. **PQ code indexing**: PQ codes indexed by original vector ID, not reordered position (fixed ADC kernel)
2. **Rerank kernel thread count**: Initial version only handled 256 candidates per query; fixed with strided processing
3. **Cluster selection**: Inner product scores require selecting LARGEST values, not smallest (consistent with distance convention)
4. **Candidate padding**: Unused candidate slots filled with sentinel distance 1e30 to avoid polluting top-k

# CUDA GEMM — Three-Run Repeatability Report

## Method

- GPU: NVIDIA Tesla T4
- Precision: FP32
- Source benchmark: `src/cuda/benchmark_three_way.cu`
- Three benchmark executions: Run 01, Run 02, Run 03
- Each execution contains 20 samples per kernel and size
- Summary: median of three `kernel_avg_ms` values
- Speedup: ratio of median execution times

## Kernel-only results

| N | Naive ms | Shared ms | Register ms | Register vs Naive | Register vs Shared |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.020001 | 0.015135 | 0.012658 | 1.580x | 1.196x |
| 256 | 0.081341 | 0.056754 | 0.042332 | 1.921x | 1.341x |
| 512 | 0.471954 | 0.300436 | 0.208596 | 2.263x | 1.440x |
| 1024 | 4.293364 | 2.676893 | 1.853836 | 2.316x | 1.444x |

## Limitations

- These are three benchmark executions, not necessarily three independent GPU sessions.
- Small matrix timings vary substantially between runs.
- Batched CUDA Events measure a launch region, which can include scheduling gaps.
- N >= 512 uses sampled independent CPU reference checks, plus full output agreement between kernels.
- Further profiling is required to identify hardware bottlenecks.

# Naive vs Shared Memory GEMM — Comparison Run 01

## Date and hardware

- Date: 2026-11-22
- Environment: Google Colab
- GPU: NVIDIA Tesla T4
- Compute Capability: 7.5
- Driver: 580.82.07
- CUDA Toolkit: 13.0
- NVCC: 13.0.88

## Source

- Repository: Mons1412/cuda-matrix-multiplication-optimizer
- Branch: feature/shared-memory-gemm
- Benchmark source commit: 5e0e2a5
- Source: src/cuda/benchmark_compare.cu
- Compiler flags: -O3 -std=c++17 -arch=sm_75

## Configuration

- Precision: FP32
- Matrix sizes: 128, 256, 512, 1024
- Block size: 16x16
- Shared memory tile: 16x16
- Warm-up: 5
- Samples: 20 per method and size
- Kernel timing: CUDA Events
- End-to-end timing: std::chrono::steady_clock

## Validation

- N <= 256: Full double-precision reference checking
- N >= 512: 512 reference samples
- All output elements checked for finite values
- Full output agreement checked between both kernels
- All benchmark cases: PASS

## Interpretation

- N=1024 kernel speedup: approximately 1.602x
- N=1024 end-to-end speedup: approximately 1.262x
- End-to-end timing excludes memory allocation and host initialization
- Kernel event timing includes the measured launch region
- Results represent one comparison run
- Repeat runs before claiming stable performance

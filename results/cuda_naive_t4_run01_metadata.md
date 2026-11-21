# Naive CUDA GEMM — Benchmark Run 01

## Environment

- GPU: NVIDIA Tesla T4
- GPU architecture: Turing
- Compute capability: 7.5
- NVIDIA driver: 580.82.07
- CUDA Toolkit: 13.0
- NVCC: 13.0.88
- Source commit: 2fa6ae1
- Branch: feature/naive-cuda
- Compiler flags: -O3 -std=c++17 -arch=sm_75

## Configuration

- Precision: FP32
- Matrix sizes: 128, 256, 512, 1024
- Thread block: 16x16
- Warm-up: 5 iterations
- Measurements: 20 samples per size
- Kernel timing: CUDA Events
- End-to-end timing: CPU steady_clock

## Validation

- N <= 256: Full reference validation.
- N >= 512: 512 sampled elements and finite-output check.
- All reported cases: PASS.
- Maximum absolute error is measured over validated elements.

## Measurement limitations

- Kernel-only and end-to-end are separate measurement methods.
- End-to-end excludes allocation and host-side initialization.
- Results represent one Colab benchmark run.
- Additional runs are needed to assess variability.
- Do not present cross-machine CPU/GPU ratios as controlled speedups.

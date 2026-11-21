#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

void cuda_check(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << operation << ": "
                  << cudaGetErrorString(error) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

// Row-major: A[M,K] * B[K,N] = C[M,N]. One thread computes one C element.
__global__ void matmul_naive(const float* A, const float* B, float* C,
                             int M, int K, int N)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

bool run_case(int M, int K, int N)
{
    const size_t a_count = static_cast<size_t>(M) * K;
    const size_t b_count = static_cast<size_t>(K) * N;
    const size_t c_count = static_cast<size_t>(M) * N;
    const size_t a_bytes = a_count * sizeof(float);
    const size_t b_bytes = b_count * sizeof(float);
    const size_t c_bytes = c_count * sizeof(float);

    std::vector<float> A(a_count), B(b_count), C(c_count);
    std::vector<float> expected(c_count, 0.0f);

    // Reproducible values, including negatives and zeros.
    for (size_t i = 0; i < a_count; ++i)
        A[i] = (static_cast<int>(i % 11) - 5) * 0.25f;
    for (size_t i = 0; i < b_count; ++i)
        B[i] = (static_cast<int>((i * 7) % 13) - 6) * 0.125f;

    // Independent reference calculation using a double accumulator.
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k)
                sum += static_cast<double>(A[row * K + k]) * B[k * N + col];
            expected[row * N + col] = static_cast<float>(sum);
        }
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_A), a_bytes), "cudaMalloc A");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_B), b_bytes), "cudaMalloc B");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_C), c_bytes), "cudaMalloc C");
    cuda_check(cudaMemcpy(d_A, A.data(), a_bytes, cudaMemcpyHostToDevice), "H2D A");
    cuda_check(cudaMemcpy(d_B, B.data(), b_bytes, cudaMemcpyHostToDevice), "H2D B");
    cuda_check(cudaMemset(d_C, 0xFF, c_bytes), "initialize output with NaN-like bytes");

    const dim3 block(16, 16);
    const dim3 grid((N + block.x - 1) / block.x,
                    (M + block.y - 1) / block.y);
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cuda_check(cudaGetLastError(), "kernel launch");
    cuda_check(cudaDeviceSynchronize(), "kernel execution");
    cuda_check(cudaMemcpy(C.data(), d_C, c_bytes, cudaMemcpyDeviceToHost), "D2H C");

    bool passed = true;
    float max_abs_error = 0.0f;
    for (size_t i = 0; i < c_count; ++i) {
        const float error = std::fabs(C[i] - expected[i]);
        if (!std::isfinite(C[i]) || error > 1e-4f + 1e-4f * std::fabs(expected[i])) {
            if (passed)
                std::cerr << "[MISMATCH] " << M << 'x' << K << " * " << K << 'x' << N
                          << " index=" << i << " expected=" << expected[i]
                          << " actual=" << C[i] << '\n';
            passed = false;
        }
        max_abs_error = std::max(max_abs_error, error);
    }

    cuda_check(cudaFree(d_A), "cudaFree A");
    cuda_check(cudaFree(d_B), "cudaFree B");
    cuda_check(cudaFree(d_C), "cudaFree C");

    std::cout << (passed ? "[PASS] " : "[FAIL] ")
              << M << 'x' << K << " * " << K << 'x' << N
              << " max_abs_error=" << max_abs_error << '\n';
    return passed;
}

int main()
{
    cuda_check(cudaSetDevice(0), "cudaSetDevice");
    struct Shape { int M, K, N; };
    const Shape cases[] = {
        {1, 1, 1}, {2, 3, 4}, {7, 5, 9},
        {17, 19, 23}, {32, 33, 31}, {64, 65, 63}, {128, 128, 128}
    };

    bool all_passed = true;
    for (const Shape& shape : cases)
        if (!run_case(shape.M, shape.K, shape.N)) all_passed = false;

    std::cout << (all_passed ? "ALL CUDA TESTS PASSED\n" : "CUDA TESTS FAILED\n");
    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

// Stage 4: shared-memory tiled GEMM, row-major A[M,K] * B[K,N] = C[M,N].
// Each 16x16 thread block computes one 16x16 output tile.
constexpr int TILE = 16;

void cuda_check(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << operation << ": "
                  << cudaGetErrorString(error) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

__global__ void matmul_shared(const float* A, const float* B, float* C,
                              int M, int K, int N)
{
    __shared__ float tile_A[TILE][TILE];
    __shared__ float tile_B[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    for (int tile = 0; tile < (K + TILE - 1) / TILE; ++tile) {
        const int a_col = tile * TILE + tx;
        const int b_row = tile * TILE + ty;

        // Every thread participates, including those outside matrix bounds.
        // Out-of-range tile entries are zero-padded.
        tile_A[ty][tx] = (row < M && a_col < K)
            ? A[row * K + a_col] : 0.0f;
        tile_B[ty][tx] = (b_row < K && col < N)
            ? B[b_row * N + col] : 0.0f;

        // Do not read either tile until all threads finish loading.
        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            sum += tile_A[ty][k] * tile_B[k][tx];
        }

        // Do not overwrite tiles until all threads finish reading them.
        __syncthreads();
    }

    // Guard only the store, AFTER all barriers.
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

struct Shape { int M, K, N; };

bool run_case(const Shape& shape, bool nonbinary)
{
    const int M = shape.M;
    const int K = shape.K;
    const int N = shape.N;
    const size_t a_count = static_cast<size_t>(M) * K;
    const size_t b_count = static_cast<size_t>(K) * N;
    const size_t c_count = static_cast<size_t>(M) * N;

    std::vector<float> A(a_count), B(b_count), C(c_count, 0.0f);

    for (size_t i = 0; i < a_count; ++i) {
        A[i] = nonbinary
            ? std::sin(static_cast<float>(i + 1) * 0.17f)
            : (static_cast<int>(i % 11) - 5) * 0.25f;
    }
    for (size_t i = 0; i < b_count; ++i) {
        B[i] = nonbinary
            ? std::cos(static_cast<float>(i + 1) * 0.13f)
            : (static_cast<int>((i * 7) % 13) - 6) * 0.125f;
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_A), a_count * sizeof(float)), "cudaMalloc A");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_B), b_count * sizeof(float)), "cudaMalloc B");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_C), c_count * sizeof(float)), "cudaMalloc C");

    cuda_check(cudaMemcpy(d_A, A.data(), a_count * sizeof(float), cudaMemcpyHostToDevice), "H2D A");
    cuda_check(cudaMemcpy(d_B, B.data(), b_count * sizeof(float), cudaMemcpyHostToDevice), "H2D B");
    cuda_check(cudaMemset(d_C, 0xFF, c_count * sizeof(float)), "initialize output sentinel");

    const dim3 block(TILE, TILE);
    const dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_shared<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cuda_check(cudaGetLastError(), "launch shared matmul");
    cuda_check(cudaDeviceSynchronize(), "execute shared matmul");
    cuda_check(cudaMemcpy(C.data(), d_C, c_count * sizeof(float), cudaMemcpyDeviceToHost), "D2H C");

    constexpr double atol = 1.0e-4;
    constexpr double rtol = 1.0e-4;
    double max_abs_error = 0.0;
    bool passed = true;

    // Independent double-precision reference; checks EVERY output element.
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            double expected = 0.0;
            for (int k = 0; k < K; ++k) {
                expected += static_cast<double>(A[static_cast<size_t>(row) * K + k])
                          * static_cast<double>(B[static_cast<size_t>(k) * N + col]);
            }

            const size_t idx = static_cast<size_t>(row) * N + col;
            const double actual = static_cast<double>(C[idx]);
            const double error = std::fabs(actual - expected);
            max_abs_error = std::max(max_abs_error, error);

            if (!std::isfinite(actual) || error > atol + rtol * std::fabs(expected)) {
                if (passed) {
                    std::cerr << "[MISMATCH] " << M << 'x' << K << " * " << K << 'x' << N
                              << " mode=" << (nonbinary ? "nonbinary" : "binary")
                              << " row=" << row << " col=" << col
                              << " expected=" << expected << " actual=" << actual
                              << " abs_error=" << error << '\n';
                }
                passed = false;
            }
        }
    }

    cuda_check(cudaFree(d_A), "cudaFree A");
    cuda_check(cudaFree(d_B), "cudaFree B");
    cuda_check(cudaFree(d_C), "cudaFree C");

    std::cout << (passed ? "[PASS] " : "[FAIL] ")
              << (nonbinary ? "nonbinary " : "binary ")
              << M << 'x' << K << " * " << K << 'x' << N
              << " max_abs_error=" << std::scientific << std::setprecision(6)
              << max_abs_error << '\n';
    return passed;
}

int main()
{
    cuda_check(cudaSetDevice(0), "select CUDA device 0");
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, 0), "read device properties");
    std::cerr << "[INFO] GPU=" << prop.name << " CC=" << prop.major << '.' << prop.minor
              << " TILE=" << TILE << " threads/block=" << (TILE * TILE)
              << " shared_bytes/block=" << (2 * TILE * TILE * sizeof(float)) << '\n';

    const Shape cases[] = {
        {1, 1, 1}, {2, 3, 4}, {7, 5, 9}, {17, 19, 23},
        {32, 33, 31}, {64, 65, 63}, {128, 128, 128}, {129, 17, 131}
    };

    bool all_passed = true;
    int passed_count = 0;
    int test_count = 0;
    for (const Shape& shape : cases) {
        for (bool nonbinary : {false, true}) {
            const bool passed = run_case(shape, nonbinary);
            all_passed = all_passed && passed;
            passed_count += passed ? 1 : 0;
            ++test_count;
        }
    }

    std::cout << (all_passed ? "ALL SHARED MEMORY CUDA TESTS PASSED "
                             : "SHARED MEMORY CUDA TESTS FAILED ")
              << '(' << passed_count << '/' << test_count << ")\n";
    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

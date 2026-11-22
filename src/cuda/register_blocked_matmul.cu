#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

// Stage 5 correctness prototype: two output elements per CUDA thread.
// Row-major A[M,K] * B[K,N] = C[M,N].
// Block: 16 x 16 threads, output tile: 32 rows x 16 columns.
// Shared tiles: A[32,16] and B[16,16] => 3072 bytes per block.
constexpr int TILE_K = 16;
constexpr int TILE_N = 16;
constexpr int ROWS_PER_THREAD = 2;
constexpr int TILE_M = TILE_K * ROWS_PER_THREAD;

void cuda_check(cudaError_t err, const char* what)
{
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << what << ": "
                  << cudaGetErrorString(err) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

__global__ void matmul_register_2x1(const float* A, const float* B, float* C,
                                    int M, int K, int N)
{
    __shared__ float a_tile[TILE_M][TILE_K];
    __shared__ float b_tile[TILE_K][TILE_N];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int col = blockIdx.x * TILE_N + tx;
    const int row0 = blockIdx.y * TILE_M + ty;
    const int row1 = row0 + TILE_K;

    // Both accumulators are private to this CUDA thread (register candidates).
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int tile = 0; tile < (K + TILE_K - 1) / TILE_K; ++tile) {
        const int a_col = tile * TILE_K + tx;
        const int b_row = tile * TILE_K + ty;

        // Every thread loads two A values and one B value.
        // Zero padding protects partial tiles at the matrix boundaries.
        a_tile[ty][tx] = (row0 < M && a_col < K)
            ? A[row0 * K + a_col] : 0.0f;
        a_tile[ty + TILE_K][tx] = (row1 < M && a_col < K)
            ? A[row1 * K + a_col] : 0.0f;
        b_tile[ty][tx] = (b_row < K && col < N)
            ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_K; ++k) {
            const float b_value = b_tile[k][tx];
            acc0 += a_tile[ty][k] * b_value;
            acc1 += a_tile[ty + TILE_K][k] * b_value;
        }

        __syncthreads();
    }

    // All threads have already passed every barrier before the guarded stores.
    if (row0 < M && col < N) C[row0 * N + col] = acc0;
    if (row1 < M && col < N) C[row1 * N + col] = acc1;
}

struct Shape { int M, K, N; };

bool run_case(const Shape& shape, bool nonbinary)
{
    const int M = shape.M, K = shape.K, N = shape.N;
    const size_t a_count = static_cast<size_t>(M) * K;
    const size_t b_count = static_cast<size_t>(K) * N;
    const size_t c_count = static_cast<size_t>(M) * N;

    std::vector<float> A(a_count), B(b_count), C(c_count);

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
    cuda_check(cudaMemcpy(d_A, A.data(), a_count * sizeof(float), cudaMemcpyHostToDevice), "copy A");
    cuda_check(cudaMemcpy(d_B, B.data(), b_count * sizeof(float), cudaMemcpyHostToDevice), "copy B");
    cuda_check(cudaMemset(d_C, 0xFF, c_count * sizeof(float)), "initialize C sentinel");

    const dim3 block(TILE_N, TILE_K);
    const dim3 grid((N + TILE_N - 1) / TILE_N,
                    (M + TILE_M - 1) / TILE_M);
    matmul_register_2x1<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cuda_check(cudaGetLastError(), "launch register 2x1");
    cuda_check(cudaDeviceSynchronize(), "execute register 2x1");
    cuda_check(cudaMemcpy(C.data(), d_C, c_count * sizeof(float), cudaMemcpyDeviceToHost), "copy C");

    constexpr double atol = 1.0e-4;
    constexpr double rtol = 1.0e-4;
    double max_error = 0.0;
    bool passed = true;

    // Validate EVERY output against an independent double-accumulator reference.
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            double expected = 0.0;
            for (int k = 0; k < K; ++k) {
                expected += static_cast<double>(A[static_cast<size_t>(row) * K + k])
                          * static_cast<double>(B[static_cast<size_t>(k) * N + col]);
            }
            const size_t idx = static_cast<size_t>(row) * N + col;
            const double actual = C[idx];
            const double error = std::fabs(actual - expected);
            max_error = std::max(max_error, error);
            if (!std::isfinite(actual) || error > atol + rtol * std::fabs(expected)) {
                if (passed) {
                    std::cerr << "[MISMATCH] "
                              << (nonbinary ? "nonbinary " : "binary ")
                              << M << 'x' << K << " * " << K << 'x' << N
                              << " row=" << row << " col=" << col
                              << " expected=" << expected << " actual=" << actual
                              << " error=" << error << '\n';
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
              << " max_abs_error=" << std::scientific << std::setprecision(7)
              << max_error << '\n';
    return passed;
}

int main()
{
    cuda_check(cudaSetDevice(0), "select GPU 0");
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, 0), "read GPU properties");
    std::cerr << "[INFO] GPU=" << prop.name << " CC=" << prop.major << '.' << prop.minor
              << " output_tile=" << TILE_M << 'x' << TILE_N
              << " threads=" << (TILE_K * TILE_N)
              << " smem_bytes=" << ((TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(float))
              << " outputs/thread=" << ROWS_PER_THREAD << '\n';

    const Shape tests[] = {
        {1, 1, 1}, {2, 3, 4}, {7, 5, 9}, {17, 19, 23},
        {31, 31, 31}, {32, 33, 31}, {33, 33, 33}, {64, 65, 63},
        {128, 128, 128}, {129, 17, 131}
    };
    int passed_count = 0, total = 0;
    bool all_passed = true;
    for (const Shape& test : tests) {
        for (bool nonbinary : {false, true}) {
            const bool passed = run_case(test, nonbinary);
            passed_count += passed ? 1 : 0;
            all_passed = passed && all_passed;
            ++total;
        }
    }
    std::cout << (all_passed ? "ALL REGISTER BLOCKING TESTS PASSED "
                             : "REGISTER BLOCKING TESTS FAILED ")
              << '(' << passed_count << '/' << total << ")\n";
    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

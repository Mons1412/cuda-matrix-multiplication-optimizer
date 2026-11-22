#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

// Stage 5: same-session comparison of naive, 16x16 shared-memory,
// and 2x1 register-blocked (32x16 output tile) FP32 GEMM.
// Row-major A[N,N] * B[N,N] -> C[N,N]. CUDA default stream, sequential.
constexpr int TILE = 16;
constexpr int WARMUP = 5;
constexpr int SAMPLES = 20;
constexpr int METHODS = 3;

enum Method { NAIVE = 0, SHARED = 1, REGISTER_2X1 = 2 };
const char* const METHOD_NAMES[METHODS] = {"naive", "shared", "register_2x1"};

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << operation << ": "
                  << cudaGetErrorString(error) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

// Same algorithm as src/cuda/naive_matmul.cu.
__global__ void matmul_naive(const float* A, const float* B, float* C,
                             int M, int K, int N) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// Same algorithm as src/cuda/shared_matmul.cu.
__global__ void matmul_shared(const float* A, const float* B, float* C,
                              int M, int K, int N) {
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
        tile_A[ty][tx] = (row < M && a_col < K)
            ? A[row * K + a_col] : 0.0f;
        tile_B[ty][tx] = (b_row < K && col < N)
            ? B[b_row * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) {
            sum += tile_A[ty][k] * tile_B[k][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

// Same algorithm as src/cuda/register_blocked_matmul.cu.
__global__ void matmul_register_2x1(const float* A, const float* B, float* C,
                                    int M, int K, int N) {
    __shared__ float a_tile[2 * TILE][TILE];
    __shared__ float b_tile[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int col = blockIdx.x * TILE + tx;
    const int row0 = blockIdx.y * (2 * TILE) + ty;
    const int row1 = row0 + TILE;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int tile = 0; tile < (K + TILE - 1) / TILE; ++tile) {
        const int a_col = tile * TILE + tx;
        const int b_row = tile * TILE + ty;
        a_tile[ty][tx] = (row0 < M && a_col < K)
            ? A[row0 * K + a_col] : 0.0f;
        a_tile[ty + TILE][tx] = (row1 < M && a_col < K)
            ? A[row1 * K + a_col] : 0.0f;
        b_tile[ty][tx] = (b_row < K && col < N)
            ? B[b_row * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) {
            const float b_value = b_tile[k][tx];
            acc0 += a_tile[ty][k] * b_value;
            acc1 += a_tile[ty + TILE][k] * b_value;
        }
        __syncthreads();
    }
    if (row0 < M && col < N) C[row0 * N + col] = acc0;
    if (row1 < M && col < N) C[row1 * N + col] = acc1;
}

void launch(Method method, const float* A, const float* B,
            float* C, int n) {
    const dim3 block(TILE, TILE);
    const dim3 grid((n + TILE - 1) / TILE,
        (n + (method == REGISTER_2X1 ? 2 * TILE : TILE) - 1)
            / (method == REGISTER_2X1 ? 2 * TILE : TILE));
    switch (method) {
        case NAIVE:
            matmul_naive<<<grid, block>>>(A, B, C, n, n, n);
            break;
        case SHARED:
            matmul_shared<<<grid, block>>>(A, B, C, n, n, n);
            break;
        case REGISTER_2X1:
            matmul_register_2x1<<<grid, block>>>(A, B, C, n, n, n);
            break;
    }
    check(cudaGetLastError(), "kernel launch");
}

struct Stats { double avg_ms, median_ms; };
Stats summarize(std::vector<double> values) {
    const double avg = std::accumulate(values.begin(), values.end(), 0.0)
                     / static_cast<double>(values.size());
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    const double med = values.size() % 2
        ? values[mid] : (values[mid - 1] + values[mid]) / 2.0;
    return {avg, med};
}

double gflops(int n, double time_ms) {
    return (2.0 * static_cast<double>(n) * n * n) / (time_ms * 1.0e6);
}

void validate(int n, const std::vector<float>& A,
              const std::vector<float>& B,
              const std::array<std::vector<float>, METHODS>& outputs) {
    const size_t count = static_cast<size_t>(n) * n;
    const size_t checks = n <= 256 ? count : 512;
    constexpr double atol = 1.0e-4;
    constexpr double rtol = 1.0e-4;
    std::array<double, METHODS> max_abs{};

    for (size_t idx = 0; idx < count; ++idx) {
        for (int m = 0; m < METHODS; ++m) {
            const double actual = outputs[m][idx];
            if (!std::isfinite(actual)) {
                std::cerr << "[FAIL] non-finite N=" << n << " method="
                          << METHOD_NAMES[m] << " index=" << idx << '\n';
                std::exit(EXIT_FAILURE);
            }
            if (m != 0) {
                const double baseline = outputs[NAIVE][idx];
                if (std::fabs(actual - baseline) > atol + rtol * std::fabs(baseline)) {
                    std::cerr << "[FAIL] disagreement N=" << n
                              << " index=" << idx << " method=" << METHOD_NAMES[m]
                              << " naive=" << baseline << " actual=" << actual << '\n';
                    std::exit(EXIT_FAILURE);
                }
            }
        }
    }

    for (size_t s = 0; s < checks; ++s) {
        const size_t idx = n <= 256 ? s : (s * 7919ULL) % count;
        const size_t row = idx / n, col = idx % n;
        double expected = 0.0;
        for (int k = 0; k < n; ++k) {
            expected += static_cast<double>(A[row * n + k])
                      * static_cast<double>(B[static_cast<size_t>(k) * n + col]);
        }
        for (int m = 0; m < METHODS; ++m) {
            const double actual = outputs[m][idx];
            const double error = std::fabs(actual - expected);
            max_abs[m] = std::max(max_abs[m], error);
            if (error > atol + rtol * std::fabs(expected)) {
                std::cerr << "[FAIL] reference N=" << n << " index=" << idx
                          << " method=" << METHOD_NAMES[m]
                          << " expected=" << expected << " actual=" << actual
                          << " abs_error=" << error << '\n';
                std::exit(EXIT_FAILURE);
            }
        }
    }
    std::cerr << "[PASS] validation N=" << n
              << " reference_checked=" << checks << '/' << count
              << " full_output_agreement=true";
    for (int m = 0; m < METHODS; ++m) {
        std::cerr << " max_abs_" << METHOD_NAMES[m] << '=' << max_abs[m];
    }
    std::cerr << '\n';
}

void run_size(int n) {
    const size_t count = static_cast<size_t>(n) * n;
    const size_t bytes = count * sizeof(float);
    const int batch = n <= 256 ? 50 : (n <= 512 ? 10 : 3);

    std::vector<float> A(count), B(count);
    for (size_t i = 0; i < count; ++i) {
        A[i] = std::sin(static_cast<float>((i % 4096) + 1) * 0.17f);
        B[i] = std::cos(static_cast<float>((i % 4096) + 1) * 0.13f);
    }
    std::array<std::vector<float>, METHODS> outputs = {
        std::vector<float>(count), std::vector<float>(count),
        std::vector<float>(count)
    };

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&d_A), bytes), "cudaMalloc A");
    check(cudaMalloc(reinterpret_cast<void**>(&d_B), bytes), "cudaMalloc B");
    check(cudaMalloc(reinterpret_cast<void**>(&d_C), bytes), "cudaMalloc C");
    check(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice), "initial H2D A");
    check(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice), "initial H2D B");

    for (int m = 0; m < METHODS; ++m) {
        check(cudaMemset(d_C, 0xFF, bytes), "initialize output sentinel");
        for (int i = 0; i < WARMUP; ++i)
            launch(static_cast<Method>(m), d_A, d_B, d_C, n);
        check(cudaDeviceSynchronize(), "validation warmup sync");
        check(cudaMemcpy(outputs[m].data(), d_C, bytes, cudaMemcpyDeviceToHost),
              "D2H validation");
    }
    validate(n, A, B, outputs);

    std::array<std::vector<double>, METHODS> kernel_times;
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "create start event");
    check(cudaEventCreate(&stop), "create stop event");

    // Rotate the measurement order between samples: N/S/R, S/R/N, R/N/S.
    for (int sample = 0; sample < SAMPLES; ++sample) {
        for (int order = 0; order < METHODS; ++order) {
            const int m = (sample + order) % METHODS;
            check(cudaEventRecord(start), "event start");
            for (int j = 0; j < batch; ++j)
                launch(static_cast<Method>(m), d_A, d_B, d_C, n);
            check(cudaEventRecord(stop), "event stop");
            check(cudaEventSynchronize(stop), "event synchronize");
            float elapsed_ms = 0.0f;
            check(cudaEventElapsedTime(&elapsed_ms, start, stop), "event elapsed");
            kernel_times[m].push_back(static_cast<double>(elapsed_ms) / batch);
        }
    }
    check(cudaEventDestroy(start), "destroy start event");
    check(cudaEventDestroy(stop), "destroy stop event");

    // Host wall-clock end-to-end excludes allocation and CPU input initialization.
    // It includes H2D A/B + 1 kernel launch + D2H C + sync.
    using Clock = std::chrono::steady_clock;
    std::array<std::vector<double>, METHODS> e2e_times;
    for (int m = 0; m < METHODS; ++m) {
        for (int i = 0; i < WARMUP; ++i) {
            check(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice), "e2e warmup H2D A");
            check(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice), "e2e warmup H2D B");
            launch(static_cast<Method>(m), d_A, d_B, d_C, n);
            check(cudaMemcpy(outputs[m].data(), d_C, bytes, cudaMemcpyDeviceToHost), "e2e warmup D2H");
        }
    }
    check(cudaDeviceSynchronize(), "e2e warmup sync");

    for (int sample = 0; sample < SAMPLES; ++sample) {
        for (int order = 0; order < METHODS; ++order) {
            const int m = (sample + order) % METHODS;
            const auto before = Clock::now();
            check(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice), "e2e H2D A");
            check(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice), "e2e H2D B");
            launch(static_cast<Method>(m), d_A, d_B, d_C, n);
            check(cudaMemcpy(outputs[m].data(), d_C, bytes, cudaMemcpyDeviceToHost), "e2e D2H");
            check(cudaDeviceSynchronize(), "e2e sync");
            const auto after = Clock::now();
            e2e_times[m].push_back(
                std::chrono::duration<double, std::milli>(after - before).count());
        }
    }

    std::array<Stats, METHODS> kernel_stats = {
        summarize(kernel_times[0]), summarize(kernel_times[1]),
        summarize(kernel_times[2])
    };
    std::array<Stats, METHODS> e2e_stats = {
        summarize(e2e_times[0]), summarize(e2e_times[1]),
        summarize(e2e_times[2])
    };

    for (int m = 0; m < METHODS; ++m) {
        std::cout << n << ',' << METHOD_NAMES[m] << ',' << WARMUP
                  << ',' << SAMPLES << ',' << batch
                  << ',' << kernel_stats[m].avg_ms << ',' << kernel_stats[m].median_ms
                  << ',' << gflops(n, kernel_stats[m].avg_ms)
                  << ',' << e2e_stats[m].avg_ms << ',' << e2e_stats[m].median_ms
                  << ',' << gflops(n, e2e_stats[m].avg_ms)
                  << ',' << kernel_stats[NAIVE].avg_ms / kernel_stats[m].avg_ms
                  << ',' << kernel_stats[SHARED].avg_ms / kernel_stats[m].avg_ms
                  << ',' << e2e_stats[NAIVE].avg_ms / e2e_stats[m].avg_ms
                  << ",PASS\n";
    }

    check(cudaFree(d_A), "cudaFree A");
    check(cudaFree(d_B), "cudaFree B");
    check(cudaFree(d_C), "cudaFree C");
}

int main() {
    check(cudaSetDevice(0), "select device");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0), "get device properties");
    std::cerr << "[INFO] GPU=" << prop.name
              << " CC=" << prop.major << '.' << prop.minor
              << " FP32, block=16x16, register output tile=32x16\n";
    std::cerr << "[INFO] N<=256 full independent double reference;"
                 " N>=512 512 reference samples + full 3-kernel agreement\n";
    std::cout << std::fixed << std::setprecision(9);
    std::cout << "N,method,warmup,samples,batch,kernel_avg_ms,kernel_median_ms,"
                 "kernel_gflops,e2e_avg_ms,e2e_median_ms,e2e_gflops,"
                 "kernel_speedup_vs_naive,kernel_speedup_vs_shared,"
                 "e2e_speedup_vs_naive,validation\n";
    for (int n : {128, 256, 512, 1024}) run_size(n);
    return EXIT_SUCCESS;
}

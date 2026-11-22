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

// Same-session controlled comparison. Both kernels are copied from the project's
// standalone Naive and Shared Memory implementations at Stage 4.
constexpr int TILE = 16;
constexpr int WARMUP = 5;
constexpr int SAMPLES = 20;

enum class Method { Naive = 0, Shared = 1 };

void check(cudaError_t err, const char* op) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << op << ": " << cudaGetErrorString(err) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

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

void launch(Method method, const float* A, const float* B, float* C,
            int n, dim3 grid, dim3 block) {
    if (method == Method::Naive)
        matmul_naive<<<grid, block>>>(A, B, C, n, n, n);
    else
        matmul_shared<<<grid, block>>>(A, B, C, n, n, n);
    check(cudaGetLastError(), "kernel launch");
}

struct Stats { double avg; double median; };
Stats summarize(std::vector<double> values) {
    const double avg = std::accumulate(values.begin(), values.end(), 0.0)
                     / static_cast<double>(values.size());
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    return {avg, values.size() % 2 ? values[mid]
                                 : (values[mid - 1] + values[mid]) / 2.0};
}

double gflops(int n, double ms) {
    return 2.0 * static_cast<double>(n) * n * n / (ms * 1.0e6);
}

void validate_outputs(int n, const std::vector<float>& A,
                      const std::vector<float>& B,
                      const std::array<std::vector<float>, 2>& C) {
    const size_t elements = static_cast<size_t>(n) * n;
    const size_t checks = n <= 256 ? elements : 512;
    constexpr double atol = 1.0e-4, rtol = 1.0e-4;
    const char* names[2] = {"naive", "shared"};
    double max_abs[2] = {0.0, 0.0};

    for (size_t idx = 0; idx < elements; ++idx) {
        for (int m = 0; m < 2; ++m) {
            if (!std::isfinite(C[m][idx])) {
                std::cerr << "[FAIL] " << names[m] << ": non-finite N=" << n
                          << " index=" << idx << '\n';
                std::exit(EXIT_FAILURE);
            }
        }
        const double err = std::fabs(static_cast<double>(C[0][idx]) - C[1][idx]);
        const double tol = atol + rtol * std::fabs(static_cast<double>(C[0][idx]));
        if (err > tol) {
            std::cerr << "[FAIL] kernel disagreement N=" << n << " index=" << idx
                      << " naive=" << C[0][idx] << " shared=" << C[1][idx] << '\n';
            std::exit(EXIT_FAILURE);
        }
    }

    for (size_t s = 0; s < checks; ++s) {
        const size_t idx = n <= 256 ? s : ((s * 7919ULL) % elements);
        const size_t row = idx / n, col = idx % n;
        double ref = 0.0;
        for (int k = 0; k < n; ++k) {
            ref += static_cast<double>(A[row * n + k])
                 * static_cast<double>(B[static_cast<size_t>(k) * n + col]);
        }
        for (int m = 0; m < 2; ++m) {
            const double error = std::fabs(static_cast<double>(C[m][idx]) - ref);
            max_abs[m] = std::max(max_abs[m], error);
            if (error > atol + rtol * std::fabs(ref)) {
                std::cerr << "[FAIL] " << names[m] << " N=" << n
                          << " index=" << idx << " expected=" << ref
                          << " actual=" << C[m][idx] << " abs_error=" << error << '\n';
                std::exit(EXIT_FAILURE);
            }
        }
    }
    std::cerr << "[PASS] validation N=" << n << " all_outputs_finite_and_agree=true"
              << " reference_checked=" << checks << '/' << elements
              << " max_abs_naive=" << max_abs[0]
              << " max_abs_shared=" << max_abs[1] << '\n';
}

void run_size(int n) {
    const size_t elements = static_cast<size_t>(n) * n;
    const size_t bytes = elements * sizeof(float);
    const int batch = n <= 256 ? 50 : (n <= 512 ? 10 : 3);
    std::vector<float> A(elements), B(elements);
    for (size_t i = 0; i < elements; ++i) {
        // Identical input generator to the Stage 3 baseline benchmark.
        A[i] = std::sin(static_cast<float>((i % 4096) + 1) * 0.17f);
        B[i] = std::cos(static_cast<float>((i % 4096) + 1) * 0.13f);
    }
    std::array<std::vector<float>, 2> outputs = {
        std::vector<float>(elements), std::vector<float>(elements)
    };

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&d_A), bytes), "cudaMalloc A");
    check(cudaMalloc(reinterpret_cast<void**>(&d_B), bytes), "cudaMalloc B");
    check(cudaMalloc(reinterpret_cast<void**>(&d_C), bytes), "cudaMalloc C");
    check(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice), "H2D A");
    check(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice), "H2D B");

    const dim3 block(TILE, TILE);
    const dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    for (int method = 0; method < 2; ++method) {
        check(cudaMemset(d_C, 0xFF, bytes), "sentinel output");
        for (int i = 0; i < WARMUP; ++i)
            launch(static_cast<Method>(method), d_A, d_B, d_C, n, grid, block);
        check(cudaDeviceSynchronize(), "warmup sync");
        check(cudaMemcpy(outputs[method].data(), d_C, bytes, cudaMemcpyDeviceToHost),
              "D2H correctness");
    }
    validate_outputs(n, A, B, outputs);

    std::array<std::vector<double>, 2> kernel_times;
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "create event start");
    check(cudaEventCreate(&stop), "create event stop");

    // Alternate A/B ordering between samples to reduce systematic clock drift.
    for (int sample = 0; sample < SAMPLES; ++sample) {
        for (int order = 0; order < 2; ++order) {
            const int m = (sample + order) % 2;
            check(cudaEventRecord(start), "record event start");
            for (int j = 0; j < batch; ++j)
                launch(static_cast<Method>(m), d_A, d_B, d_C, n, grid, block);
            check(cudaEventRecord(stop), "record event stop");
            check(cudaEventSynchronize(stop), "sync event stop");
            float elapsed_ms = 0.0f;
            check(cudaEventElapsedTime(&elapsed_ms, start, stop), "event elapsed");
            kernel_times[m].push_back(static_cast<double>(elapsed_ms) / batch);
        }
    }
    check(cudaEventDestroy(start), "destroy start event");
    check(cudaEventDestroy(stop), "destroy stop event");

    using Clock = std::chrono::steady_clock;
    std::array<std::vector<double>, 2> e2e_times;
    for (int method = 0; method < 2; ++method) {
        for (int i = 0; i < WARMUP; ++i) {
            check(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice), "H2D warmup A");
            check(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice), "H2D warmup B");
            launch(static_cast<Method>(method), d_A, d_B, d_C, n, grid, block);
            check(cudaMemcpy(outputs[method].data(), d_C, bytes, cudaMemcpyDeviceToHost), "D2H warmup");
        }
    }
    check(cudaDeviceSynchronize(), "end-to-end warmup sync");

    for (int sample = 0; sample < SAMPLES; ++sample) {
        for (int order = 0; order < 2; ++order) {
            const int m = (sample + order) % 2;
            const auto begin = Clock::now();
            check(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice), "H2D measure A");
            check(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice), "H2D measure B");
            launch(static_cast<Method>(m), d_A, d_B, d_C, n, grid, block);
            check(cudaMemcpy(outputs[m].data(), d_C, bytes, cudaMemcpyDeviceToHost), "D2H measure");
            check(cudaDeviceSynchronize(), "end-to-end sync");
            const auto end = Clock::now();
            e2e_times[m].push_back(std::chrono::duration<double, std::milli>(end - begin).count());
        }
    }

    std::array<Stats, 2> kernel_stats = {summarize(kernel_times[0]), summarize(kernel_times[1])};
    std::array<Stats, 2> e2e_stats = {summarize(e2e_times[0]), summarize(e2e_times[1])};
    for (int m = 0; m < 2; ++m) {
        std::cout << n << ',' << (m == 0 ? "naive" : "shared") << ','
                  << WARMUP << ',' << SAMPLES << ',' << batch << ','
                  << kernel_stats[m].avg << ',' << kernel_stats[m].median << ','
                  << gflops(n, kernel_stats[m].avg) << ','
                  << e2e_stats[m].avg << ',' << e2e_stats[m].median << ','
                  << gflops(n, e2e_stats[m].avg) << ','
                  << kernel_stats[0].avg / kernel_stats[m].avg << ','
                  << e2e_stats[0].avg / e2e_stats[m].avg << ",PASS\n";
    }

    check(cudaFree(d_A), "cudaFree A");
    check(cudaFree(d_B), "cudaFree B");
    check(cudaFree(d_C), "cudaFree C");
}

int main() {
    check(cudaSetDevice(0), "cudaSetDevice");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    std::cerr << "[INFO] GPU=" << prop.name << " CC=" << prop.major << '.' << prop.minor
              << " TILE=" << TILE << " default_stream=sequential\n";
    std::cerr << "[INFO] FP32 row-major; N=128,256,512,1024; correctness before timing\n";
    std::cerr << "[INFO] N<=256: full double reference; N>=512: 512 reference samples + full naive/shared agreement\n";
    std::cout << std::fixed << std::setprecision(9);
    std::cout << "N,method,warmup,samples,batch,kernel_avg_ms,kernel_median_ms,"
                 "kernel_gflops,e2e_avg_ms,e2e_median_ms,e2e_gflops,"
                 "kernel_speedup_vs_naive,e2e_speedup_vs_naive,validation\n";
    for (int n : {128, 256, 512, 1024}) run_size(n);
    return EXIT_SUCCESS;
}

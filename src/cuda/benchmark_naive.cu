#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

// CUDA Stage 3: correctness-checked naive GEMM benchmark.
// Row-major: A[N,N] * B[N,N] = C[N,N].
// Kernel timing: CUDA events around batches of launches, excluding transfers.
// End-to-end timing: host wall time for H2D A/B + one kernel + D2H C.
// Neither measurement includes cudaMalloc/cudaFree or host initialization.

void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << operation << ": "
                  << cudaGetErrorString(error) << '\n';
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

struct Statistics {
    double avg_ms;
    double median_ms;
};

Statistics summarize(std::vector<double> times_ms) {
    const double avg = std::accumulate(times_ms.begin(), times_ms.end(), 0.0)
                       / static_cast<double>(times_ms.size());
    std::sort(times_ms.begin(), times_ms.end());
    const size_t mid = times_ms.size() / 2;
    const double median = times_ms.size() % 2 == 0
        ? (times_ms[mid - 1] + times_ms[mid]) / 2.0
        : times_ms[mid];
    return {avg, median};
}

double gflops(int n, double milliseconds) {
    const double operations = 2.0 * n * n * n;
    return operations / (milliseconds * 1.0e6);
}

void run_size(int n) {
    constexpr int kWarmup = 5;
    constexpr int kIterations = 20;
    const int batch = n <= 256 ? 50 : (n <= 512 ? 10 : 3);
    const size_t elements = static_cast<size_t>(n) * n;
    const size_t bytes = elements * sizeof(float);

    std::vector<float> A(elements), B(elements), C(elements);
    for (size_t i = 0; i < elements; ++i) {
        // Reproducible, non-exact-binary inputs; modest magnitudes.
        A[i] = std::sin(static_cast<float>((i % 4096) + 1) * 0.17f);
        B[i] = std::cos(static_cast<float>((i % 4096) + 1) * 0.13f);
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_A), bytes), "cudaMalloc(A)");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_B), bytes), "cudaMalloc(B)");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_C), bytes), "cudaMalloc(C)");
    cuda_check(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice), "H2D(A)");
    cuda_check(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice), "H2D(B)");
    cuda_check(cudaMemset(d_C, 0xFF, bytes), "initialize output sentinel");

    const dim3 block(16, 16);
    const dim3 grid((n + block.x - 1) / block.x,
                    (n + block.y - 1) / block.y);

    for (int i = 0; i < kWarmup; ++i) {
        matmul_naive<<<grid, block>>>(d_A, d_B, d_C, n, n, n);
        cuda_check(cudaGetLastError(), "warmup launch");
    }
    cuda_check(cudaDeviceSynchronize(), "warmup sync");
    cuda_check(cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost),
               "D2H(validation)");

    // Full mathematical reference for N <= 256; deterministic sample for larger N.
    // Larger N are not exhaustively correctness-checked in this benchmark.
    const size_t checks = n <= 256 ? elements : 512;
    double max_abs_error = 0.0;
    for (size_t s = 0; s < checks; ++s) {
        const size_t idx = n <= 256 ? s : ((s * 7919ULL) % elements);
        const size_t row = idx / n;
        const size_t col = idx % n;
        double expected = 0.0;
        for (int k = 0; k < n; ++k) {
            expected += static_cast<double>(A[row * n + k])
                        * static_cast<double>(B[static_cast<size_t>(k) * n + col]);
        }
        const double error = std::fabs(static_cast<double>(C[idx]) - expected);
        max_abs_error = std::max(max_abs_error, error);
        if (!std::isfinite(C[idx]) ||
            error > (1.0e-4 + 1.0e-4 * std::fabs(expected))) {
            std::cerr << "[FAIL] N=" << n << " index=" << idx
                      << " expected=" << expected << " actual=" << C[idx]
                      << " absolute_error=" << error << '\n';
            std::exit(EXIT_FAILURE);
        }
    }
    // Additional finite-output sanity check over ALL elements.
    for (float value : C) {
        if (!std::isfinite(value)) {
            std::cerr << "[FAIL] N=" << n << " has a non-finite output\n";
            std::exit(EXIT_FAILURE);
        }
    }

    cudaEvent_t start = nullptr, stop = nullptr;
    cuda_check(cudaEventCreate(&start), "create start event");
    cuda_check(cudaEventCreate(&stop), "create stop event");

    std::vector<double> kernel_ms;
    kernel_ms.reserve(kIterations);
    for (int i = 0; i < kIterations; ++i) {
        cuda_check(cudaEventRecord(start), "record start");
        for (int j = 0; j < batch; ++j) {
            matmul_naive<<<grid, block>>>(d_A, d_B, d_C, n, n, n);
            cuda_check(cudaGetLastError(), "benchmark kernel launch");
        }
        cuda_check(cudaEventRecord(stop), "record stop");
        cuda_check(cudaEventSynchronize(stop), "wait stop event");
        float elapsed_ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop),
                   "event elapsed time");
        kernel_ms.push_back(static_cast<double>(elapsed_ms) / batch);
    }
    const Statistics kernel_stats = summarize(kernel_ms);

    // Warm up the full transfer-and-compute path before host-side measurements.
    for (int i = 0; i < kWarmup; ++i) {
        cuda_check(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice), "H2D warmup A");
        cuda_check(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice), "H2D warmup B");
        matmul_naive<<<grid, block>>>(d_A, d_B, d_C, n, n, n);
        cuda_check(cudaGetLastError(), "end-to-end warmup launch");
        cuda_check(cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost), "D2H warmup");
    }
    cuda_check(cudaDeviceSynchronize(), "end-to-end warmup sync");

    std::vector<double> end_to_end_ms;
    end_to_end_ms.reserve(kIterations);
    using Clock = std::chrono::steady_clock;
    for (int i = 0; i < kIterations; ++i) {
        const auto begin = Clock::now();
        cuda_check(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice), "H2D measured A");
        cuda_check(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice), "H2D measured B");
        matmul_naive<<<grid, block>>>(d_A, d_B, d_C, n, n, n);
        cuda_check(cudaGetLastError(), "end-to-end measured launch");
        cuda_check(cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost), "D2H measured");
        cuda_check(cudaDeviceSynchronize(), "end-to-end measured sync");
        const auto end = Clock::now();
        end_to_end_ms.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    }
    const Statistics full_stats = summarize(end_to_end_ms);

    std::cout << n << ',' << kWarmup << ',' << kIterations << ',' << batch
              << ',' << kernel_stats.avg_ms << ',' << kernel_stats.median_ms
              << ',' << gflops(n, kernel_stats.avg_ms)
              << ',' << full_stats.avg_ms << ',' << full_stats.median_ms
              << ',' << gflops(n, full_stats.avg_ms)
              << ",PASS," << checks << ',' << max_abs_error << '\n';

    cuda_check(cudaEventDestroy(start), "destroy start event");
    cuda_check(cudaEventDestroy(stop), "destroy stop event");
    cuda_check(cudaFree(d_A), "free A");
    cuda_check(cudaFree(d_B), "free B");
    cuda_check(cudaFree(d_C), "free C");
}

int main() {
    cuda_check(cudaSetDevice(0), "select CUDA device 0");
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, 0), "get CUDA device properties");
    std::cerr << "[INFO] GPU=" << prop.name
              << " compute_capability=" << prop.major << '.' << prop.minor << '\n';
    std::cerr << "[INFO] CUDA benchmark: sequential default stream; no concurrent work assumed.\n";
    std::cerr << "[INFO] Reference: full for N<=256; sample=512 plus finite check for larger N.\n";

    std::cout << std::fixed << std::setprecision(9);
    std::cout << "N,warmup,iterations,launches_per_event_sample,"
              << "kernel_avg_ms,kernel_median_ms,kernel_gflops,"
              << "e2e_avg_ms,e2e_median_ms,e2e_effective_gflops,"
              << "validation,validated_elements,max_abs_error\n";
    for (int n : {128, 256, 512, 1024}) {
        std::cerr << "[INFO] Benchmark N=" << n << "\n";
        run_size(n);
    }
    return EXIT_SUCCESS;
}

#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>

void matmul_cpu(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
)
{
    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            float sum = 0.0f;

            for (int k = 0; k < K; ++k)
            {
                sum += A[row * K + k]
                     * B[k * N + col];
            }

            C[row * N + col] = sum;
        }
    }
}

bool validate(
    const float* result,
    const float* expected,
    int size,
    float epsilon
)
{
    for (int i = 0; i < size; ++i)
    {
        float error = std::fabs(result[i] - expected[i]);

        if (error > epsilon)
        {
            return false;
        }
    }

    return true;
}

#ifndef MATMUL_TESTING
int main()
{
    const int N = 1024;

    std::vector<float> A(N * N, 1.0f);
    std::vector<float> B(N * N, 1.0f);
    std::vector<float> C(N * N, 0.0f);

    std::vector<float> expected(N * N, static_cast<float>(N));

    const float epsilon = 1e-5f;

    const int iterations = 10;

    const int warmup_iterations = 3;

    // Warm-up
    for (int i = 0; i < warmup_iterations; ++i)
    {
        matmul_cpu(
            A.data(),
            B.data(),
            C.data(),
            N,
            N,
            N
        );
    }

    double total_ms = 0.0;

    std::vector<double> times;

    for (int i = 0; i < iterations; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();

        matmul_cpu(
            A.data(),
            B.data(),
            C.data(),
            N,
            N,
            N
        );

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;

        total_ms += elapsed.count();

        times.push_back(elapsed.count());
    }

    double average_ms = total_ms / iterations;

    std::sort(times.begin(), times.end());

    double median_ms;

    if (iterations % 2 == 0)
    {
        median_ms =
            (times[iterations / 2 - 1]
            + times[iterations / 2]) / 2.0;
    }
    else
    {
        median_ms = times[iterations / 2];
    }

    double time_seconds = average_ms / 1000.0;

    double flops =
        2.0 * N * N * N;

    double gflops =
        flops / time_seconds / 1e9;
    bool correct = validate(
        C.data(),
        expected.data(),
        N * N,
        epsilon
    );

    std::cout << "Matrix size: "
              << N << " x " << N << '\n';

    std::cout << "Iterations: "
              << iterations
              << '\n';

    std::cout << "Average CPU time: "
              << average_ms
              << " ms\n";

    std::cout << "Throughput: "
              << gflops
              << " GFLOPS\n";

    std::cout << "Median CPU time: "
              << median_ms
              << " ms\n";
    
    std::cout << "Warm-up iterations: "
              << warmup_iterations
              << '\n';

    if (correct)
    {
        std::cout << "PASS\n";
    }
    else
    {
        std::cout << "FAIL\n";
    }

    return correct ? 0 : 1;
}
#endif // MATMUL_TESTING

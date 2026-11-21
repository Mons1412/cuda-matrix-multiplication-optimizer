#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

void matmul_cpu(const float* A, const float* B, float* C,
                int M, int K, int N);

struct Shape {
    int m;
    int k;
    int n;
};

bool check_shape(const Shape& shape, std::mt19937& rng) {
    const int M = shape.m;
    const int K = shape.k;
    const int N = shape.n;

    std::uniform_real_distribution<float> distribution(-2.0f, 2.0f);
    std::vector<float> A(static_cast<std::size_t>(M) * K);
    std::vector<float> B(static_cast<std::size_t>(K) * N);
    for (float& x : A) x = distribution(rng);
    for (float& x : B) x = distribution(rng);

    // A separate 2D double-precision reference (not the CPU function under test).
    std::vector<std::vector<double>> a2d(M, std::vector<double>(K));
    std::vector<std::vector<double>> b2d(K, std::vector<double>(N));
    for (int row = 0; row < M; ++row)
        for (int k = 0; k < K; ++k)
            a2d[row][k] = static_cast<double>(A[row * K + k]);
    for (int k = 0; k < K; ++k)
        for (int col = 0; col < N; ++col)
            b2d[k][col] = static_cast<double>(B[k * N + col]);

    std::vector<double> reference(static_cast<std::size_t>(M) * N, 0.0);
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k)
                sum += a2d[row][k] * b2d[k][col];
            reference[row * N + col] = sum;
        }
    }

    // NaN sentinels detect output entries that were never written.
    std::vector<float> result(static_cast<std::size_t>(M) * N,
                              std::numeric_limits<float>::quiet_NaN());
    matmul_cpu(A.data(), B.data(), result.data(), M, K, N);

    const double abs_tol = 1e-4;
    const double rel_tol = 1e-4;
    double max_error = 0.0;
    for (std::size_t i = 0; i < result.size(); ++i) {
        const double expected = reference[i];
        const double actual = static_cast<double>(result[i]);
        const double error = std::abs(actual - expected);
        const double allowed = abs_tol + rel_tol * std::abs(expected);
        if (!std::isfinite(actual) || error > allowed) {
            std::cerr << "[FAIL] " << M << 'x' << K << " * " << K << 'x' << N
                      << " at C[" << i / N << "," << i % N << "]"
                      << " expected=" << expected << " actual=" << actual
                      << " error=" << error << " allowed=" << allowed << '\n';
            return false;
        }
        if (error > max_error) max_error = error;
    }
    std::cout << "[PASS] " << M << 'x' << K << " * " << K << 'x' << N
              << " max_abs_error=" << std::setprecision(7) << max_error << '\n';
    return true;
}

int main() {
    std::mt19937 rng(20261121);  // Repeatable pseudo-random test inputs.
    const std::vector<Shape> shapes = {
        {1, 1, 1}, {1, 7, 1}, {2, 3, 4}, {7, 5, 9},
        {17, 19, 23}, {31, 16, 35}, {32, 33, 31}, {64, 65, 63}
    };
    bool passed = true;
    for (const auto& shape : shapes)
        if (!check_shape(shape, rng)) passed = false;
    return passed ? 0 : 1;
}

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

void matmul_cpu(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
);

struct TestCase {
    std::string name;
    int M, K, N;
    std::vector<float> A, B, expected;
};

bool run_test(const TestCase& test)
{
    if (test.A.size() != static_cast<size_t>(test.M * test.K) ||
        test.B.size() != static_cast<size_t>(test.K * test.N) ||
        test.expected.size() != static_cast<size_t>(test.M * test.N))
    {
        std::cerr << "[FAIL] Invalid test data: "
                  << test.name << '\n';
        return false;
    }

    std::vector<float> result(test.M * test.N, 0.0f);

    matmul_cpu(
        test.A.data(),
        test.B.data(),
        result.data(),
        test.M,
        test.K,
        test.N
    );

    constexpr float epsilon = 1e-5f;

    for (size_t i = 0; i < result.size(); ++i)
    {
        if (!std::isfinite(result[i]) ||
            std::fabs(result[i] - test.expected[i]) > epsilon)
        {
            std::cerr << "[FAIL] " << test.name
                      << " at index " << i
                      << ": expected=" << test.expected[i]
                      << ", actual=" << result[i] << '\n';
            return false;
        }
    }

    std::cout << "[PASS] " << test.name << '\n';
    return true;
}

int main()
{
    const std::vector<TestCase> tests = {
        {
            "Rectangular 2x3 * 3x2",
            2, 3, 2,
            {1, 2, 3, 4, 5, 6},
            {7, 8, 9, 10, 11, 12},
            {58, 64, 139, 154}
        },
        {
            "Negative numbers",
            2, 2, 2,
            {-1, 2, 3, -4},
            {5, -6, -7, 8},
            {-19, 22, 43, -50}
        },
        {
            "Identity matrix",
            3, 3, 3,
            {1, 2, 3, 4, 5, 6, 7, 8, 9},
            {1, 0, 0, 0, 1, 0, 0, 0, 1},
            {1, 2, 3, 4, 5, 6, 7, 8, 9}
        },
        {
            "Zero matrix",
            2, 2, 2,
            {1, 2, 3, 4},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        },
        {
            "Row vector * Column vector",
            1, 3, 1,
            {1, 2, 3},
            {-1, 3, 4},
            {17}
        }
    };

    bool all_passed = true;

    for (const auto& test : tests)
    {
        if (!run_test(test))
            all_passed = false;
    }

    return all_passed ? 0 : 1;
}

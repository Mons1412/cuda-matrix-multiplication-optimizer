
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

void check_cuda(cudaError_t error, const char* message)
{
    if (error != cudaSuccess)
    {
        printf(
            "CUDA Error: %s: %s\n",
            message,
            cudaGetErrorString(error)
        );

        exit(1);
    }
}

__global__ void vector_add(
    const int* A,
    const int* B,
    int* C,
    int N
)
{
    int i =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    const int N = 10;

    int h_A[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int h_B[N] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    int h_C[N] = {0};

    int* d_A = nullptr;
    int* d_B = nullptr;
    int* d_C = nullptr;

    check_cuda(
        cudaMalloc(&d_A, N * sizeof(int)),
        "cudaMalloc d_A"
    );
    check_cuda(
        cudaMalloc(&d_B, N * sizeof(int)),
        "cudaMalloc d_B"
    );

    check_cuda(
        cudaMalloc(&d_C, N * sizeof(int)),
        "cudaMalloc d_C"
    );

    check_cuda(
        cudaMemcpy(
            d_A,
            h_A,
            N * sizeof(int),
            cudaMemcpyHostToDevice
        ),
        "Copy A Host-to-Device"
    );

    check_cuda(
        cudaMemcpy(
            d_B,
            h_B,
            N * sizeof(int),
            cudaMemcpyHostToDevice
        ),
        "Copy B Host-to-Device"
    );

    const int threads_per_block = 4;

    const int blocks =
        (N + threads_per_block - 1)
        / threads_per_block;

    vector_add<<<blocks, threads_per_block>>>(
        d_A,
        d_B,
        d_C,
        N
    );

    check_cuda(
        cudaGetLastError(),
        "vector_add kernel launch"
    );

    check_cuda(
        cudaDeviceSynchronize(),
        "vector_add kernel execution"
    );

    check_cuda(
        cudaMemcpy(
            h_C,
            d_C,
            N * sizeof(int),
            cudaMemcpyDeviceToHost
        ),
        "Copy C Device-to-Host"
    );

    check_cuda(
        cudaFree(d_A),
        "cudaFree d_A"
    );

    check_cuda(
        cudaFree(d_B),
        "cudaFree d_B"
    );

    check_cuda(
        cudaFree(d_C),
        "cudaFree d_C"
    );

    for (int i = 0; i < N; ++i)
    {
        printf("%d ", h_C[i]);
    }

    printf("\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

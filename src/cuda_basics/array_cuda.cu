
#include <cstdio>
#include <cuda_runtime.h>

__global__ void fill_array(int* data)
{
    int global_id =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    data[global_id] = global_id + 100;
}

int main()
{
    const int N = 12;

    int h_data[N] = {0};

    int* d_data = nullptr;

    cudaMalloc(&d_data, N * sizeof(int));

    cudaMemcpy(
        d_data,
        h_data,
        N * sizeof(int),
        cudaMemcpyHostToDevice
    );

    fill_array<<<3, 4>>>(d_data);

    cudaDeviceSynchronize();

    cudaMemcpy(
        h_data,
        d_data,
        N * sizeof(int),
        cudaMemcpyDeviceToHost
    );

    for (int i = 0; i < N; ++i)
    {
        printf("%d ", h_data[i]);
    }

    printf("\n");

    cudaFree(d_data);

    return 0;
}


#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_kernel()
{
    int local_id = threadIdx.x;

    int global_id =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    printf(
        "Block %d | Thread %d | Global Thread %d\n",
        blockIdx.x,
        local_id,
        global_id
    );
}

int main()
{
    hello_kernel<<<3, 5>>>();

    cudaDeviceSynchronize();

    return 0;
}

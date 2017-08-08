#include <stdio.h>
#include <stdlib.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void ols_kernel(int N)
{
  int i = blockIdx.x;
  int noOfBlocks = (i + blockDim.x - 1)/blockDim.x;
  int j;

  printf("hello world\n");
  for (int ii = 0; ii < noOfBlocks; ii++)
  {
    j = threadIdx.x + ii*blockDim.x;
    // if (j >= i)
    //   return;
  }
}

int main()
{

  int N = 2000;
  dim3 blocks(4, 1);
  dim3 grids(N, 1);
  ols_kernel<<<grids, blocks>>>(N);
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}

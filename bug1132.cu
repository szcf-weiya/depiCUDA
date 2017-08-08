#include <stdio.h>
#include <stdlib.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void ols_kernel(int N, int* sum)
{
  //
  int id_j = blockIdx.y*blockDim.y + threadIdx.y;
  int id_i = blockIdx.x*blockDim.x + threadIdx.x;
  //int id_i = blockIdx.x;
  //int id_j = threadIdx.x;

  while(id_j < N && id_i < N)
  {
      printf("i = %d, j = %d\n", id_i, id_j);
      printf("hello\n");
      id_i += blockDim.x*gridDim.x;
      id_j += blockDim.y*gridDim.y;
  }


  printf("hello world\n");
}

int main()
{
  dim3 blocks(16, 16);
  dim3 grids(4, 4);
  int N = 100;
  int sum = 0;
  ols_kernel<<<grids, blocks>>>(N, &sum);
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}

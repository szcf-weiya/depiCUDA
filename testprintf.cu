#include "stdio.h"
// printf () is only supported
// for devices of compute capability 2.0 and  higher
#if defined (__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
  #define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

__global__ void helloCUDA(float f)
{
  printf("hello, thread = %d, f = %f\n", threadIdx.x, f);
}

int main(int argc, char const *argv[]) {
  helloCUDA<<<dim3(5,5), dim3(16,16)>>>(1.2345f);
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void kernel(const double *d_X, const double *d_Y, double *res, const int nn, const int pp)
{
  __shared__ double alpha, beta;
  __shared__ int n, p;
  alpha = 1.0; beta = 0.0;
  n = nn; p = pp;

  __shared__ double *d_coef;
  //d_coef = (double*)malloc(sizeof(double)*p*256);
  __syncthreads();
  if (threadIdx.x == 0)
  {
    d_coef = (double*)malloc(sizeof(double)*p*256);
    //memset(d_coef, 0, sizeof(double)*p*256);
  }
  __syncthreads();
  if(!d_coef)
  {
    printf("error\n");
    return;
  }

  //int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;
  cublasHandle_t cublasH = NULL;
  cublasStatus_t cublas_status = cublasCreate(&cublasH);

  // X'Y (X is n*p, Y is n*1)
  //double *d_coef = (double*)malloc(sizeof(double)*p);
  //memset(d_coef, 0, sizeof(double)*p);
  //printf("Thread %d got pointer: %p\n", tid, d_coef);
  //double d_coef[3];
  __syncthreads();
  cublas_status = cublasDgemv(cublasH, CUBLAS_OP_T,
                           n, p,
                           &alpha,
                           d_X, n,
                           d_Y, 1,
                           &beta,
                           d_coef+tid*p, 1);
   __syncthreads(); //应该不用加，不存在对share memory和global memory的写入。
  if (cublas_status == CUBLAS_STATUS_SUCCESS)
   {
     ;
  //   if(d_coef[0]==0)
        printf("Thread %d got pointer: %p, %f, %f, %f\n", tid, d_coef, d_coef[0+tid*p], d_coef[1+tid*p], d_coef[2+tid*p]);
    //printf("tid = %d; d_coef = %f, %f, %f\n", tid, d_coef[0], d_coef[1], d_coef[2]);
    __syncthreads();
   }
   else
   {
     printf("wrong!\n");
     __syncthreads();
   }
   for (size_t i = 0; i < p; i++)
   {
     ;
    // res[tid*p+i] = d_coef[i];
   }
   cublasDestroy(cublasH);
   //free(d_coef);
   __syncthreads();
   if (threadIdx.x == 0)
    free(d_coef);
}

int main(int argc, char const *argv[]) {
  double A[] = {1, 1, 1, 1, 2, 3, 5, 4, 3, 6, 7, 9};
  double B[] = {1, 2, 3, 4};
  double *d_A, *d_B, *d_res;

  int n = 4, p = 3;
  int threadsPerBlock = 256;
  int blocksPerGird = 40;

  double *res = (double*)malloc(sizeof(double)*p*blocksPerGird*threadsPerBlock);
  //cudaDeviceReset();
  size_t limit_stack, limit_printf, limit_heap;
  cudaThreadGetLimit(&limit_stack, cudaLimitStackSize);
  cudaThreadGetLimit(&limit_printf, cudaLimitPrintfFifoSize);
  cudaThreadGetLimit(&limit_heap, cudaLimitMallocHeapSize);
//  cudaThreadSetLimit(cudaLimitStackSize, limit*2);
  printf("%d, %d, %d\n", (int)limit_stack, (int)limit_printf, (int)limit_heap);
  cudaThreadSetLimit(cudaLimitStackSize, 1024*1024*2);
  cudaThreadSetLimit(cudaLimitPrintfFifoSize, 1024*1024*10);
  cudaThreadSetLimit(cudaLimitMallocHeapSize, 1024*1024*1024);
  cudaThreadGetLimit(&limit_stack, cudaLimitStackSize);
  cudaThreadGetLimit(&limit_printf, cudaLimitPrintfFifoSize);
  cudaThreadGetLimit(&limit_heap, cudaLimitMallocHeapSize);
//  cudaThreadSetLimit(cudaLimitStackSize, limit*2);
  printf("%d, %d, %d\n", (int)limit_stack, (int)limit_printf, (int)limit_heap);
//  cudaThreadGetLimit(&limit, cudaLimitMallocHeapSize);
//  printf("%d\n", limit);
  cudaMalloc((void**)&d_A, sizeof(double)*n*p);
  cudaMalloc((void**)&d_B, sizeof(double)*n);
  cudaMalloc((void**)&d_res, sizeof(double)*p*threadsPerBlock*blocksPerGird);
  cudaMemcpy(d_A, A, sizeof(double)*n*p, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeof(double)*n, cudaMemcpyHostToDevice);
  kernel<<<blocksPerGird, threadsPerBlock>>>(d_A, d_B, d_res, n, p);
  cudaDeviceSynchronize();
  cudaMemcpy(res, d_res, sizeof(double)*p*threadsPerBlock*blocksPerGird, cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < threadsPerBlock*blocksPerGird; i++)
  {
    ;
    //printf("%f, %f, %f\n", res[i*p], res[i*p+1], res[i*p+2]);
  }
  cudaDeviceReset();
  return 0;
}

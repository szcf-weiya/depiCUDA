
/* Includes, system */
#include <stdio.h>
#include <stdlib.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_matrix.h>

__device__ inline int solveBeta(int tid,
                          int n, int p,
                          const double *d_X,
                          const double *d_Y,
                          double *d_invXX,
                          double *d_coef)
{
  cublasHandle_t cublasH = NULL;
  cublasStatus_t cublas_status = cublasCreate_v2(&cublasH);
  int *pivotArray = (int *)malloc(p*sizeof(int));
  int *info = (int *)malloc(sizeof(int));
  info[0] = 0;
  //int info;
  int batch = 1;
  double alpha = 1, beta = 0;
  const double *palpha = &alpha, *pbeta = &beta;

  double *d_XX = (double *)malloc(sizeof(double)*p*p);
  double *d_coef2 = (double *)malloc(sizeof(double)*p);
  memset(d_coef2, 0.0, sizeof(double)*p);
  memset(d_coef, 0.0, sizeof(double)*p);
  memset(d_XX, 0.0, sizeof(double)*p*p);


  double **a = (double **)malloc(sizeof(double *));
  *a = d_XX;
  const double **aconst = (const double **)a;

  double **c = (double **)malloc(sizeof(double *));
  *c = d_invXX;
  // X'X
  cublas_status = cublasDgemm(cublasH,
                           CUBLAS_OP_T, CUBLAS_OP_N,
                           p, p, n, // DO NOT mess up the order
                           palpha,
                           d_X, n,
                           d_X, n,
                           pbeta,
                           d_XX, p);

  // inv(X'X)
  cublas_status = cublasDgetrfBatched(cublasH, p, a, p, pivotArray, info, batch);
  if (info[0] < 0)
  {
    cublasDestroy_v2(cublasH);
    return info[0];
  }
  else if (info[0] > 0)
  {
    cublasDestroy_v2(cublasH);
    return info[0];
  }
  cublas_status = cublasDgetriBatched(cublasH, p, aconst, p, pivotArray, c, p, info, batch);
  if (info[0] < 0)
  {
    cublasDestroy_v2(cublasH);
    return info[0];
  }
  // X'Y   (p*n)*(n*1) = p*1
  cublas_status = cublasDgemv(cublasH, CUBLAS_OP_T,
                           n, p,
                           palpha,
                           d_X, n,
                           d_Y, 1,
                           pbeta,
                           d_coef2, 1);
  __syncthreads();
  printf("tid = %d; inv = %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f\n", tid, d_invXX[0], d_invXX[1], d_invXX[2], d_invXX[5], d_invXX[4], d_invXX[8], d_coef2[0], d_coef2[1], d_coef2[2]);

  // (X'X)^{-1}X'Y
  cublas_status = cublasDgemv(cublasH, CUBLAS_OP_N,
                           p, p,
                           palpha,
                           d_invXX, p,
                           d_coef2, 1,
                           pbeta,
                           d_coef, 1);
  //printf("tid = %d; inv = %f, %f, %f; beta = %f, %f, %f\n", tid, d_invXX[0], d_invXX[4], d_invXX[8], d_coef[0], d_coef[1], d_coef[2]);
  //printf("tid = %d; inv = %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f\n", tid, d_invXX[0], d_invXX[1], d_invXX[2], d_invXX[5], d_invXX[4], d_invXX[8], d_coef[0], d_coef[1], d_coef[2]);
//  printf("tid = %d; inv = %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f\n", tid, d_invXX[0], d_invXX[1], d_invXX[2], d_invXX[5], d_invXX[4], d_invXX[8], d_coef2[0], d_coef2[1], d_coef2[2]);

  free(pivotArray);
  free(info);
  free(d_XX);
  free(d_coef2);
  free(a);
  free(c);
  cublasDestroy_v2(cublasH);
  return 0;
}

__global__ void kernel(int n, int p,
  const double *d_X,
  const double *d_Y,
  double *d_GinvXX,
  double *d_Gcoef)
{
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  //printf("%d\n", tid);
  double *d_coef = (double*)malloc(sizeof(double)*p);
  double *d_invXX = (double*)malloc(sizeof(double)*p*p);
  //__syncthreads(); // must add
  solveBeta(tid, n, p, d_X, d_Y, d_invXX, d_coef);
  //__syncthreads(); // must add
  /*
  if (res != 0)
  {
    free(d_coef);
    free(d_invXX);
    return;
  }
  */
  //__syncthreads(); // must add
  #pragma unroll
  for (int i = 0; i < p; i++)
  {
    d_Gcoef[tid*p+i] = d_coef[i];
  }
  //printf("tid = %d; inv = %f, %f, %f; beta = %f, %f, %f\n", tid, d_invXX[0], d_invXX[4], d_invXX[8], d_coef[0], d_coef[1], d_coef[2]);
  //__syncthreads();
  free(d_coef);
  free(d_invXX);
}

int main(int argc, char const *argv[]) {
  /* code */
  //double A[] = {1, 2, 3, 0, 2, 4, 2, 1, 5};
  double A[] = {1, 1, 1, 1, 2, 3, 5, 4, 3, 6, 7, 9};
  double B[] = {1, 2, 3, 4};
  double *d_A, *d_B, *d_invXX, *d_coef;
  int threadsPerBlock = 32;
  int blocksPerGird = 1;
  // Set a heap size of 128 megabytes. Note that this must
  // be done before any kernel is launched.
  cudaThreadSetLimit(cudaLimitMallocHeapSize, 1024*1024*1024);
  double coef[3*threadsPerBlock*blocksPerGird];
  cudaMalloc((void**)&d_A, sizeof(double)*12);
  cudaMalloc((void**)&d_B, sizeof(double)*4);
  cudaMalloc((void**)&d_invXX, sizeof(double)*9);
  cudaMalloc((void**)&d_coef, sizeof(double)*3*threadsPerBlock*blocksPerGird);
  cudaMemcpy(d_A, A, sizeof(double)*12, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeof(double)*4, cudaMemcpyHostToDevice);
  kernel<<<threadsPerBlock, blocksPerGird>>>(4, 3, d_A, d_B, d_invXX, d_coef);
  cudaDeviceSynchronize();
  cudaMemcpy(coef, d_coef, sizeof(double)*3*threadsPerBlock*blocksPerGird, cudaMemcpyDeviceToHost);

  cudaFree(d_coef);
  cudaFree(d_invXX);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaDeviceReset();
/*
  for (int i = 0; i < threadsPerBlock*blocksPerGird; i++)
  {
    for (int j = 0; j < 3; j++)
      printf("%f, \n", coef[i*3+j]);
    printf("\n");
  }
  */
  return 0;
}

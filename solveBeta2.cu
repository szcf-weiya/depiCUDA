
/* Includes, system */
#include <stdio.h>
#include <stdlib.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_matrix.h>

__device__ int solveBeta(
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
  double alpha = 1.0, beta = 0.0;

  double *d_XX = (double *)malloc(sizeof(double)*p*p);
  double *d_coef2 = (double *)malloc(sizeof(double)*p);

  double **a = (double **)malloc(sizeof(double *));
  *a = d_XX;
  const double **aconst = (const double **)a;

  double **c = (double **)malloc(sizeof(double *));
  *c = d_invXX;
  // X'X
  cublas_status = cublasDgemm(cublasH,
                           CUBLAS_OP_T, CUBLAS_OP_N,
                           p, p, n, // DO NOT mess up the order
                           &alpha,
                           d_X, n,
                           d_X, n,
                           &beta,
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
                           &alpha,
                           d_X, n,
                           d_Y, 1,
                           &beta,
                           d_coef2, 1);

  // (X'X)^{-1}X'Y
  cublas_status = cublasDgemv(cublasH, CUBLAS_OP_T,
                           p, p,
                           &alpha,
                           d_invXX, p,
                           d_coef2, 1,
                           &beta,
                           d_coef, 1);
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
  __syncthreads(); // must add
  solveBeta(n, p, d_X, d_Y, d_invXX, d_coef);
  __syncthreads(); // must add
  /*
  if (res != 0)
  {
    free(d_coef);
    free(d_invXX);
    return;
  }
  */
  __syncthreads(); // must add
  for (int i = 0; i < p; i++)
    d_Gcoef[tid*p+i] = d_coef[i];
  __syncthreads();
  free(d_coef);
  free(d_invXX);
}

int main(int argc, char const *argv[]) {
  /* code */
  //double A[] = {1, 2, 3, 0, 2, 4, 2, 1, 5};
  double A[] = {1, 1, 1, 1, 2, 3, 5, 4, 3, 6, 7, 9};
  double B[] = {1, 2, 3, 4};
  double *d_A, *d_B, *d_invXX, *d_coef;
  int threadsPerBlock = 256;
  int blocksPerGird = 1;
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
  cudaDeviceSynchronize();

  cudaFree(d_coef);
  cudaFree(d_invXX);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaDeviceReset();

  for (int i = 0; i < threadsPerBlock*blocksPerGird; i++)
  {
    for (int j = 0; j < 3; j++)
      printf("%f, \n", coef[i*3+j]);
    printf("\n");
  }
  return 0;
}

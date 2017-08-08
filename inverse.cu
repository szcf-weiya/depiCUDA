
/* Includes, system */
#include <stdio.h>
#include <stdlib.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_matrix.h>

__global__ void inverse(double* d_XX, int p, double* d_invXX)
{
  cublasHandle_t cublasH = NULL;
  cublasStatus_t cublas_status = cublasCreate_v2(&cublasH);
  int *pivotArray = (int *)malloc(p*sizeof(int));
  int *info = (int*)malloc(sizeof(int));
  info[0] = 0;
  //int info = 0;
  int batch = 1;

  /*
  if (info < 0)
  {
    cublasDestroy_v2(cublasH);
    return;
  }
  else if (info > 0)
  {
    cublasDestroy_v2(cublasH);
    return;
  }
  */

  double **a = (double **)malloc(sizeof(double *));
  *a = d_XX;
  const double **aconst = (const double **)a;
  double **c = (double **)malloc(sizeof(double *));
  *c = d_invXX;
  cublas_status = cublasDgetrfBatched(cublasH, p, a, p, pivotArray, info, batch);
  cublas_status = cublasDgetriBatched(cublasH, p, aconst, p, pivotArray, c, p, info, batch);
      /*
  if (info < 0)
  {
    cublasDestroy_v2(cublasH);
    return;
  }
  */
  cublasDestroy_v2(cublasH);
}

int main(int argc, char const *argv[]) {
  /* code */
  //double A[] = {1, 2, 3, 0, 2, 4, 2, 1, 5};
  double A[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  double B[9];
  double *d_A, *d_B;
  cudaMalloc((void**)&d_A, sizeof(double)*9);
  cudaMalloc((void**)&d_B, sizeof(double)*9);
  cudaMemcpy(d_A, A, sizeof(double)*9, cudaMemcpyHostToDevice);
  inverse<<<1, 1>>>(d_A, 3, d_B);
  cudaDeviceSynchronize();
  cudaMemcpy(B, d_B, sizeof(double)*9, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
      printf("%f, ", B[i*3+j]);
    printf("\n");
  }
  return 0;
}

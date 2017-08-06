/*
 * Routines for calling cuLUsolve in device
 */

/* Includes, system */
#include <stdio.h>
#include <stdlib.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>


#define PERR(call) \
  if (call) {\
   fprintf(stderr, "%s:%d Error [%s] on "#call"\n", __FILE__, __LINE__,\
      cudaGetErrorString(cudaGetLastError()));\
   exit(1);\
  }

#define ERRCHECK \
  if (cudaPeekAtLastError()) { \
    fprintf(stderr, "%s:%d Error [%s]\n", __FILE__, __LINE__,\
       cudaGetErrorString(cudaGetLastError()));\
    exit(1);\
  }


__global__ void ols_kernel(const double *d_X,
                            const int n,
                            const int p,
                            double *d_Y,
                            double *d_coef,
                            double *d_tscore)
{
  // create cublas handle
  cublasHandle_t cublasH = NULL;
  cublasStatus_t cublas_status = cublasCreate_v2(&cublasH);
  if (cublas_status != CUBLAS_STATUS_SUCCESS)
  {
    return;
  }
  // //////////////////
  //
  // X'X
  //
  // /////////////////
  double alpha_v = 1.0;
  double beta_v = 0.0;
  double *alpha = &alpha_v, *beta = &beta_v;
  double *d_XX = (double *)malloc(sizeof(double)*p*p);
  double *d_invXX = (double *)malloc(sizeof(double)*p*p);
  double *d_coef2 = (double *)malloc(sizeof(double)*p);
  //double *d_Yhat = (double *)malloc(sizeof(double)*n);

  cublas_status = cublasDgemm(cublasH,
                           CUBLAS_OP_T, CUBLAS_OP_N,
                           p, p, n, // DO NOT mess up the order
                           alpha,
                           d_X, n,
                           d_X, n,
                           beta,
                           d_XX, p);
  __syncthreads();

  // /////////////////////
  //
  // inv(X'X)
  //
  // ////////////////////
  int *pivotArray = (int *)malloc(p*sizeof(int));
  int *info = (int *)malloc(sizeof(int));
  int batch;

  // just one matrix
  info[0] = 0;
  batch = 1;

  double **a = (double **)malloc(sizeof(double *));
  *a = d_XX;
  const double **aconst = (const double **)a;
  double **c = (double **)malloc(sizeof(double *));
  *c = d_invXX;
  cublas_status = cublasDgetrfBatched(cublasH, p, a, p, pivotArray, info, batch);
  __syncthreads();
  printf("rf %d info %d\n", cublas_status, info[0]);
  cublas_status = cublasDgetriBatched(cublasH, p, aconst, p, pivotArray,
      c, p, info, batch);
  __syncthreads();
  printf("ri %d info %d\n", cublas_status, info[0]);

  // /////////////////////
  //
  // X'Y   (p*n)*(n*1) = p*1
  //
  // //////////////////////
  cublas_status = cublasDgemv(cublasH, CUBLAS_OP_T,
                           n, p,
                           alpha,
                           d_X, n,
                           d_Y, 1,
                           beta,
                           d_coef2, 1);
  __syncthreads();

  // /////////////////////
  //
  // (X'X)^{-1}X'Y
  //
  // //////////////////////
  cublas_status = cublasDgemv(cublasH, CUBLAS_OP_T,
                           p, p,
                           alpha,
                           d_invXX, p,
                           d_coef2, 1,
                           beta,
                           d_coef, 1);

  __syncthreads();

  // ///////////////////
  // rss
  // ///////////////////

  beta_v = -1.0;
  cublas_status = cublasDgemv(cublasH, CUBLAS_OP_N,
                           n, p,
                           alpha,
                           d_X, n,
                           d_coef, 1,
                           beta,
                           d_Y, 1);
  __syncthreads();

  // sigma ^2 = RSS/(n-p-1)
  double sigma;
  double *psigma = &sigma;
  cublasDnrm2(cublasH, n, d_Y, 1, psigma);
  //sigma = norm(n, d_Y);
  printf("%f\n", sigma);
  sigma = sigma/sqrt((n-p)*1.0);
  for (int i = 0; i < p; i++)
    d_tscore[i] = d_coef[i]/(sigma*sqrt(d_invXX[i+p*i]));
  printf("%f\n", d_invXX[0]);
  printf("%f\n", d_invXX[1]);
  printf("%f\n", d_invXX[2]);
  printf("%f\n", d_invXX[3]);

  free(d_coef2);
  free(d_invXX);
  free(d_XX);
  //free(d_Yhat);
  free(a);
  free(c);
  cublasDestroy_v2(cublasH);
}

static void
run_ols(const double *X, const double *Y, int n, int p, double *coef, double *tscore)
{
  double *d_X, *d_Y, *d_coef, *d_tscore;

  PERR(cudaMalloc(&d_X, n*p*sizeof(double)));
  PERR(cudaMalloc(&d_Y, n*sizeof(double)));
  PERR(cudaMalloc(&d_coef, p*sizeof(double)));
  PERR(cudaMalloc(&d_tscore, p*sizeof(double)));
  PERR(cudaMemcpy(d_X, X, n*p*sizeof(double), cudaMemcpyHostToDevice));
  PERR(cudaMemcpy(d_Y, Y, n*sizeof(double), cudaMemcpyHostToDevice));

  ols_kernel<<<1, 1>>>(d_X, n, p, d_Y, d_coef, d_tscore);

  cudaDeviceSynchronize();
  ERRCHECK;

  PERR(cudaMemcpy(coef, d_coef, p*sizeof(double), cudaMemcpyDeviceToHost));
  PERR(cudaMemcpy(tscore, d_tscore, p*sizeof(double), cudaMemcpyDeviceToHost));

  PERR(cudaFree(d_X));
  PERR(cudaFree(d_Y));
}

int
main(int argc, char **argv)
{
  double A[] = {1, 1, 1, 1, 2, 3, 5, 4};
  double B[] = {1, 2, 3, 4};
  double coef[2];
  double pvalue[2];

  run_ols(A, B, 4, 2, coef, pvalue);

  printf("beta0 = %f; pvalue = %f\n", coef[0], pvalue[0]);
  printf("beta1 = %f; pvalue = %f\n", coef[1], pvalue[1]);
  return 0;
}

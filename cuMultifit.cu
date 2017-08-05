#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_cdf.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cuLUsolve.h"
#include "cuMultifit.h"

int cuMultifit(const double *X, int n, int p, const double *Y, double *coef, double *pvalue)
{
  cublasHandle_t cublasH = NULL;
  cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
  cudaError_t cudaStat = cudaSuccess;

  const int lda = n;
  double *C;
  C = (double*)malloc(sizeof(double)*p*p);

  double *d_X = NULL;
  double *d_C = NULL;
  double *d_Y = NULL;
  double *d_Yhat = NULL;
  double *d_coef = NULL;
  double *d_coef2 = NULL;


  // create cublas handle
  cublas_status = cublasCreate(&cublasH);
  assert(CUBLAS_STATUS_SUCCESS == cublas_status);

  // copy to device
  cudaStat = cudaMalloc ((void**)&d_X, sizeof(double) * lda * p);
  assert(cudaSuccess == cudaStat);

  cudaStat = cudaMalloc ((void**)&d_C, sizeof(double) * p * p);
  assert(cudaSuccess == cudaStat);

  cudaStat = cudaMalloc ((void**)&d_Y, sizeof(double) * n);
  assert(cudaSuccess == cudaStat);

  cudaStat = cudaMalloc ((void**)&d_Yhat, sizeof(double) * n);
  assert(cudaSuccess == cudaStat);

  cudaStat = cudaMalloc ((void**)&d_coef, sizeof(double) * p);
  assert(cudaSuccess == cudaStat);

  cudaStat = cudaMalloc ((void**)&d_coef2, sizeof(double) * p);
  assert(cudaSuccess == cudaStat);

  cudaStat = cudaMemcpy(d_X, X, sizeof(double) * lda * p, cudaMemcpyHostToDevice);
  assert(cudaSuccess == cudaStat);
  cudaStat = cudaMemcpy(d_Y, Y, sizeof(double) * n, cudaMemcpyHostToDevice);
  assert(cudaSuccess == cudaStat);
  cudaStat = cudaMemcpy(d_Yhat, Y, sizeof(double) * n, cudaMemcpyHostToDevice);
  assert(cudaSuccess == cudaStat);
  double alpha_v = 1.0;
  double beta_v = 0.0;
  double *alpha = &alpha_v, *beta = &beta_v; //check!!
  printf("%f\n", *alpha);
  // d_C = d_X^T d_X
  cublas_status = cublasDgemm(cublasH,
                           CUBLAS_OP_T, CUBLAS_OP_N,
                           p, p, n, // DO NOT mess up the order
                           alpha,
                           d_X, n,
                           d_X, n,
                           beta,
                           d_C, p);
  cudaStat = cudaDeviceSynchronize();
  assert(cublas_status == CUBLAS_STATUS_SUCCESS);
  assert(cudaSuccess == cudaStat);
  printf("finish X'X\n");
  // copy d_C to C
  cudaStat = cudaMemcpy(C, d_C, sizeof(double)*p*p, cudaMemcpyDeviceToHost);
  assert(cudaSuccess == cudaStat);
  // inv(C)
  gsl_matrix *B = gsl_matrix_alloc(p, p);
  gsl_matrix_set_identity(B);

  cuda_LU_solve(C, p, B->data, p);
  cudaStat = cudaMemcpy(d_C, B->data, sizeof(double)*p*p, cudaMemcpyHostToDevice);
  assert(cudaSuccess == cudaStat);
  for (int i = 0; i < p*p; i++)
    printf("%f\n", B->data[i]);

  printf("finish inv(C)\n");
  printf("%f %f\n", *alpha, *beta);
  // d_Y = d_X^T * d_Y
  cublas_status = cublasDgemv(cublasH, CUBLAS_OP_T,
                           n, p,
                           alpha,
                           d_X, n,
                           d_Y, 1,
                           beta,
                           d_coef, 1);
  cudaStat = cudaDeviceSynchronize();
  assert(cublas_status == CUBLAS_STATUS_SUCCESS);
  assert(cudaSuccess == cudaStat);
  cudaStat = cudaMemcpy(coef, d_coef, sizeof(double) * p, cudaMemcpyDeviceToHost);
  assert(cudaSuccess == cudaStat);
  for (int i = 0 ; i < p ; i ++ )
    printf("%f\n", coef[i]);

  // inv(C) * d_Y
  // due to by-column in gpu while by-row in gsl, C need to be transpose
  cublas_status = cublasDgemv(cublasH, CUBLAS_OP_T,
                           p, p,
                           alpha,
                           d_C, p,
                           d_coef, 1,
                           beta,
                           d_coef2, 1);
  cudaStat = cudaDeviceSynchronize();
  assert(cublas_status == CUBLAS_STATUS_SUCCESS);
  assert(cudaSuccess == cudaStat);

  // rss
  beta_v = -1.0;
  cublas_status = cublasDgemv(cublasH, CUBLAS_OP_N,
                           n, p,
                           alpha,
                           d_X, n,
                           d_coef2, 1,
                           beta,
                           d_Yhat, 1);
  cudaStat = cudaDeviceSynchronize();
  assert(cublas_status == CUBLAS_STATUS_SUCCESS);
  assert(cudaSuccess == cudaStat);

  // sigma ^2 = RSS/(n-p-1)
  double sigma;
  double *psigma = &sigma;
  cublasDnrm2(cublasH, n, d_Yhat, 1, psigma);
  sigma = sigma/sqrt(n-p);

  // copy to coef
  cudaStat = cudaMemcpy(coef, d_coef2, sizeof(double) * p, cudaMemcpyDeviceToHost);
  assert(cudaSuccess == cudaStat);

  double tscore;
  for (int i = 0; i < p; i++)
  {
    tscore = coef[i]/(sigma*sqrt(gsl_matrix_get(B, i, i)));
    pvalue[i] = 2*(tscore < 0 ? gsl_cdf_tdist_P(tscore, n-p) : gsl_cdf_tdist_P(-tscore, n-p));
  }

  gsl_matrix_free(B);
  cudaFree(d_X);
  cudaFree(d_Y);
  cudaFree(d_Yhat);
  cudaFree(d_C);
  cudaFree(d_coef);
  cudaFree(d_coef2);


  cublasDestroy(cublasH);
  cudaDeviceReset();
  return 0;
}

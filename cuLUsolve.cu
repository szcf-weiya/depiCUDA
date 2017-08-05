#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <cuda_runtime.h>

#include <cublas_v2.h>
#include <cusolverDn.h>

#include "cuLUsolve.h"

// solve Ax = B
//
// where dim(A) = m*m
//       dim(B) = m*1
int cuda_LU_solve(const double *A, const int m, double *B, const int nrhs)
{
  cusolverDnHandle_t cusolverH = NULL;
  cublasHandle_t cublasH = NULL;
  cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
  cudaError_t cudaStat1 = cudaSuccess;
  cudaError_t cudaStat2 = cudaSuccess;
  cudaError_t cudaStat3 = cudaSuccess;
  cudaError_t cudaStat4 = cudaSuccess;

  const int lda = m;
  const int ldb = m;
  //const int nrhs = 1;
  //
  //     | 1 2 3 |
  // A = | 4 5 6 |
  //     | 2 1 1 |
  //
  // x = (1, 1, 1)'
  // b = (6, 15, 4)'
  //

  //double A[lda*m] = {1, 4, 2, 2, 5, 1, 3, 6, 1};
  //double B[ldb*nrhs] = {6, 15, 4};

  double *d_A = NULL; //linear memory of GPU
  int *devIpiv = NULL;
  double *d_B = NULL;
  int *devInfo = NULL;
  double *d_work = NULL;
  int lwork = 0;
  int info_gpu = 0;

  // step 1: create cusolver/cublas handle
  cusolver_status = cusolverDnCreate(&cusolverH);
  assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

  cublas_status = cublasCreate(&cublasH);
  assert(CUBLAS_STATUS_SUCCESS == cublas_status);

  // step 2: copy A and B to device
  cudaStat1 = cudaMalloc ((void**)&d_A  , sizeof(double) * lda * m);
  cudaStat2 = cudaMalloc ((void**)&devIpiv, sizeof(int) * m);
  cudaStat3 = cudaMalloc ((void**)&d_B  , sizeof(double) * ldb * nrhs);
  cudaStat4 = cudaMalloc ((void**)&devInfo, sizeof(int));
  assert(cudaSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat2);
  assert(cudaSuccess == cudaStat3);
  assert(cudaSuccess == cudaStat4);

  cudaStat1 = cudaMemcpy(d_A, A, sizeof(double) * lda * m   , cudaMemcpyHostToDevice);
  cudaStat2 = cudaMemcpy(d_B, B, sizeof(double) * ldb * nrhs, cudaMemcpyHostToDevice);
  assert(cudaSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat2);

  // step 3: query working space of getrf and getrs
  cusolver_status = cusolverDnDgetrf_bufferSize(cusolverH,
                      m,
                      m,
                      d_A,
                      lda,
                      &lwork );

  assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
  cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
  assert(cudaSuccess == cudaStat1);

  // step 4: compute LU decomposition
  cusolver_status = cusolverDnDgetrf(cusolverH,
           m,
           m,
           d_A,
           lda,
           d_work,
           devIpiv,
           devInfo );

  cudaStat1 = cudaDeviceSynchronize();
  assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
  assert(cudaSuccess == cudaStat1);

  // check if LU is good or not
  cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
  assert(cudaSuccess == cudaStat1);
  printf("after getrf: info_gpu = %d\n", info_gpu);
  if (info_gpu < 0)
    printf("ERROR: the %d-th parameter is wrong.\n", abs(info_gpu));
  else if (info_gpu > 0)
    printf("ERROR: U(%d, %d) = 0\n", info_gpu, info_gpu);

  assert(0 == info_gpu);

  // setp 5: solve Ax = B
  cusolver_status = cusolverDnDgetrs(cusolverH,
           CUBLAS_OP_N,
           m,
           nrhs,
           d_A,
           lda,
           devIpiv,
           d_B,
           ldb,
           devInfo );

  cudaStat1 = cudaDeviceSynchronize();
  assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
  assert(cudaSuccess == cudaStat1);

  // check if LU is good or not
  cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
  assert(cudaSuccess == cudaStat1);

  printf("after getrs: info_gpu = %d\n", info_gpu);
  if (info_gpu < 0)
    printf("ERROR: the %d-th parameter is wrong.\n", abs(info_gpu));

  assert(0 == info_gpu);


  cudaStat1 = cudaMemcpy(B, d_B, sizeof(double)*ldb*nrhs, cudaMemcpyDeviceToHost);
  assert(cudaSuccess == cudaStat1);

  //printf("X = (matlab base-1)\n");
  //printMatrix(m, nrhs, B, ldb, "X");

  // free resources
  if (d_A) cudaFree(d_A);
  if (devIpiv) cudaFree(devIpiv);
  if (d_B) cudaFree(d_B);
  if (devInfo) cudaFree(devInfo);
  if (d_work) cudaFree(d_work);

  if (cublasH ) cublasDestroy(cublasH);
  if (cusolverH) cusolverDnDestroy(cusolverH);

  //cudaDeviceReset();

  return 0;
}

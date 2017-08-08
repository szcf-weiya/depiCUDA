/*
 * Routines for calling cuLUsolve in device
 */

/* Includes, system */
#include <stdio.h>
#include <stdlib.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_matrix.h>


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

__device__ void constructX(int n,
                            const double *d_X1,
                            const double *d_X2,
                            double *d_X)
{
  cublasHandle_t cublasH = NULL;
  cublasStatus_t cublas_status = cublasCreate_v2(&cublasH);
  double *d_X3 = (double*)malloc(sizeof(double)*n);
  // elements-by-elements
  // x3 = x1.*x2
  cublas_status = cublasDdgmm(cublasH, CUBLAS_SIDE_LEFT,
                          n, 1,
                          d_X1, n,
                          d_X2, 1,
                          d_X3, n);
  // construct matrix X
  double *pone = (double*)malloc(sizeof(double));
  pone[0] = 1.0;
  //double pone = 1.0;
  cublas_status = cublasDcopy(cublasH, n,
                           pone, 0,
                           d_X, 1);
  cublas_status = cublasDcopy(cublasH, n,
                           d_X1, 1,
                           d_X+n, 1);
  cublas_status = cublasDcopy(cublasH, n,
                           d_X2, 1,
                           d_X+2*n, 1);
  cublas_status = cublasDcopy(cublasH, n,
                           d_X3, 1,
                           d_X+3*n, 1);
  free(d_X3);
  cublasDestroy_v2(cublasH);
  //free(pone);
}

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
  //free(info);
  free(d_XX);
  free(d_coef2);
  free(a);
  free(c);
  cublasDestroy_v2(cublasH);
  return 0;
}

__device__ double getPvalue(
                          const double *d_X,
                          int n, int p,
                          const double *d_GY,
                          const double *d_coef)
{
  cublasHandle_t cublasH = NULL;
  cublasStatus_t cublas_status = cublasCreate_v2(&cublasH);
  double *d_Y = (double*)malloc(sizeof(double)*n);
  cublas_status = cublasDcopy(cublasH, n,
                           d_GY, 1,
                           d_Y, 1);
  // rss
  //double *alpha = (double*)malloc(sizeof(double));
  //double *beta = (double*)malloc(sizeof(double));
  //alpha[0] = 1.0;
  //beta[0] = -1.0;
  double alpha = 1.0, beta = -1.0;
  cublas_status = cublasDgemv(cublasH, CUBLAS_OP_N,
                           n, p,
                           &alpha,
                           d_X, n,
                           d_coef, 1,
                           &beta,
                           d_Y, 1);
  double rss;
  cublas_status = cublasDnrm2(cublasH, n, d_Y, 1, &rss);
  free(d_Y);
  cublasDestroy_v2(cublasH);
  return rss;
}

__global__ void ols_kernel(const double *d_GX,
                            const int n,
                            const int p,
                            const double *d_GY,
                            double *d_Gcoef,
                            double *d_Gtscore,
                            double *d_invXX,
                            const int N)
{
  int id_i = blockIdx.x;
  int id_j = threadIdx.x;
  //int id_jj = threadIdx.x, id_j;
  //int repeat = (id_i + blockDim.x - 1)/blockDim.x;
  //__syncthreads();
  const double *d_X1, *d_X2;
  double *d_X = (double*)malloc(sizeof(double) * n * p);
  double *d_coef = (double *)malloc(sizeof(double)*p);
  // create cublas handle


  //for (int id_ii = 0; id_ii < repeat; id_ii++)
  {
  //  id_j = id_jj + id_ii*blockDim.x;
    if(id_j >= id_i)
      return;
    __syncthreads();
    d_X1 = d_GX + id_i*n;
    d_X2 = d_GX + id_j*n;
    __syncthreads();
    // construct matrix X
    constructX(n,
              d_X1,
              d_X2,
              d_X);
    __syncthreads();
    // solve beta
    int info = solveBeta(n, p,
              d_X,
              d_GY,
              d_invXX,
              d_coef);
    __syncthreads();
    if (info < 0)
    {
      printf("i = %d, j = %d, in LU decomposition, the %d parameter had an illegeal value\n", id_i, id_j, abs(info));
      return;
    }
    else if (info > 0)
    {
      printf("i = %d, j = %d, in LU decomposition, U(%d, %d) = 0\n", id_i, id_j, abs(info), abs(info));
      return;
    }
    __syncthreads();
    double sigma = getPvalue(d_X,
                            n, p,
                            d_GY,
                            d_coef);
    __syncthreads();
    sigma = sigma/sqrt((n-p)*1.0);
    int id;
    double tscore;
    __syncthreads();

    for (int i = 0; i < p; i++)
    {
      tscore = d_coef[i]/(sigma*sqrt(d_invXX[i+p*i]));
      id = id_i*N+id_j-((id_i+1)*(id_i+2))/2;
      d_Gcoef[i+p*id] = d_coef[i];
      d_Gtscore[i+p*id] = tscore;
    }
    __syncthreads();

    //printf("%d, %d, finish tscore\n", id_i, id_j);
    printf("i = %d, j = %d; beta = %f, %f, %f, %f\n", id_i, id_j, d_coef[0], d_coef[1], d_coef[2], d_coef[3]);
    //printf("i = %d, j = %d; tscore = %f, %f, %f, %f\n", id_i, id_j, d_tscore[0], d_tscore[1], d_tscore[2], d_tscore[3]);
  }
  __syncthreads();
  free(d_X);
  free(d_coef);
}

static void
run_ols(const double *G, const double *Y, int n, int p, double *coef, double *tscore, int N)
{
  double *d_G, *d_GY, *d_Gcoef, *d_Gtscore, *d_invXX, *d_coef;

  PERR(cudaMalloc(&d_G, n*N*sizeof(double)));
  PERR(cudaMalloc(&d_GY, n*sizeof(double)));
  PERR(cudaMalloc(&d_Gcoef, N*(N-1)/2*p*sizeof(double)));
  PERR(cudaMalloc(&d_Gtscore, N*(N-1)/2*p*sizeof(double)));
  PERR(cudaMalloc(&d_invXX, sizeof(double)*p*p));
  PERR(cudaMalloc(&d_coef, sizeof(double)*p));

  PERR(cudaMemcpy(d_G, G, n*N*sizeof(double), cudaMemcpyHostToDevice));
  PERR(cudaMemcpy(d_GY, Y, n*sizeof(double), cudaMemcpyHostToDevice));

  int threadsPerBlock = 10;
  int blocksPerGird = 10;
  dim3 blocks(threadsPerBlock, 1);
  dim3 grids(blocksPerGird, 1);
  //int blocks = N;
  //int grids = N;
  //dim3 blocks(16, 16);
  //dim3 grids((N+15)/16,(N+15)/16);
  //int numBlocks = (N+15)/16;
  //ols_kernel<<<1, 1>>>(d_X, n, p, d_Y, d_coef, d_tscore);
  ols_kernel<<<grids, blocks>>>(d_G, n, p, d_GY, d_Gcoef, d_Gtscore, d_invXX, N);

  cudaDeviceSynchronize();
  ERRCHECK;

  PERR(cudaMemcpy(coef, d_Gcoef, N*(N-1)/2*p*sizeof(double), cudaMemcpyDeviceToHost));
  PERR(cudaMemcpy(tscore, d_Gtscore, N*(N-1)/2*p*sizeof(double), cudaMemcpyDeviceToHost));

  PERR(cudaFree(d_G));
  PERR(cudaFree(d_GY));
  PERR(cudaFree(d_invXX));
}

int
main(int argc, char **argv)
{
  const gsl_rng_type *T;
  gsl_rng *r;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc(T);

  int N, n, p = 4;
  N = 5;
  n = 10;

  double *A = (double*)malloc(sizeof(double)*n*N);
  double *B = (double*)malloc(sizeof(double)*n);
  double *pvalue = (double*)malloc(sizeof(double)*(N*(N-1))/2*p);
  double *coef = (double*)malloc(sizeof(double)*(N*(N-1))/2*p);

  for (size_t i = 0; i < n; i++)
  {
    for (size_t j = 0; j < N; j++)
      A[j + i*N] = gsl_rng_uniform(r);
    B[i] = gsl_rng_uniform(r);
  }


  /*
  double A[] = {1, 3, 4, 5, 2, 3, 5, 4, 3, 6, 7, 9};
  double B[] = {1, 2, 3, 4};
  double coef[4*3];
  double pvalue[4*3];
  */
  run_ols(A, B, n, p, coef, pvalue, N);

  gsl_matrix_view m = gsl_matrix_view_array(coef,(N*(N-1))/2, p);
  //printf("%d, %d\n", (&m.matrix)->size1, (&m.matrix)->size2);
  free(A);
  free(B);
  free(pvalue);
  free(coef);

  return 0;
}

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


__global__ void ols_kernel(const double *d_GX,
                            const int n,
                            const int p,
                            const double *d_GY,
                            double *d_Gcoef,
                            double *d_Gtscore,
                            const int N)
{
  int id_i = blockIdx.x;
  int noOfBlocks = (N + blockDim.x - 1)/blockDim.x;
  int id_j;

  // create cublas handle
  cublasHandle_t cublasH = NULL;
  cublasStatus_t cublas_status = cublasCreate_v2(&cublasH);

  const double *d_X1, *d_X2;
  double *d_X3 = (double*)malloc(sizeof(double)*n);

  int *pivotArray = (int *)malloc(p*sizeof(int));
  int *info = (int *)malloc(sizeof(int));
  int batch;
  double *pone = (double*)malloc(sizeof(double));
  *pone = 1.0;
  // just one matrix
  info[0] = 0;
  batch = 1;

  double *d_X = (double*)malloc(sizeof(double) * n * p);
  double *d_Y = (double*)malloc(sizeof(double)*n);

  double *d_XX = (double *)malloc(sizeof(double)*p*p);
  double *d_invXX = (double *)malloc(sizeof(double)*p*p);
  double *d_coef2 = (double *)malloc(sizeof(double)*p);
  double *d_coef = (double *)malloc(sizeof(double)*p);

  double sigma;
  double *psigma = (double*)malloc(sizeof(double));
  int id;
  double tscore;

  double alpha_v = 1.0;
  double beta_v = 0.0;
  double *alpha = &alpha_v, *beta = &beta_v;

  double **a = (double **)malloc(sizeof(double *));
  *a = d_XX;
  const double **aconst = (const double **)a;

  double **c = (double **)malloc(sizeof(double *));
  *c = d_invXX;


  __syncthreads();
  for (int id_ii = 0; id_ii < noOfBlocks; id_ii++)
  {
    id_j = threadIdx.x + id_ii*blockDim.x;
    if (id_j > id_i && id_j < N)
    {
    d_X1 = d_GX + id_i*n;
    d_X2 = d_GX + id_j*n;
    __syncthreads();

    // elements-by-elements
    // x3 = x1.*x2
    cublas_status = cublasDdgmm(cublasH, CUBLAS_SIDE_LEFT,
                            n, 1,
                            d_X1, n,
                            d_X2, 1,
                            d_X3, n);
    __syncthreads();

    // copy to d_Y
    cublas_status = cublasDcopy(cublasH, n,
                             d_GY, 1,
                             d_Y, 1);
    __syncthreads();


    // construct matrix X
    cublas_status = cublasDcopy(cublasH, n,
                             pone, 0,
                             d_X, 1);
    __syncthreads();

    cublas_status = cublasDcopy(cublasH, n,
                             d_X1, 1,
                             d_X+n, 1);
    __syncthreads();

    cublas_status = cublasDcopy(cublasH, n,
                             d_X2, 1,
                             d_X+2*n, 1);
    __syncthreads();

    cublas_status = cublasDcopy(cublasH, n,
                             d_X3, 1,
                             d_X+3*n, 1);
    __syncthreads();


    // //////////////////
    //
    // X'X
    //
    // /////////////////
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

    cublas_status = cublasDgetrfBatched(cublasH, p, a, p, pivotArray, info, batch);
    __syncthreads();
    if (info[0] < 0)
    {
      printf("i = %d, j = %d, in LU decomposition, the %d parameter had an illegeal value\n", id_i, id_j, abs(info[0]));
      continue;
      //return;
    }
    else if (info[0] > 0)
    {
      printf("i = %d, j = %d, in LU decomposition, U(%d, %d) = 0\n", id_i, id_j, abs(info[0]), abs(info[0]));
      continue;
      //return;
    }
    cublas_status = cublasDgetriBatched(cublasH, p, aconst, p, pivotArray,
        c, p, info, batch);
    if (info[0] < 0)
    {
      printf("i = %d, j = %d, in LU decomposition, the %d parameter had an illegeal value\n", id_i, id_j, abs(info[0]));
      continue;
      //return;
    }
    __syncthreads();


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
    //printf("%d, %d, finish X'Y\n", id_i, id_j);
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
    //printf("%d, %d, finish beta\n", id_i, id_j);
    // ///////////////////
    // rss
    // ///////////////////
    *beta = -1.0;
    cublas_status = cublasDgemv(cublasH, CUBLAS_OP_N,
                             n, p,
                             alpha,
                             d_X, n,
                             d_coef, 1,
                             beta,
                             d_Y, 1);
    __syncthreads();
    //printf("%d, %d, finish rss\n", id_i, id_j);
    // sigma ^2 = RSS/(n-p-1)
    //psigma = &sigma;
    cublas_status = cublasDnrm2(cublasH, n, d_Y, 1, psigma);
    __syncthreads();
    sigma = *psigma;

    //printf("%f\n", sigma);
    sigma = sigma/sqrt((n-p)*1.0);
    __syncthreads();
    for (int i = 0; i < p; i++)
    {
      tscore = d_coef[i]/(sigma*sqrt(d_invXX[i+p*i]));
      id = id_i*N+id_j-((id_i+1)*(id_i+2))/2;
      d_Gcoef[i+p*id] = d_coef[i];
      d_Gtscore[i+p*id] = tscore;
    }

    //printf("%d, %d, finish tscore\n", id_i, id_j);
    printf("i = %d, j = %d; beta = %f, %f, %f, %f\n", id_i, id_j, d_coef[0], d_coef[1], d_coef[2], d_coef[3]);
    //printf("i = %d, j = %d; tscore = %f, %f, %f, %f\n", id_i, id_j, d_tscore[0], d_tscore[1], d_tscore[2], d_tscore[3]);
    }
  }
  free(pone);
  free(d_coef2);
  free(d_coef);
  free(d_invXX);
  free(d_X);
  free(d_Y);
  free(pivotArray); // DO NOT free before
  free(info); // DO NOT free before
  free(d_XX);
  free(a);
  free(psigma);
  free(c); // DO NOT free before d_invXX
  free(d_X3);
  cublasDestroy_v2(cublasH);
}

static void
run_ols(const double *G, const double *Y, int n, int p, double *coef, double *tscore, int N)
{
  double *d_G, *d_Y, *d_coef, *d_tscore;

  PERR(cudaMalloc(&d_G, n*N*sizeof(double)));
  PERR(cudaMalloc(&d_Y, n*sizeof(double)));
  PERR(cudaMalloc(&d_coef, (N*(N-1))/2*p*sizeof(double)));
  PERR(cudaMalloc(&d_tscore, (N*(N-1))/2*p*sizeof(double)));
  PERR(cudaMemcpy(d_G, G, n*N*sizeof(double), cudaMemcpyHostToDevice));
  PERR(cudaMemcpy(d_Y, Y, n*sizeof(double), cudaMemcpyHostToDevice));

  int threadsPerBlock = 4;
  int blocksPerGird = N;
//  dim3 blocks(threadsPerBlock, 1);
//  dim3 grids(blocksPerGird, 1);
  //int blocks = N;
  //int grids = N;
  //dim3 blocks(16, 16);
  //dim3 grids((N+15)/16,(N+15)/16);
  //int numBlocks = (N+15)/16;
  //ols_kernel<<<1, 1>>>(d_X, n, p, d_Y, d_coef, d_tscore);
  ols_kernel<<<blocksPerGird, threadsPerBlock>>>(d_G, n, p, d_Y, d_coef, d_tscore, N);
  cudaDeviceSynchronize();
  ERRCHECK;

  PERR(cudaMemcpy(coef, d_coef, (N*(N-1))/2*p*sizeof(double), cudaMemcpyDeviceToHost));
  PERR(cudaMemcpy(tscore, d_tscore, (N*(N-1))/2*p*sizeof(double), cudaMemcpyDeviceToHost));

  PERR(cudaFree(d_G));
  PERR(cudaFree(d_Y));
  PERR(cudaFree(d_coef));
  PERR(cudaFree(d_tscore));
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
  N = 3;
  n = 4;
  /*
  N = 4;
  n = 305;

  double *A = (double*)malloc(sizeof(double)*n*N);
  double *B = (double*)malloc(sizeof(double)*n);
  double *pvalue = (double*)malloc(sizeof(double)*(N*(N-1))/2*p);
  double *coef = (double*)malloc(sizeof(double)*(N*(N-1))/2*p);
  if (!A)
    printf("pvalue malloc error");
  if (!B)
    printf("pvalue malloc error");

  if (!pvalue)
    printf("pvalue malloc error");
  if (!coef)
    printf("coef malloc error");
  */
  /*
  for (size_t i = 0; i < n; i++)
  {
    for (size_t j = 0; j < N; j++)
      A[j + i*N] = gsl_rng_uniform(r);
    B[i] = gsl_rng_uniform(r);
  }
  */


  double A[] = {1, 3, 4, 5, 2, 3, 5, 4, 3, 6, 7, 9};
  double B[] = {1, 2, 3, 4};
  double coef[4*3];
  double pvalue[4*3];

  run_ols(A, B, n, p, coef, pvalue, N);
  /*
  gsl_matrix_view m = gsl_matrix_view_array(coef,(N*(N-1))/2, p);
  free(A);
  free(B);
  free(pvalue);
  free(coef);
  */
  printf("beta0 = %f; pvalue = %f\n", coef[0], pvalue[0]);
  printf("beta1 = %f; pvalue = %f\n", coef[1], pvalue[1]);
  printf("beta2 = %f; pvalue = %f\n", coef[2], pvalue[2]);
  printf("beta3 = %f; pvalue = %f\n", coef[3], pvalue[3]);

  printf("beta0 = %f; pvalue = %f\n", coef[4], pvalue[4]);
  printf("beta1 = %f; pvalue = %f\n", coef[5], pvalue[5]);
  printf("beta2 = %f; pvalue = %f\n", coef[6], pvalue[6]);
  printf("beta3 = %f; pvalue = %f\n", coef[7], pvalue[7]);

  printf("beta0 = %f; pvalue = %f\n", coef[8], pvalue[8]);
  printf("beta1 = %f; pvalue = %f\n", coef[9], pvalue[9]);
  printf("beta2 = %f; pvalue = %f\n", coef[10], pvalue[10]);
  printf("beta3 = %f; pvalue = %f\n", coef[11], pvalue[11]);


  return 0;
}

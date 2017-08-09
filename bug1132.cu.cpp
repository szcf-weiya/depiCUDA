for (i = 0; i < N; i++)
{
  for (j = 0; j < i; j++)
  {
    //other calculation;
  }
}


__global__ void kernel()
{
  int i = blockIdx.x;
  int j = threadIdx.x;
  if (j >= i)
    return;
  {
    //other calculation;
  }
}

dim3 blocks(N, 1);
dim3 grids(N, 1);
kernel<<<grids, blocks>>>();


__global__ void kernel2()
{
  int i = blockIdx.x;
  int noOfBlocks = (i + blockDim.x - 1)/blockDim.x;
  int j;
  for (int ii = 0; ii < noOfBlocks; ii++)
  {
    j = threadIdx.x + ii*blockDim.x;
    if (j >= i)
      return;

    {
      //other calculation;
    }
  }
}

dim3 blocks(minN, 1);
dim3 grids(N, 1);
kernel<<<grids, blocks>>>();

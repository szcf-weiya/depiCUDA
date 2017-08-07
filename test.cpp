#include <iostream>
#include <stdio.h>
#include <stdlib.h>
using namespace std;

int main()
{
  double *A = (double*)malloc(1000*sizeof(double));
  if (!A)
  {
    cout  << "malloc error" << endl;
    return 0;
  }
  cout << A[0] << endl;
  free(A);
}

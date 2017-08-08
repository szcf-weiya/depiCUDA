#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
using namespace std;

void printA(const double *a)
{
  cout << a[0] << endl;
}
void printB(const double *b)
{
  cout << b[0] << endl;
}

int main()
{
  /*
  double *A = (double*)malloc(10000000000000000000000000000000*sizeof(double));
  if (A == ENOMEM)
  {
    cout  << "malloc error" << endl;
    return 0;
  }
  cout << A[0] << endl;
  */
  double a = 1.123;
  double *b = (double*)malloc(sizeof(double));
  *b = 2.2;
  printB(b);

  printA(&a);
}

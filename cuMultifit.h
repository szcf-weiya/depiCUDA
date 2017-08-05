#ifndef _CU_MULTIFIT_H
#define _CU_MULTIFIT_H

// X should include the intercept
int cuMultifit(const double *X, int n, int p, const double *Y, double *coef, double *pvalue);

#endif

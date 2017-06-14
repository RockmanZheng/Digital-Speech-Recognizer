#ifndef MATHUTILITY_H
#define MATHUTILITY_H

#include "TypeConvertor.h"
#include <stdlib.h>
#include <time.h>

extern double LOG_LOW_BOUND;
extern double LOG_ARG_BOUND;
extern double EXP_LOW_BOUND;
extern double EXP_ARG_BOUND;


double
SafeLog(double arg);


double
SafeExp(double arg);

double
ToSafeLog(double arg);

double
LogSum(double a, double b);

double
LogGaussian(const double* x, const double *mean, const double *log_var, int dim);

void
LogEmitProb(Array *log_b, const Array *mean, const Array *log_var, const Matrix *observations);

void
LogMixEmitProb(Matrix *log_mix_b, const Array *log_b, const Matrix *log_coef);


double
GetLogTrans(const CSRMatrix *log_trans, int i, int j);

double
GetLogInit(const SparseVector *log_init, int i);

double
RandRange(double a, double b);

#endif











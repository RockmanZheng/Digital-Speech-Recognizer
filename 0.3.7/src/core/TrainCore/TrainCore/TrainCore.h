#ifndef TRAINCORE_H
#define TRAINCORE_H
#define _USE_MATH_DEFINES
#include "TypeConvertor.h"
#include <math.h>

void
Forward(Matrix *log_alpha, const CSRMatrix *log_trans, const Matrix *log_mix_b);


void
Backward(Matrix *log_beta, const CSRMatrix *log_trans, const Matrix *log_mix_b);


void
LogXi(CSRArray *log_xi, const Matrix *log_alpha, const Matrix *log_beta, const CSRMatrix *log_trans, const Matrix *log_mix_b);


void
LogGamma(Array *log_gamma, Matrix *log_gamma_T, const Matrix *log_alpha, const Matrix *log_beta, const Matrix *log_coef, const Array *log_b, const Matrix *log_mix_b);

void
UpdateLogCoef(Matrix *log_coef, const Matrix *log_gamma_T);

void
UpdateWeight(Array *weights, const Array *log_gamma, const Matrix *log_gamma_T);


void
UpdateLogVar(Array *log_var, const Matrix *observations, const Array *mean, const Array *weights);


void
UpdateMean(Array *mean, const Matrix *observations, const Array *weights);


void
UpdateLogTrans(CSRMatrix *log_trans, const CSRArray *log_xi);


double
LogPY(const Matrix *log_alpha);


void
BWTrainCore(CSRMatrix* log_trans, Matrix* log_coef, Array* mean, Array* log_var, const Matrix* observations);

void
VTrainCore(CSRMatrix *log_trans,
	Matrix *log_coef,
	Array *mean,
	Array *log_var,
	const Observation *observations,// Observations from several wave files 
	int mode// Specify initialize mode, if this is a new model, use flat start
);

// Viterbi training
static PyObject*
VTrainAPI(PyObject *self, PyObject *args);

// Baum-Welch training
static PyObject*
BWTrainAPI(PyObject* self, PyObject* args);

#endif

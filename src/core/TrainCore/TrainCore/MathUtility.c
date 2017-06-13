#include "MathUtility.h"
#include "TypeConvertor.h"
#include <math.h>

double LOG_LOW_BOUND = -1e10;
double LOG_ARG_BOUND = 1e-300;
double EXP_LOW_BOUND = 1e-300;
double EXP_ARG_BOUND = -700.0;

double
SafeLog(double arg)
{
	return arg < LOG_ARG_BOUND ? LOG_LOW_BOUND : log(arg);
}

double
SafeExp(double arg)
{
	return arg < EXP_ARG_BOUND ? EXP_LOW_BOUND : exp(arg);
}

double
ToSafeLog(double arg)
{
	return arg < LOG_LOW_BOUND ? LOG_LOW_BOUND : arg;
}

double
LogSum(double a, double b)
{
	if (a < b)
	{
		return ToSafeLog((a - b) < EXP_ARG_BOUND ? b : (b + SafeLog(1 + SafeExp(a - b))));
	}
	else
	{
		return ToSafeLog((b - a) < EXP_ARG_BOUND ? a : (a + SafeLog(1 + SafeExp(b - a))));
	}
}

static double LOG_2_PI = 1.8378770664093453;

double
LogGaussian(const double *x, const double *mean, const double *log_var, int dim)
{
	double log_prob = -LOG_2_PI * dim;
	double diff;
	int k;
	for (k = 0; k < dim; k++)
	{
		diff = x[k] - mean[k];
		log_prob -= diff * diff / SafeExp(log_var[k]);
	}
	return ToSafeLog(0.5 * log_prob);
}

void LogEmitProb(Array *log_b, const Array *mean, const Array *log_var, const Matrix *observations)
{
	int N, M, T, L;
	N = mean->row;
	M = mean->col;
	T = observations->row;
	L = observations->col;
	double temp;
	int i, m, t;
	for (i = 0; i < N; i++)
	{
		for (m = 0; m < M; m++)
		{
			for (t = 0; t < T; t++)
			{
				log_b->array[i * M * T + m * T + t] = LogGaussian(&observations->mat[t * L], &mean->array[i * M * L + m * L], &log_var->array[i * M * L + m * L], L);
			}
		}
	}
	return;
}

void LogMixEmitProb(Matrix *log_mix_b, const Array *log_b, const Matrix *log_coef)
{
	int N, M, T;
	N = log_b->row;
	M = log_b->col;
	T = log_b->lay;
	int i, m, t;
	double temp;
	for (i = 0; i < N; i++)
	{
		for (t = 0; t < T; t++)
		{
			temp = ToSafeLog(log_coef->mat[i * M] + log_b->array[i * M * T + t]);
			for (m = 1; m < M; m++)
			{
				temp = LogSum(temp, log_coef->mat[i * M + m] + log_b->array[i * M * T + m * T + t]);
			}
			log_mix_b->mat[i * T + t] = temp;
		}
	}
	return;
}

double
GetLogTrans(const CSRMatrix *log_trans, int i, int j)
{
	int k;
	for (k = log_trans->row_ptr[i]; k < log_trans->row_ptr[i + 1]; k++)
	{
		if (log_trans->col_idx[k] == j)
		{
			return log_trans->val[k];
		}
	}
	return LOG_LOW_BOUND;
}

double
GetLogInit(const SparseVector *log_init, int i)
{
	int k;
	for (k = 0; k < log_init->nnz; k++)
	{
		if (log_init->idx[k] == i)
		{
			return log_init->data[k];
		}
	}
	return LOG_LOW_BOUND;
}

double
RandRange(double a, double b)
{
	srand(time(NULL));
	return a + (b - a) * rand() * 1.0 / RAND_MAX;
}

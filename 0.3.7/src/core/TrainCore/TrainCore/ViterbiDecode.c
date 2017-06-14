//#include <Python.h>
//#include <math.h>
//#include "TypeConvertor.h"
//#include "MathUtility.h"
//#include "ViterbiDecode.h"
//
//// Viterbi decoding core
//// Recieve HMM
//// Return the most probable state sequence
//// <param> log_init: initial probability
//double
//VDecodeCore(int *states, const SparseVector *log_init, const CSRMatrix *log_trans, const Matrix *log_coef, const Array *mean, const Array *log_var, const Matrix *observations)
//{
//	int N, M, T, L;
//	N = log_trans->row;
//	M = log_coef->col;
//	T = observations->row;
//	L = observations->col;
//	// Allocate memory
//	Array *log_b = NewArray(N, M, T);
//	Matrix *log_mix_b = NewMat(N, T);
//	Matrix *log_v = NewMat(N, T);
//
//	LogEmitProb(log_b, mean, log_var, observations);
//	LogMixEmitProb(log_mix_b, log_b, log_coef);
//
//	int i, j, k, t;
//	// Initialization
//	for (i = 0; i < N; i++)
//	{
//		log_v->mat[i * T] = ToSafeLog(GetLogInit(log_init, i) + log_mix_b->mat[i * T]);
//	}
//	// Recursion
//	double log_max, temp;
//	for (t = 1; t < T; t++)
//	{
//		for (j = 0; j < N; j++)
//		{
//			log_max = GetLogTrans(log_trans, 0, j) + log_v->mat[t - 1];
//			for (i = 1; i < N; i++)
//			{
//				log_max = fmax(log_max, GetLogTrans(log_trans, i, j) + log_v->mat[i * T + t - 1]);
//			}
//			log_v->mat[j * T + t] = log_mix_b->mat[j * T + t] + log_max;
//		}
//	}
//	// Get the final state of the most probable state sequence
//	log_max = log_v->mat[T - 1];
//	k = 0;
//	for (i = 0; i < N; i++)
//	{
//		if (log_v->mat[i * T + T - 1] > log_max)
//		{
//			k = i;
//			log_max = log_v->mat[i * T + T - 1];
//		}
//	}
//	// Start back tracing
//	states[T - 1] = k;
//	for (t = T - 1; t > 0; t--)
//	{
//		log_max = GetLogTrans(log_trans, 0, states[t]) + log_v->mat[t - 1];
//		k = 0;
//		for (i = 1; i < N; i++)
//		{
//			temp = GetLogTrans(log_trans, i, states[t]) + log_v->mat[i * T + t - 1];
//			if (temp > log_max)
//			{
//				k = i;
//				log_max = temp;
//			}
//		}
//		states[t - 1] = k;
//	}
//
//	// Clean up
//	double ans = log_v->mat[N * T - 1];
//	FreeArray(log_b);
//	FreeMat(log_mix_b);
//	FreeMat(log_v);
//	return ans;
//}
//
//// Viterbi decoding API
//static PyObject *
//VDecodeAPI(PyObject *self, PyObject args)
//{
//	return NULL;
//}

#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include "TrainCore.h"
#include "MathUtility.h"
#include "ViterbiDecode.h"
#include <string.h>
#include <time.h>

/*************************** Begin Core Codes ******************************/
#define _USE_MATH_DEFINES
#include <math.h>

void Forward(Matrix *log_alpha,
	const CSRMatrix *log_trans,
	const Matrix *log_mix_b)
{
	int N, T;
	N = log_alpha->row;
	T = log_alpha->col;
	log_alpha->mat[0] = log_mix_b->mat[0];
	int i, j, t;
	for (i = 1; i < N; i++)
	{
		log_alpha->mat[i * T] = LOG_LOW_BOUND;
	}
	double temp;
	for (t = 1; t < T; t++)
	{
		for (i = 0; i < N; i++)
		{
			temp = ToSafeLog(log_alpha->mat[t - 1] + GetLogTrans(log_trans, 0, i));
			for (j = 1; j < N; j++)
			{
				temp = LogSum(temp, ToSafeLog(log_alpha->mat[j * T + t - 1] + GetLogTrans(log_trans, j, i)));
			}
			log_alpha->mat[i * T + t] = ToSafeLog(temp + log_mix_b->mat[i * T + t]);
		}
	}
	return;
}

void Backward(Matrix *log_beta,
	const CSRMatrix *log_trans,
	const Matrix *log_mix_b)
{
	int N, T;
	N = log_beta->row;
	T = log_beta->col;
	int t, i, j;
	double temp;
	for (t = T - 2; t >= 0; t--)
	{
		for (i = 0; i < N; i++)
		{
			temp = ToSafeLog(log_beta->mat[t + 1] + GetLogTrans(log_trans, i, 0) + log_mix_b->mat[t + 1]);
			for (j = 1; j < N; j++)
			{
				temp = LogSum(temp, log_beta->mat[j * T + t + 1] + GetLogTrans(log_trans, i, j) + log_mix_b->mat[j * T + t + 1]);
			}
			log_beta->mat[i * T + t] = temp;
		}
	}
	return;
}

void LogXi(CSRArray *log_xi,
	const Matrix *log_alpha,
	const Matrix *log_beta,
	const CSRMatrix *log_trans,
	const Matrix *log_mix_b)
{

	int i, j, k, t;
	int N, T;
	N = log_trans->row;
	T = log_trans->col;
	// Clear memory
	memset(log_xi->val, 0, log_xi->nnz * (T - 1) * sizeof(double));
	// Only need to compute numerator
	for (i = 0; i < N; i++)
	{
		for (k = log_xi->row_ptr[i]; k < log_xi->row_ptr[i + 1]; k++)
		{
			j = log_trans->col_idx[k];
			for (t = 0; t < log_xi->lay; t++)
			{
				log_xi->val[k * log_xi->lay + t] = ToSafeLog(log_alpha->mat[i * T + t] + log_beta->mat[j * T + t + 1] + GetLogTrans(log_trans, i, j) + log_mix_b->mat[j * T + t + 1]);
			}
		}
	}
	return;
}

double
GetLogXi(const CSRArray *log_xi,
	int i,
	int j,
	int t)
{
	int k;
	for (k = log_xi->row_ptr[i]; k < log_xi->row_ptr[i + 1]; k++)
	{
		if (log_xi->col_idx[k] == j)
		{
			return log_xi->val[k * log_xi->nnz + t];
		}
	}
	return LOG_LOW_BOUND;
}

void LogGamma(Array *log_gamma,
	Matrix *log_gamma_T,
	const Matrix *log_alpha,
	const Matrix *log_beta,
	const Matrix *log_coef,
	const Array *log_b,
	const Matrix *log_mix_b)
{
	int N, M, T;
	N = log_b->row;
	M = log_b->col;
	T = log_b->lay;

	int t, i, j, m;
	double temp;
	// Compute log_gamma
	// Only need to compute numerator
	for (i = 0; i < N; i++)
	{
		for (m = 0; m < M; m++)
		{
			for (t = 0; t < T; t++)
			{
				log_gamma->array[i * M * T + m * T + t] = ToSafeLog(log_alpha->mat[i * T + t] + log_beta->mat[i * T + t] + log_coef->mat[i * M + m] + log_b->array[i * M * T + m * T + t] - log_mix_b->mat[i * T + t]);
			}
		}
	}
	// Compute log_gamma_T
	for (i = 0; i < N; i++)
	{
		for (m = 0; m < M; m++)
		{
			temp = log_gamma->array[i * M * T + m * T];
			for (t = 1; t < T; t++)
			{
				temp = LogSum(temp, log_gamma->array[i * M * T + m * T + t]);
			}
			log_gamma_T->mat[i * M + m] = temp;
		}
	}
	return;
}

void UpdateLogCoef(Matrix *log_coef,
	const Matrix *log_gamma_T)
{
	int N, M, T;
	N = log_coef->row;
	M = log_coef->col;
	int i, m, s;
	double num, den;
	for (i = 0; i < N; i++)
	{
		for (m = 0; m < M; m++)
		{
			num = log_gamma_T->mat[i * M + m];
			den = log_gamma_T->mat[i * M];
			for (s = 1; s < M; s++)
			{
				den = LogSum(den, log_gamma_T->mat[i * M + s]);
			}
			log_coef->mat[i * M + m] = ToSafeLog(num - den);
		}
	}
	return;
}

// Used in updating log variance and mean
void UpdateWeight(Array *weights,
	const Array *log_gamma,
	const Matrix *log_gamma_T)
{
	int N, M, T;
	N = weights->row;
	M = weights->col;
	T = weights->lay;
	int i, m, t;
	for (i = 0; i < N; i++)
	{
		for (m = 0; m < M; m++)
		{
			for (t = 0; t < T; t++)
			{
				weights->array[i * M * T + m * T + t] = SafeExp(log_gamma->array[i * M * T + m * T + t] - log_gamma_T->mat[i * M + m]);
			}
		}
	}
	return;
}

void UpdateLogVar(Array *log_var,
	const Matrix *observations,
	const Array *mean,
	const Array *weights)
{
	int N, M, T, L;
	N = log_var->row;
	M = log_var->col;
	L = log_var->lay;
	T = observations->row;
	int i, m, t, l;
	double temp, diff;
	for (i = 0; i < N; i++)
	{
		for (m = 0; m < M; m++)
		{
			for (l = 0; l < L; l++)
			{
				temp = 0.0;
				for (t = 0; t < T; t++)
				{
					diff = observations->mat[t * L + l] - mean->array[i * M * L + m * L + l];
					temp += weights->array[i * M * T + m * T + t] * diff * diff;
				}
				log_var->array[i * M * L + m * L + l] = SafeLog(temp);
			}
		}
	}
	return;
}

void UpdateMean(Array *mean,
	const Matrix *observations,
	const Array *weights)
{
	int N, M, T, L;
	N = mean->row;
	M = mean->col;
	L = mean->lay;
	T = observations->row;
	// Clear memory
	memset(mean->array, 0, mean->row * mean->col * mean->lay * sizeof(double));
	int i, m, t, l;
	for (i = 0; i < N; i++)
	{
		for (m = 0; m < M; m++)
		{
			for (t = 0; t < T; t++)
			{
				for (l = 0; l < L; l++)
				{
					mean->array[i * M * L + m * L + l] += weights->array[i * M * T + m * T + t] * observations->mat[t * L + l];
				}
			}
		}
	}
	return;
}

void UpdateLogTrans(CSRMatrix *log_trans,
	const CSRArray *log_xi)
{
	int N, T;
	N = log_trans->row;
	int k, i, t;
	double num, den;
	for (i = 0; i < N; i++)
	{
		for (k = log_trans->row_ptr[i]; k < log_trans->row_ptr[i + 1]; k++)
		{
			num = log_xi->val[k * log_xi->lay];
			for (t = 1; t < log_xi->lay; t++)
			{
				num = LogSum(num, log_xi->val[k * log_xi->lay + t]);
			}
			log_trans->val[k] = num;
		}
	}
	for (i = 0; i < N; i++)
	{
		den = LOG_LOW_BOUND;
		// Sum over j to obtain denominator
		for (k = log_trans->row_ptr[i]; k < log_trans->row_ptr[i + 1]; k++)
		{
			den = LogSum(den, log_trans->val[k]);
		}
		// Divide
		for (k = log_trans->row_ptr[i]; k < log_trans->row_ptr[i + 1]; k++)
		{
			log_trans->val[k] = ToSafeLog(log_trans->val[k] - den);
		}
	}
	return;
}

double
LogPY(const Matrix *log_alpha)
{
	int N, T;
	N = log_alpha->row;
	T = log_alpha->col;
	double log_p_y;
	log_p_y = log_alpha->mat[T - 1];
	int i;
	for (i = 1; i < N; i++)
	{
		log_p_y = LogSum(log_p_y, log_alpha->mat[i * T + T - 1]);
	}
	return log_p_y;
}

void BWTrainCore(CSRMatrix *log_trans,
	Matrix *log_coef,
	Array *mean,
	Array *log_var,
	const Matrix *observations)
{
	double error = 5e-4;
	double old_log_p_y = 0.0;
	double log_p_y;
	int N, M, T, L;
	N = log_trans->row;
	M = log_coef->col;
	T = observations->row;
	L = observations->col;
	// Allocate memory
	Array *log_b = NewArray(N, M, T);
	Matrix *log_mix_b = NewMat(N, T);
	Matrix *log_alpha = NewMat(N, T);
	Matrix *log_beta = NewMat(N, T);
	CSRArray *log_xi = NewCSRArray(log_trans->nnz, N, N, T - 1);
	Array *log_gamma = NewArray(N, M, T);
	Matrix *log_gamma_T = NewMat(N, M);
	Array *weights = NewArray(N, M, T);
	//  UpdatedValue *update_val = NewUpdateVal(N,M,L);

	// Log xi has the same non zero structure as log transition
	memcpy(log_xi->row_ptr, log_trans->row_ptr, (N + 1) * sizeof(int));
	memcpy(log_xi->col_idx, log_trans->col_idx, log_trans->nnz * sizeof(int));

	while (1)
	{
		//printf("BWTrain: Preparing.\n");
		// Compute log_b
		LogEmitProb(log_b, mean, log_var, observations);
		// Compute log_mix_b
		LogMixEmitProb(log_mix_b, log_b, log_coef);
		// Forward procedure
		Forward(log_alpha, log_trans, log_mix_b);
		// Backward
		Backward(log_beta, log_trans, log_mix_b);
		// Compute log xi
		LogXi(log_xi, log_alpha, log_beta, log_trans, log_mix_b);
		// Compute log gamma
		LogGamma(log_gamma, log_gamma_T, log_alpha, log_beta, log_coef, log_b, log_mix_b);

		// Update parameters
		//printf("BWTrain: Updating parameters.\n");
		UpdateLogCoef(log_coef, log_gamma_T);
		UpdateWeight(weights, log_gamma, log_gamma_T);
		UpdateLogVar(log_var, observations, mean, weights);
		UpdateMean(mean, observations, weights);
		UpdateLogTrans(log_trans, log_xi);

		// Compute log p(y)
		log_p_y = LogPY(log_alpha);

		//printf("BWTrain: Log likelihood of observations: %f\n", log_p_y);
		// Check convergence
		double delta = fabs(old_log_p_y / log_p_y - 1.0);
		//printf("BWTrain: delta = %f\n", delta);
		if (delta < error)
		{
			break;
		}
		// Prepare for next iteration
		old_log_p_y = log_p_y;
	}
	// Clean up
	FreeArray(log_b);
	FreeMat(log_mix_b);
	FreeMat(log_alpha);
	FreeMat(log_beta);
	FreeCSRArray(log_xi);
	FreeArray(log_gamma);
	FreeMat(log_gamma_T);
	FreeArray(weights);
	return;
}

// K-means clustering algorithm
int *KMeans(const Matrix *data, // each row contains a sample vector
	int k               // specify number of clusters
						//int *cluster_idx// output cluster index indicating to which cluster each sample belongs
)
{
	int N, D;
	N = data->row;
	D = data->col;
	// Variation for each compoents in vector
	double *var = (double *)calloc(D, sizeof(double));
	double *mean = (double *)calloc(D, sizeof(double));
	int i, j, d;
	// Compute variation
	// For each dimension
	for (d = 0; d < D; d++)
	{
		// For each data
		for (i = 0; i < N; i++)
		{
			mean[d] += data->mat[i * D + d];
		}
		mean[d] /= N;
		for (i = 0; i < N; i++)
		{
			var[d] += pow(data->mat[i * D + d] - mean[d], 2);
		}
		// Unbiased estimate
		var[d] /= N - 1;
	}

	Matrix *centroids = NewMat(k, D);
	Matrix *new_centroids = NewMat(k, D);

	double dist, min_dist;
	int *cluster_idx = (int *)calloc(N, sizeof(int));
	int *cluster_size = (int *)calloc(k, sizeof(int));

	// If the number of data is no more than k
	if (N <= k)
	{
		int i;
		for (i = 0; i < N; i++)
		{
			cluster_idx[i] = i;
		}
	}
	else
	{
		int flag = 1;
		while (flag)
		{
			// Randomly generate centroids
			srand(time(NULL));
			// For each cluster
			for (i = 0; i < k; i++)
			{
				// For each dimension
				for (d = 0; d < D; d++)
				{
					centroids->mat[i * D + d] = mean[d] + (2 * rand() * (1.0 / RAND_MAX) - 1.0) * sqrt(var[d]);
				}
			}
			// Start clustering
			while (1)
			{
				// Cluster
				// For each data
				for (i = 0; i < N; i++)
				{
					min_dist = HUGE_VAL;
					// Compute its distance from each centroids
					// And assign it to the nearest one
					for (j = 0; j < k; j++)
					{
						// Compute Mahalanobis distance from this centroid
						dist = 0.0;
						for (d = 0; d < D; d++)
						{
							dist += pow(data->mat[i * D + d] - centroids->mat[j * D + d], 2) / var[d];
						}
						// Assign data to the nearest centroid
						if (dist < min_dist)
						{
							min_dist = dist;
							cluster_idx[i] = j;
						}
					}
				}
				// Update centroids
				// Clear previous centroids
				memset(new_centroids->mat, 0, sizeof(double) * k * D);
				memset(cluster_size, 0, sizeof(int) * k);
				for (i = 0; i < N; i++)
				{
					j = cluster_idx[i];
					cluster_size[j]++;
					// Sum data of the same cluster
					for (d = 0; d < D; d++)
					{
						new_centroids->mat[j * D + d] += data->mat[i * D + d];
					}
				}
				// Divide
				for (j = 0; j < k; j++)
				{
					for (d = 0; d < D; d++)
					{
						new_centroids->mat[j * D + d] /= cluster_size[j];
					}
				}
				// Iterate until convergence
				double error = 1e-7;
				// Compute the differences between old and new centroids
				dist = 0.0;
				// For each cluster
				for (j = 0; j < k; j++)
				{
					// For each dimension
					for (d = 0; d < D; d++)
					{
						dist += pow(centroids->mat[j * D + d] - new_centroids->mat[j * D + d], 2) / var[d];
					}
				}
				if (isnan(dist))
				{
					//std::cerr<<"Warning: Number of clusters is less than "<<k<<". Restarting. "<<"!\n";
					flag = 1;
					break;
				}
				else
				{
					// End
					//	std::cout<<"Error: "<<dist<<std::endl;
					if (dist < error)
					{
						flag = 0;
						break;
					}
				}

				// Update
				memcpy(centroids->mat, new_centroids->mat, sizeof(double) * k * D);
			}
		}
	}
	// Clean up memory
	FreeMat(centroids);
	FreeMat(new_centroids);
	free(var);
	free(mean);
	free(cluster_size);
	return cluster_idx;
}

// Flat start
// First segment observation uniformly
// Then initialize coefficients, means, and variances using
// data from the same group
void FlatStart(CSRMatrix *log_trans)
{
	srand(time(NULL));
	// Retrieve dimension information
	int N, M, D, R, T_sum;
	N = log_trans->row;
	// Flat start: initialize log transition probability
	int i, j, t, d, m;
	double temp;
	for (i = 0; i < N; i++)
	{
		temp = 0.0;
		for (j = log_trans->row_ptr[i]; j < log_trans->row_ptr[i + 1]; j++)
		{
			log_trans->val[j] = rand();
			temp += log_trans->val[j];
		}
		temp = log(temp);
		for (j = log_trans->row_ptr[i]; j < log_trans->row_ptr[i + 1]; j++)
		{
			log_trans->val[j] = ToSafeLog(log(log_trans->val[j]) - temp);
		}
	}
}

double
VDecode(int *state_align,
	const CSRMatrix *log_trans,
	const Matrix *log_coef,
	const Array *mean,
	const Array *log_var,
	const Matrix *observations)
{
	int N, M, T, L;
	N = log_trans->row;
	M = log_coef->col;
	T = observations->row;
	L = observations->col;
	// Allocate memory
	Array *log_b = NewArray(N, M, T);
	Matrix *log_mix_b = NewMat(N, T);
	Matrix *log_v = NewMat(N, T);

	LogEmitProb(log_b, mean, log_var, observations);
	LogMixEmitProb(log_mix_b, log_b, log_coef);

	int i, j, k, t;
	// Initialization
	log_v->mat[0] = 0.0; // Start state
	for (i = 1; i < N; i++)
	{
		log_v->mat[i * T] = SafeLog(0.0);
	}
	// Recursion
	double log_max, temp;
	for (t = 1; t < T; t++)
	{
		for (j = 0; j < N; j++)
		{
			log_max = ToSafeLog(GetLogTrans(log_trans, 0, j) + log_v->mat[t - 1]);
			for (i = 1; i < N; i++)
			{
				log_max = fmax(log_max, ToSafeLog(GetLogTrans(log_trans, i, j) + log_v->mat[i * T + t - 1]));
			}
			log_v->mat[j * T + t] = ToSafeLog(log_mix_b->mat[j * T + t] + log_max);
		}
	}
	// Get the final state of the most probable state sequence
	/* log_max = log_v->mat[T-1]; */
	/* k=0; */
	/* for(i=1;i<N;i++){ */
	/*   if(log_v->mat[i*T+T-1]>log_max){ */
	/*     k = i; */
	/*     log_max = log_v->mat[i*T+T-1]; */
	/*   } */
	/* } */
	// Start back tracing
	memset(state_align, 0, sizeof(int) * T);
	//state_align[T-1] = k;
	// Force to be final state
	state_align[T - 1] = N - 1;
	for (t = T - 1; t > 0; t--)
	{
		log_max = ToSafeLog(GetLogTrans(log_trans, 0, state_align[t]) + log_v->mat[t - 1]);
		k = 0;
		for (i = 1; i < N; i++)
		{
			temp = ToSafeLog(GetLogTrans(log_trans, i, state_align[t]) + log_v->mat[i * T + t - 1]);
			if (temp > log_max)
			{
				k = i;
				log_max = temp;
			}
		}
		state_align[t - 1] = k;
	}

	// Clean up
	double ans = log_v->mat[N * T - 1];
	FreeArray(log_b);
	FreeMat(log_mix_b);
	FreeMat(log_v);
	return ans;
}

// Estimate parameters according to states alignment
void VEstimate(const int **state_align,
	const Observation *observations,
	Matrix *log_coef,
	Array *mean,
	Array *log_var)
{
	int N, M, D, R;
	N = log_coef->row;
	M = log_coef->col;
	R = observations->R;
	D = mean->lay;
	int i, j, k, d, m, t, c, r;
	int *T = (int *)calloc(R, sizeof(int));
	for (r = 0; r < R; r++)
	{
		T[r] = observations->data[r]->row;
	}
	int **component_align = (int **)calloc(N, sizeof(int *));

	Observation *obs_group;
	obs_group = NewObservation(N);
	//int /***states_time,**states_time_ptr,*/
	int *states_num, **obs_ptr, **time_ptr, *component_num;
	states_num = (int *)calloc(N, sizeof(int));
	obs_ptr = (int **)calloc(N, sizeof(int *));
	time_ptr = (int **)calloc(N, sizeof(int *));
	// Number of observation that corresponds to the same state and component
	component_num = (int *)calloc(N * M, sizeof(int));
	// Within each state, a further alignment of observations to mixture components is made
	// Using clustering
	// For each state

	// First loop count the number of states
	memset(states_num, 0, N * sizeof(int));
	for (r = 0; r < R; r++)
	{
		for (t = 0; t < T[r]; t++)
		{
			states_num[state_align[r][t]]++;
		}
	}
	for (i = 0; i < N; i++)
	{
		obs_ptr[i] = (int *)calloc(states_num[i], sizeof(int));
		time_ptr[i] = (int *)calloc(states_num[i], sizeof(int));
	}
	// Second loop group observations of the same state together

	for (i = 0; i < N; i++)
	{
		obs_group->data[i] = NewMat(states_num[i], D);
	}
	// Use states_num as pointer to indicate how many observation corresponding to the same state we have grouped
	memset(states_num, 0, N * sizeof(int));
	for (r = 0; r < R; r++)
	{
		for (t = 0; t < T[r]; t++)
		{
			i = state_align[r][t];
			j = states_num[i];
			memcpy(obs_group->data[i]->mat + j * D, observations->data[r]->mat + t * D, D * sizeof(double));
			time_ptr[i][j] = t;
			obs_ptr[i][j] = r;
			states_num[i]++;
		}
	}
	// Use k means clustering to align components
	memset(component_num, 0, N * M * sizeof(int));
	for (i = 0; i < N; i++)
	{
		component_align[i] = KMeans(obs_group->data[i], M);
		for (j = 0; j < states_num[i]; j++)
		{
			component_num[i * M + component_align[i][j]]++;
		}
	}
	// Estimate parameters
	// Estimate log coefficients
	memset(log_coef->mat, 0, N * M * sizeof(double));
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < states_num[i]; j++)
		{
			log_coef->mat[i * M + component_align[i][j]] += 1.0;
		}
		for (m = 0; m < M; m++)
		{
			log_coef->mat[i * M + m] = ToSafeLog(log(log_coef->mat[i * M + m]) - log(states_num[i]));
		}
	}
	// Estimate mean
	memset(mean->array, 0, N * M * D * sizeof(double));
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < states_num[i]; j++)
		{
			m = component_align[i][j];
			r = obs_ptr[i][j];
			t = time_ptr[i][j];
			for (d = 0; d < D; d++)
			{
				mean->array[i * M * D + m * D + d] += observations->data[r]->mat[t * D + d];
			}
		}
	}
	for (i = 0; i < N; i++)
	{
		for (m = 0; m < M; m++)
		{
			for (d = 0; d < D; d++)
			{
				// Normalize only when the number of components is greater than 1
				if (component_num[i * M + m] > 1)
				{
					mean->array[i * M * D + m * D + d] /= component_num[i * M + m];
				}
			}
		}
	}
	// Estimate log variance
	memset(log_var->array, 0, N * M * D * sizeof(double));
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < states_num[i]; j++)
		{
			m = component_align[i][j];
			r = obs_ptr[i][j];
			t = time_ptr[i][j];
			for (d = 0; d < D; d++)
			{
				log_var->array[i * M * D + m * D + d] += pow(mean->array[i * M * D + m * D + d] - observations->data[r]->mat[t * D + d], 2);
			}
		}
	}
	for (i = 0; i < N; i++)
	{
		for (m = 0; m < M; m++)
		{
			for (d = 0; d < D; d++)
			{
				// Normalize only when the number of components is greater than 1
				if (component_num[i * M + m] > 0)
				{
					log_var->array[i * M * D + m * D + d] = ToSafeLog(log(log_var->array[i * M * D + m * D + d]) - log(component_num[i * M + m]));
				}
			}
		}
	}
	// Clean up
	for (i = 0; i < N; i++)
	{
		free(component_align[i]);
		free(obs_ptr[i]);
		free(time_ptr[i]);
	}
	free(component_align);
	free(obs_ptr);
	free(time_ptr);
	free(component_num);
	//free(states_time);
	//free(states_time_ptr);
	free(states_num);
	free(T);
	FreeObservation(obs_group);
}

// Viterbi training is used only on single model
// Use flat start instead of uniform alignment to initialize model parameters, if this is a new model
// Then use Viterbi decoding to align states with observations
// Estimate parameters
// Iterate until convergence
/***********************************************************
2017-4-21
TODO:
Do not use flat start to initialize a new model!
This will cause 0.0 emission probability, and thus lead to state alignment with all 0's.
Use random segmentation based on transition probability to initialize new model.
Or first segment observations uniformly, and then initialize coefficients, means, and variances
with observations in the same group.

 ***********************************************************/

void VTrainCore(CSRMatrix *log_trans,
	Matrix *log_coef,
	Array *mean,
	Array *log_var,
	const Observation *observations, // Observations from several wave files
	int mode                         // Specify initialize mode, if this is a new model, use flat start
)
{
	double error = 5e-4;
	double old_log_p = 0.0;
	double log_p, den;
	int N, M, D, R, T_sum;
	N = log_trans->row;
	M = log_coef->col;
	R = observations->R;
	D = mean->lay;
	int i, j, k, d, m, t, c, r;
	int *T = (int *)calloc(R, sizeof(int));
	T_sum = 0;
	for (r = 0; r < R; r++)
	{
		T[r] = observations->data[r]->row;
		T_sum += T[r];
	}
	int **state_align = (int **)calloc(R, sizeof(int *));
	for (r = 0; r < R; r++)
	{
		state_align[r] = (int *)calloc(T[r], sizeof(int));
	}
	// If this is a new model
	if (mode == 0)
	{
		// Flat start: Initialize transition probability
		FlatStart(log_trans);
		// Fake uniform alignment
		for (r = 0; r < R; r++)
		{
			for (t = 0; t < T[r]; t++)
			{
				state_align[r][t] = N * 1.0 * t / T[r];
			}
		}
		// Estimate parameters
		VEstimate(state_align, observations, log_coef, mean, log_var);
	}
	old_log_p = 0.0;
	for (r = 0; r < R; r++)
	{
		old_log_p += VDecode(state_align[r], log_trans, log_coef, mean, log_var, observations->data[r]);
	}
	double delta;
	do
	{
		// Estimate parameters
		memset(log_trans->val, 0, sizeof(double) * log_trans->nnz);
		// Update log_trans
		// Accumulate
		// Count the number of transition from i to j
		for (r = 0; r < R; r++)
		{
			for (t = 1; t < T[r]; t++)
			{
				i = state_align[r][t - 1];
				j = state_align[r][t];
				k = GetCSRIdx(log_trans, i, j);
				log_trans->val[k] += 1.0;
			}
		}
		// Normalize
		// Sum over j to get denominator
		for (i = 0; i < N; i++)
		{
			den = 0.0;
			for (j = 0; j < N; j++)
			{
				if (IsInCSRMat(log_trans, i, j))
				{
					den += GetLogTrans(log_trans, i, j);
				}
			}
			for (j = 0; j < N; j++)
			{

				if (IsInCSRMat(log_trans, i, j))
				{
					if (den > 0.0)
					{
						SetCSRMat(log_trans, i, j, ToSafeLog(log(GetLogTrans(log_trans, i, j)) - log(den)));
					}
					else
					{
						SetCSRMat(log_trans, i, j, SafeLog(0.0));
					}
				}
			}
		}

		// Estimate emission parameters
		VEstimate(state_align, observations, log_coef, mean, log_var);

		// Use Viterbi decoding to align states
		log_p = 0.0;
		for (r = 0; r < R; r++)
		{
			log_p += VDecode(state_align[r], log_trans, log_coef, mean, log_var, observations->data[r]);
		}
		// Iterate until convergence
		delta = fabs(old_log_p / log_p - 1.0);

		printf("VTrain: delta = %f\n", delta);
		// Update
		old_log_p = log_p;
	} while (delta > error);

	// Clean up
	for (r = 0; r < R; r++)
	{
		free(state_align[r]);
	}
	free(state_align);
	free(T);
}


int VDecodeCore(const Model** models, int num_models, const Matrix* observations) {
	double* log_ps = (double*)calloc(num_models, sizeof(double));
	int i;
	int *state_align = (int*)calloc(observations->row, sizeof(int));
	for (i = 0; i < num_models; i++) {
		log_ps[i] = VDecode(state_align, models[i]->log_trans, models[i]->log_coef, models[i]->mean, models[i]->log_var, observations);
		//printf("VDecode: model = %i, log_p = %f\n", i, log_ps[i]);
	}
	double max_log_p = log_ps[0];
	int ans = 0;
	for (i = 1; i < num_models; i++) {
		if (log_ps[i] > max_log_p) {
			max_log_p = log_ps[i];
			ans = i;
		}
	}
	free(state_align);
	return ans;
}

/************************** End Core Codes **********************************/

PyDoc_STRVAR(TrainCore_VTrain_doc, "Use Viterbi algorithm to roughly estimate parameters for HMM.\
VTrain(num_state, num_component, network, observations)");

static PyObject *
VTrainAPI(PyObject *self, PyObject *args)
{
	PyObject *PyNet, *PyObservations, *PyNumComponent, *PyNumState, *Pylog_trans, *Pylog_coef, *Pymean, *Pylog_var;
	if (!PyArg_ParseTuple(args, "OOOO", &PyNumState, &PyNumComponent, &PyNet, &PyObservations))
		return NULL;

	int N, M, D;
	N = PyLong_AsLong(PyNumState);
	M = PyLong_AsLong(PyNumComponent);
	D = PyList_Size(PyList_GetItem(PyList_GetItem(PyObservations, 0), 0));
	// Type conversion from Python to C

	CSRMatrix *log_trans;
	Matrix *log_coef;
	Array *mean, *log_var;
	Observation *observations;
	Network *net;
	COOMatrix *coo, *log_trans_coo;

	net = PyList2Network(PyNet, N, N);
	coo = Network2COO(net);
	log_trans = COO2CSR(coo);
	observations = PyList2Observation(PyObservations);
	log_coef = NewMat(N, M);
	mean = NewArray(N, M, D);
	log_var = NewArray(N, M, D);

	// Call train function
	VTrainCore(log_trans, log_coef, mean, log_var, observations, 0);

	// Type conversion from C to Python
	log_trans_coo = CSR2COO(log_trans);
	Pylog_trans = COOMat2PyList(log_trans_coo);
	Pylog_coef = Matrix2PyList(log_coef);
	Pymean = Array2PyList(mean);
	Pylog_var = Array2PyList(log_var);

	// Clean up
	FreeCOOMat(coo);
	FreeCOOMat(log_trans_coo);
	FreeCSRMat(log_trans);
	FreeMat(log_coef);
	FreeObservation(observations);
	FreeArray(mean);
	FreeArray(log_var);
	FreeNetwork(net);

	// Return
	return Py_BuildValue("OOOO", Pylog_trans, Pylog_coef, Pymean, Pylog_var);
}

PyDoc_STRVAR(TrainCore_BWTrain_doc, "Use Baum-Welch algorithm to train HMM.\
BWTrain(log_trans, log_coef, log_var,observations)");

// Baum-Welch training
static PyObject *
BWTrainAPI(PyObject *self, PyObject *args)
{
	PyObject *Pylog_trans, *Pylog_coef, *Pymean, *Pylog_var, *Pyobservations;
	if (!PyArg_ParseTuple(args, "OOOOO", &Pylog_trans, &Pylog_coef, &Pymean, &Pylog_var, &Pyobservations))
		return NULL;

	// Type conversion from Python to C
	COOMatrix *log_trans_coo;
	CSRMatrix *log_trans;
	Matrix *log_coef, *observations;
	Array *mean, *log_var;

	log_coef = PyList2Matrix(Pylog_coef);
	observations = PyList2Matrix(Pyobservations);
	mean = PyList2Array(Pymean);
	log_var = PyList2Array(Pylog_var);
	log_trans_coo = PyList2COOMat(Pylog_trans, log_coef->row, log_coef->row);
	log_trans = COO2CSR(log_trans_coo);
	FreeCOOMat(log_trans_coo);
	// Call train function
	BWTrainCore(log_trans, log_coef, mean, log_var, observations);

	// Type conversion from C to Python
	log_trans_coo = CSR2COO(log_trans);
	Pylog_trans = COOMat2PyList(log_trans_coo);
	Pylog_coef = Matrix2PyList(log_coef);
	Pymean = Array2PyList(mean);
	Pylog_var = Array2PyList(log_var);

	// Clean up
	FreeCOOMat(log_trans_coo);
	FreeCSRMat(log_trans);
	FreeMat(log_coef);
	FreeMat(observations);
	FreeArray(mean);
	FreeArray(log_var);

	// Return
	return Py_BuildValue("OOOO", Pylog_trans, Pylog_coef, Pymean, Pylog_var);
}

PyDoc_STRVAR(TrainCore_VDecode_doc, "Use Viterbi algorithm to decode. \
VDecode(models,observations)");

// Viterbi decoding
static PyObject *
VDecodeAPI(PyObject *self, PyObject *args)
{
	PyObject *PyModels, *PyObservations, *PyAns;
	if (!PyArg_ParseTuple(args, "OO", &PyModels, &PyObservations))
		return NULL;

	int num_models = PyList_Size(PyModels);
	Model** models = (Model**)calloc(num_models, sizeof(Model*));
	int i;
	for (i = 0; i < num_models; i++) {
		PyObject *PyModel = PyList_GetItem(PyModels, i);
		models[i] = PyList2Model(PyModel);
	}
	Matrix* observations = PyList2Matrix(PyObservations);

	int ans = VDecodeCore(models, num_models, observations);
	PyAns = PyLong_FromLong(ans);

	// Clean up
	for (i = 0; i < num_models; i++) {
		FreeModel(models[i]);
	}
	free(models);
	FreeMat(observations);
	// Return
	return Py_BuildValue("O", PyAns);
}



static PyMethodDef TrainCore_functions[] = {
	{"BWTrain", (PyCFunction)BWTrainAPI, METH_VARARGS, TrainCore_BWTrain_doc},
	{"VTrain", (PyCFunction)VTrainAPI, METH_VARARGS, TrainCore_VTrain_doc },
	{ "VDecode", (PyCFunction)VDecodeAPI, METH_VARARGS, TrainCore_VDecode_doc },
	{NULL, NULL, 0, NULL} };

/*
 * Initialize TrainCore. May be called multiple times, so avoid
 * using static state.
 */
int exec_TrainCore(PyObject *module) {
	PyModule_AddFunctions(module, TrainCore_functions);

	PyModule_AddStringConstant(module, "__author__", "rockm");
	PyModule_AddStringConstant(module, "__version__", "1.0.0");
	PyModule_AddIntConstant(module, "year", 2017);

	return 0; /* success */
}
/*
 * Documentation for TrainCore.
 */
PyDoc_STRVAR(TrainCore_doc, "The TrainCore module");


static PyModuleDef_Slot TrainCore_slots[] = {
	{ Py_mod_exec, exec_TrainCore },
	{ 0, NULL }
};

static PyModuleDef TrainCore_def = {
	PyModuleDef_HEAD_INIT,
	"TrainCore",
	TrainCore_doc,
	0,              /* m_size */
	NULL,           /* m_methods */
	TrainCore_slots,
	NULL,           /* m_traverse */
	NULL,           /* m_clear */
	NULL,           /* m_free */
};

PyMODINIT_FUNC PyInit_TrainCore() {
	return PyModuleDef_Init(&TrainCore_def);
}
//
//PyMODINIT_FUNC PyInit_TrainCore_d() {
//	return PyModuleDef_Init(&TrainCore_def);
//}

#include <Python.h>
#include "TypeConvertor.h"
#include <stdlib.h>
#include <string.h>

Array *
PyList2Array(PyObject *list)
{
	int i, j, k;
	Array *a = (Array *)calloc(1, sizeof(Array));
	PyObject *row;
	PyObject *col;
	PyObject *data;
	// Get size
	a->row = PyList_Size(list);
	row = PyList_GetItem(list, 0);
	a->col = PyList_Size(row);
	col = PyList_GetItem(row, 0);
	a->lay = PyList_Size(col);
	// Allocate memory
	a->array = (double *)calloc(a->row * a->col * a->lay, sizeof(double));
	// Copy data
	for (i = 0; i < a->row; i++)
	{
		row = PyList_GetItem(list, i);
		for (j = 0; j < a->col; j++)
		{
			col = PyList_GetItem(row, j);
			for (k = 0; k < a->lay; k++)
			{
				data = PyList_GetItem(col, k);
				a->array[i * a->col * a->lay + j * a->lay + k] = PyFloat_AsDouble(data);
			}
		}
	}
	return a;
}

PyObject *
Array2PyList(const Array *a)
{
	int i, j, k;
	// Each row is a matrix
	PyObject *row;
	PyObject *col;
	PyObject *data;
	PyObject *PyArray = PyList_New(a->row);

	for (i = 0; i < a->row; i++)
	{
		// Create new row list
		row = PyList_New(a->col);
		for (j = 0; j < a->col; j++)
		{
			// Allocate new memories for this colomn
			col = PyList_New(a->lay);
			for (k = 0; k < a->lay; k++)
			{
				data = PyFloat_FromDouble(a->array[i * a->col * a->lay + j * a->lay + k]);
				PyList_SetItem(col, k, data);
			}
			PyList_SetItem(row, j, col);
		}
		PyList_SetItem(PyArray, i, row);
	}
	return PyArray;
}

void FreeArray(Array *a)
{
	free(a->array);
	free(a);
}

Array *
NewArray(int row, int col, int lay)
{
	Array *a = (Array *)calloc(1, sizeof(Array));
	a->row = row;
	a->col = col;
	a->lay = lay;
	a->array = (double *)calloc(row * col * lay, sizeof(double));
	return a;
}

Matrix *
PyList2Matrix(PyObject *list)
{
	int i, j;
	Matrix *a = (Matrix *)calloc(1, sizeof(Matrix));
	PyObject *row;
	PyObject *data;
	// Get size
	a->row = PyList_Size(list);
	row = PyList_GetItem(list, 0);
	a->col = PyList_Size(row);
	// Allocate memory
	a->mat = (double *)calloc(a->row * a->col, sizeof(double));
	// Copy data
	for (i = 0; i < a->row; i++)
	{
		row = PyList_GetItem(list, i);
		for (j = 0; j < a->col; j++)
		{
			data = PyList_GetItem(row, j);
			a->mat[i * a->col + j] = PyFloat_AsDouble(data);
		}
	}
	return a;
}

PyObject *
Matrix2PyList(const Matrix *a)
{
	int i, j;
	// Each row is a matrix
	PyObject *row;
	PyObject *data;
	PyObject *PyMat = PyList_New(a->row);

	for (i = 0; i < a->row; i++)
	{
		// Create new row list
		row = PyList_New(a->col);
		for (j = 0; j < a->col; j++)
		{
			data = PyFloat_FromDouble(a->mat[i * a->col + j]);
			PyList_SetItem(row, j, data);
		}
		PyList_SetItem(PyMat, i, row);
	}
	return PyMat;
}

void FreeMat(Matrix *a)
{
	free(a->mat);
	free(a);
}

Matrix *
NewMat(int row, int col)
{
	Matrix *a = (Matrix *)calloc(1, sizeof(Matrix));
	a->row = row;
	a->col = col;
	a->mat = (double *)calloc(row * col, sizeof(double));
	return a;
}

COOMatrix *
NewCOOMat(int nnz, int row, int col)
{
	COOMatrix *a = (COOMatrix *)calloc(1, sizeof(COOMatrix));
	a->row_idx = (int *)calloc(nnz, sizeof(int));
	a->col_idx = (int *)calloc(nnz, sizeof(int));
	a->val = (double *)calloc(nnz, sizeof(double));
	a->nnz = nnz;
	a->row = row;
	a->col = col;
	return a;
}

void FreeCOOMat(COOMatrix *a)
{
	free(a->row_idx);
	free(a->col_idx);
	free(a->val);
	free(a);
}

COOMatrix *
PyList2COOMat(PyObject *list, int row, int col)
{
	PyObject *row_idx = PyList_GetItem(list, 0);
	PyObject *col_idx = PyList_GetItem(list, 1);
	PyObject *val = PyList_GetItem(list, 2);
	int len = PyList_Size(row_idx);
	COOMatrix *a = NewCOOMat(len, row, col);
	int k;
	a->nnz = len;
	for (k = 0; k < len; k++)
	{
		//a->row_idx[k] = PyInt_AsLong(PyList_GetItem(row_idx, k));
		a->row_idx[k] = PyLong_AsLong(PyList_GetItem(row_idx, k));

		//a->col_idx[k] = PyInt_AsLong(PyList_GetItem(col_idx, k));
		a->col_idx[k] = PyLong_AsLong(PyList_GetItem(col_idx, k));

		a->val[k] = PyFloat_AsDouble(PyList_GetItem(val, k));
	}
	return a;
}

PyObject *
COOMat2PyList(const COOMatrix *a)
{
	PyObject *row_idx = PyList_New(a->nnz);
	PyObject *col_idx = PyList_New(a->nnz);
	PyObject *val = PyList_New(a->nnz);
	PyObject *list = PyList_New(3);
	int k;
	for (k = 0; k < a->nnz; k++)
	{
		//PyList_SetItem(row_idx, k, PyInt_FromLong(a->row_idx[k]));
		//PyList_SetItem(col_idx, k, PyInt_FromLong(a->col_idx[k]));
		PyList_SetItem(row_idx, k, PyLong_FromLong(a->row_idx[k]));
		PyList_SetItem(col_idx, k, PyLong_FromLong(a->col_idx[k]));
		PyList_SetItem(val, k, PyFloat_FromDouble(a->val[k]));
	}
	PyList_SetItem(list, 0, row_idx);
	PyList_SetItem(list, 1, col_idx);
	PyList_SetItem(list, 2, val);
	return list;
}

CSRMatrix *
NewCSRMat(int nnz, int row, int col)
{
	CSRMatrix *a = (CSRMatrix *)calloc(1, sizeof(CSRMatrix));
	a->nnz = nnz;
	a->row = row;
	a->col = col;
	a->row_ptr = (int *)calloc(row + 1, sizeof(int));
	a->col_idx = (int *)calloc(nnz, sizeof(int));
	a->val = (double *)calloc(nnz, sizeof(double));
	return a;
}

void FreeCSRMat(CSRMatrix *a)
{
	free(a->row_ptr);
	free(a->col_idx);
	free(a->val);
	free(a);
}

COOMatrix *
CSR2COO(const CSRMatrix *a)
{
	COOMatrix *b = NewCOOMat(a->nnz, a->row, a->col);
	memcpy(b->col_idx, a->col_idx, sizeof(int) * a->nnz);
	memcpy(b->val, a->val, sizeof(double) * a->nnz);
	int i, k;
	for (i = 0; i < a->row; i++)
	{
		for (k = a->row_ptr[i]; k < a->row_ptr[i + 1]; k++)
		{
			b->row_idx[k] = i;
		}
	}

	return b;
}

CSRMatrix *
COO2CSR(const COOMatrix *a)
{
	CSRMatrix *b = NewCSRMat(a->nnz, a->row, a->col);
	memcpy(b->col_idx, a->col_idx, sizeof(int) * a->nnz);
	memcpy(b->val, a->val, sizeof(double) * a->nnz);
	int i = 0, k;
	b->row_ptr[0] = 0;

	for (k = 0; k < a->nnz; k++)
	{
		if (a->row_idx[k] != i)
		{
			i++;
			b->row_ptr[i] = k;
		}
	}
	i++;
	for (; i <= b->row; i++)
	{
		b->row_ptr[i] = a->nnz;
	}
	//for(i=i+1;i<=a->row;i++){
	//  b->row_ptr[i] = a->nnz;
	//}
	return b;
}

int IsInCSRMat(const CSRMatrix *a, int i, int j)
{
	int k;
	for (k = a->row_ptr[i]; k < a->row_ptr[i + 1]; k++)
	{
		if (a->col_idx[k] == j)
		{
			return 1;
		}
	}
	return 0;
}

void SetCSRMat(CSRMatrix *a, int i, int j, double val)
{
	int k;
	for (k = a->row_ptr[i]; k < a->row_ptr[i + 1]; k++)
	{
		if (a->col_idx[k] == j)
		{
			a->val[k] = val;
		}
	}
	return;
}

int GetCSRIdx(const CSRMatrix *a, int i, int j)
{
	int k;
	for (k = a->row_ptr[i]; k < a->row_ptr[i + 1]; k++)
	{
		if (a->col_idx[k] == j)
		{
			return k;
		}
	}
	return -1;
}

CSRArray *
NewCSRArray(int nnz, int row, int col, int lay)
{
	CSRArray *a = (CSRArray *)calloc(1, sizeof(CSRArray));
	a->nnz = nnz;
	a->row = row;
	a->col = col;
	a->lay = lay;
	a->row_ptr = (int *)calloc(row + 1, sizeof(int));
	a->col_idx = (int *)calloc(nnz, sizeof(int));
	a->val = (double *)calloc(nnz * lay, sizeof(double));
	return a;
}

void FreeCSRArray(CSRArray *a)
{
	free(a->row_ptr);
	free(a->col_idx);
	free(a->val);
	free(a);
	return;
}

int IsInCSRArray(const CSRArray *a, int i, int j)
{
	int k;
	for (k = a->row_ptr[i]; k < a->row_ptr[i + 1]; k++)
	{
		if (a->col_idx[k] == j)
		{
			return 1;
		}
	}
	return 0;
}

SparseVector *
NewSparseVec(int len, int nnz)
{
	SparseVector *v = (SparseVector *)calloc(1, sizeof(SparseVector));
	v->idx = (int *)calloc(nnz, sizeof(int));
	v->data = (double *)calloc(nnz, sizeof(int));
	v->len = len;
	v->nnz = nnz;
	return v;
}

void FreeSparseVec(SparseVector *v)
{
	free(v->idx);
	free(v->data);
	free(v);
	return;
}

Observation *
NewObservation(int R)
{
	Observation *obs = (Observation *)calloc(1, sizeof(Observation));
	obs->R = R;
	obs->data = (Matrix **)calloc(R, sizeof(Matrix *));
	// Each matrix's memory is allocated outside this function
	return obs;
}

void FreeObservation(Observation *obs)
{
	int r;
	for (r = 0; r < obs->R; r++)
	{
		FreeMat(obs->data[r]);
	}
	free(obs->data);
	free(obs);
}

Observation *
PyList2Observation(PyObject *list)
{
	int R, D, T;
	//  int *T = (int*)calloc(R,sizeof(int));
	PyObject *data, *row, *entry;
	Observation *obs;
	// Get size
	R = PyList_Size(list);
	obs = NewObservation(R);
	// Get first sequence of observations from the first file
	data = PyList_GetItem(list, 0);
	// Get a feature vector
	row = PyList_GetItem(data, 0);
	D = PyList_Size(row);
	int r, t, d;
	for (r = 0; r < R; r++)
	{
		data = PyList_GetItem(list, r);
		T = PyList_Size(data);
		obs->data[r] = NewMat(T, D);
		for (t = 0; t < T; t++)
		{
			row = PyList_GetItem(data, t);
			for (d = 0; d < D; d++)
			{
				entry = PyList_GetItem(row, d);
				obs->data[r]->mat[t * D + d] = PyFloat_AsDouble(entry);
			}
		}
	}
	//  free(T);
	return obs;
}

Network *
NewNetwork(int nnz, int row, int col)
{
	Network *net = (Network *)calloc(1, sizeof(Network));
	net->nnz = nnz;
	net->row = row;
	net->col = col;
	net->row_idx = (int *)calloc(nnz, sizeof(int));
	net->col_idx = (int *)calloc(nnz, sizeof(int));
	return net;
}

Network *
PyList2Network(PyObject *list, int row, int col)
{
	PyObject *row_idx, *col_idx;
	row_idx = PyList_GetItem(list, 0);
	col_idx = PyList_GetItem(list, 1);
	int nnz = PyList_Size(row_idx);
	Network *net = NewNetwork(nnz, row, col);
	int k;
	for (k = 0; k < nnz; k++)
	{
		net->row_idx[k] = PyFloat_AsDouble(PyList_GetItem(row_idx, k));
		net->col_idx[k] = PyFloat_AsDouble(PyList_GetItem(col_idx, k));
	}
	return net;
}

void FreeNetwork(Network *net)
{
	free(net->row_idx);
	free(net->col_idx);
	free(net);
}

COOMatrix *
Network2COO(Network *net)
{
	COOMatrix *coo_mat = NewCOOMat(net->nnz, net->row, net->col);
	int nnz = net->nnz;
	memcpy(coo_mat->row_idx, net->row_idx, sizeof(int) * nnz);
	memcpy(coo_mat->col_idx, net->col_idx, sizeof(int) * nnz);
	return coo_mat;
}

Model*
NewModel(int num_states, int num_components, int dim, int nnz) {
	Model * model = (Model*)calloc(1, sizeof(Model));
	model->log_trans = NewCSRMat(nnz, num_states, num_states);
	model->log_coef = NewMat(num_states, num_components);
	model->log_var = NewArray(num_states, num_components, dim);
	model->mean = NewArray(num_states, num_components, dim);
	return model;
}

void
FreeModel(Model * model) {
	FreeCSRMat(model->log_trans);
	FreeMat(model->log_coef);
	FreeArray(model->log_var);
	FreeArray(model->mean);
	free(model);
}

Model* PyList2Model(PyObject* list) {
	PyObject *PyLogTrans, *PyLogCoef, *PyMean, *PyLogVar;
	PyLogTrans = PyList_GetItem(list, 0);
	PyLogCoef = PyList_GetItem(list, 1);
	PyMean = PyList_GetItem(list, 2);
	PyLogVar = PyList_GetItem(list, 3);

	Model* model = (Model*)calloc(1, sizeof(Model));
	model->log_coef = PyList2Matrix(PyLogCoef);
	model->log_var = PyList2Array(PyLogVar);
	model->mean = PyList2Array(PyMean);
	int N = model->log_coef->row;
	COOMatrix* coo_log_trans = PyList2COOMat(PyLogTrans, N, N);
	model->log_trans = COO2CSR(coo_log_trans);

	return model;
}

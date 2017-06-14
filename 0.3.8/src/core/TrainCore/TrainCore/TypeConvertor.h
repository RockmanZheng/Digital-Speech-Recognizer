#ifndef TYPECONVERTOR_H
#define TYPECONVERTOR_H
#include <Python.h>

typedef struct {
	double* array;
	int row;
	// Colomn
	int col;
	// Layer
	int lay;
}Array;

Array*
PyList2Array(const PyObject* list);

PyObject*
Array2PyList(const Array* a);


void
FreeArray(Array *a);


Array*
NewArray(int row, int col, int lay);

typedef struct {
	double* mat;
	int row;
	// Colomn
	int col;
}Matrix;

Matrix*
PyList2Matrix(const PyObject* list);


PyObject*
Matrix2PyList(const Matrix* a);

void
FreeMat(Matrix *a);


Matrix*
NewMat(int row, int col);

// Sparse matrix in COO format
typedef struct {
	int *row_idx;
	int *col_idx;
	double *val;
	int nnz;
	int row;
	int col;
}COOMatrix;

// Sparse matrix in CSR format
typedef struct {
	int *row_ptr;
	int *col_idx;
	double *val;
	int nnz;
	int row;
	int col;
}CSRMatrix;

COOMatrix*
NewCOOMat(int nnz, int row, int col);

void
FreeCOOMat(COOMatrix *a);

COOMatrix*
PyList2COOMat(const PyObject *list, int row, int col);


PyObject*
COOMat2PyList(const COOMatrix *a);

CSRMatrix*
NewCSRMat(int nnz, int row, int col);

void
FreeCSRMat(CSRMatrix *a);

COOMatrix*
CSR2COO(const CSRMatrix *a);

CSRMatrix*
COO2CSR(const COOMatrix *a);

int
IsInCSRMat(const CSRMatrix *a, int i, int j);

void
SetCSRMat(CSRMatrix *a, int i, int j, double val);

int
GetCSRIdx(const CSRMatrix *a, int i, int j);

typedef struct {
	int *row_ptr;
	int *col_idx;
	double *val;
	// nnz is the number of non-zero 1 dimensional vector of the form (i,j,:)
	int nnz;
	int row;
	int col;
	int lay;
}CSRArray;


CSRArray*
NewCSRArray(int nnz, int row, int col, int lay);


void
FreeCSRArray(CSRArray *a);

int
IsInCSRArray(const CSRArray *a, int i, int j);

typedef struct {
	int *idx;
	double *data;
	int len;
	int nnz;
}SparseVector;

SparseVector*
NewSparseVec(int len, int nnz);

void
FreeSparseVec(SparseVector *v);


typedef struct {
	Matrix **data;// Sequence of observation data
	int R;// Number of observation files
}Observation;

Observation*
NewObservation(int R);

void
FreeObservation(Observation* obs);

Observation*
PyList2Observation(const PyObject *list);

typedef struct {
	int *row_idx;
	int *col_idx;
	int nnz;
	int row;
	int col;
}Network;

Network*
NewNetwork(int nnz, int row, int col);

Network*
PyList2Network(const PyObject *list, int row, int col);

void
FreeNetwork(Network *net);

COOMatrix*
Network2COO(Network *net);

typedef struct {
	CSRMatrix* log_trans;
	Matrix* log_coef;
	Array* log_var;
	Array* mean;
}Model;

Model*
NewModel(int num_states, int num_components, int dim, int nnz);

void
FreeModel(Model * model);

Model* PyList2Model(const PyObject*list);

#endif






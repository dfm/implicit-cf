#ifndef _CTR_BLAS_H_
#define _CTR_BLAS_H_

// Matrix matrix multiply.
void dgemm_(char* transa,
            char* transb,
            int* m,
            int* n,
            int* k,
            double* alpha,
            double* a,
            int* lda,
            double* b,
            int* ldb,
            double* beta,
            double* c,
            int* ldc);

// Linear solve.
void dgesv_(int* n,
            int* nrhs,
            double* a,
            int* lda,
            int* ipiv,
            double* b,
            int* ldb,
            int* info);

#endif

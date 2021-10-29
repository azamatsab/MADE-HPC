#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <cmath>
#include <mkl.h>
#include <mkl_vsl.h>

using namespace std;


// Task 1 related functions
void MatMul(double * A, double * B, double * C, size_t N) {
    double one = 1.0, zero = 0.0;
    int rowsA = N, common = N, colsB = N;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                rowsA, colsB, common, one, A, common, B, colsB, one, C, colsB);
}

void CopyMatrix(double * A, double * C, size_t row, size_t col) {
    #pragma omp parallel \
        default(none) \
        shared(A, C, row, col) \

    #pragma omp for collapse(2)
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
            C[j * col + i] = A[j * col + i];
    }
}

void MatPower(double * A, double * C, size_t N, size_t power) {
    if (power == 1) {
        CopyMatrix(A, C, N, N);
        return;
    } else if (power == 2) {
        MatMul(A, A, C, N);
        return;
    }

    double * temp = new double[N * N];
    if (power % 2 == 0) {
        MatPower(A, temp, N, power / 2);
        MatMul(temp, temp, C, N);
    } else {
        MatPower(A, temp, N, power - 1);
        MatMul(A, temp, C, N);
    }
    delete (temp);
}

void PrintMatrix(double * C, size_t col, size_t row) {
    for(int i = 0; i < row; i++) {
        for(int j = 0; j < col; j++)
            printf("%f  ", C[i * row + j]);
        printf("\n");
    }
    printf("\n");
}

void AdjacencyMatrix(double * A, size_t N) {
    int tid;
    #pragma omp parallel \
        default(none) \
        shared(A, N) \
        private(tid)

    tid = omp_get_thread_num();
    unsigned seed = (unsigned) time(NULL);
    seed = (seed & 0xFFFFFFF0) | (tid + 1);
    srand(seed);

    #pragma omp for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (rand() % 2) * (rand() % 2);
        }
    }
}

// Task2 related functions
void NormalizeCols(double * A, size_t rows, size_t cols) {
    int i, j, k;
    for (j = 0; j < cols; j++) {
        double col_val = 0;
        #pragma omp parallel \
            default(none) \
            shared(A, rows, cols, i, j, k) \
            reduction(+:col_val)
        #pragma omp for
        for (i = 0; i < rows; i++) {
            col_val += A[i * cols + j];
        }

        #pragma omp for
        for (k = 0; k < rows; k++) {
            if (A[k * cols + j] != 0)
                A[k * cols + j] = A[k * cols + j] / col_val;
        }
    }
}

double * NaiveRank(double * A, size_t rows, size_t cols) {
    int i, j;
    double * rank = new double[rows];
    for (i = 0; i < rows; i++) {
        double col_val = 0;
        #pragma omp parallel \
            default(none) \
            shared(A, rows, cols, i, j) \
            reduction(+:col_val)
        #pragma omp for
        for (j = 0; j < cols; j++) {
            col_val += A[i * cols + j];
        }
        #pragma omp barrier
        rank[i] = col_val;
    }

    return rank;
}

void FillMatrix(double * A, size_t N, double value) {
    #pragma omp parallel \
        default(none) \
        shared(A, N, value)

    #pragma omp for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = value;
        }
    }
}

void RandomVector(double * A, size_t N) {
    int tid;
    #pragma omp parallel \
        default(none) \
        shared(A, N) \
        private(tid)

    tid = omp_get_thread_num();
    unsigned seed = (unsigned) time(NULL);
    seed = (seed & 0xFFFFFFF0) | (tid + 1);
    srand(seed);

    #pragma omp for
    for (int i = 0; i < N; i++) {
        A[i] = double(rand()) / RAND_MAX;
    }
}

void AddMatrices(double * A, double * C, size_t N, double alpha) {
    cblas_daxpy (N * N, alpha, A, 1.0, C, 1.0);
}

void DotProd(double * A, double * B, double * C, size_t N) {
    double one = 1.0;
    int rowsA = N, common = N, colsB = 1;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                rowsA, colsB, common, one, A, common, B, colsB, one, C, colsB);
}

double CalcNorm(double * C, int N) {
    const int incx = 1;
    double norm = dnrm2(&N, C, &incx);
    return norm;
}

void DivideVector(double * A, size_t N, double alpha) {
    cblas_dscal(MKL_INT(N), 1 / alpha, A, 1);
}

void PowerMethod(double * A, double * C, size_t N, size_t num_sim) {
    double * temp = new double[N];
    for (int i = 0; i < num_sim; i++) {
        DotProd(A, C, temp, N);
        double norm = CalcNorm(temp, N);
        DivideVector(temp, N, norm);
        CopyMatrix(temp, C, N, 1);
    }
    delete (temp);
}

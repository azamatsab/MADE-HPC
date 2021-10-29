#include <cstdlib>
#include <iostream>
#include "matrix_operations.h"

using namespace std;


const int N_SIZE = 16;
const int N_ITER = 100;

int main() {
    double alpha = 0.85;
    double * A = new double[N_SIZE * N_SIZE];
    double * C = new double[N_SIZE];
    double * D = new double[N_SIZE * N_SIZE];

    RandomVector(C, N_SIZE);
    double norm = CalcNorm(C, N_SIZE);
    DivideVector(C, N_SIZE, norm);

    AdjacencyMatrix(A, N_SIZE);

    printf("Given adjacency matrix: \n");
    PrintMatrix(A, N_SIZE, N_SIZE);

    double * rank = NaiveRank(A, N_SIZE, N_SIZE);

    NormalizeCols(A, N_SIZE, N_SIZE);
    FillMatrix(D, N_SIZE, (1 - alpha) / N_SIZE);

    AddMatrices(A, D, N_SIZE, alpha);

    PowerMethod(D, C, N_SIZE, N_ITER);
    
    printf("PageRank results: \n");
    PrintMatrix(C, N_SIZE, 1);
    printf("Naive rank results: \n");
    PrintMatrix(rank, N_SIZE, 1);

    return 0;
}

#include <iostream>
#include "matmul.h"

using namespace std;

const int N_SIZE = 512;

int main()
{
    double A[N_SIZE * N_SIZE];
    double B[N_SIZE * N_SIZE];
    double C[N_SIZE * N_SIZE];
    double V[N_SIZE];
    double D[N_SIZE];

    RandomMatrix(A, N_SIZE);
    RandomMatrix(B, N_SIZE);
    RandomVector(V, N_SIZE);

    double rtime1 = CalcMatMulTime_ijk(A, B, C, N_SIZE);
    cout << rtime1 << endl;

    double rtime2 = CalcMatMulTime_ijk(A, B, C, N_SIZE);
    cout << rtime2 << endl;

    double rtime3 = CalcMatMulTime_kij(A, B, C, N_SIZE);
    cout << rtime3 << endl;

    double rtime4 = CalcMatVecMulTime(A, V, D, N_SIZE);
    cout << rtime4 << endl;
    return 0;
}

#include <iostream>

#include "matmul.h"

using namespace std;

const int N_SIZE = 512;

int main() {
  double * A = new double[N_SIZE * N_SIZE];
  double * B = new double[N_SIZE * N_SIZE];
  double * C = new double[N_SIZE * N_SIZE];
  double * V = new double[N_SIZE];
  double * D = new double[N_SIZE];

  RandomMatrix(A, N_SIZE);
  RandomMatrix(B, N_SIZE);
  RandomVector(V, N_SIZE);

  cout << "N = " << N_SIZE << endl;

  double rtime1 = CalcMatMulTime_ijk(A, B, C, N_SIZE);
  cout << "Matrix mult: ijk run time is: " << rtime1 << endl;

  double rtime2 = CalcMatMulTime_jik(A, B, C, N_SIZE);
  cout << "Matrix mult: jik run time is: " << rtime2 << endl;

  double rtime3 = CalcMatMulTime_kij(A, B, C, N_SIZE);
  cout << "Matrix mult: kij run time is: " << rtime3 << endl;

  double rtime4 = CalcMatVecMulTime(A, V, D, N_SIZE);
  cout << "Matrix to vector mult: run time is: " << rtime4 << endl;
  return 0;
}
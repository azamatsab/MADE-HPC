#include <cstdlib>
#include <iostream>
#include "matrix_operations.h"

using namespace std;

const int N_SIZE = 4;
const int POWER = 7;

int main() {
  double * A = new double[N_SIZE * N_SIZE];
  double * C = new double[N_SIZE * N_SIZE];

  AdjacencyMatrix(A, N_SIZE);
  MatPower(A, C, N_SIZE, POWER);
  printf("Initial adjacency matrix: \n");
  PrintMatrix(A, N_SIZE, N_SIZE);

  printf("Adjacency matrix to the power of %d: \n", POWER);
  PrintMatrix(C, N_SIZE, N_SIZE);
  return 0;
}

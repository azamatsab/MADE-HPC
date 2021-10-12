#include <time.h>

#include <cstdlib>

#include <iostream>

using namespace std;

void ZeroMatrix(double * A, size_t N) {
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      A[i * N + j] = 0.0;
    }
  }
}

void ZeroMatrix(double * A, size_t R, size_t C) {
  for (size_t i = 0; i < C; i++) {
    for (size_t j = 0; j < R; j++) {
      A[i * R + j] = 0.0;
    }
  }
}

void RandomMatrix(double * A, size_t N) {
  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = rand() / RAND_MAX;
    }
  }
}

void RandomVector(double * A, size_t N) {
  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    A[i] = rand() / RAND_MAX;
  }
}

double CalcMatMulTime_ijk(double * A, double * B, double * C, size_t N) {
  clock_t tStart = clock();
  size_t i, j, k;

  ZeroMatrix(C, N);

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++) {
      for (k = 0; k < N; k++)
        C[i * N + j] = C[i * N + j] + A[i * N + k] * B[k * N + j];
    }
  double rtime = (double)(clock() - tStart) / CLOCKS_PER_SEC;
  return rtime;
}

double CalcMatMulTime_jik(double * A, double * B, double * C, size_t N) {
  clock_t tStart = clock();
  size_t i, j, k;

  ZeroMatrix(C, N);

  for (j = 0; j < N; j++)
    for (i = 0; i < N; i++) {
      for (k = 0; k < N; k++)
        C[i * N + j] = C[i * N + j] + A[i * N + k] * B[k * N + j];
    }
  double rtime = (double)(clock() - tStart) / CLOCKS_PER_SEC;
  return rtime;
}

double CalcMatMulTime_kij(double * A, double * B, double * C, size_t N) {
  clock_t tStart = clock();
  size_t i, j, k;

  ZeroMatrix(C, N);

  for (k = 0; k < N; k++)
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++)
        C[i * N + j] = C[i * N + j] + A[i * N + k] * B[k * N + j];
    }
  double rtime = (double)(clock() - tStart) / CLOCKS_PER_SEC;
  return rtime;
}

double CalcMatVecMulTime(double * A, double * B, double * C, size_t N) {
  clock_t tStart = clock();
  size_t i, j;
  ZeroMatrix(C, N, 1);

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      C[i] = C[i] + A[i * N + j] * B[j];
  double rtime = (double)(clock() - tStart) / CLOCKS_PER_SEC;
  return rtime;
}
#ifndef matmul
#define matmul

void ZeroMatrix(double *A, size_t N);
void ZeroMatrix(double *A, size_t R, size_t C);
void RandomMatrix(double *A, size_t N);
void RandomVector(double *A, size_t N);

double CalcMatMulTime_ijk(double *A, double *B, double *C, size_t N);
double CalcMatMulTime_jik(double *A, double *B, double *C, size_t N);
double CalcMatMulTime_kij(double *A, double *B, double *C, size_t N);
double CalcMatVecMulTime(double *A, double *B, double *C, size_t N);

#endif

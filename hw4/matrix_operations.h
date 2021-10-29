#ifndef matrix_operations
#define matrix_operations

void MatPower(double * A, double * C, size_t N, size_t power);
void MatMul(double * A, double * B, double * C, size_t N);

void PrintMatrix(double * C, size_t col, size_t row);
void AdjacencyMatrix(double * A, size_t N);
void NormalizeCols(double * A, size_t rows, size_t cols);
double * NaiveRank(double * A, size_t rows, size_t cols);

void FillMatrix(double * A, size_t N, double value);
void RandomVector(double * A, size_t N);
void AddMatrices(double * A, double * C, size_t N, double alpha);
void CopyMatrix(double * A, double * C, size_t row, size_t col);
void DotProd(double * A, double * B, double * C, size_t N);
double CalcNorm(double * C, int N);
void DivideVector(double * A, size_t N, double alpha);
void PowerMethod(double * A, double * C, size_t N, size_t num_sim);

#endif

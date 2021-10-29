## HW4

#### Makefile создает сразу 2 исполняемых файла: task1 и task2, для первого и второго задании соответственно. Функции для обеих задач в файле matrix_operations, прошу прощения если так неудобно

void MatPower(double * A, double * C, size_t N, size_t power);  - возведение А в степень power и запись в C
void MatMul(double * A, double * B, double * C, size_t N); - матричное умножение А и В и запись результата в С

void PrintMatrix(double * C, size_t col, size_t row); - выводит содержимое матрицы в stdout
void AdjacencyMatrix(double * A, size_t N); - заполняет матрицу 0 и 1, по моей задумке соответствует матрице смежности направленного графа
void NormalizeCols(double * A, size_t rows, size_t cols); - делит каждый элемент столбца на сумму элементов столбца
double * NaiveRank(double * A, size_t rows, size_t cols); - наивное ранжирование, вычисляет входящие связи на ноду

void FillMatrix(double * A, size_t N, double value); - заполняет матрицу значением value
void RandomVector(double * A, size_t N); - заполняет вектор случайными значениями
void AddMatrices(double * A, double * C, size_t N, double alpha); - умножает A на alpha и добавляет C
void CopyMatrix(double * A, double * C, size_t row, size_t col); - копирует матрицу А в С
void DotProd(double * A, double * B, double * C, size_t N); - матричное умножение между матрицей А и вектором В, полученный вектор записывается в С
double CalcNorm(double * C, int N); - вычисление L2 нормы вектора
void DivideVector(double * A, size_t N, double alpha); - разделить вектор А на скаляр alpha
void PowerMethod(double * A, double * C, size_t N, size_t num_sim); - реализация power_method, A - матрица смежности, C случайный вектор, полученные ранги записаны в C


#### Steps to assign proper paths (for my own purpose)

            export MKLROOT=/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl


            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/lib/intel64
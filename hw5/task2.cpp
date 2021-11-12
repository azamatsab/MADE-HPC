// С помощью MPI распараллельте одномерный клеточный автомат Вольфрама (Rule110).
// Игра происходит следующим образом:
// 1) Инициализируйте одномерный массив 0 и 1 случайным образом
// 2) В зависимости от значений: левого соседа, себя, правого соседа на следующем шаге клетка либо меняет значение, либо остается той же. Посмотрите, например, что значит Rule110 (https://en.wikipedia.org/wiki/Rule_110)
// Сделайте периодические и непериодические граничные условия (5 баллов)
// Работает параллельный код на нескольких процессах (20 баллов)
// Имплементированы клетки-призраки (ghost cells) (10 балла)
// Можно поменять правило игры (сделать одно из 256) (20 баллов)
// График ускорения работы программы от кол-ва процессов (5 баллов)
// Картинка эволюции для одного правила (15 баллов)
// Итого баллов: 75  + 25 = 100 баллов за базовую часть из 2 заданий

#include <mpi.h>
#include <stdio.h>
#include <random> 

using namespace std;

struct Cell {
	int value;
	int left;
};

const int PERIODIC = 1;

void random_array(int* array, size_t size, int rank) {
    unsigned seed = (unsigned) time(NULL);
    seed = (seed & 0xFFFFFFF0) | (rank + 1);
    srand(seed);

    for (int i = 0; i < size; i++) {
        array[i] = (rand() % 2);
    }
}

void update_ghost_cell(int* array, size_t size, Cell cell) {
    if (cell.left)
        array[size - 1] = cell.value;
    else
        array[0] = cell.value;
}

Cell get_cell(int* array, int left, size_t size) {
    Cell ghost_send;
    ghost_send.left = left;
    if (left)
        ghost_send.value = array[1];
    else 
        ghost_send.value = array[size - 2];
    return ghost_send;
}

void update(int rank, size_t size, int* array, size_t array_length, MPI_Datatype mpi_cell_type, int left_neighbor, int right_neighbor, Cell left_ghost_send, Cell right_ghost_send, Cell received) {
    if (rank == 0) {
        if (PERIODIC) {
            MPI_Ssend(&left_ghost_send, 1, mpi_cell_type, left_neighbor, 0, MPI_COMM_WORLD);
            MPI_Recv(&received, 1, mpi_cell_type, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            update_ghost_cell(array, array_length, received);
        }

        MPI_Ssend(&right_ghost_send, 1, mpi_cell_type, right_neighbor, 0, MPI_COMM_WORLD);
        MPI_Recv(&received, 1, mpi_cell_type, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        update_ghost_cell(array, array_length, received);

    } else if (rank == size - 1) {
        MPI_Recv(&received, 1, mpi_cell_type, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        update_ghost_cell(array, array_length, received);
        MPI_Ssend(&left_ghost_send, 1, mpi_cell_type, left_neighbor, 0, MPI_COMM_WORLD);

        if (PERIODIC) {
            MPI_Recv(&received, 1, mpi_cell_type, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            update_ghost_cell(array, array_length, received);
            
            if (received.left)
                MPI_Ssend(&left_ghost_send, 1, mpi_cell_type, left_neighbor, 0, MPI_COMM_WORLD);
            else
                MPI_Ssend(&right_ghost_send, 1, mpi_cell_type, right_neighbor, 0, MPI_COMM_WORLD);
        } 
    } else {
        for (int t = 0; t < 2; t++) {
            MPI_Recv(&received, 1, mpi_cell_type, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            update_ghost_cell(array, array_length, received);
            if (received.left)
                MPI_Ssend(&left_ghost_send, 1, mpi_cell_type, left_neighbor, 0, MPI_COMM_WORLD);
            else
                MPI_Ssend(&right_ghost_send, 1, mpi_cell_type, right_neighbor, 0, MPI_COMM_WORLD);
        }
    }
}

int main(int argc,char **argv)
{
    const size_t N_LINE = 32;
    int* array  = new int[N_LINE + 2];
	int rank, size;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm ring_1D;
    int ndim = 1;
    int dims, periods;
    int ndims, coords;
    dims = size;
    periods = 1;

    MPI_Cart_create(MPI_COMM_WORLD, ndim, &dims, &periods, 0, &ring_1D);

    random_array(array, N_LINE, rank);

    int datasize = 2 * sizeof(int);
	MPI_Datatype mpi_cell_type;
	MPI_Type_contiguous(datasize, MPI_INT, &mpi_cell_type);
	MPI_Type_commit(&mpi_cell_type);

    int left_neighbor, right_neighbor;
    MPI_Cart_shift(ring_1D, 0, 1, &left_neighbor, &right_neighbor);
    Cell left_ghost_send = get_cell(array, 1, N_LINE);
    Cell right_ghost_send = get_cell(array, 0, N_LINE);
    Cell received;
    update(rank, size, array, N_LINE, mpi_cell_type, left_neighbor, right_neighbor, left_ghost_send, right_ghost_send, received);
	MPI_Finalize();
    return 0;
}

// 1) Начинает процессор 0. Случайным образом он выбирает другой процессор i и посылает ему сообщение со своим именем (можете случайным образом задавать имя)
// 2) Процессор i отсылает сообщение случайному процессору j (которые еще не участвовал в игре), в сообщении – все имена и ранги предыдущих процессоров в правильном порядке. Номер процессора j знает только I, так что все должны быть начеку.
// 3) Игра заканчивается через N ходов. Используйте синхронную пересылку MPI_SSend
// Напишите программу используя MPI. (25 баллов)

#include <mpi.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <random> 

using namespace std;

struct Info {
	int name = -1;
	int rank = -1;
};

bool find(vector<int> known, int num) {
	for (int i = 0; i < known.size(); i++) {
		if (known[i] == num)
			return true;
	}
	return false;
}

int get_next_rank(vector<int> known, size_t size) {
	int index = rand() % size;
	while (find(known, index)) {
		index = rand() % size;
	}
	return index;
}

int get_random_num(size_t max_val, int rank) {
	unsigned seed = (unsigned) time(NULL);
    seed = (seed & 0xFFFFFFF0) | (rank + 1);
    srand(seed);

	int number = int(rand() % max_val);
	return number;
}

int main(int argc,char **argv)
{
	int rank, size;
	MPI_Status status;
	Info message[size];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int tag = get_random_num(100, rank);
	int name = get_random_num(100, rank);

	int datasize = 2 * sizeof(int);
	MPI_Datatype mpi_info_type;
	MPI_Type_contiguous(datasize, MPI_INT, &mpi_info_type);
	MPI_Type_commit(&mpi_info_type);

	if (rank == 0) {
		vector<int> known = {0};
		int next_rank = get_next_rank(known, size);	
		message[0].name = name;
		message[0].rank = rank;

		printf("Sending from rank %d with name %d to %d\n", rank, name, next_rank);
		MPI_Ssend(&message[0], size, mpi_info_type, next_rank, tag, MPI_COMM_WORLD);
		MPI_Recv(&message[0], size, mpi_info_type, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

	} else {
		MPI_Recv(&message[0], size, mpi_info_type, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		vector<int> known = {};
		for (int i = 0; i < size; i++) {
			if (message[i].name == -1) {
				message[i].name = name;
				message[i].rank = rank;
				
				known.push_back(rank);
				printf("Rank %d received array with following info:\n", rank);
				for (int j = 0; j < i; j++) {
					printf("\tIndex %d: rank %d; name %d\n", j, message[j].rank, message[j].name);
				}

				int next_rank = 0;
				if (known.size() != size) {
					next_rank = get_next_rank(known, size);
				}

				printf("Sending from rank %d with name %d to %d\n", rank, name, next_rank);
				MPI_Ssend(&message[0], size, mpi_info_type, next_rank, tag, MPI_COMM_WORLD);
				break;
			}
			known.push_back(message[i].rank);
		}
	}

	MPI_Finalize();
}

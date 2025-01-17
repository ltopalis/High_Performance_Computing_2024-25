#include <iostream>
#include <mpi.h>

#define MY_TAG 64

void MPI_Exscan_pt2pt(int send_value, int &rec_value);

int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);

    // Initialization

    int rank, size, local_value, final_result;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    local_value = 10 * size + rank;

    MPI_Exscan_pt2pt(local_value, final_result);

    std::cout << "rank = " << rank << " size = " << size << " sent = " << local_value << " received = " << final_result << std::endl;

    MPI_Finalize();
}

void MPI_Exscan_pt2pt(int send_value, int &rec_value)
{
    int rank, size, local_rec;

    rec_value = send_value;
    if (!rank)
        rec_value = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int i = 0; i < rank; i++)
    {
        MPI_Recv(&local_rec, 1, MPI_INT, i, MY_TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        rec_value += local_rec;
    }

    for (int i = rank + 1; i < size; i++)
        MPI_Send(&send_value, 1, MPI_INT, i, MY_TAG, MPI_COMM_WORLD);

    return;
}
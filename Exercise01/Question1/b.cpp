#include <iostream>
#include <mpi.h>
#include <omp.h>

#define T 5

void MPI_Exscan_pt2pt(int send_value, int &rec_value);

int main(int argc, char **argv)
{

    int check;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &check);

    if (check < MPI_THREAD_MULTIPLE)
    {
        std::cerr << "MPI does not support MPI_THREAD_MULTIPLE. Exiting.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    omp_set_num_threads(T);

    // Initialization

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

#pragma omp parallel
    {
        int local_value = 10 * rank + omp_get_thread_num();
        int final_result;

        MPI_Exscan_pt2pt(local_value, final_result);

#pragma omp critical
        {
            std::cout << "rank = " << rank << " thread = " << omp_get_thread_num() << " sent = " << local_value << " received = " << final_result << std::endl;
        }
    }

    MPI_Finalize();
}

void MPI_Exscan_pt2pt(int send_value, int &rec_value)
{
    int rank, size;             // MPI
    int thread_id, num_threads; // OpenMP
    int local_recv = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    thread_id = omp_get_thread_num();
    num_threads = omp_get_num_threads();

    rec_value = send_value;
    if (!rank && !thread_id)
        rec_value = 0;

    for (int process = 0; process <= rank; process++)
    {
        for (int thread = 0; thread < num_threads; thread++)
        {
            if (process == rank && thread >= thread_id)
                continue;

            MPI_Recv(&local_recv, 1, MPI_INT, process, thread_id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            rec_value += local_recv;
        }
    }

    for (int process = 0; process < size; process++)
    {
        for (int thread = 0; thread < num_threads; thread++)
        {
            bool cond = (rank < process) || (process == rank && thread > thread_id);
            if (!cond)
                continue;

            MPI_Send(&send_value, 1, MPI_INT, process, thread, MPI_COMM_WORLD);
        }
    }
}

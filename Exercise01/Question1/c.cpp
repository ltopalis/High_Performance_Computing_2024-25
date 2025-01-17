#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <random>
#include <chrono>
#include <cstdio>

#define T 7
#define N 13

using std::cerr;
using std::cout;
using std::endl;

void MPI_Exscan_pt2pt(long long int send_value, long long int &rec_value);

int main(int argc, char **argv)
{

    int check;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &check);

    if (check < MPI_THREAD_MULTIPLE)
    {
        cerr << "MPI does not support MPI_THREAD_MULTIPLE. Exiting.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialization

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, "output_c.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);

    if (!rank)
        cout << "File was created" << endl;

#pragma omp parallel num_threads(T)
    {
        // initializing data

        double matrix[N][N][N];

        int thread_id = omp_get_thread_num();

        long long time_seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::seed_seq seed{static_cast<unsigned int>(time_seed),
                           static_cast<unsigned int>(rank),
                           static_cast<unsigned int>(thread_id)};
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> dis(std::numeric_limits<double>::min(),
                                                   std::numeric_limits<double>::max());

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                for (int k = 0; k < N; k++)
                    matrix[i][j][k] = dis(gen);

#pragma omp barrier
        MPI_Barrier(MPI_COMM_WORLD);
        if (!rank)
        {
#pragma omp master
            cout << "Vectors are initialized" << endl;
        }

        // writting to file
        long long int offset;
        if (!rank && !thread_id) // process 0 rank 0
            MPI_Exscan_pt2pt(0, offset);
        else
            MPI_Exscan_pt2pt(sizeof(matrix), offset);

        MPI_Offset mpi_offset = static_cast<MPI_Offset>(offset);

        MPI_File_write_at(file, offset, matrix, N * N * N, MPI_DOUBLE, MPI_STATUS_IGNORE);

#pragma omp barrier
        MPI_Barrier(MPI_COMM_WORLD);
        if (!rank)
        {
#pragma omp master
            cout << "Writing to file was completed" << endl;
        }
        MPI_File_close(&file);

        // check results

        FILE *f = fopen("./output_c.bin", "rb");

        double written[N * N * N];
        int off = 0;
        bool flag = true;

        fseek(f, offset, SEEK_SET);

        fread(&written, sizeof(written), 1, f);

        int a = 0;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                for (int k = 0; k < N; k++)
                    if (matrix[i][j][k] != written[a++])
                        flag = false;

        fclose(f);

#pragma omp barrier
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_File_close(&file);
        if (!rank)
        {
#pragma omp master
            cout << "Checking was completed" << endl;
        }

#pragma omp barrier
#pragma omp critical
        {
            if (flag)
                cout << "Rank " << rank << " thread " << thread_id << " status: success" << endl;
            else
                cout << "Rank " << rank << " thread " << thread_id << " status: failed" << endl;
        }
    }

    MPI_Finalize();
}

void MPI_Exscan_pt2pt(long long int send_value, long long int &rec_value)
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

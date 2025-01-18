#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include <time.h>
#include <string.h>

#define N 5
#define TRUE 1
#define FALSE 0


void initTables(int A[N][N], int B[N][N], int C[N][N], int D[N][N], int E[N][N], int F[N][N], int E_d[N][N], int F_d[N][N]);
int randomNumber(unsigned int *seed, int min, int max);
void printTables(const char *name, int arr[N][N]);
void arrays_multiplication_cpu(int A[N][N], int B[N][N], int C[N][N], int D[N][N], int E[N][N], int F[N][N]);
void arrays_multiplication_gpu(int A[N][N], int B[N][N], int C[N][N], int D[N][N], int E[N][N], int F[N][N]);
int cmpArrays(int cpu1[N][N], int cpu2[N][N], int gpu1[N][N], int gpu2[N][N]);

int main(int argc, char **argv)
{
    int A[N][N], B[N][N], C[N][N], D[N][N], E[N][N], F[N][N], E_d[N][N], F_d[N][N];
    int check;
    char *ans;

    initTables(A, B, C, D, E, F, E_d, F_d);
    arrays_multiplication_cpu(A, B, C, D, E, F);
    arrays_multiplication_gpu(A, B, C, D, E_d, F_d);
    check = cmpArrays(E, F, E_d, F_d);

    if (check)
        printf("Everythong is fine!\n");
    else
        printf("Arrays missmatch\n");

    printf("Do you want to print the arrays? (y/n) ");
    scanf("%s", ans);

    if (!strcmp(ans, "y") || !strcmp(ans, "Y"))
    {
        printTables("A", A);
        printTables("B", B);
        printTables("C", C);
        printTables("D", D);
        printTables("E (cpu)", E);
        printTables("F (cpu)", F);
        printTables("E (gpu)", E_d);
        printTables("F (gpu)", F_d);
    }

    return 0;
}

void initTables(int A[N][N], int B[N][N], int C[N][N], int D[N][N], int E[N][N], int F[N][N], int E_d[N][N], int F_d[N][N])
{
    unsigned int seed = (unsigned int)time(NULL);
#pragma omp parallel for simd collapse(2)
    for (long unsigned int i = 0; i < N; i++)
        for (long unsigned int j = 0; j < N; j++)
            A[i][j] = randomNumber(&seed, -90, 90);

    seed = (unsigned int)time(NULL);
#pragma omp parallel for simd collapse(2)
    for (long unsigned int i = 0; i < N; i++)
        for (long unsigned int j = 0; j < N; j++)
            B[i][j] = randomNumber(&seed, -90, 90);

    seed = (unsigned int)time(NULL);
#pragma omp parallel for simd collapse(2)
    for (long unsigned int i = 0; i < N; i++)
        for (long unsigned int j = 0; j < N; j++)
            C[i][j] = randomNumber(&seed, -90, 90);

    seed = (unsigned int)time(NULL);
#pragma omp parallel for simd collapse(2)
    for (long unsigned int i = 0; i < N; i++)
        for (long unsigned int j = 0; j < N; j++)
            D[i][j] = randomNumber(&seed, -90, 90);

#pragma omp parallel for simd collapse(2)
    for (long unsigned int i = 0; i < N; i++)
        for (long unsigned int j = 0; j < N; j++)
        {
            E[i][j] = 0;
            F[i][j] = 0;
            E_d[i][j] = 0;
            F_d[i][j] = 0;
        }
}

int randomNumber(unsigned int *seed, int min, int max)
{
    return min + rand_r(seed) % (max - min + 1);
}

void printTables(const char *name, int arr[N][N])
{
    printf("Table %s\n", name);
    printf("-----------------\n");

    for (long unsigned int i = 0; i < N; i++)
    {
        for (long unsigned int j = 0; j < N; j++)
        {
            printf("%6d ", arr[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void arrays_multiplication_cpu(int A[N][N], int B[N][N], int C[N][N], int D[N][N], int E[N][N], int F[N][N])
{
#pragma omp parallel for collapse(2) schedule(static)
    for (long unsigned int i = 0; i < N; i++)
    {
        for (long unsigned int j = 0; j < N; j++)
        {
            int sum_ac = 0;
            int sum_bd = 0;
            int sum_ad = 0;
            int sum_bc = 0;

            for (long unsigned int k = 0; k < N; k++)
            {
                sum_ac += A[i][k] * C[k][j];
                sum_bd += B[i][k] * D[k][j];
                sum_ad += A[i][k] * D[k][j];
                sum_bc += B[i][k] * C[k][j];
            }

            E[i][j] = sum_ac - sum_bd;
            F[i][j] = sum_ad + sum_bc;
        }
    }
}

void arrays_multiplication_gpu(int A[N][N], int B[N][N], int C[N][N], int D[N][N], int E[N][N], int F[N][N])
{
#pragma omp target enter data map(to : A[0 : N][0 : N], B[0 : N][0 : N], C[0 : N][0 : N], D[0 : N][0 : N]) map(alloc : E[0 : N][0 : N], F[0 : N][0 : N])

#pragma omp target
#pragma omp teams distribute parallel for collapse(2)
    for (long unsigned int i = 0; i < N; i++)
    {
        for (long unsigned int j = 0; j < N; j++)
        {
            int sum_ac = 0;
            int sum_bd = 0;
            int sum_ad = 0;
            int sum_bc = 0;

            for (long unsigned int k = 0; k < N; k++)
            {
                sum_ac += A[i][k] * C[k][j];
                sum_bd += B[i][k] * D[k][j];
                sum_ad += A[i][k] * D[k][j];
                sum_bc += B[i][k] * C[k][j];
            }

            E[i][j] = sum_ac - sum_bd;
            F[i][j] = sum_ad + sum_bc;
        }
    }

#pragma omp target exit data map(delete : A[0 : N][0 : N], B[0 : N][0 : N], C[0 : N][0 : N], D[0 : N][0 : N]) map(from : E[0 : N][0 : N], F[0 : N][0 : N])
}

int cmpArrays(int cpu1[N][N], int cpu2[N][N], int gpu1[N][N], int gpu2[N][N])
{
    for (long unsigned int i = 0; i < N; i++)
        for (long unsigned int j = 0; j < N; j++)
        {
            if (cpu1[i][j] != gpu1[i][j])
            {
                fprintf(stderr, "array E %ld %ld %d %d\n", i, j, cpu1[i][j], gpu1[i][j]);
                return FALSE;
            }
            if (cpu2[i][j] != gpu2[i][j])
            {
                fprintf(stderr, "array F %ld %ld %d %d\n", i, j, cpu2[i][j], gpu2[i][j]);
                return FALSE;
            }
        }

    return TRUE;
}
#include "multiplication.h"

void arrays_multiplication_cpu(float **A, float **B, float **C, float **D, float **E, float **F, long unsigned int n)
{
#pragma omp parallel for collapse(2) schedule(static)
    for (long unsigned int i = 0; i < n; i++)
    {
        for (long unsigned int j = 0; j < n; j++)
        {
            float sum_ac = 0.f;
            float sum_bd = 0.f;
            float sum_ad = 0.f;
            float sum_bc = 0.f;

            for (long unsigned int k = 0; k < n; k++)
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

void arrays_multiplication_gpu(float **A, float **B, float **C, float **D, float **E, float **F, long unsigned int n)
{
#pragma omp target enter data map(to : A, B, C, D)

#pragma omp target exit data map(delete : A, B, C, D) map(from : E, F)
}
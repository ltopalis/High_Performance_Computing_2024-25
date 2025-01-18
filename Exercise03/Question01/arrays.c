
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <assert.h>
#include "arrays.h"

void mem_alloc(float ***A, float ***B, float ***C, float ***D, float ***E, float ***F, long unsigned int n)
{
    *A = (float **)malloc(sizeof(float *) * n);
    *B = (float **)malloc(sizeof(float *) * n);
    *C = (float **)malloc(sizeof(float *) * n);
    *D = (float **)malloc(sizeof(float *) * n);
    *E = (float **)malloc(sizeof(float *) * n);
    *F = (float **)malloc(sizeof(float *) * n);

    assert(*A != NULL);
    assert(*B != NULL);
    assert(*C != NULL);
    assert(*D != NULL);
    assert(*E != NULL);
    assert(*F != NULL);

    for (long unsigned int i = 0; i < n; i++)
    {
        (*A)[i] = (float *)malloc(sizeof(float) * n);
        (*B)[i] = (float *)malloc(sizeof(float) * n);
        (*C)[i] = (float *)malloc(sizeof(float) * n);
        (*D)[i] = (float *)malloc(sizeof(float) * n);
        (*E)[i] = (float *)malloc(sizeof(float) * n);
        (*F)[i] = (float *)malloc(sizeof(float) * n);

        assert((*A)[i] != NULL);
        assert((*B)[i] != NULL);
        assert((*C)[i] != NULL);
        assert((*D)[i] != NULL);
        assert((*E)[i] != NULL);
        assert((*F)[i] != NULL);
    }
}

void initTables(float **A, float **B, float **C, float **D, float **E, float **F, long unsigned int n)
{
    unsigned int seed = (unsigned int)time(NULL);
#pragma omp parallel for simd collapse(2)
    for (long unsigned int i = 0; i < n; i++)
        for (long unsigned int j = 0; j < n; j++)
            A[i][j] = randomNumber(&seed, -90.f, 90.f);

    seed = (unsigned int)time(NULL);
#pragma omp parallel for simd collapse(2)
    for (long unsigned int i = 0; i < n; i++)
        for (long unsigned int j = 0; j < n; j++)
            B[i][j] = randomNumber(&seed, -90.f, 90.f);

    seed = (unsigned int)time(NULL);
#pragma omp parallel for simd collapse(2)
    for (long unsigned int i = 0; i < n; i++)
        for (long unsigned int j = 0; j < n; j++)
            C[i][j] = randomNumber(&seed, -90.f, 90.f);

    seed = (unsigned int)time(NULL);
#pragma omp parallel for simd collapse(2)
    for (long unsigned int i = 0; i < n; i++)
        for (long unsigned int j = 0; j < n; j++)
            D[i][j] = randomNumber(&seed, -90.f, 90.f);

#pragma omp parallel for simd collapse(2)
    for (long unsigned int i = 0; i < n; i++)
        for (long unsigned int j = 0; j < n; j++)
            E[i][j] = 0.f;

#pragma omp parallel for simd collapse(2)
    for (long unsigned int i = 0; i < n; i++)
        for (long unsigned int j = 0; j < n; j++)
            F[i][j] = 0.f;
}

float randomNumber(unsigned int *seed, float min, float max)
{
    return min + ((float)rand_r(seed) / RAND_MAX) * (max - min);
}

void printTables(float **A, float **B, float **C, float **D, float **E, float **F, long unsigned int n)
{
    printf("Table A\n");
    printf("-----------------\n");

    for (long unsigned int i = 0; i < n; i++)
    {
        for (long unsigned int j = 0; j < n; j++)
        {
            printf("%6.2f ", A[i][j]);
        }
        printf("\n");
    }

    printf("\n");
    printf("Table B\n");
    printf("-----------------\n");

    for (long unsigned int i = 0; i < n; i++)
    {
        for (long unsigned int j = 0; j < n; j++)
        {
            printf("%6.2f ", B[i][j]);
        }
        printf("\n");
    }

    printf("\n");
    printf("Table C\n");
    printf("-----------------\n");

    for (long unsigned int i = 0; i < n; i++)
    {
        for (long unsigned int j = 0; j < n; j++)
        {
            printf("%6.2f ", C[i][j]);
        }
        printf("\n");
    }

    printf("\n");
    printf("Table D\n");
    printf("-----------------\n");

    for (long unsigned int i = 0; i < n; i++)
    {
        for (long unsigned int j = 0; j < n; j++)
        {
            printf("%6.2f ", D[i][j]);
        }
        printf("\n");
    }

    printf("\n");
    printf("Table E\n");
    printf("-----------------\n");

    for (long unsigned int i = 0; i < n; i++)
    {
        for (long unsigned int j = 0; j < n; j++)
        {
            printf("%9.2f ", E[i][j]);
        }
        printf("\n");
    }

    printf("\n");
    printf("Table F\n");
    printf("-----------------\n");

    for (long unsigned int i = 0; i < n; i++)
    {
        for (long unsigned int j = 0; j < n; j++)
        {
            printf("%9.2f ", F[i][j]);
        }
        printf("\n");
    }
}

void mem_free(float **A, float **B, float **C, float **D, float **E, float **F, long unsigned int n)
{
    for (long unsigned int i = 0; i < n; i++)
    {
        free(A[i]);
        free(B[i]);
        free(C[i]);
        free(D[i]);
        free(E[i]);
        free(F[i]);
    }
    free(A);
    free(B);
    free(C);
    free(D);
    free(E);
    free(F);
}

int cmpArrays(float **cpu1, float **cpu2, float **gpu1, float **gpu2, long unsigned int n)
{

    for (long unsigned int i = 0; i < n; i++)
        for (long unsigned int j = 0; j < n; j++)
        {
            if (cpu1[i][j] != gpu1[i][j])
                return FALSE;
            if (cpu2[i][j] != gpu2[i][j])
                return FALSE;
        }

    return TRUE;
}
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include <time.h>

#include "arrays.h"
#include "multiplication.h"

#define N 5000

int main(int argc, char **argv)
{
    float **A, **B, **C, **D, **E, **F;

    mem_alloc(&A, &B, &C, &D, &E, &F, N);

    initTables(A, B, C, D, E, F, N);
    // printTables(A, B, C, D, E, F, N);

    arrays_multiplication_cpu(A, B, C, D, E, F, N);

    // printTables(A, B, C, D, E, F, N);

    mem_free(A, B, C, D, E, F, N);

    return 0;
}

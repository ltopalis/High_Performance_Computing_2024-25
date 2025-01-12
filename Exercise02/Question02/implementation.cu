#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 10

int *malloc_h(long unsigned int dim);
bool check_arrays(int *a, int *b, long unsigned int dim);
void matrix_mul_cpu(int *a, int *b, int *c, int *d, int *e, int *f, int n);

__global__ void matrix_mul(int *a, int *b, int *c, int *d, int *e, int *f, int n);

int main()
{
    int *a, *b, *c, *d, *e, *f, *e_cpu, *f_cpu; // host
    int *d_a, *d_b, *d_c, *d_d, *d_e, *d_f;     // device

    // memory allocation
    a = malloc_h(N);
    b = malloc_h(N);
    c = malloc_h(N);
    d = malloc_h(N);
    e = malloc_h(N);
    f = malloc_h(N);
    e_cpu = malloc_h(N);
    f_cpu = malloc_h(N);

    cudaMalloc((void **)&d_a, N * N * sizeof(int));
    cudaMalloc((void **)&d_b, N * N * sizeof(int));
    cudaMalloc((void **)&d_c, N * N * sizeof(int));
    cudaMalloc((void **)&d_d, N * N * sizeof(int));
    cudaMalloc((void **)&d_e, N * N * sizeof(int));
    cudaMalloc((void **)&d_f, N * N * sizeof(int));

    // initialize matrices
    srand(time(NULL));
    for (int i = 0; i < N * N; i++)
    {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
        c[i] = rand() % 10;
        d[i] = rand() % 10;
        e[i] = 0;
        f[i] = 0;
        e_cpu[i] = 0;
        f_cpu[i] = 0;
    }

    cudaMemcpy(d_a, a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, d, N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    matrix_mul<<<gridSize, blockSize>>>(d_a, d_b, d_c, d_d, d_e, d_f, N);
    cudaDeviceSynchronize();

    cudaMemcpy(e, d_e, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(f, d_f, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    matrix_mul_cpu(a, b, c, d, e_cpu, f_cpu, N);

    bool result = check_arrays(e, e_cpu, N) & check_arrays(f, f_cpu, N);
    if (result)
        printf("Results match!\n");
    else
        printf("Results mismatch!\n");

    // Free memory
    free(a);
    free(b);
    free(c);
    free(d);
    free(e);
    free(f);
    free(e_cpu);
    free(f_cpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
    cudaFree(d_e);
    cudaFree(d_f);

    return 0;
}

int *malloc_h(long unsigned int dim)
{
    int *p = (int *)malloc(dim * dim * sizeof(int));
    if (!p)
    {
        fprintf(stderr, "Host memory allocation failed.\n");
        exit(0);
    }
    return p;
}

bool check_arrays(int *a, int *b, long unsigned int dim)
{
    for (long unsigned int i = 0; i < dim * dim; i++)
    {
        if (a[i] != b[i])
        {
            printf("Mismatch at index %lu: GPU = %d, CPU = %d\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

void matrix_mul_cpu(int *a, int *b, int *c, int *d, int *e, int *f, int n)
{

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int sum_r = 0;
            int sum_i = 0;

            for (int k = 0; k < n; k++)
            {
                sum_r += a[i * n + k] * b[k * n + j];
                sum_i += c[i * n + k] * d[k * n + j];
            }
            e[i * n + j] = sum_r;
            f[i * n + j] = sum_i;
        }
    }
}

__global__ void matrix_mul(int *a, int *b, int *c, int *d, int *e, int *f, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        int sum_r = 0;
        int sum_i = 0;
        for (int i = 0; i < n; i++)
        {
            sum_r += a[row * n + i] * b[i * n + col];
            sum_i += c[row * n + i] * d[i * n + col];
        }
        e[row * n + col] = sum_r;
        f[row * n + col] = sum_i;
    }
}

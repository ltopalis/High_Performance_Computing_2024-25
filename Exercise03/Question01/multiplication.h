#ifndef MULTIPLICATION_H

#define MULTIPLICATION_H

void arrays_multiplication_cpu(float **A, float **B, float **C, float **D, float **E, float **F, long unsigned int n);
void arrays_multiplication_gpu(float **A, float **B, float **C, float **D, float **E, float **F, long unsigned int n);

#endif
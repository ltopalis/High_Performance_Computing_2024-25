#ifndef ARRAYS_H

#define ARRAYS_H

#define TRUE 1
#define FALSE 0

void mem_alloc(float ***A, float ***B, float ***C, float ***D, float ***E, float ***F, long unsigned int n);
void initTables(float **A, float **B, float **C, float **D, float **E, float **F, long unsigned int n);
float randomNumber(unsigned int *seed, float min, float max);
void printTables(float **A, float **B, float **C, float **D, float **E, float **F, long unsigned int n);
void mem_free(float **A, float **B, float **C, float **D, float **E, float **F, long unsigned int n);
int cmpArrays(float **cpu1, float **cpu2, float **gpu1, float **gpu2, long unsigned int n);

#endif
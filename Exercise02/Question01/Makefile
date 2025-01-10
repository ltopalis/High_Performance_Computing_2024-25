CC=gcc
CFLAGS=-O3
LIBS=-lm  -fopt-info-vec-missed  -march=native -ftree-vectorize  -fopenmp -Wall

all: bench

bench: bench.c weno.h
	$(CC) $(CFLAGS) -o bench bench.c $(LIBS)

clean:
	rm -f bench
# -fopt-info-vec -fopt-info -fopt-info-vec 
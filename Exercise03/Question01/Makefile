all:
	gcc -c arrays.c -o arrays.o -fopenmp									# compile object file arrays
	gcc -c multiplication.c -o multiplication.o -fopenmp					# compile object file multiplication
	gcc implementation.c -fPIC arrays.c -fPIC multiplication.c -o implementation -fopenmp
clean:
	rm -f implementation
	rm -f arrays.o
	rm -f multiplication.o
all:
	gcc implementation.c -o implementation -fopenmp -lgomp -foffload=nvptx-none -lcuda -foffload="-misa=sm_35" -fcf-protection=none -fno-stack-protector
clean:
	rm -f implementation
	rm -f arrays.o
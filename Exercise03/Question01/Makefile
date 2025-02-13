all:
	gcc implementation.c -o implementation -fopenmp -lgomp -foffload=nvptx-none -lcuda -foffload="-misa=sm_35" -fcf-protection=none -fno-stack-protector
test:
	gcc implementation.c -o implementation_10  -DN=10  -fopenmp -lgomp -foffload=nvptx-none -lcuda -foffload="-misa=sm_35" -fcf-protection=none -fno-stack-protector
	gcc implementation.c -o implementation_50  -DN=50  -fopenmp -lgomp -foffload=nvptx-none -lcuda -foffload="-misa=sm_35" -fcf-protection=none -fno-stack-protector
	gcc implementation.c -o implementation_100 -DN=100 -fopenmp -lgomp -foffload=nvptx-none -lcuda -foffload="-misa=sm_35" -fcf-protection=none -fno-stack-protector
	gcc implementation.c -o implementation_500 -DN=500 -fopenmp -lgomp -foffload=nvptx-none -lcuda -foffload="-misa=sm_35" -fcf-protection=none -fno-stack-protector
clean:
	rm -f implementation_*
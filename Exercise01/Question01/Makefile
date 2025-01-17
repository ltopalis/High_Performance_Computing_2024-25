all:
	mpic++ a.cpp -o a.out
	mpic++ b.cpp -o b.out -fopenmp
	mpic++ c.cpp -o c.out -fopenmp
	mpic++ d.cpp -o d.out -fopenmp -lz

clean:
	rm -f *.bin *.out
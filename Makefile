build:
	docker build -t hpc .

run:
	docker run --name hpc_con --mount type=bind,src=./,dst=/workspace -it hpc

open:
	docker exec -it hpc_con /bin/bash

clean:
	docker stop hpc_con
	docker rm hpc_con
	docker image rm hpc

all:
	docker build -t hpc .
	docker run --name hpc_con --mount type=bind,src=./,dst=/workspace -it hpc
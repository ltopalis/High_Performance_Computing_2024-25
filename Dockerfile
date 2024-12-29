FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    sudo \
    g++ \
    cmake \
    mpich \
    clang \
    zlib1g-dev \
    python3-pip

RUN mkdir /libraries 
#     cd /libraries && \
#     python3 -m venv myvenv 

ENTRYPOINT ["/bin/bash"]
WORKDIR /workspace

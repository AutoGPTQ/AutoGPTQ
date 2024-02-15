# Build with: `docker build -f Dockerfile -t autogptq .`
# Run with: `docker run --gpus all --rm -it autogptq:latest /bin/bash`

FROM nvcr.io/nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt update && \
    apt install -y wget git && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir .conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda init bash

RUN pip install --upgrade pip
RUN pip install --upgrade numpy torch setuptools wheel

RUN git clone https://github.com/AutoGPTQ/AutoGPTQ.git
WORKDIR /AutoGPTQ

RUN pip install -vvv .
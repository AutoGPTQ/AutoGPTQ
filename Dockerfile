FROM ubuntu:22.04

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

RUN DISABLE_QIGEN=1 pip install -vvv .[quality]
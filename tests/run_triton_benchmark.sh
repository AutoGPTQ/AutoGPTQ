#!/bin/bash

kernels=("default" "triton" "tritonv2" "exllama" "exllamav2")

for kernel in "${kernels[@]}"; do
    CMD="python benchmark_tritonv2_integration.py --benchmark_kernel=$kernel 2>&1 | tee ${kernel}_bench.txt"
    echo $CMD
    eval $CMD
done
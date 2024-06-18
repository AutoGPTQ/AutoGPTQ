#include <torch/all.h>

__global__ void gptq_repack_kernel(
  uint32_t* in,
  uint32_t* out,
  int m,
  int n
);

torch::Tensor gptq_repack(
    torch::Tensor W
);
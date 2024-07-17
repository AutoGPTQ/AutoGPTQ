#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <torch/all.h>
#include "../sycl_utils.h"

void gptq_repack_kernel(
  uint32_t* in,
  uint32_t* out,
  int m,
  int n,
  const sycl::nd_range<3> &item_ct1
);

torch::Tensor gptq_repack(
    torch::Tensor W
);
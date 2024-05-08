// Adapted from turboderp exllama: https://github.com/turboderp/exllama

#ifndef _q4_matmul_cuh
#define _q4_matmul_cuh

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdint>
#include <cstdio>
#include <dpct/blas_utils.hpp>
#include <dpct/lib_common_utils.hpp>

#include "q4_matrix.hpp"
#include "../tuning.h"
#include "../../sycl_utils.h"


void q4_matmul_sycl
(
    ExLlamaTuning* tuningParams,
    const sycl::half* x,
    const int x_height,
    const Q4Matrix* w,
    sycl::half* out,
    bool no_zero = false
);

void q4_matmul_recons_sycl
(
    ExLlamaTuning* tuningParams,
    const sycl::half* x,
    const int x_height,
    Q4Matrix* w,
    sycl::half* out,
    bool no_zero = false
);

#endif

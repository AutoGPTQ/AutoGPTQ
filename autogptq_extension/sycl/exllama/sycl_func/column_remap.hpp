// Adapted from turboderp exllama: https://github.com/turboderp/exllama

#ifndef _column_remap_cuh
#define _column_remap_cuh

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdint>

void column_remap_sycl
(
    const sycl::half* x,
    sycl::half* x_new,
    const int x_height,
    const int x_width,
    const uint32_t* x_map
);

#endif
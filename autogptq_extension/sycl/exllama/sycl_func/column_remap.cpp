// Adapted from turboderp exllama: https://github.com/turboderp/exllama

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "column_remap.hpp"
#include "../util.h"
#include "../../sycl_utils.h"

const int SHUF_BLOCKSIZE_X = 256;
const int SHUF_BLOCKSIZE_Y = 16;

void column_remap_kernel
(
    const sycl::half* __restrict__ x,
    sycl::half* __restrict__ x_new,
    const int x_width,
    const int x_height,
    const uint32_t* x_map,
    const sycl::nd_item<3> &item_ct1
)
{
    //auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    int x_column = SHUF_BLOCKSIZE_X * item_ct1.get_group(2) + item_ct1.get_local_id(2);
    int x_row = SHUF_BLOCKSIZE_Y * item_ct1.get_group(1);
    if (x_column >= x_width) return;
    //if (x_row >= x_height) return;

    int x_stride = x_width;
    int x_idx = x_row * x_stride + x_column;

    int x_row_end = sycl::min(x_row + SHUF_BLOCKSIZE_Y, x_height);
    int x_idx_end = x_row_end * x_stride + x_column;

    int s_column = x_map[x_column];
    int s_idx = x_row * x_stride + s_column;

    while (x_idx < x_idx_end)
    {
        x_new[x_idx] = x[s_idx];
        x_idx += x_stride;
        s_idx += x_stride;
    }
}

// Remap columns in x to correspond to sequential group index before matmul
//
// perform x -> seq_x such that seq_x @ seq_w == x @ w

void column_remap_sycl
(
    const sycl::half* x,
    sycl::half* x_new,
    const int x_height,
    const int x_width,
    const uint32_t* x_map
)
{
    sycl::range<3> threads(1, 1, SHUF_BLOCKSIZE_X);

    sycl::range<3> blocks
    (1, (x_height + SHUF_BLOCKSIZE_Y - 1) / SHUF_BLOCKSIZE_Y, (x_width + SHUF_BLOCKSIZE_X - 1) / SHUF_BLOCKSIZE_X);
    auto& q_ct1 = gptq::xpu::gptqGetQueue();
    {
      //dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.parallel_for(
        sycl::nd_range<3>(blocks * threads, threads), 
        [=](sycl::nd_item<3> item_ct1) {
          column_remap_kernel(x, x_new, x_width, x_height, x_map, item_ct1);
        });
    }
}

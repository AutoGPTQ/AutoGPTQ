#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <torch/all.h>
#include <torch/python.h>

#include <ATen/core/Tensor.h>
#include "../sycl_utils.h"

#include "marlin_repack.h"

void gptq_repack_kernel(
  uint32_t* in,
  uint32_t* out,
  int m,
  int n,
  const sycl::nd_item<3> &item_ct1
) {
  uint32_t row = item_ct1.get_group(2) * 2;
  uint32_t col = item_ct1.get_group(1) * 64;
  uint32_t t = item_ct1.get_local_id(2);

  // marlin packs 4 16x16 blocks one time;
  const int pad_len = 18;
  auto &block = *sycl::ext::oneapi::group_local_memory_for_overwrite<uint8_t[4][16][pad_len]>(item_ct1.get_group()); 

  // unpack
  int block_idx = t / 8;
  int block_offset = t % 8;
  for (int offset = block_offset; offset < 16; offset += 8) {
    uint32_t v1 = in[row * n + col + block_idx * 16 + offset];
    uint32_t v2 = in[(row + 1) * n + col + block_idx * 16 + offset];
#pragma unroll
    for (int i = 0; i < 8; i += 1) {
      block[block_idx][i][offset] = v1 & 0xf;
      v1 >>= 4;
      block[block_idx][i + 8][offset] = v2 & 0xf;
      v2 >>= 4;
    }
  }

  // repack
  // ref: _get_perms @ https://github.com/IST-DASLab/marlin/blob/master/marlin/__init__.py
  uint32_t srow = (t % 4) * 2;
  uint32_t scol = t / 4;

  uint32_t idx[8][2];
  idx[0][0] = srow;     idx[0][1] = scol;
  idx[1][0] = srow + 8; idx[1][1] = scol;
  idx[2][0] = srow;     idx[2][1] = scol + 8;
  idx[3][0] = srow + 8; idx[3][1] = scol + 8;

  idx[4][0] = srow + 1; idx[4][1] = scol;
  idx[5][0] = srow + 9; idx[5][1] = scol;
  idx[6][0] = srow + 1; idx[6][1] = scol + 8;
  idx[7][0] = srow + 9; idx[7][1] = scol + 8;

#pragma unroll
  for (int i = 0; i < 4; i += 1) {
    uint32_t v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      v[j] = block[i][idx[j][0]][idx[j][1]];
    }

    uint32_t pack = (v[7] << 28) | (v[6] << 24) | (v[5] << 20) | (v[4] << 16) |
        (v[3] << 12) | (v[2] << 8) | (v[1] << 4) | v[0];

    out[item_ct1.get_group(2) * n * 2 + item_ct1.get_group(1) * 128 + t * 4 + i] = pack;
  }
}

torch::Tensor gptq_repack(
    torch::Tensor W
) {
  int m = W.sizes()[0];
  int n = W.sizes()[1];

  assert(W.is_contiguous());
  assert(W.dtype() == at::kInt);
  assert(m % 2 == 0);
  assert(n % 64 == 0);
  auto result = at::empty(
      {m / 2, n * 2}, at::TensorOptions().dtype(at::kInt).device(W.device()));
  
  const sycl::range<3> threads(1, 1, 32);
  // marlin packs 16 x 64 block and gptq packs 8 x 1
  const sycl::range<3> blocks(1, n / 64, m / 2);
  auto& q_ct1 = gptq::xpu::gptqGetQueue();
  
  q_ct1.submit(
    [&](sycl::handler &cgh) {
      uint32_t * W_data_ptr = (uint32_t*)W.data_ptr();
      uint32_t * result_data_ptr = (uint32_t*)result.data_ptr();

      cgh.parallel_for(
        sycl::nd_range<3>(blocks * threads, threads), 
        [=](sycl::nd_item<3> item_ct1) {
          gptq_repack_kernel(W_data_ptr, result_data_ptr, m, n, item_ct1);
        });
    });
  return result;
}
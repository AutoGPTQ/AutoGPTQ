#include <cuda_runtime.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "marlin_repack.cuh"

__global__ void gptq_repack_kernel(
  uint32_t* in,
  uint32_t* out,
  int m,
  int n
) {
  uint32_t row = blockIdx.x * 2;
  uint32_t col = blockIdx.y * 64;
  uint32_t t = threadIdx.x;

  // marlin packs 4 16x16 blocks one time;
  const int pad_len = 18;
  __shared__ uint8_t block[4][16][pad_len];

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

    out[blockIdx.x * n * 2 + blockIdx.y * 128 + t * 4 + i] = pack;
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

  const at::cuda::OptionalCUDAGuard device_guard(device_of(W));
  const dim3 threads(32);
  // marlin packs 16 x 64 block and gptq packs 8 x 1
  const dim3 blocks(m / 2, n / 64);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  gptq_repack_kernel<<<blocks, threads, 0, stream>>>(
    (uint32_t*)W.data_ptr(),
    (uint32_t*)result.data_ptr(),
    m,
    n
  );
  return result;
}
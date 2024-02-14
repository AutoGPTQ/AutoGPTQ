/*
 * Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <torch/all.h>
#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "marlin_cuda_kernel.cuh"
#include "marlin_repack.cuh"

const int ERR_PROB_SHAPE = 1;
const int ERR_KERN_SHAPE = 2;

torch::Tensor gptq_repack(
    torch::Tensor W
){
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

void mul(
  const torch::Tensor& A,
  const torch::Tensor& B,
        torch::Tensor& C,
  const torch::Tensor& s,
        torch::Tensor& workspace,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 8
) {
  int prob_m = A.size(0);
  int prob_n = C.size(1);
  int prob_k = A.size(1);
  int groupsize = (s.size(0) == 1) ? -1 : prob_k / s.size(0);
  if (groupsize != -1 && groupsize * s.size(0) != prob_k)
    AT_ERROR("k=", prob_k, " not compatible with ", s.size(0), " groups.");
  if (workspace.numel() < prob_n / 128 * max_par)
    AT_ERROR("workspace must be of size at least ", prob_n / 128 * max_par, ".");
  int dev = A.get_device();
  int err = marlin_cuda(
    A.data_ptr(),
    B.data_ptr(),
    C.data_ptr(),
    s.data_ptr(),
    prob_m, prob_n, prob_k,
    workspace.data_ptr(),
    groupsize,
    dev,
    at::cuda::getCurrentCUDAStream(dev),
    thread_k,
    thread_n,
    sms,
    max_par
  );
  if (err == ERR_PROB_SHAPE) {
    AT_ERROR(
      "Problem (m=", prob_m, ", n=", prob_n, ", k=", prob_k, ")",
      " not compatible with thread_k=", thread_k, ", thread_n=", thread_n, "."
    );
  } else if (err == ERR_KERN_SHAPE) {
    AT_ERROR(
      "No kernel implementation for thread_k=", thread_k, ", thread_n=", thread_n, ", groupsize=", groupsize, "."
    );
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mul", &mul, "Marlin FP16xINT4 matmul.");
  m.def("gptq_repack", &gptq_repack, "Repack gptq for Marlin.")
}
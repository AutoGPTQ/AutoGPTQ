#include <cuda.h>
#include <cuda_runtime.h>

int marlin_cuda(
  const void* A,
  const void* B,
        void* C,
        void* s,
  int prob_m,
  int prob_n,
  int prob_k,
  void* workspace,
  int groupsize,
  int dev,
  cudaStream_t stream,
  int thread_k,
  int thread_n,
  int sms,
  int max_par
);

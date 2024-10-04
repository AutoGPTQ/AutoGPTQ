#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "../sycl_utils.h"


int marlin_sycl(
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
  dpct::queue_ptr stream,
  int thread_k,
  int thread_n,
  int sms,
  int max_par
);

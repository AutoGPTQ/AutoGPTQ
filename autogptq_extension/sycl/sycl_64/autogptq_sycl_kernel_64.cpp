#include <torch/all.h>
#include <torch/python.h>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "../sycl_utils.h"

// atomicAdd for double-precision floating-point numbers on hardware with
// compute capability < 6.0 from:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
// #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
// __device__ double atomicAdd(
//     double* address,
//     double val
// ) {
//   unsigned long long int* address_as_ull = (unsigned long long int*)address;
//   unsigned long long int old = *address_as_ull, assumed;
//
//   do {
//     assumed = old;
//     old = atomicCAS(
//       address_as_ull,
//       assumed,
//       __double_as_longlong(val + __longlong_as_double(assumed))
//     );
//
//   // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
//   } while (assumed != old);
//
//   return __longlong_as_double(old);
// }
// #endif

/*
#if (defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 700) || defined(USE_ROCM)
// adapted from https://github.com/torch/cutorch/blob/master/lib/THC/THCAtomics.cuh
__device__ __forceinline__ void atomicAdd(c10::Half* address, c10::Half val) {
    unsigned int *address_as_ui = reinterpret_cast<unsigned int *>(reinterpret_cast<char *>(address) - (reinterpret_cast<size_t>(address) & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do {
        assumed = old;
        unsigned short hsum = reinterpret_cast<size_t>(address) & 2 ? (old >> 16) : (old & 0xffff);
        hsum += val;
        old = reinterpret_cast<size_t>(address) & 2
                 ? (old & 0xffff) | (hsum << 16)
                 : (old & 0xffff0000) | hsum;
        old = atomicCAS(address_as_ui, assumed, old);

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
}
__device__ __forceinline__ void atomicAdd(__half* address, c10::Half val) {
    unsigned int * address_as_ui = (unsigned int *) ((char *)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do {
        assumed = old;
        __half_raw hsum;
        hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
        half tmpres = __hadd(hsum, val);
        hsum = __half_raw(tmpres);
        old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16) : (old & 0xffff0000) | hsum.x;
        old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
}
#endif
*/

template <typename scalar_t>
void VecQuant2MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  		int* __restrict__ zeros,
	const  	    int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
	int zero_width,
	const sycl::nd_item<3> &item_ct1,
	scalar_t *blockvec);

template <typename scalar_t>
void VecQuant3MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  		int* __restrict__ zeros,
	const  	    int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
	int zero_width,
	const sycl::nd_item<3> &item_ct1,
	scalar_t *blockvec);

template <typename scalar_t>
void VecQuant4MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  		int* __restrict__ zeros,
	const  	    int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
	int zero_width
,
	const sycl::nd_item<3> &item_ct1,
	scalar_t *blockvec);

template <typename scalar_t>
void VecQuant8MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  		int* __restrict__ zeros,
	const  	    int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
	int zero_width,
	const sycl::nd_item<3> &item_ct1,
	scalar_t *blockvec);

template <typename scalar_t>
void VecQuant2MatMulKernel_old(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  		int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize,
    const sycl::nd_item<3> &item_ct1,
    scalar_t *blockvec);

template <typename scalar_t>
void VecQuant3MatMulKernel_old(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  	int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize,
    const sycl::nd_item<3> &item_ct1,
    scalar_t *blockvec);

template <typename scalar_t>
void VecQuant4MatMulKernel_old(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  	int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize,
    const sycl::nd_item<3> &item_ct1,
    scalar_t *blockvec);

template <typename scalar_t>
void VecQuant8MatMulKernel_old(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  	int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize,
    const sycl::nd_item<3> &item_ct1,
    scalar_t *blockvec);

void VecQuant2MatMulKernelFaster_old(
    const  sycl::half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const    int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize,
    const sycl::nd_item<3> &item_ct1,
    sycl::half2 *blockvec,
    sycl::local_accessor<sycl::half2, 2> deq2);

void VecQuant3MatMulKernelFaster_old(
    const  sycl::half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const    int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize,
    const sycl::nd_item<3> &item_ct1,
    sycl::half2 *blockvec,
    sycl::local_accessor<sycl::half2, 2> deq2);

void VecQuant4MatMulKernelFaster_old(
    const  sycl::half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const    int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize,
    const sycl::nd_item<3> &item_ct1,
    sycl::half2 *blockvec,
    sycl::local_accessor<sycl::half2, 2> deq2);


const int BLOCKWIDTH  = 64;
const int BLOCKHEIGHT2 =  4;
const int BLOCKHEIGHT3 =  6;
const int BLOCKHEIGHT4 =  8;
const int BLOCKHEIGHT8 =  16;

inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

inline int as_int(int i) {
  return *reinterpret_cast<int*>(&i);
}


template <typename scalar_t>
SYCL_EXTERNAL void VecQuant2MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  		int* __restrict__ zeros,
    const   	int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
	  int zero_width,
	  const sycl::nd_item<3> &item_ct1,
    scalar_t *blockvec) {
 
  int h = BLOCKHEIGHT2 * item_ct1.get_group(2);
  int w = BLOCKWIDTH * item_ct1.get_group(1) + item_ct1.get_local_id(2);

  
  int i = width * h + w;
  int g_h = h * 16;
  int k;
  unsigned int g;
  scalar_t w_tmp;

  int z_w = w / 16;
  int z_mod = (w % 16) * 2;

  float weight[BLOCKWIDTH];

  for (k = 0; k <  BLOCKWIDTH; ++k){
	int k_w = (k / 16);
	int k_bit = (k % 16) * 2;

  g = as_int(g_idx[g_h + k]);
  scalar_t scale = scales[g * width + w];

  // Avoid overflows with & 0x0f.
  scalar_t zero = scalar_t(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod & 0x3) + 1) & 0x0f);

  w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x3);

	weight[k] = scale * (w_tmp - zero);
  }

  scalar_t res;
  for (int b = 0; b < batch; ++b){
	res = 0;

    blockvec[item_ct1.get_local_id(2)] = vec[b * vec_height + item_ct1.get_group(2) * BLOCKWIDTH + item_ct1.get_local_id(2)];
    
    item_ct1.barrier(sycl::access::fence_space::local_space);
	for (k = 0; k <  BLOCKWIDTH; ++k){
	  res += weight[k] * blockvec[k];
    }
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&mul[b * width + w], res);
    
    item_ct1.barrier(sycl::access::fence_space::local_space);
  }
}


template <typename scalar_t>
void call_vecquant2matmul_sycl(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  		int* __restrict__ zeros,
    const   	int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
	 int zero_width
){


   using sycl_t = gptq::xpu::SyclTypeTrait<scalar_t>::Type;
   sycl::range<3> blocks(1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH, (height + BLOCKHEIGHT2 - 1) / BLOCKHEIGHT2);
   sycl::range<3> threads(1, 1, BLOCKWIDTH); //grid
   auto& q_ct1 = gptq::xpu::gptqGetQueue();
   q_ct1.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<sycl_t, 1> blockvec(sycl::range<1>(BLOCKWIDTH), cgh);
      cgh.parallel_for(
          sycl::nd_range<3>(blocks * threads, threads),
          [=](sycl::nd_item<3> item_ct1) {
            VecQuant2MatMulKernel<scalar_t>(
                (const sycl_t* __restrict__)vec,
                (const int* __restrict__)mat,
                (sycl_t* __restrict__)mul,
                (const sycl_t* __restrict__)scales,
                (const int* __restrict__)zeros,
                (const int* __restrict__)g_idx,
                batch,
                vec_height,
                height,
                width,
                zero_width,
                item_ct1,
                blockvec.get_pointer());
          });
    });

}

void vecquant2matmul_sycl(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  torch::Tensor g_idx
) {
  int batch = vec.size(0);
  int vec_height = vec.size(1);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "call_vecquant2matmul_sycl", ([&] {
      call_vecquant2matmul_sycl(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(), g_idx.data<int>(),
        batch, vec_height, height, width, zero_width
      );
    })
  );
}


template <typename scalar_t>
/*
DPCT1110:3: The total declared local variable size in device function VecQuant3MatMulKernel exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total register size available and adjust the code, or use smaller sub-group size to avoid high register pressure.
*/

void VecQuant3MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const       int* __restrict__ zeros,
    const   	int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
	  int zero_width,
	  const sycl::nd_item<3> &item_ct1,
	  scalar_t *blockvec) {
     
  int h = BLOCKHEIGHT3 * item_ct1.get_group(2);
  int w = BLOCKWIDTH * item_ct1.get_group(1) + item_ct1.get_local_id(2);

  
  int i = width * h + w;
  int g_h = (h / 3) * 32;
  int k;
  unsigned int g;
  scalar_t w_tmp;

  int z_w = (w / 32) * 3;
  int z_mod = w % 32;
  int z_bit;
  unsigned int z_tmp;
  if (z_mod != 10){
    if (z_mod != 21){
      z_bit = z_mod;
      if (z_bit > 21){
        z_bit -= 22;
        z_bit *= 3;
        z_bit += 2;
        z_w += 2;
      } else if (z_bit > 10){
        z_bit -= 11;
        z_bit *= 3;
        z_bit += 1;
        z_w += 1;
      } else {
        z_bit *= 3;
      }
    } else {
      z_w += 1;
    }
  }

  float weight[BLOCKWIDTH];

  for (k = 0; k <  BLOCKWIDTH; ++k){
	int k_w = (k / 32) * 3;
	int k_mod = k % 32;
	int k_bit;

	if (k_mod != 10){
	  if (k_mod != 21){
        k_bit = k_mod;
        if (k_bit > 21){
		  k_bit -= 22;
		  k_bit *= 3;
		  k_bit += 2;
		  k_w += 2;
        } else if (k_bit > 10){
		  k_bit -= 11;
		  k_bit *= 3;
		  k_bit += 1;
		  k_w += 1;
        } else {
		  k_bit *= 3;
        }
	  } else {
        k_w += 1;
	  }
	}

    g = as_int(g_idx[g_h + k]);
    scalar_t scale = scales[g * width + w];
    scalar_t zero;
    if (z_mod == 10) {
      z_tmp = (as_unsigned(zeros[g * zero_width + z_w]) >> 30) | ((as_unsigned(zeros[g * zero_width + (z_w + 1)]) << 2) & 0x4);
      zero = scalar_t(((z_tmp) + 1) & 0x0f);  // Avoid overflows
    } else if (z_mod == 21){
      z_tmp = (as_unsigned(zeros[g * zero_width + z_w]) >> 31) | ((as_unsigned(zeros[g * zero_width + (z_w + 1)]) << 1) & 0x6);
      zero = scalar_t(((z_tmp) + 1) & 0x0f);
    } else {
      zero = scalar_t((((as_unsigned(zeros[g * zero_width + z_w]) >> z_bit) & 0x7) + 1) & 0x0f);
    }

    if (k_mod == 10) {
      w_tmp = (as_unsigned(mat[i + (k_w * width)]) >> 30) | ((as_unsigned(mat[i + ((k_w + 1)* width)]) << 2) & 0x4);
    } else if (k_mod == 21){
      w_tmp = (as_unsigned(mat[i + (k_w * width)]) >> 31) | ((as_unsigned(mat[i + ((k_w + 1)* width)]) << 1) & 0x6);
    } else {
      w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x7);
    }
	weight[k] = scale * (w_tmp - zero);
  }

  scalar_t res;
  for (int b = 0; b < batch; ++b){
	res = 0;

    blockvec[item_ct1.get_local_id(2)] = vec[b * vec_height + item_ct1.get_group(2) * BLOCKWIDTH + item_ct1.get_local_id(2)];

    item_ct1.barrier(sycl::access::fence_space::local_space);
	for (k = 0; k <  BLOCKWIDTH; ++k){
	  res += weight[k] * blockvec[k];
    }
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&mul[b * width + w], res);

    item_ct1.barrier(sycl::access::fence_space::local_space);
  }
}


template <typename scalar_t>
void call_vecquant3matmul_sycl(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  		int* __restrict__ zeros,
    const   	int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
	 int zero_width
){


   using sycl_t = gptq::xpu::SyclTypeTrait<scalar_t>::Type;
   sycl::range<3> blocks(1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH, (height + BLOCKHEIGHT2 - 1) / BLOCKHEIGHT2);
   sycl::range<3> threads(1, 1, BLOCKWIDTH); //grid
   auto& q_ct1 = gptq::xpu::gptqGetQueue();
   q_ct1.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<sycl_t, 1> blockvec(sycl::range<1>(BLOCKWIDTH), cgh);
      cgh.parallel_for(
          sycl::nd_range<3>(blocks * threads, threads),
          [=](sycl::nd_item<3> item_ct1) {
            VecQuant3MatMulKernel<sycl_t>(
                (const sycl_t* __restrict__)vec,
                (const int* __restrict__)mat,
                (sycl_t* __restrict__)mul,
                (const sycl_t* __restrict__)scales,
                (const int* __restrict__)zeros,
                (const int* __restrict__)g_idx,
                batch,
                vec_height,
                height,
                width,
                zero_width,
                item_ct1,
                blockvec.get_pointer());
          });
    });

}

void vecquant3matmul_sycl(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  torch::Tensor g_idx
) {
  int batch = vec.size(0);
  int vec_height = vec.size(1);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "call_vecquant3matmul_sycl", ([&] {
      call_vecquant3matmul_sycl<scalar_t>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(), g_idx.data<int>(),
        batch, vec_height, height, width, zero_width
      );
    })
  );
}


template <typename scalar_t>

void VecQuant4MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const       int* __restrict__ zeros,
    const   	int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
	  int zero_width,
	  const sycl::nd_item<3> &item_ct1,
	  scalar_t *blockvec) {
  
  int h = BLOCKHEIGHT4 * item_ct1.get_group(2);
  int w = BLOCKWIDTH * item_ct1.get_group(1) + item_ct1.get_local_id(2);

  
  int i = width * h + w;
  int g_h = h * 8;
  int k;
  unsigned int g;
  scalar_t w_tmp;


  int z_w = w / 8;
  int z_mod = (w % 8) * 4;

  float weight[BLOCKWIDTH];

  for (k = 0; k <  BLOCKWIDTH; ++k){
	int k_w = (k / 8);
	int k_bit = (k % 8) * 4;

    g = as_int(g_idx[g_h + k]);
    scalar_t scale = scales[g * width + w];
    scalar_t zero = scalar_t((((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xF) + 1) & 0x0f);

    w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xF);

	weight[k] = scale * (w_tmp - zero);
  }

  scalar_t res;
  for (int b = 0; b < batch; ++b){
	res = 0;

    blockvec[item_ct1.get_local_id(2)] = vec[b * vec_height + item_ct1.get_group(2) * BLOCKWIDTH + item_ct1.get_local_id(2)];
    
    item_ct1.barrier(sycl::access::fence_space::local_space);
	for (k = 0; k <  BLOCKWIDTH; ++k){
	  res += weight[k] * blockvec[k];
    }
    
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&mul[b * width + w], res);
    
    item_ct1.barrier(sycl::access::fence_space::local_space);
  }
}


template <typename scalar_t>
void call_vecquant4matmul_sycl(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  		int* __restrict__ zeros,
    const   	int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
	 int zero_width
){


   using sycl_t = gptq::xpu::SyclTypeTrait<scalar_t>::Type;
   sycl::range<3> blocks(1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH, (height + BLOCKHEIGHT2 - 1) / BLOCKHEIGHT2);
   sycl::range<3> threads(1, 1, BLOCKWIDTH); //grid
   auto& q_ct1 = gptq::xpu::gptqGetQueue();
   q_ct1.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<sycl_t, 1> blockvec(sycl::range<1>(BLOCKWIDTH), cgh);
      cgh.parallel_for(
          sycl::nd_range<3>(blocks * threads, threads),
          [=](sycl::nd_item<3> item_ct1) {
            VecQuant4MatMulKernel<sycl_t>(
                (const sycl_t* __restrict__)vec,
                (const int* __restrict__)mat,
                (sycl_t* __restrict__)mul,
                (const sycl_t* __restrict__)scales,
                (const int* __restrict__)zeros,
                (const int* __restrict__)g_idx,
                batch,
                vec_height,
                height,
                width,
                zero_width,
                item_ct1,
                blockvec.get_pointer());
          });
    });

}

void vecquant4matmul_sycl(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  torch::Tensor g_idx
) {
  int batch = vec.size(0);
  int vec_height = vec.size(1);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "call_vecquant4matmul_sycl", ([&] {
      call_vecquant4matmul_sycl<scalar_t>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(), g_idx.data<int>(),
        batch, vec_height, height, width, zero_width
      );
    })
  );
}


template <typename scalar_t>

void VecQuant8MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const       int* __restrict__ zeros,
    const   	int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
	  int zero_width,
	  const sycl::nd_item<3> &item_ct1,
	  scalar_t *blockvec) {
     
  int h = BLOCKHEIGHT8 * item_ct1.get_group(2);
  int w = BLOCKWIDTH * item_ct1.get_group(1) + item_ct1.get_local_id(2);

  
  int i = width * h + w;
  int g_h = h * 4;
  int k;
  unsigned int g;
  scalar_t w_tmp;

  int z_w = w / 4;
  int z_mod = (w % 4) * 8;

  float weight[BLOCKWIDTH];

  for (k = 0; k <  BLOCKWIDTH; ++k){
	int k_w = (k / 4);
	int k_bit = (k % 4) * 8;

    g = as_int(g_idx[g_h + k]);
    scalar_t scale = scales[g * width + w];
    scalar_t zero = scalar_t((((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xFF) + 1) & 0x0f);

    w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xFF);

	weight[k] = scale * (w_tmp - zero);
  }

  scalar_t res;
  for (int b = 0; b < batch; ++b){
	res = 0;

    blockvec[item_ct1.get_local_id(2)] = vec[b * vec_height + item_ct1.get_group(2) * BLOCKWIDTH + item_ct1.get_local_id(2)];
    
    item_ct1.barrier(sycl::access::fence_space::local_space);
	for (k = 0; k <  BLOCKWIDTH; ++k){
	  res += weight[k] * blockvec[k];
    }
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&mul[b * width + w], res);
    
    item_ct1.barrier(sycl::access::fence_space::local_space);
  }
}

template <typename scalar_t>
void call_vecquant8matmul_sycl(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  		int* __restrict__ zeros,
    const   	int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
	 int zero_width
){


   using sycl_t = gptq::xpu::SyclTypeTrait<scalar_t>::Type;
   sycl::range<3> blocks(1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH, (height + BLOCKHEIGHT2 - 1) / BLOCKHEIGHT2);
   sycl::range<3> threads(1, 1, BLOCKWIDTH); //grid
   auto& q_ct1 = gptq::xpu::gptqGetQueue();
   q_ct1.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<sycl_t, 1> blockvec(sycl::range<1>(BLOCKWIDTH), cgh);
      cgh.parallel_for(
          sycl::nd_range<3>(blocks * threads, threads),
          [=](sycl::nd_item<3> item_ct1) {
            VecQuant8MatMulKernel<sycl_t>(
                (const sycl_t* __restrict__)vec,
                (const int* __restrict__)mat,
                (sycl_t* __restrict__)mul,
                (const sycl_t* __restrict__)scales,
                (const int* __restrict__)zeros,
                (const int* __restrict__)g_idx,
                batch,
                vec_height,
                height,
                width,
                zero_width,
                item_ct1,
                blockvec.get_pointer());
          });
    });

}

void vecquant8matmul_sycl(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  torch::Tensor g_idx
) {
  int batch = vec.size(0);
  int vec_height = vec.size(1);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "call_vecquant8matmul_sycl", ([&] {
      call_vecquant8matmul_sycl<scalar_t>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(), g_idx.data<int>(),
        batch, vec_height, height, width, zero_width
      );
    })
  );
}



template <typename scalar_t>
void VecQuant2MatMulKernel_old(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  	int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize,
    const sycl::nd_item<3> &item_ct1,
    scalar_t *blockvec) {
  
  int b = item_ct1.get_group(0);
  int h = BLOCKHEIGHT2 * item_ct1.get_group(2);
  int w = BLOCKWIDTH * item_ct1.get_group(1) + item_ct1.get_local_id(2);

  
  blockvec[item_ct1.get_local_id(2)] = vec[b * vec_height + item_ct1.get_group(2) * BLOCKWIDTH + item_ct1.get_local_id(2)];
  
  item_ct1.barrier(sycl::access::fence_space::local_space);

  scalar_t res = 0;
  int i = width * h + w;
  int g_h = h * 16;
  int k = 0;

  int z_w = w / 16;
  int z_mod = (w % 16) * 2;

  unsigned int tmp;

  while (k < BLOCKWIDTH) {
    tmp = as_unsigned(mat[i]);

    int g = (g_h + k) / groupsize;
    scalar_t scale = scales[g * width + w];
    scalar_t zero = scale * scalar_t(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod & 0x3) + 1) & 0x0f);

    res += (scale * scalar_t((tmp >> 0) & 0x3) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp >> 2) & 0x3) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp >> 4) & 0x3) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp >> 6) & 0x3) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp >> 8) & 0x3) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp >> 10) & 0x3) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp >> 12) & 0x3) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp >> 14) & 0x3) - zero) * blockvec[k + 7];
    res += (scale * scalar_t((tmp >> 16) & 0x3) - zero) * blockvec[k + 8];
    res += (scale * scalar_t((tmp >> 18) & 0x3) - zero) * blockvec[k + 9];
    res += (scale * scalar_t((tmp >> 20) & 0x3) - zero) * blockvec[k + 10];
    res += (scale * scalar_t((tmp >> 22) & 0x3) - zero) * blockvec[k + 11];
    res += (scale * scalar_t((tmp >> 24) & 0x3) - zero) * blockvec[k + 12];
    res += (scale * scalar_t((tmp >> 26) & 0x3) - zero) * blockvec[k + 13];
    res += (scale * scalar_t((tmp >> 28) & 0x3) - zero) * blockvec[k + 14];
    res += (scale * scalar_t((tmp >> 30) & 0x3) - zero) * blockvec[k + 15];

    i += width;
    k += 16;
  }

  dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&mul[b * width + w], res);
}


template <typename scalar_t>
void call_vecquant2matmul_sycl_old(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  		int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
	 int zero_width,
    int groupsize
){


   using sycl_t = gptq::xpu::SyclTypeTrait<scalar_t>::Type;
   sycl::range<3> blocks(1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH, (height + BLOCKHEIGHT2 - 1) / BLOCKHEIGHT2);
   sycl::range<3> threads(1, 1, BLOCKWIDTH); //grid
   auto& q_ct1 = gptq::xpu::gptqGetQueue();
   q_ct1.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<sycl_t, 1> blockvec(sycl::range<1>(BLOCKWIDTH), cgh);
      cgh.parallel_for(
          sycl::nd_range<3>(blocks * threads, threads),
          [=](sycl::nd_item<3> item_ct1) {
            VecQuant2MatMulKernel_old<sycl_t>(
                (const sycl_t* __restrict__)vec,
                (const int* __restrict__)mat,
                (sycl_t* __restrict__)mul,
                (const sycl_t* __restrict__)scales,
                (const int* __restrict__)zeros,
                batch,
                vec_height,
                height,
                width,
                zero_width,
                groupsize,
                item_ct1,
                blockvec.get_pointer());
          });
    });

}

void vecquant2matmul_sycl_old(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  int groupsize
) {
  int batch = vec.size(0);
  int vec_height = vec.size(1);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "call_vecquant2matmul_sycl_old", ([&] {
      call_vecquant2matmul_sycl_old<scalar_t>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(),
        batch, vec_height, height, width, zero_width, groupsize
      );
    })
  );
}


template <typename scalar_t>
void VecQuant3MatMulKernel_old(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  	int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize,
    const sycl::nd_item<3> &item_ct1,
    scalar_t *blockvec) {
  
  int b = item_ct1.get_group(0);
  int h = BLOCKHEIGHT3 * item_ct1.get_group(2);
  int w = BLOCKWIDTH * item_ct1.get_group(1) + item_ct1.get_local_id(2);

  
  blockvec[item_ct1.get_local_id(2)] = vec[b * vec_height + item_ct1.get_group(2) * BLOCKWIDTH + item_ct1.get_local_id(2)];
  
  item_ct1.barrier(sycl::access::fence_space::local_space);

  scalar_t res = 0;
  int i = width * h + w;
  int g_h = (h / 3) * 32;
  int k = 0;

  int z_w = (w / 32) * 3;
  int z_mod = w % 32;
  int z_bit;

  if (z_mod != 10){
    if (z_mod != 21){
      z_bit = z_mod;
      if (z_bit > 21){
        z_bit -= 22;
        z_bit *= 3;
        z_bit += 2;
        z_w += 2;
      } else if (z_bit > 10){
        z_bit -= 11;
        z_bit *= 3;
        z_bit += 1;
        z_w += 1;
      } else {
        z_bit *= 3;
      }
    } else {
      z_w += 1;
    }
  }

  unsigned int tmp1;
  unsigned int tmp2;
  unsigned int tmp;
  unsigned int z_tmp;

  while (k < BLOCKWIDTH) {
    tmp1 = as_unsigned(mat[i]);

    int g = (g_h + k) / groupsize;
    scalar_t scale = scales[g * width + w];
    scalar_t zero;
    if (z_mod == 10) {
      z_tmp = (as_unsigned(zeros[g * zero_width + z_w]) >> 30) | ((as_unsigned(zeros[g * zero_width + (z_w + 1)]) << 2) & 0x4);
      zero = scale * scalar_t(((z_tmp) + 1) & 0x0f);
    } else if (z_mod == 21){
      z_tmp = (as_unsigned(zeros[g * zero_width + z_w]) >> 31) | ((as_unsigned(zeros[g * zero_width + (z_w + 1)]) << 1) & 0x6);
      zero = scale * scalar_t(((z_tmp) + 1) & 0x0f);
    } else {
      zero = scale * scalar_t((((as_unsigned(zeros[g * zero_width + z_w]) >> z_bit) & 0x7) + 1) & 0x0f);
    }

    res += (scale * scalar_t((tmp1 >>  0) & 0x7) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp1 >>  3) & 0x7) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp1 >>  6) & 0x7) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp1 >>  9) & 0x7) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp1 >> 12) & 0x7) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp1 >> 15) & 0x7) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp1 >> 18) & 0x7) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp1 >> 21) & 0x7) - zero) * blockvec[k + 7];
    res += (scale * scalar_t((tmp1 >> 24) & 0x7) - zero) * blockvec[k + 8];
    res += (scale * scalar_t((tmp1 >> 27) & 0x7) - zero) * blockvec[k + 9];

    i += width;
    tmp2 = as_unsigned(mat[i]);
    tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x4);
    tmp2 >>= 1;
    res += (scale * scalar_t(tmp) - zero) * blockvec[k + 10];
    k += 11;

    res += (scale * scalar_t((tmp2 >>  0) & 0x7) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp2 >>  3) & 0x7) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp2 >>  6) & 0x7) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp2 >>  9) & 0x7) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp2 >> 12) & 0x7) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp2 >> 15) & 0x7) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp2 >> 18) & 0x7) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp2 >> 21) & 0x7) - zero) * blockvec[k + 7];
    res += (scale * scalar_t((tmp2 >> 24) & 0x7) - zero) * blockvec[k + 8];
    res += (scale * scalar_t((tmp2 >> 27) & 0x7) - zero) * blockvec[k + 9];

    i += width;
    tmp1 = as_unsigned(mat[i]);
    tmp = (tmp2 >> 30) | ((tmp1 << 1) & 0x6);
    tmp1 >>= 2;
    res += (scale * scalar_t(tmp) - zero) * blockvec[k + 10];
    k += 11;

    res += (scale * scalar_t((tmp1 >>  0) & 0x7) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp1 >>  3) & 0x7) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp1 >>  6) & 0x7) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp1 >>  9) & 0x7) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp1 >> 12) & 0x7) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp1 >> 15) & 0x7) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp1 >> 18) & 0x7) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp1 >> 21) & 0x7) - zero) * blockvec[k + 7];
    res += (scale * scalar_t((tmp1 >> 24) & 0x7) - zero) * blockvec[k + 8];
    res += (scale * scalar_t((tmp1 >> 27) & 0x7) - zero) * blockvec[k + 9];

    i += width;
    k += 10;
  }

  dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&mul[b * width + w], res);
}


template <typename scalar_t>
void call_vecquant3matmul_sycl_old(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  		int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
	 int zero_width,
    int groupsize
){


   using sycl_t = gptq::xpu::SyclTypeTrait<scalar_t>::Type;
   sycl::range<3> blocks(1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH, (height + BLOCKHEIGHT2 - 1) / BLOCKHEIGHT2);
   sycl::range<3> threads(1, 1, BLOCKWIDTH); //grid
   auto& q_ct1 = gptq::xpu::gptqGetQueue();
   q_ct1.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<sycl_t, 1> blockvec(sycl::range<1>(BLOCKWIDTH), cgh);
      cgh.parallel_for(
          sycl::nd_range<3>(blocks * threads, threads),
          [=](sycl::nd_item<3> item_ct1) {
            VecQuant3MatMulKernel_old<sycl_t>(
                (const sycl_t* __restrict__)vec,
                (const int* __restrict__)mat,
                (sycl_t* __restrict__)mul,
                (const sycl_t* __restrict__)scales,
                (const int* __restrict__)zeros,
                batch,
                vec_height,
                height,
                width,
                zero_width,
                groupsize,
                item_ct1,
                blockvec.get_pointer());
          });
    });

}

void vecquant3matmul_sycl_old(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  int groupsize
) {
  int batch = vec.size(0);
  int vec_height = vec.size(1);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "call_vecquant3matmul_sycl_old", ([&] {
      call_vecquant3matmul_sycl_old<scalar_t>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(),
        batch, vec_height, height, width, zero_width, groupsize
      );
    })
  );
}



template <typename scalar_t>
void VecQuant4MatMulKernel_old(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const       int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize,
    const sycl::nd_item<3> &item_ct1,
    scalar_t *blockvec) {
  
  int b = item_ct1.get_group(0);
  int h = BLOCKHEIGHT4 * item_ct1.get_group(2);
  int w = BLOCKWIDTH * item_ct1.get_group(1) + item_ct1.get_local_id(2);

  
  blockvec[item_ct1.get_local_id(2)] = vec[b * vec_height + item_ct1.get_group(2) * BLOCKWIDTH + item_ct1.get_local_id(2)];
  
  item_ct1.barrier(sycl::access::fence_space::local_space);

  scalar_t res = 0;
  int i = width * h + w;
  int g_h = h * 8;
  int k = 0;

  int z_w = w / 8;
  int z_mod = (w % 8) * 4;

  unsigned int tmp;

  while (k < BLOCKWIDTH) {
    tmp = as_unsigned(mat[i]);

    int g = (g_h + k) / groupsize;
    scalar_t scale = scales[g * width + w];
    scalar_t zero = scale * scalar_t((((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xF) + 1) & 0x0f);

    res += (scale * scalar_t((tmp >> 0) & 0xF) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp >> 4) & 0xF) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp >> 8) & 0xF) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp >> 12) & 0xF) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp >> 16) & 0xF) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp >> 20) & 0xF) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp >> 24) & 0xF) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp >> 28) & 0xF) - zero) * blockvec[k + 7];

    i += width;
    k += 8;
  }

  dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&mul[b * width + w], res);
}


template <typename scalar_t>
void call_vecquant4matmul_sycl_old(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  		int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
	 int zero_width,
    int groupsize
){


   using sycl_t = gptq::xpu::SyclTypeTrait<scalar_t>::Type;
   sycl::range<3> blocks(1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH, (height + BLOCKHEIGHT2 - 1) / BLOCKHEIGHT2);
   sycl::range<3> threads(1, 1, BLOCKWIDTH); //grid
   auto& q_ct1 = gptq::xpu::gptqGetQueue();
   q_ct1.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<sycl_t, 1> blockvec(sycl::range<1>(BLOCKWIDTH), cgh);
      cgh.parallel_for(
          sycl::nd_range<3>(blocks * threads, threads),
          [=](sycl::nd_item<3> item_ct1) {
            VecQuant4MatMulKernel_old<sycl_t>(
                (const sycl_t* __restrict__)vec,
                (const int* __restrict__)mat,
                (sycl_t* __restrict__)mul,
                (const sycl_t* __restrict__)scales,
                (const int* __restrict__)zeros,
                batch,
                vec_height,
                height,
                width,
                zero_width,
                groupsize,
                item_ct1,
                blockvec.get_pointer());
          });
    });

}

void vecquant4matmul_sycl_old(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  int groupsize
) {
  int batch = vec.size(0);
  int vec_height = vec.size(1);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  
  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant4matmul_sycl_old", ([&] {
      call_vecquant4matmul_sycl_old<scalar_t>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(),
        batch, vec_height, height, width, zero_width, groupsize
      );
    })
  );
}


template <typename scalar_t>
void VecQuant8MatMulKernel_old(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  	int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize,
    const sycl::nd_item<3> &item_ct1,
    scalar_t *blockvec) {
  int b = item_ct1.get_group(0);
  int h = BLOCKHEIGHT8 * item_ct1.get_group(2);
  int w = BLOCKWIDTH * item_ct1.get_group(1) + item_ct1.get_local_id(2);

  
  blockvec[item_ct1.get_local_id(2)] = vec[b * vec_height + item_ct1.get_group(2) * BLOCKWIDTH + item_ct1.get_local_id(2)];
  
  item_ct1.barrier(sycl::access::fence_space::local_space);

  scalar_t res = 0;
  int i = width * h + w;
  int g_h = h * 4;
  int k = 0;

  int z_w = w / 4;
  int z_mod = (w % 4) * 8;

  unsigned int tmp;

  while (k < BLOCKWIDTH) {
    tmp = as_unsigned(mat[i]);

    int g = (g_h + k) / groupsize;
    scalar_t scale = scales[g * width + w];
    scalar_t zero = scale * scalar_t((((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xFF) + 1) & 0x0f);

    res += (scale * scalar_t((tmp >> 0) & 0xFF) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp >> 8) & 0xFF) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp >> 16) & 0xFF) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp >> 24) & 0xFF) - zero) * blockvec[k + 3];

    i += width;
    k += 4;
  }

  dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&mul[b * width + w], res);
}



template <typename scalar_t>
void call_vecquant8matmul_sycl_old(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  		int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
	 int zero_width,
    int groupsize
){


   using sycl_t = gptq::xpu::SyclTypeTrait<scalar_t>::Type;
   sycl::range<3> blocks(1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH, (height + BLOCKHEIGHT2 - 1) / BLOCKHEIGHT2);
   sycl::range<3> threads(1, 1, BLOCKWIDTH); //grid
   auto& q_ct1 = gptq::xpu::gptqGetQueue();
   q_ct1.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<sycl_t, 1> blockvec(sycl::range<1>(BLOCKWIDTH), cgh);
      cgh.parallel_for(
          sycl::nd_range<3>(blocks * threads, threads),
          [=](sycl::nd_item<3> item_ct1) {
            VecQuant8MatMulKernel_old<sycl_t>(
                (const sycl_t* __restrict__)vec,
                (const int* __restrict__)mat,
                (sycl_t* __restrict__)mul,
                (const sycl_t* __restrict__)scales,
                (const int* __restrict__)zeros,
                batch,
                vec_height,
                height,
                width,
                zero_width,
                groupsize,
                item_ct1,
                blockvec.get_pointer());
          });
    });

}

void vecquant8matmul_sycl_old(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  int groupsize
) {
  int batch = vec.size(0);
  int vec_height = vec.size(1);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  
  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant8matmul_sycl_old", ([&] {
      call_vecquant8matmul_sycl_old<scalar_t>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(),
        batch, vec_height, height, width, zero_width, groupsize
      );
    })
  );
}

void VecQuant2MatMulKernelFaster_old(
    const  sycl::half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const  	 int* __restrict__ zeros,
	  int batch,
	  int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize,
    const sycl::nd_item<3> &item_ct1,
    sycl::half2 *blockvec,
    sycl::local_accessor<sycl::half2, 2> deq2) {
  
  const int blockwidth2 = BLOCKWIDTH / 2;
  int b = item_ct1.get_group(0);
  int h = BLOCKHEIGHT2 * item_ct1.get_group(2);
  int w = BLOCKWIDTH * item_ct1.get_group(1) + item_ct1.get_local_id(2);

  
  if (item_ct1.get_local_id(2) < blockwidth2)
    blockvec[item_ct1.get_local_id(2)] = vec[b * vec_height + item_ct1.get_group(2) * blockwidth2 + item_ct1.get_local_id(2)];

  
  int val = item_ct1.get_local_id(2) / 16;
  int off = item_ct1.get_local_id(2) % 16;
  for (; val < 16; val += BLOCKWIDTH / 16) {
    deq2[val][off] = sycl::half2(sycl::vec<int, 1>(val & 0x3).convert<sycl::half, sycl::rounding_mode::rte>()[0], sycl::vec<int, 1>(val >> 2).convert<sycl::half, sycl::rounding_mode::rte>()[0]);
  }

  int i = width * h + w;
  int g_h = h * 16;
  int k = 0;

  int z_w = w / 16;
  int z_mod = (w % 16) * 2;

  float res = 0;
  sycl::half2 res2;

  unsigned int tmp;

  
  item_ct1.barrier(sycl::access::fence_space::local_space);

  while (k < blockwidth2) {
    int g = (g_h + (k * 2)) / groupsize;
	float scale_f = scales[g * width + w];
    sycl::half2 scale = sycl::float2(scale_f).convert<sycl::half, sycl::rounding_mode::rte>();
    sycl::half2 zero = sycl::float2(-(scale_f * ((((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0x3) + 1) & 0x0f))).convert<sycl::half, sycl::rounding_mode::rte>();

    std::memset(&res2, 0, sizeof(sycl::half2));
    tmp = as_unsigned(mat[i]);
    res2 = sycl::fma(sycl::fma(deq2[(tmp >>  0) & 0xf][off], scale, zero), blockvec[k + 0], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp >>  4) & 0xf][off], scale, zero), blockvec[k + 1], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp >>  8) & 0xf][off], scale, zero), blockvec[k + 2], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp >> 12) & 0xf][off], scale, zero), blockvec[k + 3], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp >> 16) & 0xf][off], scale, zero), blockvec[k + 4], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp >> 20) & 0xf][off], scale, zero), blockvec[k + 5], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp >> 24) & 0xf][off], scale, zero), blockvec[k + 6], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp >> 28) & 0xf][off], scale, zero), blockvec[k + 7], res2);
	i += width;
    k += 8;
    res += res2[0] + res2[1];
  }

  dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&mul[b * width + w], res);
}

template <typename scalar_t>
void vecquant2matmul_faster_cuda_old(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  int groupsize,
  int vec_height
) {
  int batch = vec.size(0);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  
  
  using sycl_t = gptq::xpu::SyclTypeTrait<scalar_t>::Type;
  sycl::range<3> blocks(1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH, (height + BLOCKHEIGHT2 - 1) / BLOCKHEIGHT2);
  sycl::range<3> threads(1, 1, BLOCKWIDTH); //grid
  auto& q_ct1 = gptq::xpu::gptqGetQueue();
  q_ct1.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<sycl_t, 1> blockvec(sycl::range<1>(BLOCKWIDTH), cgh);
      sycl::local_accessor<sycl::half2, 2> deq2(sycl::range<2>(16, 16), cgh);

      sycl::half2 * vec_data_ptr_ct0 = (sycl::half2*) vec.data_ptr();
      const int *__restrict mat_data_ptr_int_ct1 = mat.data_ptr<int>();
      float *__restrict mul_data_ptr_float_ct2 = mul.data_ptr<float>();
      const float *__restrict scales_data_ptr_float_ct3 = scales.data_ptr<float>();
      const int *__restrict zeros_data_ptr_int_ct4 = zeros.data_ptr<int>();

      cgh.parallel_for(
          sycl::nd_range<3>(blocks * threads, threads),
          [=](sycl::nd_item<3> item_ct1) {
          VecQuant2MatMulKernelFaster_old(vec_data_ptr_ct0, 
                mat_data_ptr_int_ct1, mul_data_ptr_float_ct2,
                scales_data_ptr_float_ct3, zeros_data_ptr_int_ct4, 
                batch, vec_height, height, width, zero_width, groupsize, 
                item_ct1, blockvec.get_pointer(), deq2);
          });
      });
  
}





void VecQuant3MatMulKernelFaster_old(
    const  sycl::half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const  	 int* __restrict__ zeros,
	  int batch,
	  int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize,
    const sycl::nd_item<3> &item_ct1,
    sycl::half2 *blockvec,
    sycl::local_accessor<sycl::half2, 2> deq2) {
  const int blockwidth2 = BLOCKWIDTH / 2;
  int b = item_ct1.get_group(0);
  int h = BLOCKHEIGHT3 * item_ct1.get_group(2);
  int w = BLOCKWIDTH * item_ct1.get_group(1) + item_ct1.get_local_id(2);

  
  if (item_ct1.get_local_id(2) < blockwidth2)
    blockvec[item_ct1.get_local_id(2)] = vec[b * vec_height + item_ct1.get_group(2) * blockwidth2 + item_ct1.get_local_id(2)];

  
  int val = item_ct1.get_local_id(2) / 32;
  int off = item_ct1.get_local_id(2) % 32;
  for (; val < 64; val += BLOCKWIDTH / 32) {
    deq2[val][off] = sycl::half2(sycl::vec<int, 1>(val & 0x7).convert<sycl::half, sycl::rounding_mode::rte>()[0], sycl::vec<int, 1>(val >> 3).convert<sycl::half, sycl::rounding_mode::rte>()[0]);
  }

  int i = width * h + w;
  int g_h = (h / 3) * 32;
  int k = 0;

  int z_w = (w / 32) * 3;
  int z_mod = w % 32;
  int z_bit;

  if (z_mod != 10){
    if (z_mod != 21){
      z_bit = z_mod;
      if (z_bit > 21){
        z_bit -= 22;
        z_bit *= 3;
        z_bit += 2;
        z_w += 2;
      } else if (z_bit > 10){
        z_bit -= 11;
        z_bit *= 3;
        z_bit += 1;
        z_w += 1;
      } else {
        z_bit *= 3;
      }
    } else {
      z_w += 1;
    }
  }

  float res = 0;
  sycl::half2 res2;

  unsigned int tmp1;
  unsigned int tmp2;
  unsigned int tmp;
  unsigned int z_tmp;

  
  item_ct1.barrier();

  while (k < blockwidth2) {
    int g = (g_h + (k * 2)) / groupsize;
	float scale_f = scales[g * width + w];
    sycl::half2 scale = sycl::float2(scale_f).convert<sycl::half, sycl::rounding_mode::rte>();
    sycl::half2 zero;
    if (z_mod == 10) {
      z_tmp = (as_unsigned(zeros[g * zero_width + z_w]) >> 30) | ((as_unsigned(zeros[g * zero_width + (z_w + 1)]) << 2) & 0x4);
      zero = sycl::float2(-(scale_f * (((z_tmp) + 1) & 0x0f))).convert<sycl::half, sycl::rounding_mode::rte>();
    } else if (z_mod == 21){
      z_tmp = (as_unsigned(zeros[g * zero_width + z_w]) >> 31) | ((as_unsigned(zeros[g * zero_width + (z_w + 1)]) << 1) & 0x6);
      zero = sycl::float2(-(scale_f * (((z_tmp) + 1) & 0x0f))).convert<sycl::half, sycl::rounding_mode::rte>();
    } else {
      zero = sycl::float2(-(scale_f * ((((as_unsigned(zeros[g * zero_width + z_w]) >> z_bit) & 0x7) + 1) & 0x0f))).convert<sycl::half, sycl::rounding_mode::rte>();
    }

    std::memset(&res2, 0, sizeof(sycl::half2));
    tmp1 = as_unsigned(mat[i]);
    res2 = sycl::fma(sycl::fma(deq2[(tmp1 >>  0) & 0x3f][off], scale, zero), blockvec[k + 0], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp1 >>  6) & 0x3f][off], scale, zero), blockvec[k + 1], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp1 >> 12) & 0x3f][off], scale, zero), blockvec[k + 2], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp1 >> 18) & 0x3f][off], scale, zero), blockvec[k + 3], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp1 >> 24) & 0x3f][off], scale, zero), blockvec[k + 4], res2);
    i += width;
    tmp2 = as_unsigned(mat[i]);
    tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x3c);
    res2 = sycl::fma(sycl::fma(deq2[tmp][off], scale, zero), blockvec[k + 5], res2);
    tmp2 >>= 4;
    k += 6;
    res2 = sycl::fma(sycl::fma(deq2[(tmp2 >>  0) & 0x3f][off], scale, zero), blockvec[k + 0], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp2 >>  6) & 0x3f][off], scale, zero), blockvec[k + 1], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp2 >> 12) & 0x3f][off], scale, zero), blockvec[k + 2], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp2 >> 18) & 0x3f][off], scale, zero), blockvec[k + 3], res2);
    i += width;
    tmp1 = as_unsigned(mat[i]);
    tmp = (tmp2 >> 24) | ((tmp1 << 4) & 0x30);
    res2 = sycl::fma(sycl::fma(deq2[tmp][off], scale, zero), blockvec[k + 4], res2);
    tmp1 >>= 2;
    k += 5;
    res2 = sycl::fma(sycl::fma(deq2[(tmp1 >>  0) & 0x3f][off], scale, zero), blockvec[k + 0], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp1 >>  6) & 0x3f][off], scale, zero), blockvec[k + 1], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp1 >> 12) & 0x3f][off], scale, zero), blockvec[k + 2], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp1 >> 18) & 0x3f][off], scale, zero), blockvec[k + 3], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp1 >> 24) & 0x3f][off], scale, zero), blockvec[k + 4], res2);
    i += width;
    k += 5;
    res += res2[0] + res2[1];
  }

  dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&mul[b * width + w], res);
}

template <typename scalar_t>
void vecquant3matmul_faster_sycl_old(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  int groupsize,
  int vec_height
) {
  int batch = vec.size(0);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  
   using sycl_t = gptq::xpu::SyclTypeTrait<scalar_t>::Type;
   sycl::range<3> blocks(1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH, (height + BLOCKHEIGHT2 - 1) / BLOCKHEIGHT2);
   sycl::range<3> threads(1, 1, BLOCKWIDTH); //grid
   auto& q_ct1 = gptq::xpu::gptqGetQueue();
   q_ct1.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<sycl::half2, 1> blockvec(sycl::range<1>(BLOCKWIDTH), cgh);
        sycl::local_accessor<sycl::half2, 2> deq2(sycl::range<2>(64, 32), cgh);

        sycl::half2 * vec_data_ptr_ct0 = (sycl::half2*) vec.data_ptr();
        const int *__restrict mat_data_ptr_int_ct1 = mat.data_ptr<int>();
        float *__restrict mul_data_ptr_float_ct2 = mul.data_ptr<float>();
        const float *__restrict scales_data_ptr_float_ct3 = scales.data_ptr<float>();
        const int *__restrict zeros_data_ptr_int_ct4 = zeros.data_ptr<int>();

        cgh.parallel_for(
          sycl::nd_range<3>(blocks * threads, threads), 
          [=](sycl::nd_item<3> item_ct1) {
            VecQuant3MatMulKernelFaster_old(vec_data_ptr_ct0, mat_data_ptr_int_ct1, 
            mul_data_ptr_float_ct2, scales_data_ptr_float_ct3, 
            zeros_data_ptr_int_ct4, batch, vec_height, height, width, zero_width, 
            groupsize, item_ct1, blockvec.get_pointer(), deq2);
          });
      });
  
}


void VecQuant4MatMulKernelFaster_old(
    const  sycl::half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const  	 int* __restrict__ zeros,
	 int batch,
	 int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize,
    const sycl::nd_item<3> &item_ct1,
    sycl::half2 *blockvec,
    sycl::local_accessor<sycl::half2, 2> deq2) {
    
  const int blockwidth2 = BLOCKWIDTH / 2;
  int b = item_ct1.get_group(0);
  int h = BLOCKHEIGHT4 * item_ct1.get_group(2);
  int w = BLOCKWIDTH * item_ct1.get_group(1) + item_ct1.get_local_id(2);

  
  if (item_ct1.get_local_id(2) < blockwidth2)
    blockvec[item_ct1.get_local_id(2)] = vec[b * vec_height + item_ct1.get_group(2) * blockwidth2 + item_ct1.get_local_id(2)];

  
  int val = item_ct1.get_local_id(2) / 8;
  int off = item_ct1.get_local_id(2) % 8;
  for (; val < 256; val += BLOCKWIDTH / 8) {
    deq2[val][off] = sycl::half2(sycl::vec<int, 1>(val & 0xF).convert<sycl::half, sycl::rounding_mode::rte>()[0], sycl::vec<int, 1>(val >> 4).convert<sycl::half, sycl::rounding_mode::rte>()[0]);
  }

  int i = width * h + w;
  int g_h = h * 8;
  int k = 0;

  int z_w = w / 8;
  int z_mod = (w % 8) * 4;

  float res = 0;
  sycl::half2 res2;

  unsigned int tmp;

  
  item_ct1.barrier(sycl::access::fence_space::local_space);

  while (k < blockwidth2) {
    int g = (g_h + (k * 2)) / groupsize;
	float scale_f = scales[g * width + w];

    sycl::half2 scale = sycl::float2(scale_f).convert<sycl::half, sycl::rounding_mode::rte>();
    sycl::half2 zero = sycl::float2(-(scale_f * ((((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xF) + 1) & 0x0f))).convert<sycl::half, sycl::rounding_mode::rte>();

    //std::memset(&res2, 0, sizeof(half2));

    //res2 = __float2half2_rn((float)0.);

    std::memset(&res2, 0, sizeof(sycl::half2));
    tmp = as_unsigned(mat[i]);
    res2 = sycl::fma(sycl::fma(deq2[(tmp >>  0) & 0xff][off], scale, zero), blockvec[k + 0], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp >>  8) & 0xff][off], scale, zero), blockvec[k + 1], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp >> 16) & 0xff][off], scale, zero), blockvec[k + 2], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp >> 24) & 0xff][off], scale, zero), blockvec[k + 3], res2);
	i += width;
    k += 4;

    res += res2[0] + res2[1];

  }

  dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&mul[b * width + w], res);
}


template <typename scalar_t>
void vecquant4matmul_faster_sycl_old(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  int groupsize,
  int vec_height
) {
  int batch = vec.size(0);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  
   using sycl_t = gptq::xpu::SyclTypeTrait<scalar_t>::Type;
   sycl::range<3> blocks(1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH, (height + BLOCKHEIGHT2 - 1) / BLOCKHEIGHT2);
   sycl::range<3> threads(1, 1, BLOCKWIDTH); //grid
   auto& q_ct1 = gptq::xpu::gptqGetQueue();
   q_ct1.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<sycl::half2, 1> blockvec(sycl::range<1>(BLOCKWIDTH), cgh);
        sycl::local_accessor<sycl::half2, 2> deq2(sycl::range<2>(256, 8), cgh);

        sycl::half2 * vec_data_ptr_ct0 = (sycl::half2*) vec.data_ptr();
        const int *__restrict mat_data_ptr_int_ct1 = mat.data_ptr<int>();
        float *__restrict mul_data_ptr_float_ct2 = mul.data_ptr<float>();
        const float *__restrict scales_data_ptr_float_ct3 = scales.data_ptr<float>();
        const int *__restrict zeros_data_ptr_int_ct4 = zeros.data_ptr<int>();

        cgh.parallel_for(
          sycl::nd_range<3>(blocks * threads, threads), 
          [=](sycl::nd_item<3> item_ct1) {
            VecQuant4MatMulKernelFaster_old(vec_data_ptr_ct0, mat_data_ptr_int_ct1, 
            mul_data_ptr_float_ct2, scales_data_ptr_float_ct3, zeros_data_ptr_int_ct4, 
            batch, vec_height, height, width, zero_width, 
            groupsize, item_ct1, blockvec.get_pointer(), deq2);
          });
      });
  
}

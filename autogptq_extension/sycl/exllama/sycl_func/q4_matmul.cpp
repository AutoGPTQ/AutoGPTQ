// Adapted from turboderp exllama: https://github.com/turboderp/exllama

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "q4_matmul.hpp"
#include "column_remap.hpp"
#include "../util.h"
#include "../matrix.h"
#include "../sycl_compat.hpp"
#include "../sycl_buffers.hpp"
#include <dpct/blas_utils.hpp>
#include <dpct/lib_common_utils.hpp>
#include "../../sycl_utils.h"


const int THREADS_X = 32;       // Block size and thread count along columns in w and out
const int THREADS_Y = 1;        // Block size and thread count along rows in x and out

typedef void (*fp_q4_matmul_kernel)
(
    const sycl::half*,
    const uint32_t*,
    sycl::half*,
    const sycl::half*,
    const uint32_t*,
    const int,
    const int,
    const int,
    const int,
    const int,
    const uint32_t*,
    bool,
    const sycl::nd_item<3> &
);

template<bool use_half2, bool use_groupsize, bool use_x_map>
/*
DPCT1110:1: The total declared local variable size in device function q4_matmul_kernel exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total register size available and adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void q4_matmul_kernel
(
    const sycl::half* __restrict__ x,
    const uint32_t* __restrict__ w,
    sycl::half* __restrict__ out,
    const sycl::half* __restrict__ w_scales,
    const uint32_t* __restrict__ w_zeros,
    const int height,
    const int dim,
    const int width,
    const int groupsize,
    const int block_size_z,
    const uint32_t* __restrict__ x_map,
    bool no_zero,
    const sycl::nd_item<3> &item_ct1
)
{
    // Start of block

    //auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    int x_column = block_size_z * item_ct1.get_group(0);
    int x_column_end = fmin(dim, (int)(block_size_z * (item_ct1.get_group(0) + 1)));

    int w_column = THREADS_X * item_ct1.get_group(2) + item_ct1.get_local_id(2);
    int x_row = THREADS_Y * item_ct1.get_group(1) + item_ct1.get_local_id(1);

    int iterations = (x_column_end - x_column) / 8;

    // Views

    MatrixView_half x_(x, height, dim);
    MatrixView_half w_scales_(w_scales, dim / groupsize, width);
    MatrixView_q4_row w_zeros_(w_zeros, dim / groupsize, width);
    MatrixView_q4_column w_(w, dim, width);
    MatrixView_half_rw out_(out, height, width);

    // Zero output

    if (!no_zero && item_ct1.get_group(0) == 0 && (item_ct1.get_local_id(2) & 1) == 0)
    {
        *((uint32_t*) out_.item_ptr(x_row, w_column)) = 0;
        /*
        DPCT1118:2: SYCL group functions and algorithms must be encountered in converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    // Loop over part of x row (and w column)

    sycl::half2 acc = {};
    sycl::half acc_h = {};

    if constexpr (use_groupsize)
    {
        // For quant matrices where groupsize divides BLOCK_SIZE_Z we always start on a group boundary, so this
        // could be slightly faster

        for (int k = x_column, group = x_column / groupsize; k < x_column + iterations * 8; group++, k += groupsize)
        {
            if constexpr (use_half2)
            {
                sycl::half2 w_scale = w_scales_.item_half2half2(group, w_column);
                uint32_t w_zero = (w_zeros_.item(group, w_column) + 1) & 0x0f;  // Avoid overflows.

                if constexpr (use_x_map) acc = dot_product_8_x_map(acc, x_, x_row, k, w_, k, w_column, w_scale, w_zero, groupsize / 8, x_map);
                else                     acc = dot_product_8      (acc, x_, x_row, k, w_, k, w_column, w_scale, w_zero, groupsize / 8);
            }
            else
            {
                sycl::half w_scale = w_scales_.item(group, w_column);
                uint32_t w_zero = (w_zeros_.item(group, w_column) + 1) & 0x0f;  // Avoid overflows.

                if constexpr (use_x_map) acc_h = dot_product_8_x_map_h(acc_h, x_, x_row, k, w_, k, w_column, w_scale, w_zero, groupsize / 8, x_map);
                else                     acc_h = dot_product_8_h      (acc_h, x_, x_row, k, w_, k, w_column, w_scale, w_zero, groupsize / 8);
            }
        }
    }
    else
    {
        // Otherwise assume groupsize is a multiple of 8, do 8 columns per iteration and trust the cache

        for (int k = x_column; k < x_column + iterations * 8; k += 8)
        {
            if constexpr (use_half2)
            {
                int group = k / groupsize;
                sycl::half2 w_scale = w_scales_.item_half2half2(group, w_column);
                uint32_t w_zero = (w_zeros_.item(group, w_column) + 1) & 0x0f;  // Avoid overflows.

                if constexpr (use_x_map) acc = dot_product_8_x_map(acc, x_, x_row, k, w_, k, w_column, w_scale, w_zero, 1, x_map);
                else                     acc = dot_product_8      (acc, x_, x_row, k, w_, k, w_column, w_scale, w_zero, 1);
            }
            else
            {
                int group = k / groupsize;
                sycl::half w_scale = w_scales_.item(group, w_column);
                uint32_t w_zero = (w_zeros_.item(group, w_column) + 1) & 0x0f;  // Avoid overflows.

                if constexpr (use_x_map) acc_h = dot_product_8_x_map_h(acc_h, x_, x_row, k, w_, k, w_column, w_scale, w_zero, 1, x_map);
                else                     acc_h = dot_product_8_h      (acc_h, x_, x_row, k, w_, k, w_column, w_scale, w_zero, 1);
            }
        }
    }

    // Add to block result

    if constexpr (use_half2)
    {
        sycl::half result = acc[0] + acc[1];
        /*
        DPCT1007:6: Migration of half version of atomicAdd is not supported.
        */
        atomicAdd(out_.item_ptr(x_row, w_column), result);
        //dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(out_.item_ptr(x_row, w_column), result);
    }
    else
    {
        /*
        DPCT1007:7: Migration of half version of atomicAdd is not supported.
        */
        atomicAdd(out_.item_ptr(x_row, w_column), acc_h);
        //dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(out_.item_ptr(x_row, w_column), acc_h);
    }
}

fp_q4_matmul_kernel q4_matmul_kernel_pick(ExLlamaTuning* tuningParams, int block_size_z, int groupsize, uint32_t* x_map)
{
    // <bool use_half2, bool use_groupsize, bool use_x_map>
    if (tuningParams->matmul_no_half2) {
        if (block_size_z % groupsize == 0) {
            if (x_map) return q4_matmul_kernel<false, true,  true >;
            else       return q4_matmul_kernel<false, true,  false>;
        } else {
            if (x_map) return q4_matmul_kernel<false, false, true >;
            else       return q4_matmul_kernel<false, false, false>;
        }
    } else {
        if (block_size_z % groupsize == 0)
        {
            if (x_map) return q4_matmul_kernel<true,  true,  true >;
            else       return q4_matmul_kernel<true,  true,  false>;
        } else {
            if (x_map) return q4_matmul_kernel<true,  false, true >;
            else       return q4_matmul_kernel<true,  false, false>;
        }
    }
};

// Compute y = x @ w

void q4_matmul_sycl
(
    ExLlamaTuning* tuningParams,
    const sycl::half* x,
    const int x_height,
    const Q4Matrix* w,
    sycl::half* out,
    bool no_zero
)
{
    int height = x_height;
    int dim = w->height;
    int width = w->width;
    auto &alt_stream = gptq::xpu::gptqGetQueue();
    /*
    DPCT1093:8: The "w->device" device may be not the one intended for use. Adjust the selected device if needed.
    */
    dpct::select_device(w->device);

    uint32_t* x_map = w->sycl_x_map;
    const sycl::half* x_mapped = x;
    if (x_map && !tuningParams->matmul_fused_remap)
    {
        SyclBuffers* buffers = get_buffers(w->device);
        column_remap_sycl(x, buffers->temp_state, x_height, dim, w->sycl_x_map);
        x_mapped = buffers->temp_state;
        x_map = NULL;
    }

    int block_size_z;
    if (w->width == 4096) block_size_z = 384;           // 7B
    else if (w->width == 11008) block_size_z = 256;
    else if (w->width == 5120) block_size_z = 384;      // 13B
    else if (w->width == 13824) block_size_z = 256;
    else if (w->width == 6656) block_size_z = 256;      // 33B
    else if (w->width == 17920) block_size_z = 128;
    else block_size_z = 256;

    //if (!no_zero) cudaMemsetAsync(out, 0, x_height * w->width * sizeof(half));

    sycl::range<3> threads(1, THREADS_Y, THREADS_X);

    sycl::range<3> blocks
    ((dim + block_size_z - 1) / block_size_z, (height + threads[1] - 1) / threads[1], (width + threads[2] - 1) / threads[2]);

    fp_q4_matmul_kernel kernel = q4_matmul_kernel_pick(tuningParams, block_size_z, w->groupsize, x_map);

    /*
    DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    alt_stream.submit(
      [&](sycl::handler &cgh) {
        const uint32_t * w_sycl_qweight_ct1 = w->sycl_qweight;
        const sycl::half * w_sycl_scales_ct3 = w->sycl_scales;
        const uint32_t * w_sycl_qzeros_ct4 = w->sycl_qzeros;
        auto w_groupsize_ct8 = w->groupsize;

        cgh.parallel_for(
          sycl::nd_range<3>(blocks * threads, threads), 
          [=](sycl::nd_item<3> item_ct1) {
           q4_matmul_kernel<true,  true,  true >(x_mapped, w_sycl_qweight_ct1, out, w_sycl_scales_ct3, w_sycl_qzeros_ct4, height, dim, width, w_groupsize_ct8, block_size_z, x_map, no_zero, item_ct1);
          });
      });
}
/*
 Disable reconstruction temporarily
void q4_matmul_recons_sycl
(
    ExLlamaTuning* tuningParams,
    const sycl::half* x,
    const int x_height,
    Q4Matrix* w,
    sycl::half* out,
    bool no_zero
)
{
    int height = x_height;
    int dim = w->height;
    int width = w->width;
    dpct::blas::descriptor_ptr handle = new dpct::blas::descriptor();
    
    dpct::select_device(w->device);
    SyclBuffers* buffers = get_buffers(w->device);

    const sycl::half* x_mapped = x;
    if (w->sycl_x_map)
    {
        TORCH_CHECK(buffers->temp_state_size >= x_height * dim, "The temp_state buffer is too small in the exllama backend for GPTQ with act-order. Please call the exllama_set_max_input_length function to increase the buffer size for a sequence length >=", x_height, ":\nfrom auto_gptq import exllama_set_max_input_length\nmodel = exllama_set_max_input_length(model, max_input_length=", x_height, ")");
        column_remap_sycl(x, buffers->temp_state, x_height, dim, w->sycl_x_map);
        x_mapped = buffers->temp_state;
    }

    w->reconstruct(buffers->temp_dq);
    handle->set_queue(alt_stream);

#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 700
    const float alpha = 1.0f;
    const float beta = no_zero ? 1.0f : 0.0f;
    dpct::blas::gemm(handle,  sycl::ext::oneapi::mkl::transpose::nontrans,   sycl::ext::oneapi::mkl::transpose::nontrans, width, height, dim, &alpha,  dpct::library_data_t::real_half, width, x_mapped,  dpct::library_data_t::real_half, dim, &beta, out,  dpct::library_data_t::real_half, width,  dpct::library_data_t::real_float);
    
    //cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, width, height, dim, &alpha, buffers->temp_dq, CUDA_R_16F, width,
    //              x_mapped, CUDA_R_16F, dim, &beta, out, CUDA_R_16F, width);
#else
    const sycl::half alpha = sycl::vec<float, 1>(1.0f).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    const sycl::half beta = no_zero ? sycl::vec<float, 1>(1.0f).convert<sycl::half, sycl::rounding_mode::automatic>()[0] : sycl::vec<float, 1>(0.0f).convert<sycl::half, sycl::rounding_mode::automatic>()[0]; 
    sycl::ext::oneapi::mkl::blas::column_major::gemm(handle->get_queue(),  sycl::ext::oneapi::mkl::transpose::nontrans,  sycl::ext::oneapi::mkl::transpose::nontrans, width, height, dim, &alpha, buffers->temp_dq, width, x_mapped, dim, &beta, out , width );
    
    
    //cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, width, height, dim, &alpha, buffers->temp_dq, width, x_mapped, dim, &beta, out, width);
#endif
}
*/
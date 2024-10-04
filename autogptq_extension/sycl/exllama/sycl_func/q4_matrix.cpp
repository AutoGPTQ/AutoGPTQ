// Adapted from turboderp exllama: https://github.com/turboderp/exllama

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "q4_matrix.hpp"
#include <vector>
#include "../util.h"
#include "../matrix.h"
#include "../../sycl_utils.h"

using namespace std;

const int UNSHUF_BLOCKSIZE_X = 64;

const int RECONS_THREADS_X = 64;      // Block size and thread count along columns in out, each thread converts 1 column
const int RECONS_THREADS_Y = 1;       // Block size and thread count along rows in x and out, each thread converts 8 rows

vector<Q4Matrix*> g_q4_matrices;

void g_q4_keep_matrix(Q4Matrix* m)
{
    g_q4_matrices.push_back(m);
}

void g_q4_free_matrices()
{
    for (const auto& m : g_q4_matrices) delete m;
    g_q4_matrices.clear();
}

Q4Matrix::Q4Matrix
(
    const int _height,
    const int _width,
    const int _groups,

    uint32_t* _qweight,
    uint32_t* _qzeros,
    sycl::half* _scales,
    uint32_t* _g_idx,

    const int _device
) :
    height(_height),
    width(_width),
    groups(_groups),
    device(_device)
{
    /*
    DPCT1093:5: The "device" device may be not the one intended for use. Adjust the selected device if needed.
    */
    dpct::select_device(device);

    sycl_qweight = _qweight;
    sycl_qzeros = _qzeros;
    sycl_scales = _scales;

    groupsize = height / groups;

    if (_g_idx) make_sequential(_g_idx);
}

Q4Matrix::~Q4Matrix()
{
} 

// Make sequential

void make_sequential_kernel
(
    const uint32_t* __restrict__ w,
    uint32_t* __restrict__ w_new,
    const uint32_t* __restrict__ x_map,
    const int w_height,
    const int w_width,
    const sycl::nd_item<3> &item_ct1
)
{
    //auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    const uint64_t* w2 = (uint64_t*) w;
    uint64_t* w_new2 = (uint64_t*) w_new;
    int w2_stride = w_width >> 1;

    int w2_column = UNSHUF_BLOCKSIZE_X * item_ct1.get_group(2) + item_ct1.get_local_id(2);
    if (w2_column >= w2_stride) return;

    int w_new2_row = item_ct1.get_group(1);

    int x_map_idx = w_new2_row << 3;

    uint64_t dst = 0;

    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
        int source_row = x_map[x_map_idx++];

        int w2_row = source_row >> 3;
        int w2_subrow = source_row & 0x07;
        int w2_row_shift = w2_subrow << 2;
        int wnew2_row_shift = i << 2;

        uint64_t src = w2[w2_row * w2_stride + w2_column];
        src >>= w2_row_shift;
        src &= 0x0000000f0000000f;
        src <<= wnew2_row_shift;
        dst |= src;
    }

    w_new2[w_new2_row * w2_stride + w2_column] = dst;
}

void Q4Matrix::make_sequential(const uint32_t* cpu_g_idx)
{
    uint32_t* sycl_new_qweight = NULL;
    auto& q_ct1 = gptq::xpu::gptqGetQueue();
    sycl_new_qweight = sycl::malloc_device<uint32_t>(height / 8 * width, q_ct1);
    sycl_x_map = sycl::malloc_device<uint32_t>(height, q_ct1);  // TODO: Should probably be allocated in PyTorch

    uint32_t* cpu_g_idx_map = (uint32_t*) calloc(groups, sizeof(uint32_t));
    uint32_t* cpu_x_map = (uint32_t*) malloc(height * sizeof(uint32_t));
    uint32_t* cpu_x_map_inv = (uint32_t*) malloc(height * sizeof(uint32_t));

    // Group histogram

    for (int i = 0; i < height; i++) cpu_g_idx_map[cpu_g_idx[i]]++;

    // Group map

    for (int i = 0, acc = 0; i < groups; i++)
    {
        short tmp = cpu_g_idx_map[i];
        cpu_g_idx_map[i] = acc;
        acc += tmp;
    }

    // X map (inverse)

    for (int row = 0; row < height; row++)
    {
        uint32_t target_group = cpu_g_idx[row];
        uint32_t target_row = cpu_g_idx_map[target_group];
        cpu_g_idx_map[target_group]++;
        cpu_x_map_inv[row] = target_row;
    }

    // X map

    for (int row = 0; row < height; row++) cpu_x_map[cpu_x_map_inv[row]] = row;

    // Move to CUDA

    q_ct1.memcpy(sycl_x_map, cpu_x_map, height * sizeof(uint32_t));

    // Rearrange rows in w

    sycl::range<3> threads(1, 1, UNSHUF_BLOCKSIZE_X);
    sycl::range<3> blocks
    (1, height / 8, (width + UNSHUF_BLOCKSIZE_X * 2 - 1) / (UNSHUF_BLOCKSIZE_X * 2));

    q_ct1.submit(
      [&](sycl::handler &cgh) {
        const uint32_t * sycl_qweight_ct0 = sycl_qweight;
        const uint32_t * sycl_x_map_ct2 = sycl_x_map;
        const int height_ct3 = height / 8;
        const int width_ct4 = width;

        cgh.parallel_for(
          sycl::nd_range<3>(blocks * threads, threads), 
          [=](sycl::nd_item<3> item_ct1) {
            make_sequential_kernel(sycl_qweight_ct0, sycl_new_qweight, sycl_x_map_ct2, height_ct3, width_ct4, item_ct1);
          });
      });

    // Replace qweights

    q_ct1.memcpy(sycl_qweight, sycl_new_qweight, height / 8 * width * sizeof(uint32_t));

    // Cleanup

    q_ct1.wait_and_throw();
    sycl::free(sycl_new_qweight, q_ct1);
    free(cpu_g_idx_map);
    free(cpu_x_map);
    free(cpu_x_map_inv);
}

void reconstruct_kernel
(
    const uint32_t* __restrict__ w,
    sycl::half* __restrict__ out,  // (y)
    const sycl::half* __restrict__ w_scales,
    const uint32_t* __restrict__ w_zeros,
    const int height,
    const int width,
    const int groupsize,
    const sycl::nd_item<3> &item_ct1
)
{
    // Start of block

    //auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    int column = RECONS_THREADS_X * item_ct1.get_group(2) + item_ct1.get_local_id(2);
    int row = (RECONS_THREADS_Y * item_ct1.get_group(1) + item_ct1.get_local_id(1)) * 8;
    if (column >= width) return;
    
    // Views

    MatrixView_q4_column w_(w, height, width);
    MatrixView_half_rw out_(out, height, width);
    MatrixView_half w_scales_(w_scales, height / groupsize, width);
    MatrixView_q4_row w_zeros_(w_zeros, height / groupsize, width);

    // Groupsize version

    int group = row / groupsize;

    sycl::half w_scale = w_scales_.item(group, column);
    uint32_t w_zero = (w_zeros_.item(group, column) + 1) & 0x0f;  // Avoid overflows.

    uint32_t w_read = w_.item_uint32_t(row, column);
    sycl::half* out_ptr = out_.item_ptr(row, column);

    #pragma unroll
    for (int s = 0; s < 32; s += 4)
    {
        sycl::half w_item = sycl::vec<int, 1>((int)((w_read >> s) & 0x0f) - w_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0] * w_scale;
        *out_ptr = w_item; out_ptr += out_.width;
    }
}

void Q4Matrix::reconstruct(sycl::half* out)
{
    auto& q_ct1 = gptq::xpu::gptqGetQueue();
    sycl::range<3> threads(1, RECONS_THREADS_Y, RECONS_THREADS_X);

    sycl::range<3> blocks
    (1, (height / 8 + threads[1] - 1) / threads[1], (width + threads[2] - 1) / threads[2]);

    /*
    DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
      //dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
        [&](sycl::handler &cgh) {
          const uint32_t * sycl_qweight_ct0 = sycl_qweight;
          const sycl::half * sycl_scales_ct2 = sycl_scales;
          const uint32_t * sycl_qzeros_ct3 = sycl_qzeros;
          const int height_ct4 = height / 8;
          const int width_ct5 = width;
          const int groupsize_ct6 = groupsize;

          cgh.parallel_for(
            sycl::nd_range<3>(blocks * threads, threads), 
            [=](sycl::nd_item<3> item_ct1) {
              reconstruct_kernel(sycl_qweight_ct0, out, sycl_scales_ct2, sycl_qzeros_ct3, height_ct4, width_ct5, groupsize_ct6, item_ct1);
            });
        });
    }
}

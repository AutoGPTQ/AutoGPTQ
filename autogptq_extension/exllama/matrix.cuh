// Adapted from turboderp exllama: https://github.com/turboderp/exllama

#ifndef _matrix_cuh
#define _matrix_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>

class MatrixView_half
{
public:
    const half* data;
    const int height;
    const int width;

    __device__ __forceinline__ MatrixView_half(const half* data, const int height, const int width)
        : data(data), height(height), width(width)
    { }

    __device__ __forceinline__ half item(int row, int column) const { return data[row * width + column]; }
    __device__ __forceinline__ half2 item_half2(int row, int column) const { return ((half2*)data)[(row * width + column) / 2]; }
    __device__ __forceinline__ half2 item_half2half2(int row, int column) const { return __half2half2(data[row * width + column]); }
    __device__ __forceinline__ const half* item_ptr(int row, int column) const { return &data[row * width + column]; }
};

class MatrixView_half_rw
{
public:
    half* data;
    const int height;
    const int width;

    __device__ __forceinline__ MatrixView_half_rw(half* data, const int height, const int width)
        : data(data), height(height), width(width)
    { }

    __device__ __forceinline__ half item(int row, int column) const { return data[row * width + column]; }
    __device__ __forceinline__ half2 item_half2(int row, int column) const { return ((half2*)data)[(row * width + column) / 2]; }
    __device__ __forceinline__ half* item_ptr(int row, int column) { return &data[row * width + column]; }
    __device__ __forceinline__ void set(int row, int column, half value) { data[row * width + column] = value; }
    __device__ __forceinline__ void set_half2(int row, int column, half2 value) { ((half2*)data)[(row * width + column) / 2] = value; }
};

class MatrixView_q4_row
{
public:
    const uint32_t* data;
    const int height;
    const int width;

    __device__ __forceinline__ MatrixView_q4_row(const uint32_t* data, const int height, const int width)
        : data(data), height(height), width(width)
    { }

    __device__ __forceinline__ int item(int row, int column) const
    {
        int shift = (column & 0x07) * 4;
        return (data[row * width / 8 + column / 8] >> shift) & 0x0f;
    }
};

class MatrixView_q4_column
{
public:
    const uint32_t* data;
    const int height;
    const int width;

    __device__ __forceinline__ MatrixView_q4_column(const uint32_t* data, const int height, const int width)
        : data(data), height(height), width(width)
    { }

    __device__ __forceinline__ int item(int row, int column) const
    {
        int shift = (row & 0x07) * 4;
        return (data[row / 8 * width + column] >> shift) & 0x0f;
    }

    __device__ __forceinline__ uint32_t item_uint32_t(int row, int column) { return data[row / 8 * width + column]; }
    __device__ __forceinline__ const uint32_t* item_uint32_ptr(int row, int column) { return &data[row / 8 * width + column]; }
};

// TODO: Rewrite all these dot product functions using functors or something, move to q4_matmul.cu

// Accumulated dot product of 8-element row vectors in h and quantized column vectors in v, constant zero/scale

__device__ __forceinline__ half2 dot_product_8
(
    const half2 acc,
    const half2* h_ptr,
    MatrixView_q4_column& v_,
    const int v_row,                    // divisible by 8
    const int v_column,
    const half2 v_scale_2,
    const uint32_t v_zero,
    const int count
)
{
    const uint32_t* v_ptr = (const uint32_t*) v_.item_uint32_ptr(v_row, v_column);
    half2 result = acc;

    for (int i = 0; i < count; i++)
    {
        uint32_t v_read = *v_ptr; v_ptr += v_.width;

        half v_0 = __int2half_rn((int)((v_read      ) & 0x0f) - v_zero);
        half v_1 = __int2half_rn((int)((v_read >>  4) & 0x0f) - v_zero);
        half v_2 = __int2half_rn((int)((v_read >>  8) & 0x0f) - v_zero);
        half v_3 = __int2half_rn((int)((v_read >> 12) & 0x0f) - v_zero);
        half v_4 = __int2half_rn((int)((v_read >> 16) & 0x0f) - v_zero);
        half v_5 = __int2half_rn((int)((v_read >> 20) & 0x0f) - v_zero);
        half v_6 = __int2half_rn((int)((v_read >> 24) & 0x0f) - v_zero);
        half v_7 = __int2half_rn((int)((v_read >> 28)       ) - v_zero);

        half2 v_01 = __halves2half2(v_0, v_1);
        half2 v_23 = __halves2half2(v_2, v_3);
        half2 v_45 = __halves2half2(v_4, v_5);
        half2 v_67 = __halves2half2(v_6, v_7);

//         half2 v_01 = q4_table[v_zero - 1][(v_read      ) & 0xff]; // (constant memory is too slow apparently)
//         half2 v_23 = q4_table[v_zero - 1][(v_read >>  8) & 0xff];
//         half2 v_45 = q4_table[v_zero - 1][(v_read >> 16) & 0xff];
//         half2 v_67 = q4_table[v_zero - 1][(v_read >> 24)       ];

        half2 tmp = __hmul2(*h_ptr++, v_01);
        tmp = __hfma2(*h_ptr++, v_23, tmp);
        tmp = __hfma2(*h_ptr++, v_45, tmp);
        tmp = __hfma2(*h_ptr++, v_67, tmp);
        result = __hfma2(v_scale_2, tmp, result);
    }

    return result;
}

__device__ __forceinline__ half dot_product_8_h
(
    const half acc,
    const half* h_ptr,
    MatrixView_q4_column& v_,
    const int v_row,                    // divisible by 8
    const int v_column,
    const half v_scale,
    const uint32_t v_zero,
    const int count
)
{
    const uint32_t* v_ptr = (const uint32_t*) v_.item_uint32_ptr(v_row, v_column);
    half result = acc;

    for (int i = 0; i < count; i++)
    {
        uint32_t v_read = *v_ptr; v_ptr += v_.width;

        half v_0 = __int2half_rn((int)((v_read      ) & 0x0f) - v_zero);
        half v_1 = __int2half_rn((int)((v_read >>  4) & 0x0f) - v_zero);
        half v_2 = __int2half_rn((int)((v_read >>  8) & 0x0f) - v_zero);
        half v_3 = __int2half_rn((int)((v_read >> 12) & 0x0f) - v_zero);
        half v_4 = __int2half_rn((int)((v_read >> 16) & 0x0f) - v_zero);
        half v_5 = __int2half_rn((int)((v_read >> 20) & 0x0f) - v_zero);
        half v_6 = __int2half_rn((int)((v_read >> 24) & 0x0f) - v_zero);
        half v_7 = __int2half_rn((int)((v_read >> 28)       ) - v_zero);

        half tmp = __hmul(*h_ptr++, v_0);
        tmp = __hfma(*h_ptr++, v_1, tmp);
        tmp = __hfma(*h_ptr++, v_2, tmp);
        tmp = __hfma(*h_ptr++, v_3, tmp);
        tmp = __hfma(*h_ptr++, v_4, tmp);
        tmp = __hfma(*h_ptr++, v_5, tmp);
        tmp = __hfma(*h_ptr++, v_6, tmp);
        tmp = __hfma(*h_ptr++, v_7, tmp);
        result = __hfma(v_scale, tmp, result);
    }

    return result;
}

#endif

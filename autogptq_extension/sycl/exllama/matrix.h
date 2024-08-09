// Adapted from turboderp exllama: https://github.com/turboderp/exllama

#ifndef _matrix_cuh
#define _matrix_cuh

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

class MatrixView_half
{
public:
    const sycl::half* data;
    const int height;
    const int width;

    __dpct_inline__ MatrixView_half(const sycl::half* data, const int height, const int width)
        : data(data), height(height), width(width)
    { }

    __dpct_inline__ sycl::half item(int row, int column) const { return data[row * width + column]; }
    __dpct_inline__ sycl::half2 item_half2(int row, int column) const { return ((sycl::half2*)data)[(row * width + column) / 2]; }
    __dpct_inline__ sycl::half2 item_half2half2(int row, int column) const { return sycl::half2(data[row * width + column]); }
    __dpct_inline__ const sycl::half* item_ptr(int row, int column) const { return &data[row * width + column]; }
};

class MatrixView_half_rw
{
public:
    sycl::half* data;
    const int height;
    const int width;

    __dpct_inline__ MatrixView_half_rw(sycl::half* data, const int height, const int width)
        : data(data), height(height), width(width)
    { }

    __dpct_inline__ sycl::half item(int row, int column) const { return data[row * width + column]; }
    __dpct_inline__ sycl::half2 item_half2(int row, int column) const { return ((sycl::half2*)data)[(row * width + column) / 2]; }
    __dpct_inline__ sycl::half* item_ptr(int row, int column) { return &data[row * width + column]; }
    __dpct_inline__ void set(int row, int column, sycl::half value) { data[row * width + column] = value; }
    __dpct_inline__ void set_half2(int row, int column, sycl::half2 value) { ((sycl::half2*)data)[(row * width + column) / 2] = value; }
};

class MatrixView_q4_row
{
public:
    const uint32_t* data;
    const int height;
    const int width;

    __dpct_inline__ MatrixView_q4_row(const uint32_t* data, const int height, const int width)
        : data(data), height(height), width(width)
    { }

    __dpct_inline__ int item(int row, int column) const
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

    __dpct_inline__ MatrixView_q4_column(const uint32_t* data, const int height, const int width)
        : data(data), height(height), width(width)
    { }

    __dpct_inline__ int item(int row, int column) const
    {
        int shift = (row & 0x07) * 4;
        return (data[row / 8 * width + column] >> shift) & 0x0f;
    }

    __dpct_inline__ uint32_t item_uint32_t(int row, int column) { return data[row / 8 * width + column]; }
    __dpct_inline__ const uint32_t* item_uint32_ptr(int row, int column) { return &data[row / 8 * width + column]; }
};

// TODO: Rewrite all these dot product functions using functors or something, move to q4_matmul.cu
// Todo: Change to dnn for matmuls

// Accumulated dot product of 8-element row vectors in h and quantized column vectors in v, constant zero/scale

__dpct_inline__ sycl::half2 dot_product_8
(
    const sycl::half2 acc,
    MatrixView_half& h_,
    const int h_row,
    const int h_column,                 // divisible by 8
    MatrixView_q4_column& v_,
    const int v_row,                    // divisible by 8
    const int v_column,
    const sycl::half2 v_scale_2,
    const uint32_t v_zero,              // + 1 (!!)
    const int count
)
{
    const sycl::half2* h_ptr = (const sycl::half2*) h_.item_ptr(h_row, h_column);
    const uint32_t* v_ptr = (const uint32_t*) v_.item_uint32_ptr(v_row, v_column);
    sycl::half2 result = acc;

    for (int i = 0; i < count; i++)
    {
        uint32_t v_read = *v_ptr; v_ptr += v_.width;

        sycl::half v_0 = sycl::vec<int, 1>((int)((v_read      ) & 0x0f) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];
        sycl::half v_1 = sycl::vec<int, 1>((int)((v_read >>  4) & 0x0f) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];
        sycl::half v_2 = sycl::vec<int, 1>((int)((v_read >>  8) & 0x0f) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];
        sycl::half v_3 = sycl::vec<int, 1>((int)((v_read >> 12) & 0x0f) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];
        sycl::half v_4 = sycl::vec<int, 1>((int)((v_read >> 16) & 0x0f) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];
        sycl::half v_5 = sycl::vec<int, 1>((int)((v_read >> 20) & 0x0f) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];
        sycl::half v_6 = sycl::vec<int, 1>((int)((v_read >> 24) & 0x0f) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];
        sycl::half v_7 = sycl::vec<int, 1>((int)((v_read >> 28)       ) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];

        sycl::half2 v_01 = sycl::half2(v_0, v_1);
        sycl::half2 v_23 = sycl::half2(v_2, v_3);
        sycl::half2 v_45 = sycl::half2(v_4, v_5);
        sycl::half2 v_67 = sycl::half2(v_6, v_7);

//         half2 v_01 = q4_table[v_zero - 1][(v_read      ) & 0xff]; // (constant memory is too slow apparently)
//         half2 v_23 = q4_table[v_zero - 1][(v_read >>  8) & 0xff];
//         half2 v_45 = q4_table[v_zero - 1][(v_read >> 16) & 0xff];
//         half2 v_67 = q4_table[v_zero - 1][(v_read >> 24)       ];

        sycl::half2 tmp = *h_ptr++ * v_01;
        tmp = sycl::fma(*h_ptr++, v_23, tmp);
        tmp = sycl::fma(*h_ptr++, v_45, tmp);
        tmp = sycl::fma(*h_ptr++, v_67, tmp);
        result = sycl::fma(v_scale_2, tmp, result);
    }

    return result;
}

__dpct_inline__ sycl::half dot_product_8_h
(
    const sycl::half acc,
    MatrixView_half& h_,
    const int h_row,
    const int h_column,                 // divisible by 8
    MatrixView_q4_column& v_,
    const int v_row,                    // divisible by 8
    const int v_column,
    const sycl::half v_scale,
    const uint32_t v_zero,              // + 1 (!!)
    const int count
)
{
    const sycl::half* h_ptr = h_.item_ptr(h_row, h_column);
    const uint32_t* v_ptr = (const uint32_t*) v_.item_uint32_ptr(v_row, v_column);
    sycl::half result = acc;

    for (int i = 0; i < count; i++)
    {
        uint32_t v_read = *v_ptr; v_ptr += v_.width;

        sycl::half v_0 = sycl::vec<int, 1>((int)((v_read      ) & 0x0f) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];
        sycl::half v_1 = sycl::vec<int, 1>((int)((v_read >>  4) & 0x0f) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];
        sycl::half v_2 = sycl::vec<int, 1>((int)((v_read >>  8) & 0x0f) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];
        sycl::half v_3 = sycl::vec<int, 1>((int)((v_read >> 12) & 0x0f) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];
        sycl::half v_4 = sycl::vec<int, 1>((int)((v_read >> 16) & 0x0f) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];
        sycl::half v_5 = sycl::vec<int, 1>((int)((v_read >> 20) & 0x0f) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];
        sycl::half v_6 = sycl::vec<int, 1>((int)((v_read >> 24) & 0x0f) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];
        sycl::half v_7 = sycl::vec<int, 1>((int)((v_read >> 28)       ) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];

        sycl::half tmp = *h_ptr++ * v_0;
        tmp = sycl::fma(*h_ptr++, v_1, tmp);
        tmp = sycl::fma(*h_ptr++, v_2, tmp);
        tmp = sycl::fma(*h_ptr++, v_3, tmp);
        tmp = sycl::fma(*h_ptr++, v_4, tmp);
        tmp = sycl::fma(*h_ptr++, v_5, tmp);
        tmp = sycl::fma(*h_ptr++, v_6, tmp);
        tmp = sycl::fma(*h_ptr++, v_7, tmp);
        result = sycl::fma(v_scale, tmp, result);
    }

    return result;
}

// Accumulated dot product of 8-element row vectors in h and quantized column vectors in v, constant zero/scale, with x_map

__dpct_inline__ sycl::half2 dot_product_8_x_map
(
    const sycl::half2 acc,
    MatrixView_half& h_,
    const int h_row,
    const int h_column,                 // divisible by 8
    MatrixView_q4_column& v_,
    const int v_row,                    // divisible by 8
    const int v_column,
    const sycl::half2 v_scale_2,
    const uint32_t v_zero,              // + 1 (!!)
    const int count,
    const uint32_t* x_map
)
{
    const sycl::half* h_ptr = h_.item_ptr(h_row, 0);
    const uint32_t* x_map_ptr = x_map + h_column;
    const uint32_t* v_ptr = (const uint32_t*) v_.item_uint32_ptr(v_row, v_column);
    sycl::half2 result = acc;

    for (int i = 0; i < count; i++)
    {
        uint32_t v_read = *v_ptr; v_ptr += v_.width;

        sycl::half v_0 = sycl::vec<int, 1>((int)((v_read      ) & 0x0f) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];
        sycl::half v_1 = sycl::vec<int, 1>((int)((v_read >>  4) & 0x0f) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];
        sycl::half v_2 = sycl::vec<int, 1>((int)((v_read >>  8) & 0x0f) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];
        sycl::half v_3 = sycl::vec<int, 1>((int)((v_read >> 12) & 0x0f) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];
        sycl::half v_4 = sycl::vec<int, 1>((int)((v_read >> 16) & 0x0f) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];
        sycl::half v_5 = sycl::vec<int, 1>((int)((v_read >> 20) & 0x0f) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];
        sycl::half v_6 = sycl::vec<int, 1>((int)((v_read >> 24) & 0x0f) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];
        sycl::half v_7 = sycl::vec<int, 1>((int)((v_read >> 28)       ) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];

        sycl::half2 v_01 = sycl::half2(v_0, v_1);
        sycl::half2 v_23 = sycl::half2(v_2, v_3);
        sycl::half2 v_45 = sycl::half2(v_4, v_5);
        sycl::half2 v_67 = sycl::half2(v_6, v_7);

        sycl::half h_0 = h_ptr[*x_map_ptr++];
        sycl::half h_1 = h_ptr[*x_map_ptr++];
        sycl::half h_2 = h_ptr[*x_map_ptr++];
        sycl::half h_3 = h_ptr[*x_map_ptr++];
        sycl::half h_4 = h_ptr[*x_map_ptr++];
        sycl::half h_5 = h_ptr[*x_map_ptr++];
        sycl::half h_6 = h_ptr[*x_map_ptr++];
        sycl::half h_7 = h_ptr[*x_map_ptr++];

        sycl::half2 h_01 = sycl::half2(h_0, h_1);
        sycl::half2 h_23 = sycl::half2(h_2, h_3);
        sycl::half2 h_45 = sycl::half2(h_4, h_5);
        sycl::half2 h_67 = sycl::half2(h_6, h_7);

        sycl::half2 tmp = h_01 * v_01;
        tmp = sycl::fma(h_23, v_23, tmp);
        tmp = sycl::fma(h_45, v_45, tmp);
        tmp = sycl::fma(h_67, v_67, tmp);
        result = sycl::fma(v_scale_2, tmp, result);
    }

    return result;
}

__dpct_inline__ sycl::half dot_product_8_x_map_h
(
    const sycl::half acc,
    MatrixView_half& h_,
    const int h_row,
    const int h_column,                 // divisible by 8
    MatrixView_q4_column& v_,
    const int v_row,                    // divisible by 8
    const int v_column,
    const sycl::half v_scale,
    const uint32_t v_zero,              // + 1 (!!)
    const int count,
    const uint32_t* x_map
)
{
    const sycl::half* h_ptr = h_.item_ptr(h_row, 0);
    const uint32_t* x_map_ptr = x_map + h_column;
    const uint32_t* v_ptr = (const uint32_t*) v_.item_uint32_ptr(v_row, v_column);
    sycl::half result = acc;

    for (int i = 0; i < count; i++)
    {
        uint32_t v_read = *v_ptr; v_ptr += v_.width;

        sycl::half v_0 = sycl::vec<int, 1>((int)((v_read      ) & 0x0f) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];
        sycl::half v_1 = sycl::vec<int, 1>((int)((v_read >>  4) & 0x0f) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];
        sycl::half v_2 = sycl::vec<int, 1>((int)((v_read >>  8) & 0x0f) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];
        sycl::half v_3 = sycl::vec<int, 1>((int)((v_read >> 12) & 0x0f) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];
        sycl::half v_4 = sycl::vec<int, 1>((int)((v_read >> 16) & 0x0f) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];
        sycl::half v_5 = sycl::vec<int, 1>((int)((v_read >> 20) & 0x0f) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];
        sycl::half v_6 = sycl::vec<int, 1>((int)((v_read >> 24) & 0x0f) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];
        sycl::half v_7 = sycl::vec<int, 1>((int)((v_read >> 28)       ) - v_zero).convert<sycl::half, sycl::rounding_mode::rte>()[0];

        sycl::half tmp = h_ptr[*x_map_ptr++] * v_0;
        tmp = sycl::fma(h_ptr[*x_map_ptr++], v_1, tmp);
        tmp = sycl::fma(h_ptr[*x_map_ptr++], v_2, tmp);
        tmp = sycl::fma(h_ptr[*x_map_ptr++], v_3, tmp);
        tmp = sycl::fma(h_ptr[*x_map_ptr++], v_4, tmp);
        tmp = sycl::fma(h_ptr[*x_map_ptr++], v_5, tmp);
        tmp = sycl::fma(h_ptr[*x_map_ptr++], v_6, tmp);
        tmp = sycl::fma(h_ptr[*x_map_ptr++], v_7, tmp);
        result = sycl::fma(v_scale, tmp, result);
    }

    return result;
}

#endif

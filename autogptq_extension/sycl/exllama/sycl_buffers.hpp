// Adapted from turboderp exllama: https://github.com/turboderp/exllama

#ifndef _sycl_buffers_cuh
#define _sycl_buffers_cuh

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdint>
#include <cstdio>

const int SYCL_MAX_DEVICES = 12;

// #ifndef _sycl_buffers_cu
// extern __constant__ sycl::half2 q4_table[16][256];
// #endif

class SyclBuffers
{
public:
    int device;

    sycl::half* temp_state;           // [max_hidden_rows * intermediate_size]
    int temp_state_size;
    sycl::half* temp_dq;              // size of largest quant tensor * 8

    dpct::queue_ptr alt_stream_1;
    dpct::queue_ptr alt_stream_2;
    dpct::queue_ptr alt_stream_3;
    dpct::event_ptr alt_stream_1_done;
    dpct::event_ptr alt_stream_2_done;
    dpct::event_ptr alt_stream_3_done;

    SyclBuffers
    (
        int _device,
        int _temp_state_size,
        sycl::half* _temp_state,
        sycl::half* _temp_dq
    );
    ~SyclBuffers();
};

SyclBuffers* get_buffers(const int device_index);

void prepare_buffers_sycl
(
    int _device,
    int _temp_state_size,
    sycl::half* _temp_state,
    sycl::half* _temp_dq
);

void cleanup_buffers_sycl();

#endif

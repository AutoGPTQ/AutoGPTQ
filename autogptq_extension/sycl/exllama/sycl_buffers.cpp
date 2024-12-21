// Adapted from turboderp exllama: https://github.com/turboderp/exllama

#define _sycl_buffers_cu
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "sycl_buffers.hpp"

SyclBuffers* g_buffers[SYCL_MAX_DEVICES] = {NULL};
// __constant__ sycl::half2 q4_table[16][256];
// sycl::half2 q4_table_host[16][256];
// bool q4_table_init = false;

SyclBuffers::SyclBuffers
(
    int _device,
    int _temp_state_size,
    sycl::half* _temp_state,
    sycl::half* _temp_dq
) :
    device(_device),
    temp_state_size(_temp_state_size),
    temp_state(_temp_state),
    temp_dq(_temp_dq)
{
    
    dpct::select_device(_device);

    alt_stream_1 = dpct::get_current_device().create_queue();
    alt_stream_2 = dpct::get_current_device().create_queue();
    alt_stream_3 = dpct::get_current_device().create_queue();
    alt_stream_1_done = new sycl::event();
    alt_stream_2_done = new sycl::event();
    alt_stream_3_done = new sycl::event();
}

SyclBuffers::~SyclBuffers()
{
    dpct::get_current_device().destroy_queue(alt_stream_1);
    dpct::get_current_device().destroy_queue(alt_stream_2);
    dpct::get_current_device().destroy_queue(alt_stream_3);
    dpct::destroy_event(alt_stream_1_done);
    dpct::destroy_event(alt_stream_2_done);
    dpct::destroy_event(alt_stream_3_done);
}

SyclBuffers* get_buffers(const int device_index)
{
    return g_buffers[device_index];
}

void prepare_buffers_sycl
(
    int _device,
    int _temp_state_size,
    sycl::half* _temp_state,
    sycl::half* _temp_dq
)
{
    SyclBuffers* buffers = new SyclBuffers
    (
        _device,
        _temp_state_size,
        _temp_state,
        _temp_dq
    );

    g_buffers[_device] = buffers;
}

void cleanup_buffers_sycl()
{
    for (int i = 0; i < SYCL_MAX_DEVICES; i++)
    {
        if (!g_buffers[i]) continue;
        delete g_buffers[i];
        g_buffers[i] = NULL;
    }
}

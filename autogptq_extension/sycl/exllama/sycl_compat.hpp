#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
// Adapted from turboderp exllama: https://github.com/turboderp/exllama

#ifndef _sycl_compat_cuh
#define _sycl_compat_cuh

// atomicAdd for half types, to support CC < 7.x

__dpct_inline__ void atomicAdd_half(sycl::half* address, sycl::half val)
{
    unsigned int * address_as_ui = (unsigned int *) ((char *)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do
    {
        assumed = old;
        uint16_t hsum;
        hsum = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
        sycl::half tmpres = hsum + val;
        hsum = uint16_t(tmpres);
        old = (size_t)address & 2 ? (old & 0xffff) | (hsum << 16) : (old & 0xffff0000) | hsum;
        old = dpct::atomic_compare_exchange_strong<sycl::access::address_space::generic_space>(address_as_ui, assumed, old);
    }
    while (assumed != old);
}

// atomicAdd for half2 types

__dpct_inline__ void atomicAdd_half2(sycl::half2* address, sycl::half2 val)
{
    unsigned int* address_as_ui = (unsigned int*)address;
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    do
    {
        assumed = old;
        sycl::half2 old_val = *((sycl::half2*)&old);
        sycl::half2 new_val = old_val + val;
        old = dpct::atomic_compare_exchange_strong<sycl::access::address_space::generic_space>(address_as_ui, assumed, *((unsigned int*)&new_val));
    }
    while (assumed != old);
}

//

#if defined(DPCT_COMPATIBILITY_TEMP) || defined(USE_ROCM)
#if DPCT_COMPATIBILITY_TEMP < 700 || defined(USE_ROCM)

inline void atomicAdd(sycl::half* address, sycl::half val) { atomicAdd_half(address, val); }

#if __CUDA_ARCH__ < 600 || defined(USE_ROCM)
inline void atomicAdd(sycl::half2* address, sycl::half2 val) { atomicAdd_half2(address, val); }
#endif

#endif
#endif

#endif

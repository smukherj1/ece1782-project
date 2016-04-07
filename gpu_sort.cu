/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */



#include "sort.h"
#include "utils.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <stdio.h>
#include <stdlib.h>

#define checkCudaOK(val) {\
    if(val != cudaSuccess) {\
        fprintf(stderr, "Cuda check failed at %s:%d '%s'\n", __FILE__, __LINE__, #val); \
        fprintf(stderr, "%s\n", cudaGetErrorString(val));\
        exit(EXIT_FAILURE);\
    }\
}

__device__ inline void swap(int & a, int & b)
{
    // Alternative swap doesn't use a temporary register:
    // a ^= b;
    // b ^= a;
    // a ^= b;

    int tmp = a;
    a = b;
    b = tmp;
}

__global__ static void bitonic_sort_kernel(int * values, int size)
{
    extern __shared__ int shared[];

    const int bid = blockIdx.x * blockDim.x;
    const int tid = bid + threadIdx.x;

    // Copy input to shared mem.
    shared[threadIdx.x] = values[tid];

    __syncthreads();

    // Parallel bitonic sort.
    for (int k = 2; k <= size; k *= 2)
    {
        // Bitonic merge:
        for (int j = k / 2; j>0; j /= 2)
        {
            int ixj = tid ^ j;

            __syncthreads();

            if (ixj > tid)
            {
                ixj -= bid;
                if ((tid & k) == 0)
                {
                    if (shared[threadIdx.x] > shared[ixj])
                    {
                        swap(shared[threadIdx.x], shared[ixj]);
                    }
                }
                else
                {
                    if (shared[threadIdx.x] < shared[ixj])
                    {
                        swap(shared[threadIdx.x], shared[ixj]);
                    }
                }
            }
        }
    }

    // Write result.
    values[tid] = shared[threadIdx.x];
}

bool is_bitonic_sort_allowed(const VECT_T& vect)
{
    const int *v = &vect[0];
    int size = static_cast<int>(vect.size());

    if((size & (size - 1)) != 0)
    {
        return false;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int arr_max_size = (prop.sharedMemPerBlock * 2) / sizeof(int);

    if(size > arr_max_size)
    {
        return false;
    }

    return true;
}

/**
 * Inplace bitonic sort using CUDA.
 */
void gpu_bitonic_sort(VECT_T& vect)
{
    int *d_v;

    int *v = &vect[0];
    int size = static_cast<int>(vect.size());

    if((size & (size - 1)) != 0)
    {
        printf("Error: size %d is not a power of 2. Bitonic sort needs it to be a power of 2\n", size);
        exit(EXIT_FAILURE);
    }

    int num_cuda_devices;
    cudaGetDeviceCount(&num_cuda_devices);
    if(num_cuda_devices <= 0)
    {
        printf("Error: No CUDA capable devices were detected.\n");
        exit(EXIT_FAILURE);
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int num_blocks, num_threads = size;

    if(num_threads <= 1024)
    {
        num_blocks = 1;
    }
    else
    {
        num_blocks = num_threads / 1024;
        num_threads = 1024;
    }

    int arr_max_size = (prop.sharedMemPerBlock * 2) / sizeof(int);
    if(size > arr_max_size)
    {
        printf("Error: Size %d exceeds capabilities of GPU. Must be limited to %d\n", size, arr_max_size);
        exit(EXIT_FAILURE);
    }

    cudaMalloc((void**) &d_v, size * sizeof(int));

    dim3 blocks(num_blocks,1);
    dim3 threads(num_threads,1);

    PROFILE_BIN_T bin = create_bin();
    cudaMemcpy(d_v, v, size * sizeof(int), cudaMemcpyHostToDevice);
    bitonic_sort_kernel<<<blocks, threads, prop.sharedMemPerBlock>>>(d_v, size);
    cudaMemcpy(v, d_v, size * sizeof(int), cudaMemcpyDeviceToHost);
    checkCudaOK(cudaDeviceSynchronize());
    double elapsed = get_elapsed(bin);
    printf("gpu_bitonic_sort took %.6f ms\n", elapsed);
    destroy_bin(bin);
    cudaFree(d_v);
}

void gpu_thrust_sort(VECT_T& v)
{
    thrust::device_vector<int> d_v(v.size());

    PROFILE_BIN_T bin = create_bin();
    d_v = v;
    thrust::sort(d_v.begin(), d_v.end());
    thrust::copy(d_v.begin(), d_v.end(), v.begin());
    double elapsed = get_elapsed(bin);
    printf("gpu_thrust_sort took %.6f ms\n", elapsed);
    destroy_bin(bin);
}

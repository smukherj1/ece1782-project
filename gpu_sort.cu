#include "gpu_sort.h"

#include <stdio.h>

#define checkCudaOK(val) {\
    if(val != cudaSuccess) {\
        fprintf(stderr, "Cuda check failed at %s:%d '%s'\n", __FILE__, __LINE__, #val); \
        fprintf(stderr, "%s\n", cudaGetErrorString(val));\
        exit(EXIT_FAILURE);\
    }\
}

__global__ void bitonic_sort_kernel()
{
	printf("Hello World from thread %d in the GPU!\n", threadIdx.x);
}


void gpu_sort()
{
	bitonic_sort_kernel<<<1, 1>>> ();
	checkCudaOK(cudaDeviceSynchronize());
}
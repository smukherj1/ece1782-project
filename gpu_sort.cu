#include "sort.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>

#define checkCudaOK(val) {\
    if(val != cudaSuccess) {\
        fprintf(stderr, "Cuda check failed at %s:%d '%s'\n", __FILE__, __LINE__, #val); \
        fprintf(stderr, "%s\n", cudaGetErrorString(val));\
        exit(EXIT_FAILURE);\
    }\
}

__global__ void bitonic_sort_kernel(int *d_v, int j, int k, int size)
{
  unsigned int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i^j;

  if(i >= size || ixj >= size)
  {
  	return;
  }

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i) {
    if ((i&k)==0) {
      /* Sort ascending */
      if (d_v[i]>d_v[ixj]) {
        /* exchange(i,ixj); */
        int temp = d_v[i];
        d_v[i] = d_v[ixj];
        d_v[ixj] = temp;
      }
    }
    if ((i&k)!=0) {
      /* Sort descending */
      if (d_v[i]<d_v[ixj]) {
        /* exchange(i,ixj); */
        int temp = d_v[i];
        d_v[i] = d_v[ixj];
        d_v[ixj] = temp;
      }
    }
  }
}

/**
 * Inplace bitonic sort using CUDA.
 */
 void gpu_sort(int *v, int size)
 {
 	int *d_v;

 	cudaMalloc((void**) &d_v, size * sizeof(int));
 	cudaMemcpy(d_v, v, size * sizeof(int), cudaMemcpyHostToDevice);

 	int num_blocks, num_threads;

 	if(size < 1024)
 	{
 		num_threads = size;
 		num_blocks = 1;
 	}
 	else
 	{
 		num_threads = 1024;
 		num_blocks = (size / num_threads);
 		if(size % num_threads != 0)
 		{
 			num_blocks++;
 		}
 	}

	dim3 blocks(num_blocks,1);
	dim3 threads(num_threads,1);

 	int j, k;

 	PROFILE_BIN_T bin = create_bin();
	/* Major step */
 	for (k = 2; k <= size; k <<= 1) {
    /* Minor step */
 		for (j=k>>1; j>0; j=j>>1) {
 			bitonic_sort_kernel<<<blocks, threads>>>(d_v, j, k, size);
 		}
 	}
 	checkCudaOK(cudaDeviceSynchronize());
 	double elapsed = get_elapsed(bin);
 	printf("gpu_sort took %.4f seconds\n", elapsed);
 	destroy_bin(bin);
 	cudaMemcpy(v, d_v, size * sizeof(int), cudaMemcpyDeviceToHost);
 	cudaFree(d_v);
 }
#include "sort.h"
#include <omp.h>
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

// Taken from https://github.com/gtgbansal/openmp/blob/master/qsort_parallel.c


void qsort_serial(int *a, int l,int r){
	if(r>l){
		int pivot=a[r],tmp;
		int less=l-1,more;
		for(more=l;more<=r;more++){
			if(a[more]<=pivot){
				less++;
				 tmp=a[less];
				a[less]=a[more];
				a[more]=tmp;
			}
		}
		qsort_serial(a, l,less-1);
		qsort_serial(a, less+1,r);
	}
}
void qsort_parallel(int *a, int l,int r){
	if(r>l){

		int pivot=a[r],tmp;
		int less=l-1,more;
		for(more=l;more<=r;more++){
			if(a[more]<=pivot){
				less++;
				tmp=a[less];
				a[less]=a[more];
				a[more]=tmp;
			}
		}
		if((r-l)<1000){
			qsort_serial(a, l,less-1);
			qsort_serial(a, less+1,r);
		}
		else{
			#pragma omp task
			qsort_parallel(a, l,less-1);
			#pragma omp task
			qsort_parallel(a, less+1,r);
			#pragma omp taskwait
		}
	}
}

void cpu_sort(int *v, int size)
{
	PROFILE_BIN_T bin = create_bin();
	#pragma omp parallel
	{
		#pragma omp single
		qsort_parallel(v, 0, size);
	}
	double elapsed = get_elapsed(bin);
	printf("cpu_sort took %.6f ms\n", elapsed);
	destroy_bin(bin);
}

void fpga_sort(int *v, int size)
{
	printf("Error: Not implemented\n");
	exit(EXIT_FAILURE);
}
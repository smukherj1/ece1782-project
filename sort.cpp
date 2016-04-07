#include <iostream>

#include "sort.h"
#include "utils.h"

void run_cpu_sort(VECT_T& v)
{
	VECT_T v1 = v;
	cpu_sort(v1);
	verify_sort(v1);

	v1 = v;
	cpu_tbb_sort(v1);
	verify_sort(v1);
}

void run_gpu_sort(VECT_T& v)
{
	VECT_T v1 = v;
	gpu_thrust_sort(v1);
	verify_sort(v1);

	if(is_bitonic_sort_allowed(v))
	{
		v1 = v;
		gpu_bitonic_sort(v1);
		verify_sort(v1);
	}

	//v1 = v;
	//gpu_cudapp_sort(v1);
	//verify_sort(v1);
}

void run_fpga_sort(VECT_T& v)
{
	//fpga_sort(&v[0], v.size());
	//verify_sort(v);
}

int main(int argc, char *argv[])
{
	parse_cmd_opts(argc, argv);

	const CMD_OPTS opts = get_opts();
	int size = opts.size;
	VECT_T v;

	fill_array(v, size);
	if(opts.target == TARGET::CPU)
	{
		run_cpu_sort(v);
	}
	else if(opts.target == TARGET::GPU)
	{
		run_gpu_sort(v);
	}
	else if(opts.target == TARGET::FPGA)
	{
		run_fpga_sort(v);
	}
	else if(opts.target == TARGET::ALL)
	{
		run_cpu_sort(v);
		run_gpu_sort(v);
		run_fpga_sort(v);
	}

	return 0;
}
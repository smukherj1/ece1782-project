#include <iostream>

#include "sort.h"
#include "utils.h"

int main(int argc, char *argv[])
{
	parse_cmd_opts(argc, argv);

	const CMD_OPTS opts = get_opts();
	int size = (1 << 20) * opts.size;
	int *v = new int[size];

	fill_array(v, size);
	if(opts.target == TARGET::CPU)
	{
		cpu_sort(v, size);
	}
	else if(opts.target == TARGET::GPU)
	{
		gpu_sort(v, size);
	}
	else if(opts.target == TARGET::FPGA)
	{
		fpga_sort(v, size);
	}

	verify_sort(v, size);

	delete []v;
	return 0;
}
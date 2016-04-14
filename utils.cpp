#include "utils.h"
#include <chrono>
#include <map>
#include <string>
#include <cassert>
#include <mutex>
#include <stdio.h>
#include <cstdlib>

typedef std::map<PROFILE_BIN_T, std::chrono::high_resolution_clock::time_point> PROFILE_BIN_MAP;

static PROFILE_BIN_T s_next_bin_id = 0;
static PROFILE_BIN_MAP s_pmap;
static std::mutex s_mtx; 
static CMD_OPTS s_opts;

void parse_cmd_opts(int argc, char *argv[])
{
	if(argc != 3)
	{
		printf("Exactly 2 command line arguments expected\n");
		printf("./sort <target> <size>\n");
		printf("target=(cpu|gpu|fpga|all)\n");
		printf("size=non zero positive integer\n");
		exit(EXIT_FAILURE);
	}

	std::string target = std::string(argv[1]);
	if(target == "cpu")
	{
		s_opts.target = TARGET::CPU;
	}
	else if(target == "gpu")
	{
		s_opts.target = TARGET::GPU;
	}
	else if(target == "fpga")
	{
		s_opts.target = TARGET::FPGA;
	}
	else if(target == "all")
	{
		s_opts.target = TARGET::ALL;
	}
	else
	{
		printf("Illegal target %s. Choices are (cpu|gpu|fpga|all)\n", target.c_str());
		exit(EXIT_FAILURE);
	}


	int size = std::atoi(argv[2]);
	if(size <= 0)
	{
		printf("Illegal size '%s'. Must be a non zero positive integer\n", argv[2]);
		exit(EXIT_FAILURE);
	}
	s_opts.size = size;
}

const CMD_OPTS get_opts()
{
	return s_opts;
}

PROFILE_BIN_T create_bin()
{
	std::lock_guard<std::mutex> lock(s_mtx);
	assert(s_pmap.find(s_next_bin_id) == s_pmap.end());
	s_pmap[s_next_bin_id] = std::chrono::high_resolution_clock::now();
	PROFILE_BIN_T ret = s_next_bin_id;
	++s_next_bin_id;
	return ret;
}

void destroy_bin(PROFILE_BIN_T bin)
{
	std::lock_guard<std::mutex> lock(s_mtx);
	s_pmap.erase(bin);
}

double get_elapsed(PROFILE_BIN_T bin)
{
	std::lock_guard<std::mutex> lock(s_mtx);
	auto now = std::chrono::high_resolution_clock::now();
	auto before = s_pmap.at(bin);
	double elapsed = std::chrono::duration<double>(now - before).count();
	return elapsed * 1000;
}

void fill_array(VECT_T& v, int size)
{
	srand(0);
	for(int i = 0; i < size; ++i)
	{
		v.push_back(rand());
	}
}

void dump_array(const VECT_T& v, const char *filename)
{
	FILE *fp = fopen(filename, "w");
	if(fp == NULL)
	{
		printf("Error: Could not dump array. Failed to open file for writing\n");
		return;
	}

	printf("Info: Dumping array to %s...\n", filename);
	for(int i = 0; i < static_cast<int>(v.size()); ++i)
	{
		fprintf(fp, "[%d]\t%d\n", i, v[i]);
	}
	fclose(fp);
}

void verify_sort(const VECT_T& v, const VECT_T& orig_v)
{
	if(v.empty())
	{
		return;
	}
	size_t size = v.size() - 1;
	for(size_t i = 0; i < size; ++i)
	{
		if(v[i] > v[i + 1])
		{
			printf("Error: Array was not sorted properly. v[%lu] <= v[%lu] failed. Array size is %lu\n", 
				i, i + 1, size + 1);
			dump_array(v, "new.txt");
			dump_array(orig_v, "old.txt");
			exit(EXIT_FAILURE);
		}
	}
}
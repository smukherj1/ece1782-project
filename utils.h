#ifndef UTILS_H
#define UTILS_H

typedef int PROFILE_BIN_T;

enum class TARGET {
	CPU,
	GPU,
	FPGA
};

struct CMD_OPTS
{
	TARGET target;

	// No. of M elements where M = 2 ^ 20
	int size;
};


// Command line processing
void parse_cmd_opts(int argc, char *argv[]);
const CMD_OPTS get_opts();

// Timing
PROFILE_BIN_T create_bin();
void destroy_bin(PROFILE_BIN_T bin);
double get_elapsed(PROFILE_BIN_T bin);

void fill_array(int *v, int size);

#endif /* UTILS_H */
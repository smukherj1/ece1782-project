#ifndef SORT_H
#define SORT_H

#include <vector>

typedef std::vector<int> VECT_T;

void cpu_sort(VECT_T& v);
void cpu_tbb_sort(VECT_T& v);

bool is_bitonic_sort_allowed(const VECT_T& v);
void gpu_bitonic_sort(VECT_T& v);
void gpu_merge_bitonic_sort(VECT_T& vect);
void gpu_thrust_sort(VECT_T& v);

void fpga_sort(int *v, int size);

#endif /* SORT_H */
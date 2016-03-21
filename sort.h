#ifndef SORT_H
#define SORT_H

void cpu_sort(int *v, int size);
void gpu_sort(int *v, int size);
void fpga_sort(int *v, int size);

#endif /* SORT_H */
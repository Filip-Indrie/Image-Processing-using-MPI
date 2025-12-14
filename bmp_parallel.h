#ifndef BMP_PARALLEL

#define BMP_PARALLEL

#include "bmp_common.h"

Image *readBMP_MPI(const char *file_name, int my_rank, int num_processes, int halo_dim, int *true_start, int *true_end);
Image *compose_BMP(Image *img, int my_rank, int num_processes);

#endif
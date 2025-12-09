#ifndef BMP_PARALLEL

#define BMP_PARALLEL

#include "bmp_common.h"

Image *readBMP_MPI(const char *file_name, int my_rank, int num_processes);

#endif
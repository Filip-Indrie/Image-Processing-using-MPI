#ifndef BMP_PARALLEL

#define BMP_PARALLEL

#include "bmp_common.h"
#include "mpi.h"

MPI_Datatype create_mpi_datatype_for_RGB();
Image *readBMP_serial(const char *filename);
Image *readBMP_MPI(const char *file_name, int my_rank, int num_processes, int halo_dim, int *true_start, int *true_end);
Image *compose_BMP(Image *img, int my_rank, int num_processes);
int scatter_data(Image *img, RGB **data, int my_rank, int num_processes, int width, int *local_height, int halo_dim);
int saveBMP(const char *filename, const Image *img);

#endif
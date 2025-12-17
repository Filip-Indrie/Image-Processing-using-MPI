#ifndef BMP_PARALLEL

#define BMP_PARALLEL

#include "bmp_common.h"
#include "mpi.h"

MPI_Datatype create_mpi_datatype_for_RGB();
Image *readBMP_serial(const char *filename);
Image *readBMP_MPI(const char *file_name, int my_rank, int num_processes, int halo_dim, int *true_start, int *true_end);
Image *compose_BMP(Image *img, int my_rank, int num_processes);
FILE *open_BMP(const char *filename, int *height, int *width, int *data_start, int *padding);
Image *readBMP_chunk(FILE *image_file, int halo_dim, int chunk_size, int height, int width, int padding, int data_start, int *offset, int *true_start, int *true_end);
// move scatter data to image_processing
int scatter_data(Image *img, RGB **data, int my_rank, int num_processes, int width, int *local_height, int halo_dim);
int saveBMP(const char *filename, const Image *img);

#endif
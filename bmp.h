#ifndef BMP_PARALLEL

#define BMP_PARALLEL

#include "bmp_common.h"

Image *read_BMP_serial(const char *filename);
Image *read_BMP_MPI(const char *file_name, int my_rank, int num_processes, int halo_dim, int *true_start, int *true_end);
Image *compose_BMP(Image *img, int my_rank, int num_processes);
FILE *open_BMP(const char *filename, int *height, int *width, int *data_start, int *padding);
Image *read_BMP_chunk(FILE *image_file, int halo_dim, int chunk_size, int height, int width, int padding, int data_start, int *offset, int *true_start, int *true_end);
int save_BMP(const char *filename, const Image *img);

#endif
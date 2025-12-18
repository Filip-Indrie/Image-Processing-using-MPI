#ifndef BMP_COMMON

#define BMP_COMMON

#include "mpi.h"

typedef struct
{
    unsigned char r, g, b;
} RGB; // one RGB point

typedef struct
{
    int width;
    int height;
    RGB *data;
} Image; // a BMP image as an array of RGB points

typedef struct{
	int true_start, true_end, height, width, num_threads;
	int operation; // enum has the same size as an int
}send_block_t;

void copy_RGB(const RGB *src, RGB *dest);
int equal_RGB(const RGB rgb1, const RGB rgb2);
int min(int a, int b);
int max(int a, int b);

MPI_Datatype create_mpi_datatype_for_RGB();
MPI_Datatype create_mpi_datatype_for_send_block_t();
int deallocate_MPI_datatype(MPI_Datatype *type, int my_rank);

#endif
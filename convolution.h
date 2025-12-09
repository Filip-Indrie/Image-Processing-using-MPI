#ifndef CONVOLUTION

#define CONVOLUTION

#include "bmp.h"

typedef enum{
	RIDGE,
	EDGE,
	SHARPEN,
	BOXBLUR,
	GAUSSBLUR3,
	GAUSSBLUR5,
	UNSHARP5
}operation_t;

Image* perform_convolution_serial(const Image *img, const operation_t operation);
Image* perform_convolution_MPI(const Image *img, const operation_t operation, const int shared_file_tree);

#endif
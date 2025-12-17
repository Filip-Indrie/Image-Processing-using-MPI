#ifndef CONVOLUTION

#define CONVOLUTION

#include "bmp_common.h"

typedef enum{
	RIDGE,
	EDGE,
	SHARPEN,
	BOXBLUR,
	GAUSSBLUR3,
	GAUSSBLUR5,
	UNSHARP5
}operation_t;

operation_t string_to_operation(char *string);
int get_kernel_size(const operation_t operation);
Image* perform_convolution_serial(const Image *img, const operation_t operation);
Image* perform_convolution_parallel(const Image *img, const operation_t operation, const int true_start, const int true_end, const int threads);

#endif
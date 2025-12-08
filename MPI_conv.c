#include "convolution.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void copy_RGB(RGB *src, RGB *dest){
	dest->r = src->r;
	dest->g = src->g;
	dest->b = src->b;
}

int* generate_kernel(const operation_t operation, int *size){
	/**
	*	TO BE IMPLEMENTED
	*/
	return NULL;
}

Image* perform_convolution_MPI(const Image, const operation_t operation, const int shared_file_tree){
	/**
	*	TO BE IMPLEMENTED
	*/
	return NULL;
}
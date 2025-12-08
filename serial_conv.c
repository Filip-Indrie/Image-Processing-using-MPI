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

Image* perform_convolution_serial(const Image, const operation_t operation){
	/**
	*	TO BE IMPLEMENTED
	*/
	return NULL;
}

/**
================================================================================================
	TESTBENCH
	
int main(){
	Image *img = readBMP("Photos\\Large.bmp");
	saveBMP("Photos\\Padded_Large.bmp", editted_img);
	return 0;
}
*/
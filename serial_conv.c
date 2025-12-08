#include "convolution.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void copy_RGB(RGB *src, RGB *dest){
	dest->r = src->r;
	dest->g = src->g;
	dest->b = src->b;
}

double* generate_kernel(const operation_t operation, int *size){
	switch(operation){
		case RIDGE:{
			*size = 3;
			
			double *kernel = (double*)malloc(3 * 3 * sizeof(double));
			if(kernel == NULL){
				fprintf(stderr, "Error in generate_kernel while allocating memory\n");
				exit(-1);
			}
			
			kernel[0] = 0;
			kernel[1] = -1;
			kernel[2] = 0;
			kernel[3] = -1;
			kernel[4] = 4;
			kernel[5] = -1;
			kernel[6] = 0;
			kernel[7] = -1;
			kernel[8] = 0;
			return kernel;
		}
		case EDGE:{
			*size = 3;
			
			double *kernel = (double*)malloc(3 * 3 * sizeof(double));
			if(kernel == NULL){
				fprintf(stderr, "Error in generate_kernel while allocating memory\n");
				exit(-1);
			}
			
			kernel[0] = -1;
			kernel[1] = -1;
			kernel[2] = -1;
			kernel[3] = -1;
			kernel[4] = 8;
			kernel[5] = -1;
			kernel[6] = -1;
			kernel[7] = -1;
			kernel[8] = -1;
			return kernel;
		}
		case SHARPEN:{
			*size = 3;
			
			double *kernel = (double*)malloc(3 * 3 * sizeof(double));
			if(kernel == NULL){
				fprintf(stderr, "Error in generate_kernel while allocating memory\n");
				exit(-1);
			}
			
			kernel[0] = 0;
			kernel[1] = -1;
			kernel[2] = 0;
			kernel[3] = -1;
			kernel[4] = 5;
			kernel[5] = -1;
			kernel[6] = 0;
			kernel[7] = -1;
			kernel[8] = 0;
			return kernel;
		}
		case BOXBLUR:{
			*size = 3;
			
			double *kernel = (double*)malloc(3 * 3 * sizeof(double));
			if(kernel == NULL){
				fprintf(stderr, "Error in generate_kernel while allocating memory\n");
				exit(-1);
			}
			
			kernel[0] = (double)1/9;
			kernel[1] = (double)1/9;
			kernel[2] = (double)1/9;
			kernel[3] = (double)1/9;
			kernel[4] = (double)1/9;
			kernel[5] = (double)1/9;
			kernel[6] = (double)1/9;
			kernel[7] = (double)1/9;
			kernel[8] = (double)1/9;
			return kernel;
		}
		case GAUSSBLUR3:{
			*size = 3;
			
			double *kernel = (double*)malloc(3 * 3 * sizeof(double));
			if(kernel == NULL){
				fprintf(stderr, "Error in generate_kernel while allocating memory\n");
				exit(-1);
			}
			
			kernel[0] = (double)1/16;
			kernel[1] = (double)2/16;
			kernel[2] = (double)1/16;
			kernel[3] = (double)2/16;
			kernel[4] = (double)4/16;
			kernel[5] = (double)2/16;
			kernel[6] = (double)1/16;
			kernel[7] = (double)2/16;
			kernel[8] = (double)1/16;
			return kernel;
		}
		case GAUSSBLUR5:{
			*size = 5;
			
			double *kernel = (double*)malloc(5 * 5 * sizeof(double));
			if(kernel == NULL){
				fprintf(stderr, "Error in generate_kernel while allocating memory\n");
				exit(-1);
			}
			
			kernel[0] = (double)1/256;
			kernel[1] = (double)4/256;
			kernel[2] = (double)6/256;
			kernel[3] = (double)4/256;
			kernel[4] = (double)1/256;
			kernel[5] = (double)4/256;
			kernel[6] = (double)16/256;
			kernel[7] = (double)24/256;
			kernel[8] = (double)16/256;
			kernel[9] = (double)4/256;
			kernel[10] = (double)6/256;
			kernel[11] = (double)24/256;
			kernel[12] = (double)36/256;
			kernel[13] = (double)24/256;
			kernel[14] = (double)6/256;
			kernel[15] = (double)4/256;
			kernel[16] = (double)16/256;
			kernel[17] = (double)24/256;
			kernel[18] = (double)16/256;
			kernel[19] = (double)4/256;
			kernel[20] = (double)1/256;
			kernel[21] = (double)4/256;
			kernel[22] = (double)6/256;
			kernel[23] = (double)4/256;
			kernel[24] = (double)1/256;
			return kernel;
		}
		case UNSHARP5:{
			*size = 5;
			
			double *kernel = (double*)malloc(5 * 5 * sizeof(double));
			if(kernel == NULL){
				fprintf(stderr, "Error in generate_kernel while allocating memory\n");
				exit(-1);
			}
			
			kernel[0] = (double)-1/256;
			kernel[1] = (double)-4/256;
			kernel[2] = (double)-6/256;
			kernel[3] = (double)-4/256;
			kernel[4] = (double)-1/256;
			kernel[5] = (double)-4/256;
			kernel[6] = (double)-16/256;
			kernel[7] = (double)-24/256;
			kernel[8] = (double)-16/256;
			kernel[9] = (double)-4/256;
			kernel[10] = (double)-6/256;
			kernel[11] = (double)-24/256;
			kernel[12] = (double)476/256;
			kernel[13] = (double)-24/256;
			kernel[14] = (double)-6/256;
			kernel[15] = (double)-4/256;
			kernel[16] = (double)-16/256;
			kernel[17] = (double)-24/256;
			kernel[18] = (double)-16/256;
			kernel[19] = (double)-4/256;
			kernel[20] = (double)-1/256;
			kernel[21] = (double)-4/256;
			kernel[22] = (double)-6/256;
			kernel[23] = (double)-4/256;
			kernel[24] = (double)-1/256;
			return kernel;
		}
		default:{
			printf("Invalid operation\n");
			exit(-1);
		}
	}
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
*/
int main(){
	//Image *img = readBMP("Photos\\Large.bmp");
	int kernel_size;
	double *kernel = generate_kernel(UNSHARP5, &kernel_size);
	for(int i=0;i<kernel_size;++i){
		for(int j=0;j<kernel_size;++j){
			printf("%f ", kernel[i*kernel_size + j]);
		}
		printf("\n");
	}
	//saveBMP("Photos\\Padded_Large.bmp", editted_img);
	return 0;
}

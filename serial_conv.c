#include "convolution.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void copy_RGB(const RGB *src, RGB *dest){
	dest->r = src->r;
	dest->g = src->g;
	dest->b = src->b;
}

int min(int a, int b){
	return (a < b) ? a : b;
}

int max(int a, int b){
	return (a > b) ? a : b;
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

Image* perform_convolution_serial(const Image *img, const operation_t operation){
	Image *new_img = (Image*)malloc(sizeof(Image));
	if(new_img == NULL){
		fprintf(stderr, "Error in perform_convolution_serial while allocating memory\n");
		exit(-1);
	}
	
	RGB *new_data = (RGB*)malloc(img->height * img->width * sizeof(RGB));
	if(new_data == NULL){
		fprintf(stderr, "Error in perform_convolution_serial while allocating memory\n");
		exit(-1);
	}
	
	RGB *old_data = img->data;
	int height = img->height;
	int width = img->width;
	int kernel_size;
	double *kernel = generate_kernel(operation, &kernel_size);
	
	for(int i = 0; i < img->height; ++i){
		for(int j = 0; j < img->width; ++j){
			double r = 0, g = 0, b = 0;
			for(int m = -kernel_size / 2; m <= kernel_size / 2; ++m){
				for(int n = -kernel_size / 2; n <= kernel_size / 2; ++n){
					if(i + m >= 0 && i + m < height && j + n >= 0 && j + n < width){
						// if the pixel coresponding to kernel[m][n] is not outside the image
						double weight = kernel[(m + kernel_size / 2) * kernel_size + (n + kernel_size / 2)];
						RGB pixel = old_data[(i + m) * width + (j + n)];
						r += pixel.r * weight;
						g += pixel.g * weight;
						b += pixel.b * weight;
					}
				}
			}
			
			// keeping the results from overflowing when casting to unsigned char
			if(r < 0) r = 0;
			else if(r > 255) r = 255;
			
			if(g < 0) g = 0;
			else if(g > 255) g = 255;
			
			if(b < 0) b = 0;
			else if(b > 255) b = 255;
			
			RGB result;
			result.r = r;
			result.g = g;
			result.b = b;
			copy_RGB(&result, &new_data[i * width + j]);
		}
	}
	
	new_img->height = height;
	new_img->width = width;
	new_img->data = new_data;
	free(kernel);
	return new_img;
}

/**
================================================================================================
	TESTBENCH

int main(){
	Image *img = readBMP("Photos\\XL.bmp");
	Image *editted_img = perform_convolution_serial(img, UNSHARP5);
	saveBMP("Photos\\Editted_XL.bmp", editted_img);
	free(img->data);
	free(img);
	return 0;
}
*/
#include "convolution.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void copy_RGB(RGB *src, RGB *dest){
	dest->r = src->r;
	dest->g = src->g;
	dest->b = src->b;
}

Image* pad_image(Image *img, int padding){
	if(padding < 0 || padding > img->height || padding > img->width){
		printf("Incompatible padding\n");
		exit(-1);
	}
	
	if(padding == 0) return img;
	
	int old_width = img->width;
	int old_height = img->height;
	RGB *old_data = img->data;
	
	int new_width = old_width + 2 * padding;
	int new_height = old_height + 2 * padding;
	
	RGB *new_data = (RGB*)malloc(new_width * new_height * sizeof(RGB));
	if(new_data == NULL){
		fprintf(stderr, "Error in pad_image while allocating memory for RGB data\n");
		exit(-1);
	}
	
	Image *new_img = (Image*)malloc(sizeof(Image));
	if(new_img == NULL){
		fprintf(stderr, "Error in pad_image while alocating memory for image\n");
		exit(-1);
	}
	
	// filling in the inner data
	for(int i = 0; i < old_height; ++i){
		for(int j = 0; j < old_width; ++j){
			copy_RGB(&old_data[i * old_width + j], &new_data[(i + padding) * new_width + (j + padding)]);
		}
	}
	
	// filling in the up and down paddings
	for(int i = 0; i < padding; ++i){
		for(int j = 0; j < old_width; ++j){
			copy_RGB(&old_data[i * old_width + j], &new_data[(padding - 1 - i) * new_width + (j + padding)]);
			copy_RGB(&old_data[(old_height - 1 - i) * old_width + j], &new_data[(new_height - 1 - (padding - 1 - i)) * new_width + (j + padding)]);
		}
	}
	
	//filling in the left and right paddings
	for(int j = 0; j < padding; ++j){
		for(int i = 0; i < old_height; ++i){
			copy_RGB(&old_data[i * old_width + j], &new_data[(i + padding) * new_width + (padding - 1 - j)]);
			copy_RGB(&old_data[i * old_width + (old_width - 1 - j)], &new_data[(i + padding) * new_width + (new_width - 1 - (padding - 1 - j))]);
		}
	}
	
	//filling in the corners
	for(int i = 0; i < padding; ++i){
		for(int j = 0; j < padding; ++j){
			copy_RGB(&old_data[(padding - 1 - i) * old_width + (padding - 1 - j)], &new_data[j * new_width + i]);
			copy_RGB(&old_data[(old_height - 1 - i) * old_width + (old_width - 1 - j)], &new_data[(new_height - 1 - (padding - 1 - j)) * new_width + (new_width - 1 - (padding - 1 - i))]);
			copy_RGB(&old_data[i * old_width + (old_width - 1 - (padding - 1 - j))], &new_data[j * new_width + (new_width - 1 - (padding - 1 - i))]);
			copy_RGB(&old_data[(old_height - 1 - (padding - 1 - i)) * old_width + j], &new_data[(new_height - 1 - (padding - 1 - j)) * new_width + i]);
		}
	}
	
	new_img->height = new_height;
	new_img->width = new_width;
	new_img -> data = new_data;
	return new_img;
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
	Image *editted_img = pad_image(img, 100);
	saveBMP("Photos\\Padded_Large.bmp", editted_img);
	return 0;
}
*/
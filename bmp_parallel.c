#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "bmp_parallel.h"

Image *readBMP_MPI(const char *file_name, int my_rank, int num_processes){
	int check, local_error_flag, global_error_flag;
	MPI_File image_file_handler;
	
	check = MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &image_file_handler);
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Error in main while opening file %s\n", file_name);
		return NULL;
	}
	
	unsigned char header[54];
	
	check = MPI_File_read_at_all(image_file_handler, 0, header, 54, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Error in main while reading from file file %s\n", file_name);
		return NULL;
	}
	
	if (header[0] != 'B' || header[1] != 'M'){
		if(my_rank == 0) fprintf(stderr, "Error: Not a valid BMP file\n");
		return NULL;
	}
	
	int bitsPerPixel = *(short *)&header[28];
	if (bitsPerPixel != 24){
		if(my_rank == 0) fprintf(stderr, "Error: Only 24-bit BMPs are supported\n");
		return NULL;
	}
	
	int width = *(int *)&header[18];
	int height = *(int *)&header[22];
	int data_offset = *(int *)&header[10];
	
	int row_padded = (width * 3 + 3) & (~3);
	int pixel_data_size = width * 3;
	int padding_size = row_padded - pixel_data_size;
	
	unsigned char *row_pixels = (unsigned char *)malloc(pixel_data_size);
	if(row_pixels == NULL) local_error_flag = 1;
	
	check = MPI_Reduce(&local_error_flag, &global_error_flag, 1, MPI_INT, MPI_LOR, 0, MPI_COMM_WORLD);
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Error in main while comunicating error flag\n");
		return NULL;
	}
	
	if(my_rank == 0 && global_error_flag == 1){
		fprintf(stderr, "Error in main while allocating memory\n");
		return NULL;
	}
	
	int local_rows = height / num_processes;
	int local_offset = data_offset + local_rows * width * my_rank;
	if(my_rank == num_processes - 1) local_rows += height % num_processes;
	
	RGB *data = (RGB *)malloc(width * local_rows * sizeof(RGB));
	if(data == NULL) local_error_flag = 1;
	
	check = MPI_Reduce(&local_error_flag, &global_error_flag, 1, MPI_INT, MPI_LOR, 0, MPI_COMM_WORLD);
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Error in main while comunicating error flag\n");
		return NULL;
	}
	
	if(my_rank == 0 && global_error_flag == 1){
		fprintf(stderr, "Error in main while allocating memory\n");
		return NULL;
	}
	
	for (int y = 0; y < local_rows; y++){
		check = MPI_File_read_at(image_file_handler, local_offset, row_pixels, pixel_data_size, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
		if(check != MPI_SUCCESS){
			if(my_rank == 0) fprintf(stderr, "Error in main while reading from file file %s\n", file_name);
			return NULL;
		}
		
		local_offset += pixel_data_size + padding_size;
		
		for (int x = 0; x < width; x++){
			data[(local_rows - 1 - y) * width + x].b = row_pixels[x * 3];
			data[(local_rows - 1 - y) * width + x].g = row_pixels[x * 3 + 1];
			data[(local_rows - 1 - y) * width + x].r = row_pixels[x * 3 + 2];
		}
	}
	
	free(row_pixels);
	check = MPI_File_close(&image_file_handler);
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Error in main while closing file %s\n", file_name);
		return NULL;
	}
	
	Image *img = (Image*)malloc(sizeof(Image));
	if(img == NULL) local_error_flag = 1;
	
	check = MPI_Reduce(&local_error_flag, &global_error_flag, 1, MPI_INT, MPI_LOR, 0, MPI_COMM_WORLD);
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Error in main while comunicating error flag\n");
		return NULL;
	}
	
	if(my_rank == 0 && global_error_flag == 1){
		fprintf(stderr, "Error in main while allocating memory\n");
		return NULL;
	}
	
	img->width = width;
	img->height = local_rows;
	img->data = data;
	return img;
}
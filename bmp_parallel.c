#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "bmp_parallel.h"

MPI_Datatype create_mpi_datatype_for_RGB(){
	RGB rgb;
	MPI_Datatype mpi_rgb;
	MPI_Datatype types[3] = {MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR};
	int block_lengths[3] = {1, 1, 1};
	MPI_Aint displacements[3];
	
	// getting field addresses
	MPI_Get_address(&rgb.r, &displacements[0]);
	MPI_Get_address(&rgb.g, &displacements[1]);
	MPI_Get_address(&rgb.b, &displacements[2]);
	
	// making displacements relative to the first field
	displacements[2] -= displacements[0];
	displacements[1] -= displacements[0];
	displacements[0] -= displacements[0];
	
	// creating struct
	MPI_Type_create_struct(3, block_lengths, displacements, types, &mpi_rgb);
	MPI_Type_commit(&mpi_rgb);
	return mpi_rgb;
}

Image *readBMP_MPI(const char *file_name, int my_rank, int num_processes, int halo_dim, int *true_start, int *true_end){
	int check, local_error_flag = 0, global_error_flag = 0;
	MPI_File image_file_handler;
	
	check = MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &image_file_handler);
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Rank %d: Error in readBMP_MPI while opening file %s\n", my_rank, file_name);
		return NULL;
	}
	
	unsigned char header[54];
	
	check = MPI_File_read_at_all(image_file_handler, 0, header, 54, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Rank %d: Error in readBMP_MPI while reading from file file %s\n", my_rank, file_name);
		return NULL;
	}
	
	if (header[0] != 'B' || header[1] != 'M'){
		if(my_rank == 0) fprintf(stderr, "Rank %d: Error in readBMP_MPI: Not a valid BMP file\n", my_rank);
		return NULL;
	}
	
	int bitsPerPixel = *(short *)&header[28];
	if (bitsPerPixel != 24){
		if(my_rank == 0) fprintf(stderr, "Rank %d: Error in readBMP_MPI: Only 24-bit BMPs are supported\n", my_rank);
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
		fprintf(stderr, "Rank %d: Error in readBMP_MPI while comunicating error flag\n", my_rank);
		return NULL;
	}
	
	if(my_rank == 0 && global_error_flag == 1){
		fprintf(stderr, "Rank %d: Error in readBMP_MPI while allocating memory\n", my_rank);
		return NULL;
	}
	
	int local_rows;
	int true_rows = height / num_processes;
	int remainder = height % num_processes;
	int rows_read_until_now = true_rows * my_rank + ((my_rank < remainder) ? my_rank : remainder);
	MPI_Offset local_offset = data_offset + rows_read_until_now * (pixel_data_size + padding_size);
	if(my_rank != 0) local_offset -= halo_dim * (pixel_data_size + padding_size); // reading the halos as well
	if(my_rank < remainder) ++true_rows; // distributing remainder uniformly
	
	// adding halo rows
	if(my_rank == 0 || my_rank == num_processes - 1) local_rows = true_rows + halo_dim;
	else local_rows = true_rows + 2 * halo_dim;

	if(my_rank > 0) *true_end = local_rows - halo_dim - 1; // end is refering to the bottom of the matrix since the bottom has the highest index
	else *true_end = local_rows - 1;
	
	if(my_rank < num_processes - 1) *true_start = halo_dim; // start is refering to the top of the matrix since the top has the lowest index
	else *true_start = 0;
	
	RGB *data = (RGB *)malloc(width * local_rows * sizeof(RGB));
	if(data == NULL) local_error_flag = 1;
	
	check = MPI_Reduce(&local_error_flag, &global_error_flag, 1, MPI_INT, MPI_LOR, 0, MPI_COMM_WORLD);
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Rank %d: Error in readBMP_MPI while comunicating error flag\n", my_rank);
		return NULL;
	}
	
	if(my_rank == 0 && global_error_flag == 1){
		fprintf(stderr, "Rank %d: Error in readBMP_MPI while allocating memory\n", my_rank);
		return NULL;
	}
	
	for (int y = 0; y < local_rows; y++){
		check = MPI_File_read_at(image_file_handler, local_offset, row_pixels, pixel_data_size, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
		if(check != MPI_SUCCESS){
			if(my_rank == 0) fprintf(stderr, "Rank %d: Error in readBMP_MPI while reading from file file %s\n", my_rank, file_name);
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
		fprintf(stderr, "Rank %d: Error in readBMP_MPI while closing file %s\n", my_rank, file_name);
		return NULL;
	}
	
	Image *img = (Image*)malloc(sizeof(Image));
	if(img == NULL) local_error_flag = 1;
	
	check = MPI_Reduce(&local_error_flag, &global_error_flag, 1, MPI_INT, MPI_LOR, 0, MPI_COMM_WORLD);
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Rank %d: Error in readBMP_MPI while comunicating error flag\n", my_rank);
		return NULL;
	}
	
	if(my_rank == 0 && global_error_flag == 1){
		fprintf(stderr, "Rank %d: Error in readBMP_MPI while allocating memory\n", my_rank);
		return NULL;
	}
	
	img->width = width;
	img->height = local_rows;
	img->data = data;
	return img;
}

Image *compose_BMP(Image *img, int my_rank, int num_processes){
	int check;
	int width = img->width;
	int total_height = 0;
	int *heights = NULL;
	int *receives = NULL;
	int *displacements = NULL;
	Image *new_img = NULL;
	RGB *data = NULL;
	
	if(my_rank == 0){
		heights = (int*)malloc(num_processes * sizeof(int));
		if(heights == NULL){
			fprintf(stderr, "Rank %d: Error in compose_BMP while allocating memory\n", my_rank);
			return NULL;
		}
	}
	
	check = MPI_Gather(&(img->height), 1, MPI_INT, heights, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Rank %d: Error in compose_BMP while comunicating height\n", my_rank);
		return NULL;
	}
	
	if(my_rank == 0){
		new_img = (Image*)malloc(sizeof(Image));
		if(new_img == NULL){
			fprintf(stderr, "Rank %d: Error in compose_BMP while allocating memory\n", my_rank);
			return NULL;
		}
		
		displacements = (int*)malloc(num_processes * sizeof(int));
		if(displacements == NULL){
			fprintf(stderr, "Rank %d: Error in compose_BMP while allocating memory\n", my_rank);
			return NULL;
		}
		
		receives = (int*)malloc(num_processes * sizeof(int));
		if(receives == NULL){
			fprintf(stderr, "Rank %d: Error in compose_BMP while allocating memory\n", my_rank);
			return NULL;
		}

		for(int i = 0; i < num_processes; ++i){
			total_height += heights[i];
		}
		
		data = (RGB *)malloc(width * total_height * sizeof(RGB));
		if(data == NULL){
			fprintf(stderr, "Rank %d: Error in compose_BMP while allocating memory\n", my_rank);
			return NULL;
		}

		MPI_Offset offset = width * total_height;
		for(int i = 0; i < num_processes; ++i){
			offset -= heights[i] * width;
			displacements[i] = offset;
			receives[i] = heights[i] * width;
		}
	}
	
	MPI_Datatype mpi_rgb = create_mpi_datatype_for_RGB();
	
	check = MPI_Gatherv(
		img->data, img->height * width, mpi_rgb,
		data, receives, displacements, mpi_rgb,
		0, MPI_COMM_WORLD
	);
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Rank %d: Error in compose_BMP while comunicating data\n", my_rank);
		return NULL;
	}
	
	check = MPI_Type_free(&mpi_rgb);
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Rank %d: Error in compose_BMP while de-allocating data type\n", my_rank);
		return NULL;
	}
	
	if(my_rank == 0){
		new_img->data = data;
		new_img->width = width;
		new_img->height = total_height;
		return new_img;
	}
	else{
		return img;
	}
}
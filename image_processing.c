#include <stdio.h>
#include <stdlib.h>
#include "bmp.h"
#include "convolution.h"

int image_processing_serial(const char *in_file_name, const char *out_file_name, operation_t operation){
	Image *img = readBMP_serial(in_file_name);
	if(img == NULL){ // error message was printed by the called function
		return -1;
	}
	
	Image *editted_img = perform_convolution_serial(img, operation);
	free(img->data);
	free(img);
	
	if(editted_img == NULL){ // error message was printed by the called function
		return -1;
	}
	
	int exit_code = saveBMP(out_file_name, editted_img);
	free(editted_img->data);
	free(editted_img);
	
	if(exit_code == 0){ // error message was printed by the called function
		return -1;
	}
	
	return 0;
}

int image_processing_parallel_sft(const char *in_file_name, const char *out_file_name, operation_t operation, int my_rank, int num_processes, int num_cores){
	int kernel_size = get_kernel_size(operation);
	int halo_dim = kernel_size / 2;
	int true_start, true_end;
	
	Image *img = readBMP_MPI(in_file_name, my_rank, num_processes, halo_dim, &true_start, &true_end);
	if(img == NULL){ // error message was printed by the called function
		return -1;
	}
	
	int num_threads = max(1, num_cores / num_processes);
	Image *edited_img = perform_convolution_parallel(img, operation, true_start, true_end, num_threads);
	if(edited_img == NULL){ // error message was printed by the called function
		return -1;
	}
	
	free(img->data);
	free(img);
	
	Image *composed_img = compose_BMP(edited_img, my_rank, num_processes);
	if(composed_img == NULL){ // error message was printed by the called function
		return -1;
	}
	
	free(edited_img->data);
	free(edited_img);
	
	if(my_rank == 0){
		int exit_code = saveBMP(out_file_name, composed_img);
		if(exit_code == 0){ // error message was printed by the called function
			return -1;
		}
		
		free(composed_img->data);
		free(composed_img);
	}
	
	return 0;
}

int image_processing_parallel_no_sft(const char *in_file_name, const char *out_file_name, operation_t operation, int my_rank, int num_processes, int num_cores, int num_workstations){
	int kernel_size = get_kernel_size(operation);
	int halo_dim = kernel_size / 2;
	int check;
	int height, width;
	int local_height;
	RGB *data = NULL;
	Image *img = NULL;
	
	if(my_rank == 0){
		img = readBMP_serial(in_file_name);
		if(img == NULL){ // error message was printed by the called function
			return -1;
		}
		
		width = img->width;
		height = img->height;
	}
	
	// broadcasting width
	check = MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Rank %d: Error in main while broadcasting width\n", my_rank);
		return -1;
	}
	
	check = scatter_data(img, &data, my_rank, num_processes, width, &local_height, halo_dim);
	if(check != 0){ // error message was printed by the called function
		return -1;
	}
	
	if(my_rank == 0){
		free(img->data);
		free(img);
	}
	
	Image *new_image = (Image*)malloc(sizeof(Image));
	if(new_image == NULL){
		fprintf(stderr, "Rank %d: Error in main while allocating memory\n", my_rank);
		return -1;
	}
	
	new_image->height = local_height;
	new_image->width = width;
	new_image->data = data;
	
	int true_start, true_end;
	if(my_rank > 0) true_start = halo_dim;
	else true_start = 0;
	if(my_rank < num_processes - 1) true_end = local_height - halo_dim - 1;
	else true_end = local_height - 1;
	
	int num_threads = max(1, num_cores / (num_processes / num_workstations));
	
	Image *edited_img = perform_convolution_parallel(new_image, operation, true_start, true_end, num_threads);
	if(edited_img == NULL){ // error message was printed by the called function
		return -1;
	}
	
	Image *composed_img = compose_BMP(edited_img, my_rank, num_processes);
	if(composed_img == NULL){ // error message was printed by the called function
		return -1;
	}
	
	free(edited_img->data);
	free(edited_img);
	
	if(my_rank == 0){
		int exit_code = saveBMP(out_file_name, composed_img);
		if(exit_code == 0){ // error message was printed by the called function
			return -1;
		}

		free(composed_img->data);
		free(composed_img);
	}
	
	return 0;
}
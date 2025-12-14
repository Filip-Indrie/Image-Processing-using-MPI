#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "bmp.h"
#include "convolution.h"

Image *image_processing_serial(const char *in_file_name, const char *out_file_name, operation_t operation, int save){
	Image *img = readBMP_serial(in_file_name);
	if(img == NULL){ // error message was printed by the called function
		return NULL;
	}
	
	Image *edited_img = perform_convolution_serial(img, operation);
	free(img->data);
	free(img);
	
	if(edited_img == NULL){ // error message was printed by the called function
		return NULL;
	}
	
	if(save == 1){
		int exit_code = saveBMP(out_file_name, edited_img);
		if(exit_code == 0){ // error message was printed by the called function
			return NULL;
		}
	}
	return edited_img;
}

Image *image_processing_parallel_sft(const char *in_file_name, const char *out_file_name, operation_t operation, int my_rank, int num_processes, int num_cores, int save){
	int kernel_size = get_kernel_size(operation);
	int halo_dim = kernel_size / 2;
	int true_start, true_end;
	
	Image *img = readBMP_MPI(in_file_name, my_rank, num_processes, halo_dim, &true_start, &true_end);
	if(img == NULL){ // error message was printed by the called function
		return NULL;
	}
	
	int num_threads = max(1, num_cores / num_processes);
	Image *edited_img = perform_convolution_parallel(img, operation, true_start, true_end, num_threads);
	if(edited_img == NULL){ // error message was printed by the called function
		return NULL;
	}
	
	free(img->data);
	free(img);
	
	Image *composed_img = compose_BMP(edited_img, my_rank, num_processes);
	if(composed_img == NULL){ // error message was printed by the called function
		return NULL;
	}
	
	free(edited_img->data);
	free(edited_img);
	
	if(save == 1 && my_rank == 0){
		int exit_code = saveBMP(out_file_name, composed_img);
		if(exit_code == 0){ // error message was printed by the called function
			return NULL;
		}
	}
	
	return composed_img;
}

Image *image_processing_parallel_no_sft(const char *in_file_name, const char *out_file_name, operation_t operation, int my_rank, int num_processes, int num_cores, int num_workstations, int save){
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
			return NULL;
		}
		
		width = img->width;
		height = img->height;
	}
	
	// broadcasting width
	check = MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Rank %d: Error in main while broadcasting width\n", my_rank);
		fflush(stderr);
		return NULL;
	}
	
	check = scatter_data(img, &data, my_rank, num_processes, width, &local_height, halo_dim);
	if(check != 0){ // error message was printed by the called function
		return NULL;;
	}
	
	if(my_rank == 0){
		free(img->data);
		free(img);
	}
	
	Image *new_image = (Image*)malloc(sizeof(Image));
	if(new_image == NULL){
		fprintf(stderr, "Rank %d: Error in main while allocating memory\n", my_rank);
		fflush(stderr);
		return NULL;
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
		return NULL;
	}
	
	Image *composed_img = compose_BMP(edited_img, my_rank, num_processes);
	if(composed_img == NULL){ // error message was printed by the called function
		return NULL;
	}
	
	free(edited_img->data);
	free(edited_img);
	
	if(save == 1 && my_rank == 0){
		int exit_code = saveBMP(out_file_name, composed_img);
		if(exit_code == 0){ // error message was printed by the called function
			return NULL;
		}
	}
	
	return composed_img;
}

int images_are_identical(Image *img1, Image *img2){
	if(img1->height != img2->height || img1->width != img2->width) return 0;
	
	for(int i = 0; i < img1->height; ++i){
		for(int j = 0; j < img1->width; ++j){
			if(equal_RGB((img1->data)[i * img1->width + j], (img2->data)[i * img2->width + j]) == 0) return 0;
		}
	}
	
	return 1;
}

int image_is_correct(Image *img, char *in_file_name, char *out_file_name, operation_t operation, double parallel_time){
	Image *serial_edited_image;
	double start_serial, end_serial;
	
	start_serial = omp_get_wtime();
	serial_edited_image = image_processing_serial(in_file_name, out_file_name, operation, 0);
	end_serial = omp_get_wtime();
	if(serial_edited_image == NULL){ // error message was printed by the called function
		return -1;
	}
	
	if(images_are_identical(serial_edited_image, img) == 0){
		fprintf(stdout, "The parallel edited image and the serial edited image are NOT identical!\n");
		fflush(stdout);
		
		char file_name[256] = "";
		char *aux = strrchr(out_file_name, '\\');
		int index = aux - out_file_name;
		strncat(file_name, out_file_name, index + 1);
		file_name[index + 1] = '\0';
		strcat(file_name, "Serial_");
		strcat(file_name, aux + 1);
		fprintf(stdout, "Saving the serial edited image under the name Serial_%s...\n", file_name);
		fflush(stdout);
		
		int exit_code = saveBMP(file_name, serial_edited_image);
		if(exit_code == 0){ // error message was printed by the called function
			return -1;
		}
	}
	else{
		fprintf(stdout, "The parallel edited image and the serial edited image are identical.\n");
		fprintf(stdout, "Serial Time: %f\n", end_serial - start_serial);
		fprintf(stdout, "Parallel Time: %f\n", parallel_time);
		fprintf(stdout, "Speedup: %f\n\n", (end_serial - start_serial) / parallel_time);
		fflush(stdout);
	}
	
	return 0;
}
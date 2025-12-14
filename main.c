#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "mpi.h"
#include "convolution.h"
#include "bmp_parallel.h"
#include "bmp_serial.h"

#define NUM_CORES 16
#define NUM_WORKSTATIONS 5 

int main(int argc, char **argv){
	MPI_Init(&argc, &argv);
	int my_rank, num_processes;
	operation_t operation;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
	
	if(argc != 5 && argc != 6){
		if(my_rank == 0) printf("Usage: %s [version = {`serial`, `parallel`}] [file_in] [file_out] [operation = {`RIDGE`, `EDGE`, `SHARPEN`, `BOXBLUR`, `GAUSSBLUR3`, `GAUSSBLUR5`, `UNSHARP5`}] [shared_file_tree = {`0` = False, `1` = True}]", argv[0]);
		MPI_Finalize();
		return 0;
	}
	
	if((stricmp(argv[1], "serial") != 0 && argc == 5) || (stricmp(argv[1], "parallel") != 0 && argc == 6)){
		if(my_rank == 0) printf("Invalid version\n");
		MPI_Finalize();
		return 0;
	}
	
	operation = string_to_operation(argv[4]);
	if(operation == -1){
		if(my_rank == 0) printf("Invalid operation\n");
		MPI_Finalize();
		return 0;
	}
	
	int shared_file_tree;
	if(stricmp(argv[5], "0") == 0) shared_file_tree = 0;
	else if (stricmp(argv[5], "1") == 0) shared_file_tree = 1;
	else shared_file_tree = -1;
	if(stricmp(argv[1], "parallel") == 0 && shared_file_tree == -1){
		if(my_rank == 0) printf("Invalid shared_file_tree argument\n");
		MPI_Finalize();
		return 0;
	}
	
	if(argc == 5){
		if(my_rank != 0){
			MPI_Finalize();
			return 0;
		}
		
		Image *img = readBMP_serial(argv[2]);
		if(img == NULL){ // error message was printed by the called function
			MPI_Abort(MPI_COMM_WORLD, -1);
		}
		
		Image *editted_img = perform_convolution_serial(img, operation);
		if(editted_img == NULL){ // error message was printed by the called function
			MPI_Abort(MPI_COMM_WORLD, -1);
		}
		
		int exit_code = saveBMP(argv[3], editted_img);
		if(exit_code == 0){ // error message was printed by the called function
			MPI_Abort(MPI_COMM_WORLD, -1);
		}
		
		free(img->data);
		free(img);
	}
	else{
		if(shared_file_tree == 1){
			int kernel_size = get_kernel_size(operation);
			int halo_dim = kernel_size / 2;
			int check;
			
			int true_start, true_end; // rows at which THIS PROCESSES image start and ends (inclusive)
			// start is refering to the top of the matrix since the top has the lowest index
			// end is refering to the bottom of the matrix since the bottom has the highest index
			
			Image *img = readBMP_MPI(argv[2], my_rank, num_processes, halo_dim, &true_start, &true_end);
			if(img == NULL){ // error message was printed by the called function
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
			
			int num_threads = max(1, NUM_CORES / num_processes);
			Image *edited_img = perform_convolution_parallel(img, operation, true_start, true_end, num_threads);
			if(edited_img == NULL){ // error message was printed by the called function
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
			
			Image *composed_img = compose_BMP(edited_img, my_rank, num_processes);
			if(composed_img == NULL){ // error message was printed by the called function
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
			
			if(my_rank == 0){
				int exit_code = saveBMP(argv[3], composed_img);
				if(exit_code == 0){ // error message was printed by the called function
					MPI_Abort(MPI_COMM_WORLD, -1);
				}
			}
			free(img->data);
			free(img);
			free(edited_img->data);
			free(edited_img);
			if(my_rank == 0){
				free(composed_img->data);
				free(composed_img);
			}
		}
		else{
			/**
			*	TO BE IMPLEMENTED
			*/
		}
	}
	
	MPI_Finalize();
	return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "mpi.h"
#include "convolution.h"
#include "bmp.h"
#include "image_processing.h"

#define NUM_CORES 16
#define NUM_WORKSTATIONS 5 

#define OPTIMAL_CHUNK_SIZE 200

int main(int argc, char **argv){
	MPI_Init(&argc, &argv);
	int my_rank, num_processes;
	operation_t operation;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
	
	if(argc != 5 && argc != 6){
		if(my_rank == 0){
			fprintf(stdout, "Usage: %s [version = {`serial`, `parallel`, `master`}] [file_in] [file_out] [operation = {`RIDGE`, `EDGE`, `SHARPEN`, `BOXBLUR`, `GAUSSBLUR3`, `GAUSSBLUR5`, `UNSHARP5`}] [shared_file_tree = {`0` = False, `1` = True}]", argv[0]);
			fflush(stdout);
		}
		MPI_Finalize();
		return 0;
	}
	
	if(((stricmp(argv[1], "serial") != 0 && stricmp(argv[1], "master") != 0) && argc == 5) || (stricmp(argv[1], "parallel") != 0 && argc == 6)){
		if(my_rank == 0){
			fprintf(stdout, "Invalid version\n");
			fflush(stdout);
		}
		MPI_Finalize();
		return 0;
	}
	
	operation = string_to_operation(argv[4]);
	if(operation == -1){
		if(my_rank == 0){
			fprintf(stdout, "Invalid operation\n");
			fflush(stdout);
		}
		MPI_Finalize();
		return 0;
	}
	
	int shared_file_tree;
	if(stricmp(argv[5], "0") == 0) shared_file_tree = 0;
	else if (stricmp(argv[5], "1") == 0) shared_file_tree = 1;
	else shared_file_tree = -1;
	
	if(stricmp(argv[1], "parallel") == 0 && shared_file_tree == -1){
		if(my_rank == 0){
			fprintf(stdout, "Invalid shared_file_tree argument\n");
			fflush(stdout);
		}
		MPI_Finalize();
		return 0;
	}
	
	/**
	*	mode = 0 --> serial
	*	mode = 1 --> master/worker
	*/
	int mode;
	if(argc == 5){
		if(stricmp(argv[1], "serial") == 0) mode = 0;
		else mode = 1;
	}
	
	if(argc == 5){
		if(mode == 0){ // serial
			if(my_rank != 0){
				MPI_Finalize();
				return 0;
			}
			
			Image *edited_img = image_processing_serial(argv[2], argv[3], operation, 1);
			if(edited_img == NULL){ // error message was printed by the called function
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
			
			free(edited_img->data);
			free(edited_img);
		}
		else{ // master/worker
			int chunk = 100;
			Image *parallel_edited_image;
			double start_parallel, end_parallel;
			
			if(my_rank == 0) start_parallel = omp_get_wtime();
			parallel_edited_image = image_processing_master(argv[2], argv[3], operation, OPTIMAL_CHUNK_SIZE, my_rank, num_processes, NUM_CORES, NUM_WORKSTATIONS, 1);
			if(my_rank == 0) end_parallel = omp_get_wtime();
			if(parallel_edited_image == NULL){ // error message was printed by the called function
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
			
			if(my_rank != 0){
				free(parallel_edited_image);
				MPI_Finalize();
				return 0;
			}
			
			int check = image_is_correct(parallel_edited_image, argv[2], argv[3], operation, end_parallel - start_parallel);
			if(check != 0){
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
			
			free(parallel_edited_image->data);
			free(parallel_edited_image);
		}
	}
	else{
		if(shared_file_tree == 1){
			Image *parallel_edited_image;
			double start_parallel, end_parallel;
			
			if(my_rank == 0) start_parallel = omp_get_wtime();
			parallel_edited_image = image_processing_parallel_sft(argv[2], argv[3], operation, my_rank, num_processes, NUM_CORES, 1);
			if(my_rank == 0) end_parallel = omp_get_wtime();
			if(parallel_edited_image == NULL){ // error message was printed by the called function
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
			
			if(my_rank != 0){
				// processes already deallocated parallel_edited_image
				MPI_Finalize();
				return 0;
			}
			
			int check = image_is_correct(parallel_edited_image, argv[2], argv[3], operation, end_parallel - start_parallel);
			if(check != 0){
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
			
			free(parallel_edited_image->data);
			free(parallel_edited_image);
		}
		else{
			Image *parallel_edited_image;
			double start_parallel, end_parallel;
			
			if(my_rank == 0) start_parallel = omp_get_wtime();
			parallel_edited_image = image_processing_parallel_no_sft(argv[2], argv[3], operation, my_rank, num_processes, NUM_CORES, NUM_WORKSTATIONS, 1);
			if(my_rank == 0) end_parallel = omp_get_wtime();
			if(parallel_edited_image == NULL){ // error message was printed by the called function
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
			
			if(my_rank != 0){
				MPI_Finalize();
				return 0;
			}
			
			int check = image_is_correct(parallel_edited_image, argv[2], argv[3], operation, end_parallel - start_parallel);
			if(check != 0){
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
			
			free(parallel_edited_image->data);
			free(parallel_edited_image);
		}
	}
	
	MPI_Finalize();
	return 0;
}
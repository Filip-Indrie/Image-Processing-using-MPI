#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "mpi.h"
#include "convolution.h"
#include "bmp_parallel.h"
#include "bmp_serial.h"

/**
*	TODO:
*		- make the operation selectable through command line args
		- choose between serial and parallel through command line args
		- if parallel, specify if there is a shared file tree or not through command line args
		- make different versions of pad_image and generate_kernel since when being paralleled, when a process encounteres an error, process 0 should terminate all its children and exit.
*/

// main 'serial'/'parallel' file_in file_out operation [shared_file_tree = {0 = False, 1 = True}]

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
			Image *img = readBMP_MPI(argv[2], my_rank, num_processes);
			if(img == NULL){ // error message was printed by the called function
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
			if(my_rank == 0){
				// THIS WILL SAVE ONLY THE DATA THIS PROCESS READ (THE FIRST HEIGHT/NUM_PROCESSES ROWS)
				int exit_code = saveBMP(argv[3], img);
				if(exit_code == 0){ // error message was printed by the called function
					MPI_Abort(MPI_COMM_WORLD, -1);
				}
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
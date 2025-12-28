#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "mpi.h"
#include "convolution.h"
#include "bmp.h"
#include "image_processing.h"

#define NUM_CORES 16
#define NUM_WORKSTATIONS 1 

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
		MPI_Abort(MPI_COMM_WORLD, -1);
	}
	
	if(((stricmp(argv[1], "serial") != 0 && stricmp(argv[1], "master") != 0) && argc == 5) || (stricmp(argv[1], "parallel") != 0 && argc == 6)){
		if(my_rank == 0){
			fprintf(stdout, "Invalid version\n");
			fflush(stdout);
		}
		MPI_Abort(MPI_COMM_WORLD, -1);
	}
	
	operation = string_to_operation(argv[4]);
	if(operation == -1){
		if(my_rank == 0){
			fprintf(stdout, "Invalid operation\n");
			fflush(stdout);
		}
		MPI_Abort(MPI_COMM_WORLD, -1);
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
		MPI_Abort(MPI_COMM_WORLD, -1);
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
			
			Image *edited_img = image_processing_serial(argv[2], operation);
			if(edited_img == NULL){ // error message was printed by the called function
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
			
			int check = save_BMP(argv[3], edited_img);
			if(check == -1){ // error message was printed by the called function
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
			
			free(edited_img->data);
			free(edited_img);
		}
		else{ // master/worker
			int chunk = 100;
			Image *parallel_edited_image, *serial_edited_image;
			double parallel_time, serial_time;
			int check;
			
			if(my_rank == 0) parallel_time = omp_get_wtime();
			parallel_edited_image = image_processing_master(argv[2], operation, OPTIMAL_CHUNK_SIZE, my_rank, num_processes, NUM_CORES, NUM_WORKSTATIONS);
			if(my_rank == 0) parallel_time = omp_get_wtime() - parallel_time;
			if(parallel_edited_image == NULL){ // error message was printed by the called function
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
			
			if(my_rank != 0){
				free(parallel_edited_image);
				MPI_Finalize();
				return 0;
			}
			
			check = save_BMP(argv[3], parallel_edited_image);
			if(check == -1){ // error message was printed by the called function
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
			
			serial_time = omp_get_wtime();
			serial_edited_image = image_processing_serial(argv[2], operation);
			serial_time = omp_get_wtime() - serial_time;
			if(serial_edited_image == NULL){ // error message was printed by the called function
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
			
			if(images_are_identical(serial_edited_image, parallel_edited_image) == 1){
				fprintf(stdout, "The serial and parallel edited images are identical.\n");
				fprintf(stdout, "Serial time: %f\n", serial_time);
				fprintf(stdout, "Master/Worker time: %f\n", parallel_time);
				fprintf(stdout, "Speedup: %f\n\n", serial_time / parallel_time);
				fflush(stdout);
			}
			else{
				fprintf(stdout, "The serial and master/worker edited images are NOT identical!\n");
				fflush(stdout);
				
				char file_name[256] = "";
				char *aux = strrchr(argv[3], '\\');
				int index = aux - argv[3];
				strncat(file_name, argv[3], index + 1);
				file_name[index + 1] = '\0';
				strcat(file_name, "Serial_");
				strcat(file_name, aux + 1);
				fprintf(stdout, "Saving the serial edited image at %s...\n", file_name);
				fflush(stdout);

				int exit_code = save_BMP(file_name, serial_edited_image);
				if(exit_code == -1){ // error message was printed by the called function
					MPI_Abort(MPI_COMM_WORLD, -1);
				}
			}
			
			free(parallel_edited_image->data);
			free(parallel_edited_image);
			free(serial_edited_image->data);
			free(serial_edited_image);
		}
	}
	else{
		if(shared_file_tree == 1){
			Image *parallel_edited_image, *serial_edited_image;
			double parallel_time, serial_time;
			int check;
			
			if(my_rank == 0) parallel_time = omp_get_wtime();
			parallel_edited_image = image_processing_parallel_sft(argv[2], operation, my_rank, num_processes, NUM_CORES);
			if(my_rank == 0) parallel_time = omp_get_wtime() - parallel_time;
			if(parallel_edited_image == NULL){ // error message was printed by the called function
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
			
			if(my_rank != 0){
				// processes already deallocated parallel_edited_image
				MPI_Finalize();
				return 0;
			}
			
			check = save_BMP(argv[3], parallel_edited_image);
			if(check == -1){ // error message was printed by the called function
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
			
			serial_time = omp_get_wtime();
			serial_edited_image = image_processing_serial(argv[2], operation);
			serial_time = omp_get_wtime() - serial_time;
			if(serial_edited_image == NULL){ // error message was printed by the called function
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
			
			if(images_are_identical(serial_edited_image, parallel_edited_image) == 1){
				fprintf(stdout, "The serial and parallel edited images are identical.\n");
				fprintf(stdout, "Serial time: %f\n", serial_time);
				fprintf(stdout, "Parallel SFT time: %f\n", parallel_time);
				fprintf(stdout, "Speedup: %f\n\n", serial_time / parallel_time);
				fflush(stdout);
			}
			else{
				fprintf(stdout, "The serial and parallel sft edited images are NOT identical!\n");
				fflush(stdout);
				
				char file_name[256] = "";
				char *aux = strrchr(argv[3], '\\');
				int index = aux - argv[3];
				strncat(file_name, argv[3], index + 1);
				file_name[index + 1] = '\0';
				strcat(file_name, "Serial_");
				strcat(file_name, aux + 1);
				fprintf(stdout, "Saving the serial edited image at %s...\n", file_name);
				fflush(stdout);

				int exit_code = save_BMP(file_name, serial_edited_image);
				if(exit_code == -1){ // error message was printed by the called function
					MPI_Abort(MPI_COMM_WORLD, -1);
				}
			}
			
			free(parallel_edited_image->data);
			free(parallel_edited_image);
			free(serial_edited_image->data);
			free(serial_edited_image);
		}
		else{
			Image *parallel_edited_image, *serial_edited_image;
			double serial_time, parallel_time;
			int check;
			
			if(my_rank == 0) parallel_time = omp_get_wtime();
			parallel_edited_image = image_processing_parallel_no_sft(argv[2], operation, my_rank, num_processes, NUM_CORES, NUM_WORKSTATIONS);
			if(my_rank == 0) parallel_time = omp_get_wtime() - parallel_time;
			if(parallel_edited_image == NULL){ // error message was printed by the called function
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
			
			if(my_rank != 0){
				// processes already deallocated parallel_edited_image
				MPI_Finalize();
				return 0;
			}
			
			check = save_BMP(argv[3], parallel_edited_image);
			if(check == -1){ // error message was printed by the called function
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
			
			serial_time = omp_get_wtime();
			serial_edited_image = image_processing_serial(argv[2], operation);
			serial_time = omp_get_wtime() - serial_time;
			if(serial_edited_image == NULL){ // error message was printed by the called function
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
			
			if(images_are_identical(serial_edited_image, parallel_edited_image) == 1){
				fprintf(stdout, "The serial and parallel edited images are identical.\n");
				fprintf(stdout, "Serial time: %f\n", serial_time);
				fprintf(stdout, "Parallel NO SFT time: %f\n", parallel_time);
				fprintf(stdout, "Speedup: %f\n\n", serial_time / parallel_time);
				fflush(stdout);
			}
			else{
				fprintf(stdout, "The serial and parallel no sft edited images are NOT identical!\n");
				fflush(stdout);
				
				char file_name[256] = "";
				char *aux = strrchr(argv[3], '\\');
				int index = aux - argv[3];
				strncat(file_name, argv[3], index + 1);
				file_name[index + 1] = '\0';
				strcat(file_name, "Serial_");
				strcat(file_name, aux + 1);
				fprintf(stdout, "Saving the serial edited image at %s...\n", file_name);
				fflush(stdout);

				int exit_code = save_BMP(file_name, serial_edited_image);
				if(exit_code == -1){ // error message was printed by the called function
					MPI_Abort(MPI_COMM_WORLD, -1);
				}
			}
			
			free(parallel_edited_image->data);
			free(parallel_edited_image);
			free(serial_edited_image->data);
			free(serial_edited_image);
		}
	}
	
	MPI_Finalize();
	return 0;
}
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

#define CHUNK_START 20
#define CHUNK_STEP 10
#define CHUNK_END 1000

#define OPERATION GAUSSBLUR5

int experiment_with_image(char *in_file_name, char *out_file_name, char *measurements_file, int my_rank, int num_processes){
	Image *serial_edited_img, *parallel_sft_edited_img, *parallel_no_sft_edited_img, *master_edited_img;
	int num_chunk_sizes = (CHUNK_END - CHUNK_START) / CHUNK_STEP + 1;
	double serial_time, parallel_sft_time, parallel_no_sft_time, master_time[num_chunk_sizes];
	double optimal_chunk_time = INT_MAX;
	int optimal_chunk_size = -1;
	
	if(my_rank == 0){
		fprintf(stdout, "%s --> Serial\n", in_file_name);
		fflush(stdout);
		
		serial_time = omp_get_wtime();
		serial_edited_img = image_processing_serial(in_file_name, OPERATION);
		serial_time = omp_get_wtime() - serial_time;
		if(serial_edited_img == NULL){ // error message was printed by the called function
			return -1;
		}
	}
	
	if(my_rank == 0){
		fprintf(stdout, "%s --> Parallel SFT\n", in_file_name);
		fflush(stdout);
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	if(my_rank == 0) parallel_sft_time = omp_get_wtime();
	parallel_sft_edited_img = image_processing_parallel_sft(in_file_name, OPERATION, my_rank, num_processes, NUM_CORES);
	if(my_rank == 0) parallel_sft_time = omp_get_wtime() - parallel_sft_time;
	if(parallel_sft_edited_img == NULL){ // error message was printed by the called function
		return -1;
	}
	
	if(my_rank == 0 && images_are_identical(serial_edited_img, parallel_sft_edited_img) == 0){
		fprintf(stdout, "The serial and parallel sft edited images are NOT identical!\n");
		fflush(stdout);
		
		fprintf(stdout, "Saving the parallel sft edited image...\n");
		fflush(stdout);
		
		int exit_code;
		exit_code = save_BMP(out_file_name, parallel_sft_edited_img);
		if(exit_code == -1){ // error message was printed by the called function
			return -1;
		}
		
		char file_name[256] = "";
		char *aux = strrchr(out_file_name, '\\');
		int index = aux - out_file_name;
		strncat(file_name, out_file_name, index + 1);
		file_name[index + 1] = '\0';
		strcat(file_name, "Serial_");
		strcat(file_name, aux + 1);
		
		fprintf(stdout, "Saving the serial edited image at %s...\n", file_name);
		fflush(stdout);

		exit_code = save_BMP(file_name, serial_edited_img);
		if(exit_code == -1){ // error message was printed by the called function
			return -1;
		}
	}
	
	if(my_rank == 0){
		free(parallel_sft_edited_img->data);
		free(parallel_sft_edited_img);
	}
	
	if(my_rank == 0){
		fprintf(stdout, "%s --> Parallel NO SFT\n", in_file_name);
		fflush(stdout);
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	if(my_rank == 0) parallel_no_sft_time = omp_get_wtime();
	parallel_no_sft_edited_img = image_processing_parallel_no_sft(in_file_name, OPERATION, my_rank, num_processes, NUM_CORES, NUM_WORKSTATIONS);
	if(my_rank == 0) parallel_no_sft_time = omp_get_wtime() - parallel_no_sft_time;
	if(parallel_no_sft_edited_img == NULL){ // error message was printed by the called function
		return -1;
	}
	
	if(my_rank == 0 && images_are_identical(serial_edited_img, parallel_no_sft_edited_img) == 0){
		fprintf(stdout, "The serial and parallel no sft edited images are NOT identical!\n");
		fflush(stdout);
		
		fprintf(stdout, "Saving the parallel no sft edited image...\n");
		fflush(stdout);
		
		int exit_code;
		exit_code = save_BMP(out_file_name, parallel_no_sft_edited_img);
		if(exit_code == -1){ // error message was printed by the called function
			return -1;
		}
		
		char file_name[256] = "";
		char *aux = strrchr(out_file_name, '\\');
		int index = aux - out_file_name;
		strncat(file_name, out_file_name, index + 1);
		file_name[index + 1] = '\0';
		strcat(file_name, "Serial_");
		strcat(file_name, aux + 1);
		
		fprintf(stdout, "Saving the serial edited image at %s...\n", file_name);
		fflush(stdout);

		exit_code = save_BMP(file_name, serial_edited_img);
		if(exit_code == -1){ // error message was printed by the called function
			return -1;
		}
	}
	
	if(my_rank == 0){
		free(parallel_no_sft_edited_img->data);
		free(parallel_no_sft_edited_img);
	}
	
	if(my_rank == 0){
		fprintf(stdout, "%s --> Master/Worker\n", in_file_name);
		fflush(stdout);
	}
	
	for(int chunk = CHUNK_START; chunk <= CHUNK_END; chunk += CHUNK_STEP){
		int index = (chunk - CHUNK_START) / CHUNK_STEP;
		if(my_rank == 0){
			fprintf(stdout, "Chunk %d\n", chunk);
			fflush(stdout);
		}
		
		MPI_Barrier(MPI_COMM_WORLD);
		if(my_rank == 0) master_time[index] = omp_get_wtime();
		master_edited_img = image_processing_master(in_file_name, OPERATION, chunk, my_rank, num_processes, NUM_CORES, NUM_WORKSTATIONS);
		if(my_rank == 0) master_time[index] = omp_get_wtime() - master_time[index];
		if(master_edited_img == NULL){ // error message was printed by the called function
			return -1;
		}
		
		if(my_rank == 0 && images_are_identical(serial_edited_img, master_edited_img) == 0){
			fprintf(stdout, "The serial and master/worker edited images are NOT identical!\n");
			fflush(stdout);
			
			fprintf(stdout, "Saving the master/worker edited image...\n");
			fflush(stdout);
			
			int exit_code;
			exit_code = save_BMP(out_file_name, master_edited_img);
			if(exit_code == -1){ // error message was printed by the called function
				return -1;
			}
			
			char file_name[256] = "";
			char *aux = strrchr(out_file_name, '\\');
			int index = aux - out_file_name;
			strncat(file_name, out_file_name, index + 1);
			file_name[index + 1] = '\0';
			strcat(file_name, "Serial_");
			strcat(file_name, aux + 1);
			
			fprintf(stdout, "Saving the serial edited image at %s...\n", file_name);
			fflush(stdout);

			exit_code = save_BMP(file_name, serial_edited_img);
			if(exit_code == -1){ // error message was printed by the called function
				return -1;
			}
		}
		
		if(my_rank == 0) free(master_edited_img->data);
		free(master_edited_img);
		
		if(master_time[index] < optimal_chunk_time){
			optimal_chunk_time = master_time[index];
			optimal_chunk_size = chunk;
		}
	}
	
	if(my_rank == 0){
		FILE *f = fopen(measurements_file, "a");
		if(f == NULL){
			fprintf(stdout, "Could not open file measurements.txt\n");
			fflush(stdout);
			return -1;
		}
		
		fprintf(f, "%s\n\n", in_file_name);
		fprintf(f, "Serial Time: %f\n", serial_time);
		fprintf(f, "Parallel SFT Time: %f\tSpeedup: %f\n", parallel_sft_time, serial_time / parallel_sft_time);
		fprintf(f, "Parallel NO SFT Time: %f\tSpeedup: %f\n", parallel_no_sft_time, serial_time / parallel_no_sft_time);
		fprintf(f, "Mater/Worker Time: %f\tSpeedup: %f\tOptimal Chunk Size: %d\n\n", optimal_chunk_time, serial_time / optimal_chunk_time, optimal_chunk_size);
		
		fprintf(f, "Master/Worker Chunks\n");
		for(int chunk = CHUNK_START; chunk <= CHUNK_END; chunk += CHUNK_STEP){
			int index = (chunk - CHUNK_START) / CHUNK_STEP;
			fprintf(f, "Chunk Size: %d\t\tTime: %f\tSpeedup: %f\n", chunk, master_time[index], serial_time / master_time[index]);
		}
		fprintf(f, "\n============================================================\n");
		
		fflush(f);
		fclose(f);
	}
}

int main(int argc, char **argv){
	MPI_Init(&argc, &argv);
	int my_rank, num_processes, check;
	double run_time;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
	
	if(argc != 2){
		if(my_rank == 0){
			fprintf(stdout, "Usage: %s [file_out]", argv[0]);
			fflush(stdout);
		}
		MPI_Abort(MPI_COMM_WORLD, -1);
	}
	
	if(my_rank == 0){
		// overwriting the file
		
		FILE *f = fopen(argv[1], "w");
		if(f == NULL){
			fprintf(stdout, "Could not open file measurements.txt\n");
			fflush(stdout);
			MPI_Abort(MPI_COMM_WORLD, -1);
		}
		fclose(f);
	}
	
	if(my_rank == 0) run_time = omp_get_wtime();
	
	check = experiment_with_image("Photos\\Large.bmp", "Photos\\Edited_Large.bmp", argv[1], my_rank, num_processes);
	if(check == -1){
		MPI_Abort(MPI_COMM_WORLD, -1);
	}
	
	check = experiment_with_image("Photos\\XL.bmp", "Photos\\Edited_XL.bmp", argv[1], my_rank, num_processes);
	if(check == -1){
		MPI_Abort(MPI_COMM_WORLD, -1);
	}
	
	check = experiment_with_image("Photos\\XXL.bmp", "Photos\\Edited_XXL.bmp", argv[1], my_rank, num_processes);
	if(check == -1){
		MPI_Abort(MPI_COMM_WORLD, -1);
	}
	
	if(my_rank == 0){
		run_time = omp_get_wtime() - run_time;
		
		FILE *f = fopen(argv[1], "a");
		if(f == NULL){
			fprintf(stdout, "Could not open file measurements.txt\n");
			fflush(stdout);
			return -1;
		}
		fprintf(f, "Total Run Time: %f\n", run_time);
		fflush(f);
		fclose(f);
	}
	
	MPI_Finalize();
	return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "bmp.h"
#include "convolution.h"

#define WORK_HEADER_SEND_TAG 1
#define WORK_DATA_SEND_TAG 2
#define WORK_HEADER_RECEIVE_TAG 3
#define WORK_DATA_RECEIVE_TAG 4
#define TERMINATE_TAG 5

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

typedef struct{
	int true_start, true_end, height, width, num_threads;
	operation_t operation;
}send_block_t;

MPI_Datatype create_mpi_datatype_for_send_block_t(){
	send_block_t block;
	MPI_Datatype mpi_block;
	MPI_Datatype types[6] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT};
	int block_lengths[6] = {1, 1, 1, 1, 1, 1};
	MPI_Aint displacements[6];
	
	// getting field addresses
	MPI_Get_address(&block.true_start, &displacements[0]);
	MPI_Get_address(&block.true_end, &displacements[1]);
	MPI_Get_address(&block.height, &displacements[2]);
	MPI_Get_address(&block.width, &displacements[3]);
	MPI_Get_address(&block.num_threads, &displacements[4]);
	MPI_Get_address(&block.operation, &displacements[5]);
	
	// making displacements relative to the first field
	displacements[5] -= displacements[0];
	displacements[4] -= displacements[0];
	displacements[3] -= displacements[0];
	displacements[2] -= displacements[0];
	displacements[1] -= displacements[0];
	displacements[0] -= displacements[0];
	
	// creating struct
	MPI_Type_create_struct(6, block_lengths, displacements, types, &mpi_block);
	MPI_Type_commit(&mpi_block);
	return mpi_block;
}

void send_work(int worker_process, int operation, int *work_done, FILE *image_file, int halo_dim, int chunk_size, int height, int width, int padding, int data_start, int *offset, int num_threads){
	int true_start, true_end, check;
	Image *chunk_image = NULL;
	MPI_Datatype mpi_send_block = create_mpi_datatype_for_send_block_t();
	MPI_Datatype mpi_rgb = create_mpi_datatype_for_RGB();
	
	chunk_image = readBMP_chunk(image_file, halo_dim, chunk_size, height, width, padding, data_start, offset, &true_start, &true_end);
	if(chunk_image == NULL){ // error message was printed by the called function
		MPI_Abort(MPI_COMM_WORLD, -1);
	}

	if(chunk_image->data == NULL){
		*work_done = 1;
		fclose(image_file);
		free(chunk_image);
		return;
	}

	send_block_t block;
	block.true_start = true_start;
	block.true_end = true_end;
	block.height = chunk_image->height;
	block.width = width;
	block.operation = operation;
	block.num_threads = num_threads;

	check = MPI_Send(&block, 1, mpi_send_block, worker_process, WORK_HEADER_SEND_TAG, MPI_COMM_WORLD);
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Rank 0: Error in master_process while sending work header\n");
		fflush(stderr);
		MPI_Abort(MPI_COMM_WORLD, -1);
	}

	check = MPI_Send(chunk_image->data, chunk_image->height * chunk_image->width, mpi_rgb, worker_process, WORK_DATA_SEND_TAG, MPI_COMM_WORLD);
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Rank 0: Error in master_process while sending work data\n");
		fflush(stderr);
		MPI_Abort(MPI_COMM_WORLD, -1);
	}
	
	free(chunk_image->data);
	free(chunk_image);
}

Image *master_process(const char *in_file_name, const char *out_file_name, operation_t operation, int chunk, int num_processes, int num_threads, int save){
	int kernel_size = get_kernel_size(operation);
	int halo_dim = kernel_size / 2;
	int check;
	int active_workers = 0;
	int height, width, data_start, padding;
	int offset;
	int work_from_rows[num_processes];
	MPI_Datatype mpi_send_block = create_mpi_datatype_for_send_block_t();
	MPI_Datatype mpi_rgb = create_mpi_datatype_for_RGB();
	
	FILE *image_file = open_BMP(in_file_name, &height, &width, &data_start, &padding);
	if(image_file == NULL){ // error message was printed by the called function
		MPI_Abort(MPI_COMM_WORLD, -1);
	}
	
	offset = data_start;
	
	RGB *new_data = (RGB*)malloc(width * height * sizeof(RGB));
	if(new_data == NULL){
		fprintf(stderr, "Rank 0: Error in master_process while allocating memory\n");
		fflush(stderr);
		MPI_Abort(MPI_COMM_WORLD, -1);
	}
	
	Image *new_image = (Image*)malloc(sizeof(Image));
	if(new_image == NULL){
		fprintf(stderr, "Rank 0: Error in master_process while allocating memory\n");
		fflush(stderr);
		MPI_Abort(MPI_COMM_WORLD, -1);
	}
	
	// distributing initial work to all the processes
	int work_done = 0;
	for(int i = 1; i < num_processes; ++i){
		if(work_done == 0){
			++active_workers;
			work_from_rows[i] = (offset - data_start) / (3 * width + padding);
			send_work(i, operation, &work_done, image_file, halo_dim, chunk, height, width, padding, data_start, &offset, num_threads);
			if(work_done == 1){ // remove the worker that was terminated 
				--active_workers;
				check = MPI_Send(NULL, 0, MPI_BYTE, i, TERMINATE_TAG, MPI_COMM_WORLD);
				if(check != MPI_SUCCESS){
					fprintf(stderr, "Rank 0: Error in master_process while sending terminate order\n");
					fflush(stderr);
					MPI_Abort(MPI_COMM_WORLD, -1);
				}
			}
		}
		else{
			check = MPI_Send(NULL, 0, MPI_BYTE, i, TERMINATE_TAG, MPI_COMM_WORLD);
			if(check != MPI_SUCCESS){
				fprintf(stderr, "Rank 0: Error in master_process while sending terminate order\n");
				fflush(stderr);
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
		}
	}
	
	while(active_workers != 0 || work_done == 0){
		int worker_rank;
		int chunk_size;
		int data_offset;
		send_block_t header;
		MPI_Status status;
		
		check = MPI_Recv(&header, 1, mpi_send_block, MPI_ANY_SOURCE, WORK_HEADER_RECEIVE_TAG, MPI_COMM_WORLD, &status);
		if(check != MPI_SUCCESS){
			fprintf(stderr, "Rank 0: Error in master_process while receiving work header\n");
			fflush(stderr);
			MPI_Abort(MPI_COMM_WORLD, -1);
		}
		
		worker_rank = status.MPI_SOURCE;
		chunk_size = header.height;
		
		data_offset = height - work_from_rows[worker_rank] - chunk_size;
		
		check = MPI_Recv(new_data + data_offset * width, chunk_size * width, mpi_rgb, worker_rank, WORK_DATA_RECEIVE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if(check != MPI_SUCCESS){
			fprintf(stderr, "Rank 0: Error in master_process while receiving work data\n");
			fflush(stderr);
			MPI_Abort(MPI_COMM_WORLD, -1);
		}
		
		if(work_done == 1){
			check = MPI_Send(NULL, 0, MPI_BYTE, worker_rank, TERMINATE_TAG, MPI_COMM_WORLD);
			if(check != MPI_SUCCESS){
				fprintf(stderr, "Rank 0: Error in master_process while sending terminate order\n");
				fflush(stderr);
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
			--active_workers;
		}
		else{
			work_from_rows[worker_rank] = (offset - data_start) / (3 * width + padding);
			send_work(worker_rank, operation, &work_done, image_file, halo_dim, chunk, height, width, padding, data_start, &offset, num_threads);
			if(work_done == 1){ // remove and terminate the worker
				--active_workers;
				check = MPI_Send(NULL, 0, MPI_BYTE, worker_rank, TERMINATE_TAG, MPI_COMM_WORLD);
				if(check != MPI_SUCCESS){
					fprintf(stderr, "Rank 0: Error in master_process while sending terminate order\n");
					fflush(stderr);
					MPI_Abort(MPI_COMM_WORLD, -1);
				}
			}
		}
	}
	
	fclose(image_file);
	
	new_image->height = height;
	new_image->width = width;
	new_image->data = new_data;
	return new_image;
}

void worker_process(int my_rank){
	MPI_Status status;
	MPI_Datatype mpi_send_block = create_mpi_datatype_for_send_block_t();
	MPI_Datatype mpi_rgb = create_mpi_datatype_for_RGB();
	send_block_t header;
	int working = 1;
	int check;
	
	while(working){
		check = MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		if(check != MPI_SUCCESS){
			fprintf(stderr, "Rank %d: Error in worker_process while probing for incoming messages\n", my_rank);
			fflush(stderr);
			MPI_Abort(MPI_COMM_WORLD, -1);
		}
		
		if(status.MPI_TAG == TERMINATE_TAG){
			working = 0;
			check = MPI_Recv(NULL, 0, MPI_BYTE, 0, TERMINATE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if(check != MPI_SUCCESS){
				fprintf(stderr, "Rank %d: Error in worker_process while consuming termination message\n", my_rank);
				fflush(stderr);
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
		}
		else if(status.MPI_TAG == WORK_HEADER_SEND_TAG){
			check = MPI_Recv(&header, 1, mpi_send_block, 0, WORK_HEADER_SEND_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if(check != MPI_SUCCESS){
				fprintf(stderr, "Rank %d: Error in worker_process while receiving work header\n", my_rank);
				fflush(stderr);
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
			
			RGB *data = (RGB*)malloc(header.height * header.width * sizeof(RGB));
			if(data == NULL){
				fprintf(stderr, "Rank %d: Error in worker_process while allocating memory\n", my_rank);
				fflush(stderr);
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
			
			Image *img = (Image*)malloc(sizeof(Image));
			if(img == NULL){
				fprintf(stderr, "Rank %d: Error in worker_process while allocating memory\n", my_rank);
				fflush(stderr);
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
			
			check = MPI_Recv(data, header.height * header.width, mpi_rgb, 0, WORK_DATA_SEND_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if(check != MPI_SUCCESS){
				fprintf(stderr, "Rank %d: Error in worker_process while receiving work data\n", my_rank);
				fflush(stderr);
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
			
			img->height = header.height;
			img->width = header.width;
			img->data = data;
			
			Image *new_image = perform_convolution_parallel(img, header.operation, header.true_start, header.true_end, header.num_threads);
			if(new_image == NULL){ // error message was printed by the called function
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
			
			free(img->data);
			free(img);
			
			header.true_start = 0;
			header.true_end = new_image->height - 1;
			header.height = new_image->height;
			header.width = new_image->width;
			
			check = MPI_Send(&header, 1, mpi_send_block, 0, WORK_HEADER_RECEIVE_TAG, MPI_COMM_WORLD);
			if(check != MPI_SUCCESS){
				fprintf(stderr, "Rank %d: Error in worker_process while sending work header\n", my_rank);
				fflush(stderr);
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
			
			check = MPI_Send(new_image->data, header.height * header.width, mpi_rgb, 0, WORK_DATA_RECEIVE_TAG, MPI_COMM_WORLD);
			if(check != MPI_SUCCESS){
				fprintf(stderr, "Rank %d: Error in worker_process while sending work data\n", my_rank);
				fflush(stderr);
				MPI_Abort(MPI_COMM_WORLD, -1);
			}
			
			free(new_image->data);
			free(new_image);
		}
	}
	
	MPI_Finalize();
}

Image *image_processing_master(const char *in_file_name, const char *out_file_name, operation_t operation, int chunk_size, int my_rank, int num_processes, int num_cores, int num_workstations, int save){
	if(my_rank == 0){ // MASTER
		Image *img = NULL;
		int num_threads = max(1, num_cores / (num_processes / num_workstations));

		// can't return NULL
		img = master_process(in_file_name, out_file_name, operation, chunk_size, num_processes, num_threads, save);
		
		if(save == 1){
			int exit_code = saveBMP(out_file_name, img);
			if(exit_code == 0){ // error message was printed by the called function
				return NULL;
			}
		}
		
		return img;
	}
	else{ // WORKER
		worker_process(my_rank);
		return NULL;
	}
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
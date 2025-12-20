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

/**
*	IMAGE PROCESSING SERIAL
*/

Image *image_processing_serial(const char *in_file_name, operation_t operation){
	/**
	*	Takes in a file path to the file to edit and an operation_t.
	*	It reads the .bmp file, edits the Image and returns the edited Image.
	*/
	
	Image *img = read_BMP_serial(in_file_name);
	if(img == NULL){ // error message was printed by the called function
		return NULL;
	}
	
	Image *edited_img = perform_convolution_serial(img, operation);
	free(img->data);
	free(img);
	
	if(edited_img == NULL){ // error message was printed by the called function
		return NULL;
	}

	return edited_img;
}



/**
*	IMAGE PROCESSING PARALLEL SFT
*/

Image *image_processing_parallel_sft(const char *in_file_name, operation_t operation, int my_rank, int num_processes, int num_cores){
	/**
	*	Takes in a file path to the file to edit, an operation_t, 
	*	this process's rank, the total number of processes and the number of cores on this workstation.
	*	It reads the process's associated chunk of the .bmp file, edits the Image chunk
	*	and, if rank != 0, sends the edited Image chunk to process 0 and returns the edited Image chunk,
	*	and if rank == 0, returns the whole edited Image.
	*/
	
	int kernel_size = get_kernel_size(operation);
	int halo_dim = kernel_size / 2;
	int true_start, true_end;
	
	Image *img = read_BMP_MPI(in_file_name, my_rank, num_processes, halo_dim, &true_start, &true_end);
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
	
	return composed_img;
}



/**
*	IMAGE PROCESSING NO PARALLEL SFT
*/

int scatter_data(Image *img, RGB **data, int my_rank, int num_processes, int width, int *local_height, int halo_dim){
	/**
	*	Takes in an Image, a RGB data buffer, this process's rank, the total number of processes,
	*	the width of the Image and the size of the halo.
	*	If rank == 0, the process calculates how many rows (local_height) of the image each process gets,
	*	informs each process about their height and sends them (including process 0) local_height rows of the Image.
	*	If rank != 0, the process receives their respective number of rows (local_height) and their respective rows.
	*/
	
	int check = 0;
	int *heights = NULL;
	int *displacements = NULL;
	int *sends = NULL;
	RGB *old_data = (img != NULL) ? img->data : NULL;
	
	if(my_rank == 0){
		int individual_height = img->height / num_processes;
		int remainder = img->height % num_processes;
		
		heights = (int*)malloc(num_processes * sizeof(int));
		if(heights == NULL){
			fprintf(stderr, "Rank %d: Error in main while distributing height\n", my_rank);
			fflush(stderr);
			return -1;
		}
		
		displacements = (int*)malloc(num_processes * sizeof(int));
		if(displacements == NULL){
			fprintf(stderr, "Rank %d: Error in main while distributing height\n", my_rank);
			fflush(stderr);
			return -1;
		}
		
		for(int i = 0; i < num_processes; ++i){
			heights[i] = individual_height + ((i < remainder) ? 1 : 0);
			if(i == 0 || i == num_processes - 1) heights[i] += halo_dim;
			else heights[i] += 2 * halo_dim;
		}
	}
	
	// sending local_height to every process
	check = MPI_Scatter(
		heights, 1, MPI_INT,
		local_height, 1, MPI_INT,
		0, MPI_COMM_WORLD
	);
	
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Rank %d: Error in main while distributing height\n", my_rank);
		fflush(stderr);
		return -1;
	}
	
	if(my_rank == 0){
		sends = (int*)malloc(num_processes * sizeof(int));
		if(sends == NULL){
			fprintf(stderr, "Rank %d: Error in main while distributing height\n", my_rank);
			fflush(stderr);
			return -1;
		}
		
		int offset = 0;
		for(int i = 0; i < num_processes; ++i){
			displacements[i] = offset;
			offset += (heights[i] - 2 * halo_dim) * width;
			sends[i] = heights[i] * width;
		}
	}
	
	// allocating data
	*data = (RGB *)malloc(width * (*local_height) * sizeof(RGB));
	if(data == NULL){
		fprintf(stderr, "Rank %d: Error in main while allocating memory\n", my_rank);
		fflush(stderr);
		return -1;
	}

	MPI_Datatype mpi_rgb = create_mpi_datatype_for_RGB();
	
	check = MPI_Scatterv(
		old_data, sends, displacements, mpi_rgb,
		*data, (*local_height) * width, mpi_rgb,
		0, MPI_COMM_WORLD
	);
	
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Rank %d: Error in main while comunicating data\n", my_rank);
		fflush(stderr);
		return -1;
	}
	
	check = MPI_Type_free(&mpi_rgb);
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Rank %d: Error in compose_BMP while de-allocating data type\n", my_rank);
		fflush(stderr);
		return -1;
	}
	return 0;
}

Image *image_processing_parallel_no_sft(const char *in_file_name, operation_t operation, int my_rank, int num_processes, int num_cores, int num_workstations){
	/**
	*	Takes in a file path to the file to edit, an operation_t, this process's rank,
	*	the total number of processes, the number of cores on this workstation
	*	and the total number of workstations.
	*	If rank == 0, the process reads the whole .bmp file, distributes chunks of the Image to each process (including process 0),
	*	edits its respective chunk, composes the whole edited Image and returns it.
	*	If rank != 0, the process receives its respective chunk, edits it, sends the edited chunk to process 0
	*	and returns the edited Image chunk.
	*/
	
	int kernel_size = get_kernel_size(operation);
	int halo_dim = kernel_size / 2;
	int check;
	int height, width;
	int local_height;
	RGB *data = NULL;
	Image *img = NULL;
	
	if(my_rank == 0){
		img = read_BMP_serial(in_file_name);
		if(img == NULL){ // error message was printed by the called function
			return NULL;
		}
		
		width = img->width;
		height = img->height;
	}
	
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
	
	return composed_img;
}



/**
*	IMAGE PROCESSING MASTER/WORKER
*/

int send_work(int worker_process, operation_t operation, int *work_done, FILE *image_file, int halo_dim, int chunk_size, int height, int width, int padding, int data_start, int *offset, int num_threads){
	/**
	*	Takes in the rank of the procees which needs to receive work, an operation_t,
	*	a FILE* coresponding to the open .bmp file, the size of the halo, the size of a chunk,
	*	the height, width, data_start and padding of the Image, the offset at which to read
	* 	and the number of threads the worker process can use.
	*	It reads a chunk of the Image and sends it, along with the required data, to the worker process.
	*	If there is no more work to be done, it sets work_done to 1 and returns without sending any work to the worker process.
	*/
	
	int true_start, true_end, check;
	Image *chunk_image = NULL;
	MPI_Datatype mpi_send_block = create_mpi_datatype_for_send_block_t();
	MPI_Datatype mpi_rgb = create_mpi_datatype_for_RGB();
	
	chunk_image = read_BMP_chunk(image_file, halo_dim, chunk_size, height, width, padding, data_start, offset, &true_start, &true_end);
	if(chunk_image == NULL){ // error message was printed by the called function
		return -1;
	}

	if(chunk_image->data == NULL){
		*work_done = 1;
		fclose(image_file);
		free(chunk_image);
		return 0;
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
		
		check = deallocate_MPI_datatype(&mpi_send_block, 0);
		if(check == -1){ // error message was printed by the called function
			return -1;
		}
		
		check = deallocate_MPI_datatype(&mpi_rgb, 0);
		if(check == -1){ // error message was printed by the called function
			return -1;
		}
		
		return -1;
	}

	check = MPI_Send(chunk_image->data, chunk_image->height * chunk_image->width, mpi_rgb, worker_process, WORK_DATA_SEND_TAG, MPI_COMM_WORLD);
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Rank 0: Error in master_process while sending work data\n");
		fflush(stderr);
		
		check = deallocate_MPI_datatype(&mpi_send_block, 0);
		if(check == -1){ // error message was printed by the called function
			return -1;
		}
		
		check = deallocate_MPI_datatype(&mpi_rgb, 0);
		if(check == -1){ // error message was printed by the called function
			return -1;
		}
		
		return -1;
	}
	
	check = deallocate_MPI_datatype(&mpi_send_block, 0);
	if(check == -1){ // error message was printed by the called function
		return -1;
	}
	
	check = deallocate_MPI_datatype(&mpi_rgb, 0);
	if(check == -1){ // error message was printed by the called function
		return -1;
	}
	
	free(chunk_image->data);
	free(chunk_image);
	return 0;
}

Image *master_process(const char *in_file_name, operation_t operation, int chunk, int num_processes, int num_threads){
	/**
	*	Takes in a file path to the file to edit, an operation_t, the size of a chunk,
	*	the total number of processes and the number of threads on available to each process.
	*	It opens the .bmp file and reads and sends a chunk to each worker process. After that, to each worker process
	*	that finishes its work, it collects the edited chunk and sends another chunk to the worker process
	*	until there are no more chunks to process. At that point, it waits for all the worker processes to finish their work,
	*	collects their edited chunks and terminates the process. It then returns the whole edited Image.
	*/
	
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
		return NULL;
	}
	
	offset = data_start;
	
	RGB *new_data = (RGB*)malloc(width * height * sizeof(RGB));
	if(new_data == NULL){
		fprintf(stderr, "Rank 0: Error in master_process while allocating memory\n");
		fflush(stderr);
		fclose(image_file);
		
		check = deallocate_MPI_datatype(&mpi_send_block, 0);
		if(check == -1){ // error message was printed by the called function
			return NULL;
		}
		
		check = deallocate_MPI_datatype(&mpi_rgb, 0);
		if(check == -1){ // error message was printed by the called function
			return NULL;
		}
		
		return NULL;
	}
	
	Image *new_image = (Image*)malloc(sizeof(Image));
	if(new_image == NULL){
		fprintf(stderr, "Rank 0: Error in master_process while allocating memory\n");
		fflush(stderr);
		free(new_data);
		fclose(image_file);
		
		check = deallocate_MPI_datatype(&mpi_send_block, 0);
		if(check == -1){ // error message was printed by the called function
			return NULL;
		}
		
		check = deallocate_MPI_datatype(&mpi_rgb, 0);
		if(check == -1){ // error message was printed by the called function
			return NULL;
		}
		
		return NULL;
	}
	
	// distributing initial work to all the processes
	int work_done = 0;
	for(int i = 1; i < num_processes; ++i){
		if(work_done == 0){
			++active_workers;
			work_from_rows[i] = (offset - data_start) / (3 * width + padding);
			
			check = send_work(i, operation, &work_done, image_file, halo_dim, chunk, height, width, padding, data_start, &offset, num_threads);
			if(check == -1){ // error message was printed by the called function
				free(new_data);
				free(new_image);
				fclose(image_file);
				
				check = deallocate_MPI_datatype(&mpi_send_block, 0);
				if(check == -1){ // error message was printed by the called function
					return NULL;
				}
				
				check = deallocate_MPI_datatype(&mpi_rgb, 0);
				if(check == -1){ // error message was printed by the called function
					return NULL;
				}
				
				return NULL;
			}
			
			if(work_done == 1){ // remove and terminate the current worker
				--active_workers;
				check = MPI_Send(NULL, 0, MPI_BYTE, i, TERMINATE_TAG, MPI_COMM_WORLD);
				if(check != MPI_SUCCESS){
					fprintf(stderr, "Rank 0: Error in master_process while sending terminate order\n");
					fflush(stderr);
					free(new_data);
					free(new_image);
					fclose(image_file);
					
					check = deallocate_MPI_datatype(&mpi_send_block, 0);
					if(check == -1){ // error message was printed by the called function
						return NULL;
					}
					
					check = deallocate_MPI_datatype(&mpi_rgb, 0);
					if(check == -1){ // error message was printed by the called function
						return NULL;
					}
					
					return NULL;
				}
			}
		}
		else{
			check = MPI_Send(NULL, 0, MPI_BYTE, i, TERMINATE_TAG, MPI_COMM_WORLD);
			if(check != MPI_SUCCESS){
					fprintf(stderr, "Rank 0: Error in master_process while sending terminate order\n");
					fflush(stderr);
					free(new_data);
					free(new_image);
					fclose(image_file);
					
					check = deallocate_MPI_datatype(&mpi_send_block, 0);
					if(check == -1){ // error message was printed by the called function
						return NULL;
					}
					
					check = deallocate_MPI_datatype(&mpi_rgb, 0);
					if(check == -1){ // error message was printed by the called function
						return NULL;
					}
					
					return NULL;
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
			free(new_data);
			free(new_image);
			fclose(image_file);
			
			check = deallocate_MPI_datatype(&mpi_send_block, 0);
			if(check == -1){ // error message was printed by the called function
				return NULL;
			}
			
			check = deallocate_MPI_datatype(&mpi_rgb, 0);
			if(check == -1){ // error message was printed by the called function
				return NULL;
			}
			
			return NULL;
		}
		
		worker_rank = status.MPI_SOURCE;
		chunk_size = header.height;
		
		data_offset = height - work_from_rows[worker_rank] - chunk_size;
		
		check = MPI_Recv(new_data + data_offset * width, chunk_size * width, mpi_rgb, worker_rank, WORK_DATA_RECEIVE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if(check != MPI_SUCCESS){
			fprintf(stderr, "Rank 0: Error in master_process while receiving work data\n");
			fflush(stderr);
			free(new_data);
			free(new_image);
			fclose(image_file);
			
			check = deallocate_MPI_datatype(&mpi_send_block, 0);
			if(check == -1){ // error message was printed by the called function
				return NULL;
			}
			
			check = deallocate_MPI_datatype(&mpi_rgb, 0);
			if(check == -1){ // error message was printed by the called function
				return NULL;
			}
			
			return NULL;
		}
		
		if(work_done == 1){
			check = MPI_Send(NULL, 0, MPI_BYTE, worker_rank, TERMINATE_TAG, MPI_COMM_WORLD);
			if(check != MPI_SUCCESS){
				fprintf(stderr, "Rank 0: Error in master_process while sending terminate order\n");
				fflush(stderr);
				free(new_data);
				free(new_image);
				fclose(image_file);
				
				check = deallocate_MPI_datatype(&mpi_send_block, 0);
				if(check == -1){ // error message was printed by the called function
					return NULL;
				}
				
				check = deallocate_MPI_datatype(&mpi_rgb, 0);
				if(check == -1){ // error message was printed by the called function
					return NULL;
				}
				
				return NULL;
			}
			--active_workers;
		}
		else{
			work_from_rows[worker_rank] = (offset - data_start) / (3 * width + padding);
			
			check = send_work(worker_rank, operation, &work_done, image_file, halo_dim, chunk, height, width, padding, data_start, &offset, num_threads);
			if(check == -1){ // error message was printed by the called function
				free(new_data);
				free(new_image);
				fclose(image_file);
				
				check = deallocate_MPI_datatype(&mpi_send_block, 0);
				if(check == -1){ // error message was printed by the called function
					return NULL;
				}
				
				check = deallocate_MPI_datatype(&mpi_rgb, 0);
				if(check == -1){ // error message was printed by the called function
					return NULL;
				}
				
				return NULL;
			}
			
			if(work_done == 1){ // remove and terminate the current worker
				--active_workers;
				
				check = MPI_Send(NULL, 0, MPI_BYTE, worker_rank, TERMINATE_TAG, MPI_COMM_WORLD);
				if(check != MPI_SUCCESS){
					fprintf(stderr, "Rank 0: Error in master_process while sending terminate order\n");
					fflush(stderr);
					free(new_data);
					free(new_image);
					fclose(image_file);
					
					check = deallocate_MPI_datatype(&mpi_send_block, 0);
					if(check == -1){ // error message was printed by the called function
						return NULL;
					}
					
					check = deallocate_MPI_datatype(&mpi_rgb, 0);
					if(check == -1){ // error message was printed by the called function
						return NULL;
					}
					
					return NULL;
				}
			}
		}
	}
	
	fclose(image_file);
	check = deallocate_MPI_datatype(&mpi_send_block, 0);
	if(check == -1){ // error message was printed by the called function
		return NULL;
	}
	
	check = deallocate_MPI_datatype(&mpi_rgb, 0);
	if(check == -1){ // error message was printed by the called function
		return NULL;
	}
	
	new_image->height = height;
	new_image->width = width;
	new_image->data = new_data;
	return new_image;
}

int worker_process(int my_rank){
	/**
	*	Takes in this process's rank.
	*	It receives a chunk of an Image from process 0, which it edits and sends back to process 0.
	*/
	
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
			
			check = deallocate_MPI_datatype(&mpi_send_block, my_rank);
			if(check == -1){ // error message was printed by the called function
				return -1;
			}
			
			check = deallocate_MPI_datatype(&mpi_rgb, my_rank);
			if(check == -1){ // error message was printed by the called function
				return -1;
			}
			
			return -1;
		}
		
		if(status.MPI_TAG == TERMINATE_TAG){
			working = 0;
			check = MPI_Recv(NULL, 0, MPI_BYTE, 0, TERMINATE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if(check != MPI_SUCCESS){
				fprintf(stderr, "Rank %d: Error in worker_process while consuming termination message\n", my_rank);
				fflush(stderr);
				
				check = deallocate_MPI_datatype(&mpi_send_block, my_rank);
				if(check == -1){ // error message was printed by the called function
					return -1;
				}
				
				check = deallocate_MPI_datatype(&mpi_rgb, my_rank);
				if(check == -1){ // error message was printed by the called function
					return -1;
				}
				
				return -1;
			}
		}
		else if(status.MPI_TAG == WORK_HEADER_SEND_TAG){
			check = MPI_Recv(&header, 1, mpi_send_block, 0, WORK_HEADER_SEND_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if(check != MPI_SUCCESS){
				fprintf(stderr, "Rank %d: Error in worker_process while receiving work header\n", my_rank);
				fflush(stderr);
				
				check = deallocate_MPI_datatype(&mpi_send_block, my_rank);
				if(check == -1){ // error message was printed by the called function
					return -1;
				}
				
				check = deallocate_MPI_datatype(&mpi_rgb, my_rank);
				if(check == -1){ // error message was printed by the called function
					return -1;
				}
				
				return -1;
			}
			
			RGB *data = (RGB*)malloc(header.height * header.width * sizeof(RGB));
			if(data == NULL){
				fprintf(stderr, "Rank %d: Error in worker_process while allocating memory\n", my_rank);
				fflush(stderr);
				
				check = deallocate_MPI_datatype(&mpi_send_block, my_rank);
				if(check == -1){ // error message was printed by the called function
					return -1;
				}
				
				check = deallocate_MPI_datatype(&mpi_rgb, my_rank);
				if(check == -1){ // error message was printed by the called function
					return -1;
				}
				
				return -1;
			}
			
			Image *img = (Image*)malloc(sizeof(Image));
			if(img == NULL){
				fprintf(stderr, "Rank %d: Error in worker_process while allocating memory\n", my_rank);
				fflush(stderr);
				free(data);
				
				check = deallocate_MPI_datatype(&mpi_send_block, my_rank);
				if(check == -1){ // error message was printed by the called function
					return -1;
				}
				
				check = deallocate_MPI_datatype(&mpi_rgb, my_rank);
				if(check == -1){ // error message was printed by the called function
					return -1;
				}
				
				return -1;
			}
			
			check = MPI_Recv(data, header.height * header.width, mpi_rgb, 0, WORK_DATA_SEND_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if(check != MPI_SUCCESS){
				fprintf(stderr, "Rank %d: Error in worker_process while receiving work data\n", my_rank);
				fflush(stderr);
				free(data);
				free(img);
				
				check = deallocate_MPI_datatype(&mpi_send_block, my_rank);
				if(check == -1){ // error message was printed by the called function
					return -1;
				}
				
				check = deallocate_MPI_datatype(&mpi_rgb, my_rank);
				if(check == -1){ // error message was printed by the called function
					return -1;
				}
				
				return -1;
			}
			
			img->height = header.height;
			img->width = header.width;
			img->data = data;
			
			Image *new_image = perform_convolution_parallel(img, header.operation, header.true_start, header.true_end, header.num_threads);
			if(new_image == NULL){ // error message was printed by the called function
				free(img->data);
				free(img);
				
				check = deallocate_MPI_datatype(&mpi_send_block, my_rank);
				if(check == -1){ // error message was printed by the called function
					return -1;
				}
				
				check = deallocate_MPI_datatype(&mpi_rgb, my_rank);
				if(check == -1){ // error message was printed by the called function
					return -1;
				}
				
				return -1;
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
				free(new_image->data);
				free(new_image);
				
				check = deallocate_MPI_datatype(&mpi_send_block, my_rank);
				if(check == -1){ // error message was printed by the called function
					return -1;
				}
				
				check = deallocate_MPI_datatype(&mpi_rgb, my_rank);
				if(check == -1){ // error message was printed by the called function
					return -1;
				}
				
				return -1;
			}
			
			check = MPI_Send(new_image->data, header.height * header.width, mpi_rgb, 0, WORK_DATA_RECEIVE_TAG, MPI_COMM_WORLD);
			if(check != MPI_SUCCESS){
				fprintf(stderr, "Rank %d: Error in worker_process while sending work data\n", my_rank);
				fflush(stderr);
				free(new_image->data);
				free(new_image);
				
				check = deallocate_MPI_datatype(&mpi_send_block, my_rank);
				if(check == -1){ // error message was printed by the called function
					return -1;
				}
				
				check = deallocate_MPI_datatype(&mpi_rgb, my_rank);
				if(check == -1){ // error message was printed by the called function
					return -1;
				}
				
				return -1;
			}
			
			free(new_image->data);
			free(new_image);
		}
	}
	
	check = deallocate_MPI_datatype(&mpi_send_block, my_rank);
	if(check == -1){ // error message was printed by the called function
		return -1;
	}
	
	check = deallocate_MPI_datatype(&mpi_rgb, my_rank);
	if(check == -1){ // error message was printed by the called function
		return -1;
	}

	return 0;
}

Image *image_processing_master(const char *in_file_name, operation_t operation, int chunk_size, int my_rank, int num_processes, int num_cores, int num_workstations){
	/**
	*	Takes in a file path to the file to edit, an operation_t, the size of a chunk,
	*	this process's rank, the total number of processes,
	*	the number of available cores on this workstation and the number of workstations.
	*	If rank == 0, it calls master_process with the appropriate arguments and returns the whole edited Image.
	*	If rank != 0, it calls worker_process with the appropriate arguments and returns a `dummy` Image.
	*/
	
	if(my_rank == 0){ // MASTER
		Image *img = NULL;
		int num_threads = max(1, num_cores / (num_processes / num_workstations));

		img = master_process(in_file_name, operation, chunk_size, num_processes, num_threads);
		if(img == NULL){ // error message was printed by the called function
			return NULL;
		}
		
		return img;
	}
	else{ // WORKER
		int check = worker_process(my_rank);
		if(check == -1){
			return NULL;
		}
		
		Image *dummy = (Image*)malloc(sizeof(Image));
		if(dummy == NULL){
			fprintf(stderr, "Rank %d: Error in image_processing_master while allocating memory\n", my_rank);
			fflush(stderr);
			return NULL;
		}
		
		dummy->data = NULL;
		return dummy;
	}
}



/**
*	HELPER FUNCTION
*/

int images_are_identical(Image *img1, Image *img2){
	/**
	*	Takes in 2 Images and returns 1 if they are identical and 0 otherwise.
	*/
	
	if(img1->height != img2->height || img1->width != img2->width) return 0;
	
	for(int i = 0; i < img1->height; ++i){
		for(int j = 0; j < img1->width; ++j){
			if(equal_RGB((img1->data)[i * img1->width + j], (img2->data)[i * img2->width + j]) == 0) return 0;
		}
	}
	
	return 1;
}
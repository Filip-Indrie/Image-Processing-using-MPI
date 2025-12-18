#include <stdio.h>
#include "bmp_common.h"

void copy_RGB(const RGB *src, RGB *dest){
	dest->r = src->r;
	dest->g = src->g;
	dest->b = src->b;
}

int equal_RGB(const RGB rgb1, const RGB rgb2){
	return rgb1.r == rgb2.r && rgb1.g == rgb2.g && rgb1.b == rgb2.b;
}

int min(int a, int b){
	return (a < b) ? a : b;
}

int max(int a, int b){
	return (a > b) ? a : b;
}

MPI_Datatype create_mpi_datatype_for_RGB(){
	/**
	*	This function creates a custom MPI_Datatype for the RGB type.
	*	It is used for sending messages with RGB data as a whole.
	*/
	
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


MPI_Datatype create_mpi_datatype_for_send_block_t(){
	/**
	*	This function creates a custom MPI_Datatype for the send_block_t type.
	*	It is used for sending messages with send_block_t data as a whole.
	*/
	
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

int deallocate_MPI_datatype(MPI_Datatype *type, int my_rank){
	/**
	*	Takes in a MPI_Datatype and deallocates it.
	*/
	
	int check = MPI_Type_free(type);
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Rank %d: Error in deallocate_MPI_datatype while de-allocating data type\n", my_rank);
		fflush(stderr);
		return -1;
	}
	
	return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bmp.h"
#include "bmp_common.h"

Image *read_BMP_serial(const char *filename){
	/**
	*	Takes in a file path and returns the bitmap inside as an Image.
	*/
	
    FILE *image_file = fopen(filename, "rb");
    if (image_file == NULL){
        fprintf(stderr, "Error in readBMP_serial: Could not open file %s\n", filename);
		fflush(stderr);
        return NULL;
    }

    unsigned char header[54];
    if (fread(header, sizeof(unsigned char), 54, image_file) != 54){
        fprintf(stderr, "Error in readBMP_serial: Invalid BMP header\n");
		fflush(stderr);
        fclose(image_file);
        return NULL;
    }

    if (header[0] != 'B' || header[1] != 'M'){
        fprintf(stderr, "Error in readBMP_serial: Not a valid BMP file\n");
		fflush(stderr);
        fclose(image_file);
        return NULL;
    }

    int width = *(int *)&header[18];
    int height = *(int *)&header[22];
    int bits_per_pixel = *(short *)&header[28];

	int data_offset = *(int *)&header[10];
	fseek(image_file, data_offset, SEEK_SET);

    if (bits_per_pixel != 24){
        fprintf(stderr, "Error in readBMP_serial: Only 24-bit BMPs are supported\n");
		fflush(stderr);
        fclose(image_file);
        return NULL;
    }

    int row_padded = (width * 3 + 3) & (~3);
	int pixel_data_size = width * 3;
	int padding_size = row_padded - pixel_data_size;
	
    unsigned char *row_pixels = (unsigned char*)malloc(pixel_data_size);
	if(row_pixels == NULL){
		fprintf(stderr, "Error in readBMP_serial while allocating memory\n");
		fflush(stderr);
        fclose(image_file);
        return NULL;
	}
	
    RGB *data = (RGB*)malloc(width * height * sizeof(RGB));
    if(data == NULL){
		fprintf(stderr, "Error in readBMP_serial while allocating memory\n");
		fflush(stderr);
        fclose(image_file);
        return NULL;
	}

    for (int y = 0; y < height; ++y){
        fread(row_pixels, sizeof(unsigned char), pixel_data_size, image_file);
        for (int x = 0; x < width; ++x){
            data[(height - 1 - y) * width + x].b = row_pixels[x * 3];
            data[(height - 1 - y) * width + x].g = row_pixels[x * 3 + 1];
            data[(height - 1 - y) * width + x].r = row_pixels[x * 3 + 2];
        }
		if (padding_size > 0){
			fseek(image_file, padding_size, SEEK_CUR);
		}
    }

    free(row_pixels);
    fclose(image_file);

    Image *img = (Image *)malloc(sizeof(Image));
	if(img == NULL){
		fprintf(stderr, "Error in readBMP_serial while allocating memory\n");
		fflush(stderr);
        fclose(image_file);
        return NULL;
	}
	
    img->width = width;
    img->height = height;
    img->data = data;
    return img;
}

Image *read_BMP_MPI(const char *file_name, int my_rank, int num_processes, int halo_dim, int *true_start, int *true_end){
	/**
	*	Takes in a file path, this process's rank, the total number of processes and
	*	the size of the halo (depending on the size of the kernel).
	*	It returns the chunk of the bmp coresponding to each process and sets true_start and true_end
	*	to mark the positions at which the chunk (not including the halos) starts and ends.
	*/
	
	int check;
	MPI_File image_file_handler;
	
	check = MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &image_file_handler);
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Rank %d: Error in readBMP_MPI while opening file %s\n", my_rank, file_name);
		fflush(stderr);
		return NULL;
	}
	
	unsigned char header[54];
	
	check = MPI_File_read_at_all(image_file_handler, 0, header, 54, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Rank %d: Error in readBMP_MPI while reading from file file %s\n", my_rank, file_name);
		fflush(stderr);
		return NULL;
	}
	
	if (header[0] != 'B' || header[1] != 'M'){
		if(my_rank == 0){
			fprintf(stderr, "Rank %d: Error in readBMP_MPI: Not a valid BMP file\n", my_rank);
			fflush(stderr);
		}
		return NULL;
	}
	
	int bitsPerPixel = *(short *)&header[28];
	if (bitsPerPixel != 24){
		if(my_rank == 0){
			fprintf(stderr, "Rank %d: Error in readBMP_MPI: Only 24-bit BMPs are supported\n", my_rank);
			fflush(stderr);
		}
		return NULL;
	}
	
	int width = *(int *)&header[18];
	int height = *(int *)&header[22];
	int data_offset = *(int *)&header[10];
	
	int row_padded = (width * 3 + 3) & (~3);
	int pixel_data_size = width * 3;
	int padding_size = row_padded - pixel_data_size;
	
	unsigned char *row_pixels = (unsigned char*)malloc(pixel_data_size);
	if(row_pixels == NULL){
		fprintf(stderr, "Rank %d: Error in readBMP_MPI while allocating memory\n", my_rank);
		fflush(stderr);
		
		check = MPI_File_close(&image_file_handler);
		if(check != MPI_SUCCESS){
			fprintf(stderr, "Rank %d: Error in readBMP_MPI while closing file %s\n", my_rank, file_name);
			fflush(stderr);
			return NULL;
		}
		
		return NULL;
	}
	
	int virtual_rank = num_processes - my_rank - 1;
	int local_rows;
	int true_rows = height / num_processes;
	int remainder = height % num_processes;
	
	int rows_read_until_now = true_rows * virtual_rank + ((virtual_rank < remainder) ? virtual_rank : remainder);
	MPI_Offset local_offset = data_offset + rows_read_until_now * (pixel_data_size + padding_size);
	
	if(virtual_rank != 0) local_offset -= halo_dim * (pixel_data_size + padding_size); // reading the halos as well
	if(virtual_rank < remainder) ++true_rows; // distributing remainder uniformly
	
	// adding halo rows
	if(virtual_rank == 0 || virtual_rank == num_processes - 1) local_rows = true_rows + halo_dim;
	else local_rows = true_rows + 2 * halo_dim;

	if(virtual_rank > 0) *true_end = local_rows - halo_dim - 1; // end is refering to the bottom of the matrix since the bottom has the highest index
	else *true_end = local_rows - 1;
	
	if(virtual_rank < num_processes - 1) *true_start = halo_dim; // start is refering to the top of the matrix since the top has the lowest index
	else *true_start = 0;
	
	RGB *data = (RGB *)malloc(width * local_rows * sizeof(RGB));
	if(data == NULL){
		fprintf(stderr, "Rank %d: Error in readBMP_MPI while allocating memory\n", my_rank);
		fflush(stderr);
		free(row_pixels);
		
		check = MPI_File_close(&image_file_handler);
		if(check != MPI_SUCCESS){
			fprintf(stderr, "Rank %d: Error in readBMP_MPI while closing file %s\n", my_rank, file_name);
			fflush(stderr);
			return NULL;
		}
		
		return NULL;
	}
	
	for (int y = 0; y < local_rows; y++){
		check = MPI_File_read_at(image_file_handler, local_offset, row_pixels, pixel_data_size, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
		if(check != MPI_SUCCESS){
			fprintf(stderr, "Rank %d: Error in readBMP_MPI while reading from file file %s\n", my_rank, file_name);
			fflush(stderr);
			free(row_pixels);
			free(data);
		
			check = MPI_File_close(&image_file_handler);
			if(check != MPI_SUCCESS){
				fprintf(stderr, "Rank %d: Error in readBMP_MPI while closing file %s\n", my_rank, file_name);
				fflush(stderr);
				return NULL;
			}
			
			return NULL;
		}
		
		local_offset += pixel_data_size + padding_size;
		
		for (int x = 0; x < width; x++){
			data[(local_rows - 1 - y) * width + x].b = row_pixels[x * 3];
			data[(local_rows - 1 - y) * width + x].g = row_pixels[x * 3 + 1];
			data[(local_rows - 1 - y) * width + x].r = row_pixels[x * 3 + 2];
		}
	}
	
	free(row_pixels);
	check = MPI_File_close(&image_file_handler);
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Rank %d: Error in readBMP_MPI while closing file %s\n", my_rank, file_name);
		fflush(stderr);
		return NULL;
	}
	
	Image *img = (Image*)malloc(sizeof(Image));
	if(img == NULL){
		fprintf(stderr, "Rank %d: Error in readBMP_MPI while allocating memory\n", my_rank);
		fflush(stderr);
		free(data);
		return NULL;
	}
	
	img->width = width;
	img->height = local_rows;
	img->data = data;
	return img;
}

Image *compose_BMP(Image *img, int my_rank, int num_processes){
	/**
	*	Takes in each process's Image and rank, as well as the total number of processes
	*	and, for rank 0, returns the Image resulted from concatenating the Images of
	*	each process, and, for ranks != 0, returns the process's original Image.
	*/
	
	int check;
	int width = img->width;
	int total_height = 0;
	int *heights = NULL;
	int *receives = NULL;
	int *displacements = NULL;
	Image *new_img = NULL;
	RGB *data = NULL;
	
	if(my_rank == 0){
		heights = (int*)malloc(num_processes * sizeof(int));
		if(heights == NULL){
			fprintf(stderr, "Rank %d: Error in compose_BMP while allocating memory\n", my_rank);
			fflush(stderr);
			return NULL;
		}
	}
	
	check = MPI_Gather(&(img->height), 1, MPI_INT, heights, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Rank %d: Error in compose_BMP while comunicating height\n", my_rank);
		fflush(stderr);
		return NULL;
	}
	
	if(my_rank == 0){
		new_img = (Image*)malloc(sizeof(Image));
		if(new_img == NULL){
			fprintf(stderr, "Rank %d: Error in compose_BMP while allocating memory\n", my_rank);
			fflush(stderr);
			free(heights);
			return NULL;
		}
		
		displacements = (int*)malloc(num_processes * sizeof(int));
		if(displacements == NULL){
			fprintf(stderr, "Rank %d: Error in compose_BMP while allocating memory\n", my_rank);
			fflush(stderr);
			free(heights);
			free(new_img);
			return NULL;
		}
		
		receives = (int*)malloc(num_processes * sizeof(int));
		if(receives == NULL){
			fprintf(stderr, "Rank %d: Error in compose_BMP while allocating memory\n", my_rank);
			fflush(stderr);
			free(heights);
			free(new_img);
			free(displacements);
			return NULL;
		}

		for(int i = 0; i < num_processes; ++i){
			total_height += heights[i];
		}
		
		data = (RGB *)malloc(width * total_height * sizeof(RGB));
		if(data == NULL){
			fprintf(stderr, "Rank %d: Error in compose_BMP while allocating memory\n", my_rank);
			fflush(stderr);
			free(heights);
			free(new_img);
			free(displacements);
			free(receives);
			return NULL;
		}
		
		MPI_Offset offset = 0;
		for(int i = 0; i < num_processes; ++i){
			displacements[i] = offset;
			offset += heights[i] * width;
			receives[i] = heights[i] * width;
		}
	}
	
	MPI_Datatype mpi_rgb = create_mpi_datatype_for_RGB();
	
	check = MPI_Gatherv(
		img->data, img->height * width, mpi_rgb,
		data, receives, displacements, mpi_rgb,
		0, MPI_COMM_WORLD
	);
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Rank %d: Error in compose_BMP while comunicating data\n", my_rank);
		fflush(stderr);
		
		if(my_rank == 0){
			free(heights);
			free(new_img);
			free(displacements);
			free(receives);
			free(data);
		}
		
		check = deallocate_MPI_datatype(&mpi_rgb, my_rank);
		if(check != 0){ // error message was printed by the called function
			return NULL;
		}
		
		return NULL;
	}
	
	check = deallocate_MPI_datatype(&mpi_rgb, my_rank);
	if(check != 0){ // error message was printed by the called function
		if(my_rank == 0){
			free(heights);
			free(new_img);
			free(displacements);
			free(receives);
			free(data);
		}
		return NULL;
	}
	
	if(my_rank == 0){
		new_img->data = data;
		new_img->width = width;
		new_img->height = total_height;
		return new_img;
	}
	else{
		return img;
	}
}

FILE *open_BMP(const char *filename, int *height, int *width, int *data_start, int *padding){
	/**
	*	Takes in a file path and returns a FILE* associated with the opened file.
	*	It also sets height, width, data_start and padding according to the file's header.
	*/
	
	FILE *image_file = fopen(filename, "rb");
    if (image_file == NULL){
        fprintf(stderr, "Error in open_BMP: Could not open file %s\n", filename);
		fflush(stderr);
        return NULL;
    }

    unsigned char header[54];
    if (fread(header, sizeof(unsigned char), 54, image_file) != 54){
        fprintf(stderr, "Error in open_BMP: Invalid BMP header\n");
		fflush(stderr);
        fclose(image_file);
        return NULL;
    }

    if (header[0] != 'B' || header[1] != 'M'){
        fprintf(stderr, "Error in open_BMP: Not a valid BMP file\n");
		fflush(stderr);
        fclose(image_file);
        return NULL;
    }
	
	int bitsPerPixel = *(short *)&header[28];
	if (bitsPerPixel != 24){
        fprintf(stderr, "Error in open_BMP: Only 24-bit BMPs are supported\n");
		fflush(stderr);
        fclose(image_file);
        return NULL;
    }
	
	*height = *(int *)&header[22];
	*width = *(int *)&header[18];
	*data_start = *(int *)&header[10];
	
	int row_padded = (*width * 3 + 3) & (~3);
	int pixel_data_size = *width * 3;
	
	*padding = row_padded - pixel_data_size;
	
	return image_file;
}

Image *read_BMP_chunk(FILE *image_file, int halo_dim, int chunk_size, int height, int width, int padding, int data_start, int *offset, int *true_start, int *true_end){
	/**
	*	Takes in a FILE* coresponding to an open .bmp file, the size of the halo, the size of a chunk,
	*	the height, width, data_start and padding of the image and the offset at which to start reading
	*	and returns the read chunk as an Image, setting true_start and true_end to represent the start
	* 	and end of the image chunk (not including the halos).
	*/
	
	int pixel_data_size = width * 3;
	int offset_stride = width * 3 + padding;
	int next_row = (*offset - data_start) / offset_stride;
	int halo_available = halo_dim;
	int rows_to_read;
	
	if(next_row >= height){ // no more rows to read
		Image *dummy_image = (Image*)malloc(sizeof(Image));
		dummy_image->data = NULL;
		return dummy_image;
	}
	
	if(next_row == 0){ // first chunk doesnt have an end halo
		if(chunk_size >= height){ // if the chunk is bigger than the image
			rows_to_read = height;
			*true_start = 0;
			*true_end = rows_to_read - 1;
		}
		else if(chunk_size + halo_dim >= height){ // if the chunk + halo is bigger than the image
			halo_available = halo_dim - (chunk_size + halo_dim - height);
			rows_to_read = chunk_size + halo_available;
			*true_start = halo_available;
			*true_end = rows_to_read - 1;
		}
		else{ // chunk + halo is smaller than the image
			rows_to_read = chunk_size + halo_dim;
			*true_start = halo_dim;
			*true_end = rows_to_read - 1;
		}
	}
	else if(next_row + chunk_size >= height){ // last chunk doesnt have a start halo
		rows_to_read = height - next_row + halo_dim;
		*true_start = 0;
		*true_end = rows_to_read - 1 - halo_dim;
		*offset -= halo_dim * offset_stride;
	}
	else if(next_row + chunk_size + halo_dim >= height){ // second to last chunk may not have the entire start halo
		halo_available = halo_dim - (next_row + chunk_size + halo_dim - height);
		rows_to_read = chunk_size + halo_dim + halo_available;
		*true_start = halo_available;
		*true_end = rows_to_read - 1 - halo_dim;
		*offset -= halo_dim * offset_stride;
	}
	else{ // any other chunks have both halos
		rows_to_read = chunk_size + 2 * halo_dim;
		*true_start = halo_dim;
		*true_end = rows_to_read - 1 - halo_dim;
		*offset -= halo_dim * offset_stride;
	}
	
	unsigned char *row_pixels = (unsigned char*)malloc(pixel_data_size * sizeof(unsigned char));
	if(row_pixels == NULL){
		fprintf(stderr, "Rank 0: Error in readBMP_chunk while allocating memory\n");
		fflush(stderr);
		fclose(image_file);
		return NULL;
	}
	
	RGB *data = (RGB*)malloc(rows_to_read * width * sizeof(RGB));
	if(data == NULL){
		fprintf(stderr, "Rank 0: Error in readBMP_chunk while allocating memory\n");
		fflush(stderr);
		
		free(row_pixels);
		fclose(image_file);
		
		return NULL;
	}
	
	fseek(image_file, *offset, SEEK_SET); // moving file cursor to the offset
	
	for(int y = 0; y < rows_to_read; ++y){
		fread(row_pixels, sizeof(unsigned char), pixel_data_size, image_file);
		for (int x = 0; x < width; x++){
            data[(rows_to_read - 1 - y) * width + x].b = row_pixels[x * 3];
            data[(rows_to_read - 1 - y) * width + x].g = row_pixels[x * 3 + 1];
            data[(rows_to_read - 1 - y) * width + x].r = row_pixels[x * 3 + 2];
        }
		if(padding > 0){
			fseek(image_file, padding, SEEK_CUR);
		}
		
		*offset += offset_stride;
	}
	
	if(*true_start != 0){ // if true_start == 0 it means that this is the last chunk and that we didnt read a start halo
		*offset -= halo_available * offset_stride;
	}
	
	free(row_pixels);
	
	Image *image_chunk = (Image*)malloc(sizeof(Image));
	image_chunk->width = width;
	image_chunk->height = rows_to_read;
	image_chunk->data = data;
	return image_chunk;
}

int save_BMP(const char *filename, const Image *img){
	/**
	*	Takes in a file path and an Image, and save the Image at the given file path.
	*/
	
    FILE *image_file = fopen(filename, "wb");
    if (image_file == NULL){
        fprintf(stderr, "Error in save_BMP: Could not create file %s\n", filename);
		fflush(stderr);
        return -1;
    }

    int width = img->width;
    int height = img->height;
	int pixel_data_size = width * 3;
    int row_padded = (width * 3 + 3) & (~3);
	int padding_size = row_padded - pixel_data_size;
    int file_size = 54 + row_padded * height;

    unsigned char header[54] = {
        'B', 'M',    // Signature
        0, 0, 0, 0,  // File size
        0, 0, 0, 0,  // Reserved
        54, 0, 0, 0, // Data offset
        40, 0, 0, 0, // Header size
        0, 0, 0, 0,  // Width
        0, 0, 0, 0,  // Height
        1, 0,        // Planes
        24, 0,       // Bits per pixel
        0, 0, 0, 0,  // Compression (none)
        0, 0, 0, 0,  // Image size (can be 0 for no compression)
        0, 0, 0, 0,  // X pixels per meter
        0, 0, 0, 0,  // Y pixels per meter
        0, 0, 0, 0,  // Total colors
        0, 0, 0, 0   // Important colors
    };

    // Fill in width, height, and file size
    *(int *)&header[2] = file_size;
    *(int *)&header[18] = width;
    *(int *)&header[22] = height;

    fwrite(header, sizeof(unsigned char), 54, image_file);

	unsigned char padding[3] = {0, 0, 0};
    unsigned char *row_pixels = (unsigned char *)malloc(pixel_data_size);
    if (row_pixels == NULL){
        fprintf(stderr, "Error in save_BMP while allocating memory\n");
		fflush(stderr);
        fclose(image_file);
        return -1;
    }

    // Write pixel data bottom-to-top
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            RGB pixel = img->data[(height - 1 - y) * width + x];
            row_pixels[x * 3] = pixel.b;
            row_pixels[x * 3 + 1] = pixel.g;
            row_pixels[x * 3 + 2] = pixel.r;
        }
        fwrite(row_pixels, sizeof(unsigned char), pixel_data_size, image_file);
		if(padding_size > 0)
		{
			fwrite(padding, sizeof(unsigned char), padding_size, image_file);
		}
    }

    free(row_pixels);
    fclose(image_file);
    return 0;
}
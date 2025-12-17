#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include "bmp.h"
#include "bmp_common.h"

MPI_Datatype create_mpi_datatype_for_RGB(){
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

Image *readBMP_serial(const char *filename)
{
    FILE *f = fopen(filename, "rb");
    if (!f)
    {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
		fflush(stderr);
        return NULL;
    }

    unsigned char header[54];
    if (fread(header, sizeof(unsigned char), 54, f) != 54)
    {
        fprintf(stderr, "Error: Invalid BMP header\n");
		fflush(stderr);
        fclose(f);
        return NULL;
    }

    if (header[0] != 'B' || header[1] != 'M')
    {
        fprintf(stderr, "Error: Not a valid BMP file\n");
		fflush(stderr);
        fclose(f);
        return NULL;
    }

    int width = *(int *)&header[18];
    int height = *(int *)&header[22];
    int bitsPerPixel = *(short *)&header[28];

	int data_offset = *(int *)&header[10];
	fseek(f, data_offset, SEEK_SET);

    if (bitsPerPixel != 24)
    {
        fprintf(stderr, "Error: Only 24-bit BMPs are supported\n");
		fflush(stderr);
        fclose(f);
        return NULL;
    }

    int row_padded = (width * 3 + 3) & (~3);
	int pixel_data_size = width * 3;
	int padding_size = row_padded - pixel_data_size;
    unsigned char *row_pixels = (unsigned char *)malloc(pixel_data_size);
    RGB *data = (RGB *)malloc(width * height * sizeof(RGB));
    if (!data || !row_pixels)
    {
        fprintf(stderr, "Error: Memory allocation failed\n");
		fflush(stderr);
        free(data);
        free(row_pixels);
        fclose(f);
        return NULL;
    }

    for (int y = 0; y < height; y++)
    {
        fread(row_pixels, sizeof(unsigned char), pixel_data_size, f);
        for (int x = 0; x < width; x++)
        {
            data[(height - 1 - y) * width + x].b = row_pixels[x * 3];
            data[(height - 1 - y) * width + x].g = row_pixels[x * 3 + 1];
            data[(height - 1 - y) * width + x].r = row_pixels[x * 3 + 2];
        }
		if (padding_size > 0)
		{
			// fseek moves the file pointer (f) by padding_size bytes from the current position (SEEK_CUR)
			fseek(f, padding_size, SEEK_CUR);
		}
    }

    free(row_pixels);
    fclose(f);

    Image *img = (Image *)malloc(sizeof(Image));
    img->width = width;
    img->height = height;
    img->data = data;
    return img;
}

Image *readBMP_MPI(const char *file_name, int my_rank, int num_processes, int halo_dim, int *true_start, int *true_end){
	int check, local_error_flag = 0, global_error_flag = 0;
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
	
	unsigned char *row_pixels = (unsigned char *)malloc(pixel_data_size);
	if(row_pixels == NULL) local_error_flag = 1;
	
	check = MPI_Reduce(&local_error_flag, &global_error_flag, 1, MPI_INT, MPI_LOR, 0, MPI_COMM_WORLD);
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Rank %d: Error in readBMP_MPI while comunicating error flag\n", my_rank);
		fflush(stderr);
		return NULL;
	}
	
	if(my_rank == 0 && global_error_flag == 1){
		fprintf(stderr, "Rank %d: Error in readBMP_MPI while allocating memory\n", my_rank);
		fflush(stderr);
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
	if(data == NULL) local_error_flag = 1;
	
	check = MPI_Reduce(&local_error_flag, &global_error_flag, 1, MPI_INT, MPI_LOR, 0, MPI_COMM_WORLD);
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Rank %d: Error in readBMP_MPI while comunicating error flag\n", my_rank);
		fflush(stderr);
		return NULL;
	}
	
	if(my_rank == 0 && global_error_flag == 1){
		fprintf(stderr, "Rank %d: Error in readBMP_MPI while allocating memory\n", my_rank);
		fflush(stderr);
		return NULL;
	}
	
	for (int y = 0; y < local_rows; y++){
		check = MPI_File_read_at(image_file_handler, local_offset, row_pixels, pixel_data_size, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
		if(check != MPI_SUCCESS){
			if(my_rank == 0){
				fprintf(stderr, "Rank %d: Error in readBMP_MPI while reading from file file %s\n", my_rank, file_name);
				fflush(stderr);
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
	if(img == NULL) local_error_flag = 1;
	
	check = MPI_Reduce(&local_error_flag, &global_error_flag, 1, MPI_INT, MPI_LOR, 0, MPI_COMM_WORLD);
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Rank %d: Error in readBMP_MPI while comunicating error flag\n", my_rank);
		fflush(stderr);
		return NULL;
	}
	
	if(my_rank == 0 && global_error_flag == 1){
		fprintf(stderr, "Rank %d: Error in readBMP_MPI while allocating memory\n", my_rank);
		fflush(stderr);
		return NULL;
	}
	
	img->width = width;
	img->height = local_rows;
	img->data = data;
	return img;
}

Image *compose_BMP(Image *img, int my_rank, int num_processes){
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
			return NULL;
		}
		
		displacements = (int*)malloc(num_processes * sizeof(int));
		if(displacements == NULL){
			fprintf(stderr, "Rank %d: Error in compose_BMP while allocating memory\n", my_rank);
			fflush(stderr);
			return NULL;
		}
		
		receives = (int*)malloc(num_processes * sizeof(int));
		if(receives == NULL){
			fprintf(stderr, "Rank %d: Error in compose_BMP while allocating memory\n", my_rank);
			fflush(stderr);
			return NULL;
		}

		for(int i = 0; i < num_processes; ++i){
			total_height += heights[i];
		}
		
		data = (RGB *)malloc(width * total_height * sizeof(RGB));
		if(data == NULL){
			fprintf(stderr, "Rank %d: Error in compose_BMP while allocating memory\n", my_rank);
			fflush(stderr);
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
		return NULL;
	}
	
	check = MPI_Type_free(&mpi_rgb);
	if(check != MPI_SUCCESS){
		fprintf(stderr, "Rank %d: Error in compose_BMP while de-allocating data type\n", my_rank);
		fflush(stderr);
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
	FILE *f = fopen(filename, "rb");
    if (!f){
        fprintf(stderr, "Error: Could not open file %s\n", filename);
		fflush(stderr);
        return NULL;
    }

    unsigned char header[54];
    if (fread(header, sizeof(unsigned char), 54, f) != 54){
        fprintf(stderr, "Error: Invalid BMP header\n");
		fflush(stderr);
        fclose(f);
        return NULL;
    }

    if (header[0] != 'B' || header[1] != 'M'){
        fprintf(stderr, "Error: Not a valid BMP file\n");
		fflush(stderr);
        fclose(f);
        return NULL;
    }
	
	int bitsPerPixel = *(short *)&header[28];
	if (bitsPerPixel != 24){
        fprintf(stderr, "Error: Only 24-bit BMPs are supported\n");
		fflush(stderr);
        fclose(f);
        return NULL;
    }
	
	*height = *(int *)&header[22];
	*width = *(int *)&header[18];
	*data_start = *(int *)&header[10];
	
	int row_padded = (*width * 3 + 3) & (~3);
	int pixel_data_size = *width * 3;
	
	*padding = row_padded - pixel_data_size;
	
	return f;
}

Image *readBMP_chunk(FILE *image_file, int halo_dim, int chunk_size, int height, int width, int padding, int data_start, int *offset, int *true_start, int *true_end){
	int pixel_data_size = width * 3;
	int offset_stride = width * 3 + padding;
	int next_row = (*offset - data_start) / offset_stride;
	int halo_available = halo_dim;
	int rows_to_read;
	
	if(next_row >= height){
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
		MPI_Abort(MPI_COMM_WORLD, -1);
	}
	
	RGB *data = (RGB*)malloc(rows_to_read * width * sizeof(RGB));
	if(data == NULL){
		fprintf(stderr, "Rank 0: Error in readBMP_chunk while allocating memory\n");
		fflush(stderr);
		MPI_Abort(MPI_COMM_WORLD, -1);
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

int scatter_data(Image *img, RGB **data, int my_rank, int num_processes, int width, int *local_height, int halo_dim){
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

int saveBMP(const char *filename, const Image *img)
{
    FILE *f = fopen(filename, "wb");
    if (!f)
    {
        fprintf(stderr, "Error: Could not create file %s\n", filename);
		fflush(stderr);
        return 0;
    }

    int width = img->width;
    int height = img->height;
	int pixel_data_size = width * 3;
    int row_padded = (width * 3 + 3) & (~3);
	int padding_size = row_padded - pixel_data_size;
    int fileSize = 54 + row_padded * height;

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
    *(int *)&header[2] = fileSize;
    *(int *)&header[18] = width;
    *(int *)&header[22] = height;

    fwrite(header, sizeof(unsigned char), 54, f);

    unsigned char *row_pixels = (unsigned char *)malloc(pixel_data_size);
	unsigned char padding[3] = {0, 0, 0};
    if (!row_pixels)
    {
        fprintf(stderr, "Error: Memory allocation failed\n");
		fflush(stderr);
        fclose(f);
        return 0;
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
        fwrite(row_pixels, sizeof(unsigned char), pixel_data_size, f);
		if(padding_size > 0)
		{
			fwrite(padding, sizeof(unsigned char), padding_size, f);
		}
    }

    free(row_pixels);
    fclose(f);
    return 1;
}
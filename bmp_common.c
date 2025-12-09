#include <stdio.h>
#include <stdlib.h>
#include "bmp_common.h"

void copy_RGB(const RGB *src, RGB *dest){
	dest->r = src->r;
	dest->g = src->g;
	dest->b = src->b;
}

int saveBMP(const char *filename, const Image *img)
{
    FILE *f = fopen(filename, "wb");
    if (!f)
    {
        fprintf(stderr, "Error: Could not create file %s\n", filename);
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
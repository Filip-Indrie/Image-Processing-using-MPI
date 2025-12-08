#include "bmp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Image *readBMP(const char *filename)
{
    FILE *f = fopen(filename, "rb");
    if (!f)
    {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return NULL;
    }

    unsigned char header[54];
    if (fread(header, sizeof(unsigned char), 54, f) != 54)
    {
        fprintf(stderr, "Error: Invalid BMP header\n");
        fclose(f);
        return NULL;
    }

    if (header[0] != 'B' || header[1] != 'M')
    {
        fprintf(stderr, "Error: Not a valid BMP file\n");
        fclose(f);
        return NULL;
    }

    int width = *(int *)&header[18];
    int height = *(int *)&header[22];
    int bitsPerPixel = *(short *)&header[28];

	int dataOffset = *(int *)&header[10];
	fseek(f, dataOffset, SEEK_SET);

    if (bitsPerPixel != 24)
    {
        fprintf(stderr, "Error: Only 24-bit BMPs are supported\n");
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
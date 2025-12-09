#include <stdio.h>
#include <stdlib.h>
#include "bmp_serial.h"

Image *readBMP_serial(const char *filename)
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

	int data_offset = *(int *)&header[10];
	fseek(f, data_offset, SEEK_SET);

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
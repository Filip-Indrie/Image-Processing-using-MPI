#ifndef BMP_HEADER

#define BMP_HEADER

typedef struct
{
    unsigned char r, g, b;
} RGB; // one RGB point

typedef struct
{
    int width;
    int height;
    RGB *data;
} Image; // a BMP image as an array of RGB points

/* Read BMP file, build and return Image struct */
Image *readBMP(const char *filename);

/* Save Image in file in BMP format */
int saveBMP(const char *filename, const Image *img);

#endif
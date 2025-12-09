#ifndef BMP_COMMON

#define BMP_COMMON

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

void copy_RGB(const RGB *src, RGB *dest);
int saveBMP(const char *filename, const Image *img);

#endif
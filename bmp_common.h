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

/*
*	static - makes the function private to all the files compilng it
*	inline - wherever this function is called it replaces the function call with the body of the function
*/
static inline void copy_RGB(const RGB *src, RGB *dest){
	dest->r = src->r;
	dest->g = src->g;
	dest->b = src->b;
}

static inline int equal_RGB(const RGB rgb1, const RGB rgb2){
	return rgb1.r == rgb2.r && rgb1.g == rgb2.g && rgb1.b == rgb2.b;
}

#endif
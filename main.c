#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "convolution.h"

/**
*	TODO:
*		- make the operation selectable through command line args
		- choose between serial and parallel through command line args
		- if parallel, specify if there is a shared file tree or not through command line args
		- make different versions of pad_image and generate_kernel since when being paralleled, when a process encounteres an error, process 0 should terminate all its children and exit.
*/

int main(){
	printf("da\n");
	return 0;
}
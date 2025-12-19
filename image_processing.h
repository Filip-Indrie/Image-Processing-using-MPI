#ifndef IMAGE_PROCESSING

#define IMAGE_PROCESSING

Image *image_processing_serial(const char *in_file_name, operation_t operation);
Image *image_processing_parallel_sft(const char *in_file_name, operation_t operation, int my_rank, int num_processes, int num_cores);
Image *image_processing_parallel_no_sft(const char *in_file_name, operation_t operation, int my_rank, int num_processes, int num_cores, int num_workstations);
Image *image_processing_master(const char *in_file_name, operation_t operation, int chunk_size, int my_rank, int num_processes, int num_cores, int num_workstations);
int images_are_identical(Image *img1, Image *img2);

#endif
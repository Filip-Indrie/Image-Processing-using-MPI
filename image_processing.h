#ifndef IMAGE_PROCESSING

#define IMAGE_PROCESSING

Image *image_processing_serial(const char *in_file_name, const char *out_file_name, operation_t operation, int save);
Image *image_processing_parallel_sft(const char *in_file_name, const char *out_file_name, operation_t operation, int my_rank, int num_processes, int num_cores, int save);
Image *image_processing_parallel_no_sft(const char *in_file_name, const char *out_file_name, operation_t operation, int my_rank, int num_processes, int num_cores, int num_workstations, int save);
int image_is_correct(Image *img, char *in_file_name, char *out_file_name, operation_t operation, double parallel_time);

#endif
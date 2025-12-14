#ifndef IMAGE_PROCESSING

#define IMAGE_PROCESSING

int image_processing_serial(const char *in_file_name, const char *out_file_name, operation_t operation);
int image_processing_parallel_sft(const char *in_file_name, const char *out_file_name, operation_t operation, int my_rank, int num_processes, int num_cores);
int image_processing_parallel_no_sft(const char *in_file_name, const char *out_file_name, operation_t operation, int my_rank, int num_processes, int num_cores, int num_workstations);

#endif
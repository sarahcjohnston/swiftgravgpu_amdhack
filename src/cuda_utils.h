#ifndef SWIFT_CUDA_UTILS
#define SWIFT_CUDA_UTILS

#include <stdbool.h>
#include <error.h>
#include <cuda_runtime.h>

#define CRASH_ON_CUDA_ERROR(fn_call, err_str) if(fn_call != cudaSuccess) error(err_str)

struct device_host_pair_float {
  float *device;
  float *host;
  int size;
};

struct device_host_pair_float init_device_host_pair_float(int number_of_items);
void free_device_host_pair_float(struct device_host_pair_float *in);
void host_to_device_float(const struct device_host_pair_float *in, bool is_async, cudaStream_t stream);
void device_to_host_float(const struct device_host_pair_float *in, bool is_async, cudaStream_t stream);

struct device_host_pair_int {
  int *device;
  int *host;
  int size;
};

struct device_host_pair_int init_device_host_pair_int(int number_of_items);
void free_device_host_pair_int(struct device_host_pair_int *in);
void host_to_device_int(const struct device_host_pair_int *in, bool is_async, cudaStream_t stream);
void device_to_host_int(const struct device_host_pair_int *in, bool is_async, cudaStream_t stream);

#endif

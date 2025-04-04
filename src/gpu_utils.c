#include "gpu_utils.h"

/**
 * @brief 
 *
 * @param number_of_items Number of items to be allocated. Not number of bytes.
 */
struct device_host_pair_float init_device_host_pair_float(int number_of_items) {
  struct device_host_pair_float result = {
    .device = NULL,
    .host = NULL,
    .size = number_of_items,
  };

	CRASH_ON_CUDA_ERROR(cudaMallocHost((void **)&result.host, number_of_items * sizeof(float)), "Failed to allocate host memory");
	CRASH_ON_CUDA_ERROR(cudaMalloc((void **)&result.device, number_of_items * sizeof(float)), "Failed to allocate device memory");

  return result;
}

void free_device_host_pair_float(struct device_host_pair_float *in) {
  if(in->host != NULL) {
    CRASH_ON_CUDA_ERROR(cudaFreeHost(in->host), "Failed to free host memory");
    in->host = NULL;
  }
  if(in->device != NULL) {
    CRASH_ON_CUDA_ERROR(cudaFree(in->device), "Failed to free device memory");
    in->device = NULL;
  }
}

void host_to_device_float(const struct device_host_pair_float *in, bool is_async, cudaStream_t stream) {
  if(is_async) {
    CRASH_ON_CUDA_ERROR(cudaMemcpyAsync(in->device, in->host, in->size * sizeof(float), cudaMemcpyHostToDevice, stream), "Failed to host_to_device cudaMemcpyAsync");
  } else {
    CRASH_ON_CUDA_ERROR(cudaMemcpy(in->device, in->host, in->size * sizeof(float), cudaMemcpyHostToDevice), "Failed to host_to_device cudaMemcpyAsync");
  }
}

void device_to_host_float(const struct device_host_pair_float *in, bool is_async, cudaStream_t stream) {
  if(is_async) {
    CRASH_ON_CUDA_ERROR(cudaMemcpyAsync(in->host, in->device, in->size * sizeof(float), cudaMemcpyDeviceToHost, stream), "Failed to device_to_host cudaMemcpyAsync");
  } else {
    CRASH_ON_CUDA_ERROR(cudaMemcpy(in->host, in->device, in->size * sizeof(float), cudaMemcpyDeviceToHost), "Failed to device_to_host cudaMemcpyAsync");
  }
}

/**
 * @brief 
 *
 * @param number_of_items Number of items to be allocated. Not number of bytes.
 */
struct device_host_pair_int init_device_host_pair_int(int number_of_items) {
  struct device_host_pair_int result = {
    .device = NULL,
    .host = NULL,
    .size = number_of_items,
  };

	CRASH_ON_CUDA_ERROR(cudaMallocHost((void **)&result.host, number_of_items * sizeof(int)), "Failed to allocate host memory");
	CRASH_ON_CUDA_ERROR(cudaMalloc((void **)&result.device, number_of_items * sizeof(int)), "Failed to allocate device memory");

  return result;
}

void host_to_device_int(const struct device_host_pair_int *in, bool is_async, cudaStream_t stream) {
  if(is_async) {
    CRASH_ON_CUDA_ERROR(cudaMemcpyAsync(in->device, in->host, in->size * sizeof(int), cudaMemcpyHostToDevice, stream), "Failed to host_to_device cudaMemcpyAsync");
  } else {
    CRASH_ON_CUDA_ERROR(cudaMemcpy(in->device, in->host, in->size * sizeof(int), cudaMemcpyHostToDevice), "Failed to host_to_device cudaMemcpyAsync");
  }
}

void device_to_host_int(const struct device_host_pair_int *in, bool is_async, cudaStream_t stream) {
  if(is_async) {
    CRASH_ON_CUDA_ERROR(cudaMemcpyAsync(in->host, in->device, in->size * sizeof(int), cudaMemcpyDeviceToHost, stream), "Failed to device_to_host cudaMemcpyAsync");
  } else {
    CRASH_ON_CUDA_ERROR(cudaMemcpy(in->host, in->device, in->size * sizeof(int), cudaMemcpyDeviceToHost), "Failed to device_to_host cudaMemcpyAsync");
  }
}

void free_device_host_pair_int(struct device_host_pair_int *in) {
  if(in->host != NULL) {
    CRASH_ON_CUDA_ERROR(cudaFreeHost(in->host), "Failed to free host memory");
    in->host = NULL;
  }
  if(in->device != NULL) {
    CRASH_ON_CUDA_ERROR(cudaFree(in->device), "Failed to free device memory");
    in->device = NULL;
  }
}

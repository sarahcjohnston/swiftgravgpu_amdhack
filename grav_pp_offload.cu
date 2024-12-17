#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>

#include "externalfunctions.cu"
#include "multipole_struct.h"

/* Local Cuda includes */
#include "src/gpu_params.h"

extern "C" void gpu_device_props(struct gpu_info *gpu_info) {

  /* Set the device ID */
  cudaGetDevice(&gpu_info->device_id);

  /* Get the device properties */
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, gpu_info->device_id);

  /* Set the number of streaming multiprocessors */
  gpu_info->nr_sm = deviceProp.multiProcessorCount;

  /* Set the maximum number of threads per SM */
  gpu_info->max_threads_per_sm = deviceProp.maxThreadsPerMultiProcessor;

  /* Set the maximum number of threads per block */
  gpu_info->max_threads_per_block = deviceProp.maxThreadsPerBlock;

  /* Set the maximum number of blocks per SM */
  gpu_info->max_blocks_per_sm = deviceProp.maxBlocksPerMultiProcessor;

  /* Set the maximum amount of shared memory per SM */
  gpu_info->max_shared_memory_per_sm = deviceProp.sharedMemPerMultiprocessor;

  /* Set the maximum amount of shared memory per block */
  gpu_info->max_shared_memory_per_block = deviceProp.sharedMemPerBlock;

  /* Set the maximum number of registers per block */
  gpu_info->max_registers_per_block = deviceProp.regsPerBlock;

  /* Set the warp size */
  gpu_info->warp_size = deviceProp.warpSize;

  /* Set the maximum number of threads per block dimension */
  gpu_info->max_threads_per_block_dimension = deviceProp.maxThreadsDim[0];

  /* Set the maximum grid size */
  gpu_info->max_grid_size = deviceProp.maxGridSize[0];

  /* Set the maximum number of threads per block dimension x */
  gpu_info->max_threads_per_block_dimension_x = deviceProp.maxThreadsDim[0];

  /* Set the maximum number of threads per block dimension y */
  gpu_info->max_threads_per_block_dimension_y = deviceProp.maxThreadsDim[1];

  /* Set the maximum number of threads per block dimension z */
  gpu_info->max_threads_per_block_dimension_z = deviceProp.maxThreadsDim[2];
}

//PP ALL INTERACTIONS
__global__ void pair_grav_pp(int periodic, const float *CoM_i, const float *CoM_j, float rmax_i, float rmax_j, double min_trunc, int *active_i, int *active_j, float dim_0, float dim_1, float dim_2, float *h_i, float *h_j, float *mass_i_arr, float *mass_j_arr, const float r_s_inv, const float *x_i, const float *x_j, const float *y_i, const float *y_j, const float *z_i, const float *z_j, float *a_x_i, float *a_y_i, float *a_z_i, float *a_x_j, float *a_y_j, float *a_z_j, float *pot_i, float *pot_j, int gcount_i, int gcount_padded_i, int gcount_j, int gcount_padded_j, int ci_active, int cj_active, const int symmetric, float *epsilon) {

   int max_r_decision = 0;

    /* GPU-ported copy of the existing SWIFT decision tree
       "Can we use the Newtonian version or do we need the truncated one ?"

	NON-PERIODIC BC
       "Not periodic -> Can always use Newtonian potential
       Let's updated the active cell(s) only" */

      /* Full P2P */
      grav_pp_full(active_i, dim_0, dim_1, dim_2, h_i, h_j, mass_j_arr, r_s_inv, x_i, x_j, y_i, y_j, z_i, z_j, a_x_i, a_y_i, a_z_i, pot_i, gcount_i, gcount_padded_j, periodic, ci_active, 0, symmetric, max_r_decision);

      /* No M2P in GPU version */

      /* Full P2P */
      grav_pp_full(active_j, dim_0, dim_1, dim_2, h_j, h_i, mass_i_arr, r_s_inv, x_j, x_i, y_j, y_i, z_j, z_i, a_x_j, a_y_j, a_z_j, pot_j, gcount_j, gcount_padded_i, periodic, 0, cj_active, symmetric, max_r_decision);

    /* Periodic BC */

    /* Get the relative distance between the CoMs */
    double d[3] = {CoM_j[0] - CoM_i[0], CoM_j[1] - CoM_i[1],
                    CoM_j[2] - CoM_i[2]};

    /* Correct for periodic BCs */
    d[0] = nearestf1(d[0], dim_0);
    d[1] = nearestf1(d[1], dim_1);
    d[2] = nearestf1(d[2], dim_2);

    const double r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];

    /* Get the maximal distance between any two particles */
    const double max_r = sqrt(r2) + rmax_i + rmax_j;

    /* Apply decision for whether you need to use truncated interactions or not */
    if (max_r > min_trunc) {
	max_r_decision = 0;}
    else {
	max_r_decision = 1;}

    /* "Do we need to use the truncated interactions ?
    

       Periodic but far-away cells must use the truncated potential 

       Let's updated the active cell(s) only" */

        /* Truncated P2P - cell i */
        grav_pp_truncated(active_i, dim_0, dim_1, dim_2, h_i, h_j, mass_j_arr, r_s_inv, x_i, x_j, y_i, y_j, z_i, z_j, a_x_i, a_y_i, a_z_i, pot_i, gcount_i, gcount_padded_j, periodic, ci_active, 0, symmetric, max_r_decision);
	

        /* Truncated P2P - cell j */
	grav_pp_truncated(active_j, dim_0, dim_1, dim_2, h_j, h_i, mass_i_arr, r_s_inv, x_j, x_i, y_j, y_i, z_j, z_i, a_x_j, a_y_j, a_z_j, pot_j, gcount_j, gcount_padded_i, periodic, 0, cj_active, symmetric, max_r_decision);

        

      /* "Periodic but close-by cells can use the full Newtonian potential

       Let's updated the active cell(s) only" */

        /* Full P2P - cell i */
        grav_pp_full(active_i, dim_0, dim_1, dim_2, h_i, h_j, mass_j_arr, r_s_inv, x_i, x_j, y_i, y_j, z_i, z_j, a_x_i, a_y_i, a_z_i, pot_i, gcount_i, gcount_padded_j, periodic, ci_active, 0, symmetric, max_r_decision);

        /* Full P2P - cell j */
        grav_pp_full(active_j, dim_0, dim_1, dim_2, h_j, h_i, mass_i_arr, r_s_inv, x_j, x_i, y_j, y_i, z_j, z_i, a_x_j, a_y_j, a_z_j, pot_j, gcount_j, gcount_padded_i, periodic, 0, cj_active, symmetric, max_r_decision);

}


//Main definition of pp_offload function to go into normal SWIFT C code
extern "C" void pp_offload(int periodic, const float *CoM_i, const float *CoM_j, float rmax_i, float rmax_j, double min_trunc, int* active_i, int* active_j, float *dim, const float *x_i, const float *x_j_arr, const float *y_i, const float *y_j_arr, const float *z_i, const float *z_j_arr, float *pot_i, float *pot_j, float *a_x_i, float *a_y_i, float *a_z_i, float *a_x_j, float *a_y_j, float *a_z_j, float *mass_i_arr, float *mass_j_arr, const float *r_s_inv, float *h_i, float *h_j_arr, const int *gcount_i, const int *gcount_padded_i, const int *gcount_j, const int *gcount_padded_j, int ci_active, int cj_active, const int symmetric, float *epsilon){
	
	float a_x_i_new[*gcount_i];
	float a_y_i_new[*gcount_i];
	float a_z_i_new[*gcount_i];
	float pot_i_new[*gcount_i];

	float a_x_j_new[*gcount_j];
	float a_y_j_new[*gcount_j];
	float a_z_j_new[*gcount_j];
	float pot_j_new[*gcount_j];

	//create device pointers
	float *d_h_i;
	float *d_h_j;
	float *d_mass_i;
	float *d_mass_j;
	float *d_x_i;
	float *d_x_j;
	float *d_y_i;
	float *d_y_j;
	float *d_z_i;
	float *d_z_j;
	float *d_a_x_i;
	float *d_a_y_i;
	float *d_a_z_i;
	float *d_a_x_j;
	float *d_a_y_j;
	float *d_a_z_j;
	float *d_pot_i;
	float *d_pot_j;
	int *d_active_i;
	int *d_active_j;
	float *d_CoM_i;
	float *d_CoM_j;

	//allocate memory on device
	cudaMalloc(&d_h_i, *gcount_i * sizeof(float));
	cudaMalloc(&d_h_j, *gcount_padded_j * sizeof(float));
	cudaMalloc(&d_mass_i, *gcount_padded_i * sizeof(float));
	cudaMalloc(&d_mass_j, *gcount_padded_j * sizeof(float));
	cudaMalloc(&d_x_i, *gcount_i * sizeof(float));
	cudaMalloc(&d_x_j, *gcount_padded_j * sizeof(float));
	cudaMalloc(&d_y_i, *gcount_i * sizeof(float));
	cudaMalloc(&d_y_j, *gcount_padded_j * sizeof(float));
	cudaMalloc(&d_z_i, *gcount_i * sizeof(float));
	cudaMalloc(&d_z_j, *gcount_padded_j * sizeof(float));
	cudaMalloc(&d_a_x_i, *gcount_i * sizeof(float));
	cudaMalloc(&d_a_y_i, *gcount_i * sizeof(float));
	cudaMalloc(&d_a_z_i, *gcount_i * sizeof(float));
	cudaMalloc(&d_a_x_j, *gcount_j * sizeof(float));
	cudaMalloc(&d_a_y_j, *gcount_j * sizeof(float));
	cudaMalloc(&d_a_z_j, *gcount_j * sizeof(float));
	cudaMalloc(&d_pot_i, *gcount_i * sizeof(float));
	cudaMalloc(&d_pot_j, *gcount_j * sizeof(float));
	cudaMalloc(&d_active_i, *gcount_i * sizeof(int));
	cudaMalloc(&d_active_j, *gcount_j * sizeof(int));
	cudaMalloc(&d_CoM_i, 3 * sizeof(float));
	cudaMalloc(&d_CoM_j, 3 * sizeof(float));

	//copy data to device
	cudaMemcpyAsync(d_h_i, h_i, *gcount_i * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_h_j, h_j_arr, *gcount_padded_j * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_mass_i, mass_i_arr, *gcount_padded_i * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_mass_j, mass_j_arr, *gcount_padded_j * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_x_i, x_i, *gcount_i * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_x_j, x_j_arr, *gcount_padded_j * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_y_i, y_i, *gcount_i * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_y_j, y_j_arr, *gcount_padded_j * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_z_i, z_i, *gcount_i * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_z_j, z_j_arr, *gcount_padded_j * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_a_x_i, a_x_i, *gcount_i * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_a_y_i, a_y_i, *gcount_i * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_a_z_i, a_z_i, *gcount_i * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_a_x_j, a_x_j, *gcount_j * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_a_y_j, a_y_j, *gcount_j * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_a_z_j, a_z_j, *gcount_j * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_pot_i, pot_i, *gcount_i * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_pot_j, pot_j, *gcount_j * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_active_i, active_i, *gcount_i * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_active_j, active_j, *gcount_j * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_CoM_i, CoM_i, 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_CoM_j, CoM_j, 3 * sizeof(float), cudaMemcpyHostToDevice);
	
	cudaError_t err = cudaGetLastError();
    	if (err != cudaSuccess) 
	printf("Error1: %s\n", cudaGetErrorString(err));

	//call kernel function
	pair_grav_pp<<<32,256>>>(periodic, d_CoM_i, d_CoM_j, rmax_i, rmax_j, min_trunc, d_active_i, d_active_j, dim[0], dim[1], dim[2], d_h_i, d_h_j, d_mass_i, d_mass_j, *r_s_inv, d_x_i, d_x_j, d_y_i, d_y_j, d_z_i, d_z_j, d_a_x_i, d_a_y_i, d_a_z_i, d_a_x_j, d_a_y_j, d_a_z_j, d_pot_i, d_pot_j, *gcount_i, *gcount_padded_i, *gcount_j, *gcount_padded_j, ci_active, cj_active, symmetric, epsilon);


	cudaError_t err2 = cudaGetLastError();
    	if (err2 != cudaSuccess)
	printf("Error2: %s\n", cudaGetErrorString(err2));

	//copy data from device
	cudaMemcpyAsync(&a_x_i_new, d_a_x_i, *gcount_i*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(&a_y_i_new, d_a_y_i, *gcount_i*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(&a_z_i_new, d_a_z_i, *gcount_i*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(&pot_i_new, d_pot_i, *gcount_i*sizeof(float), cudaMemcpyDeviceToHost);

	cudaMemcpyAsync(&a_x_j_new, d_a_x_j, *gcount_j*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(&a_y_j_new, d_a_y_j, *gcount_j*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(&a_z_j_new, d_a_z_j, *gcount_j*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(&pot_j_new, d_pot_j, *gcount_j*sizeof(float), cudaMemcpyDeviceToHost);

	cudaError_t err3 = cudaGetLastError();
    	if (err3 != cudaSuccess)
	printf("Error3: %s\n", cudaGetErrorString(err3));

	for (int i = 0; i < *gcount_i; i++){
		a_x_i[i] += a_x_i_new[i];
		a_y_i[i] += a_y_i_new[i];
		a_z_i[i] += a_z_i_new[i];
		pot_i[i] += pot_i_new[i];}
		
	for (int i = 0; i < *gcount_j; i++){
		a_x_j[i] += a_x_j_new[i];
		a_y_j[i] += a_y_j_new[i];
		a_z_j[i] += a_z_j_new[i];
		pot_j[i] += pot_j_new[i];}

	//free memory
	cudaFree(d_h_i);
	cudaFree(d_h_j);
	cudaFree(d_mass_i);
	cudaFree(d_mass_j);
	cudaFree(d_x_i);
	cudaFree(d_x_j);
	cudaFree(d_y_i);
	cudaFree(d_y_j);
	cudaFree(d_z_i);
	cudaFree(d_z_j);
	cudaFree(d_a_x_i);
	cudaFree(d_a_y_i);
	cudaFree(d_a_z_i);
	cudaFree(d_a_x_j);
	cudaFree(d_a_y_j);
	cudaFree(d_a_z_j);
	cudaFree(d_pot_i);
	cudaFree(d_pot_j);
	cudaFree(d_active_i);
	cudaFree(d_active_j);
	cudaFree(d_CoM_i);
	cudaFree(d_CoM_j);

	cudaError_t err4 = cudaGetLastError();
    	if (err4 != cudaSuccess)
	printf("Error4: %s\n", cudaGetErrorString(err4));
	
}

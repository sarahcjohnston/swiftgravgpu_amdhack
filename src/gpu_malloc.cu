#include <cuda.h>
#include <hip_runtime.h>
#include <stdio.h>

/* In GPU-land need to allocate memory for the values */
/*
#ifdef __cplusplus
extern "C" {
#endif
void gpu_malloc(struct swift_params *params,  float *d_h_i, float *d_h_j, float *d_mass_i, float *d_mass_j, float *d_x_i, float *d_x_j, float *d_y_i, float *d_y_j, float *d_z_i, float *d_z_j, float *d_a_x_i, float *d_a_y_i, float *d_a_z_i, float *d_a_x_j, float *d_a_y_j, float *d_a_z_j, float *d_pot_i, float *d_pot_j, int *d_active_i, int *d_active_j, float *d_CoM_i, float *d_CoM_j){
  //create device pointers
	/*float *d_h_i;
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
	float *d_CoM_j;*/
	
	/*float max_cell_size = parser_get_opt_param_int(params, "Scheduler:cell_split_size", space_splitsize);

	//allocate memory on device
	cudaMalloc(&d_h_i, max_cell_size * sizeof(float));
	cudaMalloc(&d_h_j, max_cell_size * sizeof(float));
	cudaMalloc(&d_mass_i, max_cell_size * sizeof(float));
	cudaMalloc(&d_mass_j, max_cell_size * sizeof(float));
	cudaMalloc(&d_x_i, max_cell_size * sizeof(float));
	cudaMalloc(&d_x_j, max_cell_size * sizeof(float));
	cudaMalloc(&d_y_i, max_cell_size * sizeof(float));
	cudaMalloc(&d_y_j, max_cell_size * sizeof(float));
	cudaMalloc(&d_z_i, max_cell_size * sizeof(float));
	cudaMalloc(&d_z_j, max_cell_size * sizeof(float));
	cudaMalloc(&d_a_x_i, max_cell_size * sizeof(float));
	cudaMalloc(&d_a_y_i, max_cell_size * sizeof(float));
	cudaMalloc(&d_a_z_i, max_cell_size * sizeof(float));
	cudaMalloc(&d_a_x_j, max_cell_size * sizeof(float));
	cudaMalloc(&d_a_y_j, max_cell_size * sizeof(float));
	cudaMalloc(&d_a_z_j, max_cell_size * sizeof(float));
	cudaMalloc(&d_pot_i, max_cell_size * sizeof(float));
	cudaMalloc(&d_pot_j, max_cell_size * sizeof(float));
	cudaMalloc(&d_active_i, max_cell_size * sizeof(int));
	cudaMalloc(&d_active_j, max_cell_size * sizeof(int));
	cudaMalloc(&d_CoM_i, 3 * sizeof(float));
	cudaMalloc(&d_CoM_j, 3 * sizeof(float));
}
#ifdef __cplusplus
}
#endif*/

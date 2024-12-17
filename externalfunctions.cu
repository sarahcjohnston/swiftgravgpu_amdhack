#include <config.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include "multipole_struct.h"
#include "error.h"
#include "gravity_derivatives.h"

__device__ float nearestf1(
    float dx, const float box_size) {

    return ((dx > 0.5 * box_size)
              ? (dx - box_size)
              : ((dx < -0.5 * box_size) ? (dx + box_size) : dx));


    /*float signx = round(dx/abs(dx+0.0000000000001));
    float adjustx = round(dx/box_size);
    dx -= adjustx*signx*box_size;

  return dx;*/
}

__device__ float grav_force_eval(const float u){

  float W;
#ifdef GADGET2_SOFTENING_CORRECTION
  float W_f_less = 10.6666667f + u * u * (32.f * u - 38.4f);
  float W_f_more = 21.3333333f - 48.f * u + 38.4f * u * u - 10.6666667f * u * u * u - 0.06666667f / (u * u * u);
  W = abs(round(round(round(u)/u)-1))*W_f_less + round(round(u)/u)*W_f_more;
#else

  /* W(u) = 21u^5 - 90u^4 + 140u^3 - 84u^2 + 14 */
  W = 21.f * u - 90.f;
  W = W * u + 140.f;
  W = W * u - 84.f;
  W = W * u;
  W = W * u + 14.f;
#endif
  return W;
}

__device__ float grav_pot_eval(const float u) {

  float W;
#ifdef GADGET2_SOFTENING_CORRECTION
  float W_pot_less = -2.8f + u * u * (5.333333333333f + u * u * (6.4f * u - 9.6f));
  float W_pot_more = -3.2f + 0.066666666667f / u + u * u * (10.666666666667f + u * (-16.f + u * (9.6f - 2.133333333333f * u)));
  W = abs(round(round(round(u)/u)-1))*W_pot_less + round(round(u)/u)*W_pot_more;
#else

  /* W(u) = 3u^7 - 15u^6 + 28u^5 - 21u^4 + 7u^2 - 3 */
  W = 3.f * u - 15.f;
  W = W * u + 28.f;
  W = W * u - 21.f;
  W = W * u;
  W = W * u + 7.f;
  W = W * u;
  W = W * u - 3.f;
#endif
  return W;
}

__device__ float long_grav_eval(const float r_over_r_s, float *corr_f, float *corr_pot){
#ifdef GADGET2_LONG_RANGE_CORRECTION

  const float two_over_sqrt_pi = ((float)M_2_SQRTPI);

  const float u = 0.5f * r_over_r_s;
  const float u2 = u * u;
  const float exp_u2 = expf(-u2);

  /* Compute erfcf(u) using eq. 7.1.26 of
   * Abramowitz & Stegun, 1972.
   *
   * This has a *relative* error of less than 3.4e-3 over
   * the range of interest (0 < u < 5)\
   *
   * This is a good approximation to use since we already
   * need exp(-u2) */

  const float t = 1.f / (1.f + 0.3275911f * u);

  const float a1 = 0.254829592f;
  const float a2 = -0.284496736f;
  const float a3 = 1.421413741f;
  const float a4 = -1.453152027;
  const float a5 = 1.061405429f;

  /* a1 * t + a2 * t^2 + a3 * t^3 + a4 * t^4 + a5 * t^5 */
  float a = a5 * t + a4;
  a = a * t + a3;
  a = a * t + a2;
  a = a * t + a1;
  a = a * t;

  const float erfc_u = a * exp_u2;

  *corr_pot = erfc_u;
  *corr_f = erfc_u + two_over_sqrt_pi * u * exp_u2;

#else
  const float x = 2.f * r_over_r_s;
  const float exp_x = expf(x);  // good_approx_expf(x);
  const float alpha = 1.f / (1.f + exp_x);

  /* We want 2 - 2 exp(x) * alpha */
  float W = 1.f - alpha * exp_x;
  W = W * 2.f;

  *corr_pot = W;

  /* We want 2*(x*alpha - x*alpha^2 - exp(x)*alpha + 1) */
  W = 1.f - alpha;
  W = W * x - exp_x;
  W = W * alpha + 1.f;
  W = W * 2.f;

  *corr_f = W;
#endif
}


__device__ void iact_grav_pp_full(const float r2, const float h2, const float h_inv, const float h_inv3, const float mass, float *f_ij, float *pot_ij) {

  /* Get the inverse distance */
  const float r_inv = 1.f / sqrtf(r2 + FLT_MIN);

  /* Should we soften ? */
  if (r2 >= h2) {

    /* Get Newtonian gravity */
    *f_ij = mass * r_inv * r_inv * r_inv;
    *pot_ij = -mass * r_inv;

  } else {

    const float r = r2 * r_inv;
    const float ui = r * h_inv;
    const float W_f_ij = grav_force_eval(ui);
    const float W_pot_ij = grav_pot_eval(ui);

    /* Get softened gravity */
    *f_ij = mass * h_inv3 * W_f_ij;
    *pot_ij = mass * h_inv * W_pot_ij;
  }
  

  /* Get the inverse distance */
  //const float r_inv = 1.f / sqrtf(r2); //no buffer

  /* Should we soften ? */
  /*float f_ij_full = mass * r_inv * r_inv * r_inv;
  float pot_ij_full = -mass * r_inv;

    const float r = r2 * r_inv;
    const float ui = r * h_inv;
    const float W_f_ij = grav_force_eval(ui);
    const float W_pot_ij = grav_pot_eval(ui);*/

    /* Get softened gravity */
    /*float f_ij_soft = mass * h_inv3 * W_f_ij;
    float pot_ij_soft = mass * h_inv * W_pot_ij;
  
  
  *f_ij = f_ij_full;
  *pot_ij = pot_ij_full;

  if (r2 < h2){
  	*f_ij = f_ij_soft;
  	*pot_ij = pot_ij_soft;
	}*/
}

__device__ void iact_grav_pp_truncated(const float r2, const float h2, const float h_inv, const float h_inv3, const float mass, const float r_s_inv, float *f_ij, float *pot_ij){

  /* Get the inverse distance */
  const float r_inv = 1.f / sqrtf(r2 + FLT_MIN);
  const float r = r2 * r_inv;

  /* Should we soften ? */
  if (r2 >= h2) {
    
    /* Get Newtonian gravity */
    *f_ij = mass * r_inv * r_inv * r_inv;
    *pot_ij = -mass * r_inv;

  } else {

    const float ui = r * h_inv;
    const float W_f_ij = grav_force_eval(ui);
    const float W_pot_ij = grav_pot_eval(ui);

    /* Get softened gravity */
    *f_ij = mass * h_inv3 * W_f_ij;
    *pot_ij = mass * h_inv * W_pot_ij;
  }

  /* Get long-range correction */
  const float u_lr = r * r_s_inv;
  float corr_f_lr, corr_pot_lr;
  long_grav_eval(u_lr, &corr_f_lr, &corr_pot_lr);
  *f_ij *= corr_f_lr;
  *pot_ij *= corr_pot_lr;

   ////////////////////////////////////////////////////

  /* Get the inverse distance */
  /*const float r_inv = 1.f / sqrtf(r2); //no buffer
  const float r = r2 * r_inv;*/

  /* Should we soften ? */

    /* Get Newtonian gravity */
    /*float f_ij_full = mass * r_inv * r_inv * r_inv;
    float pot_ij_full = -mass * r_inv;

    const float ui = r * h_inv;
    const float W_f_ij = grav_force_eval(ui);
    const float W_pot_ij = grav_pot_eval(ui);*/

    /* Get softened gravity */
    /*float f_ij_soft = mass * h_inv3 * W_f_ij;
    float pot_ij_soft = mass * h_inv * W_pot_ij;

  *f_ij = f_ij_full;
  *pot_ij = pot_ij_full;

  if (r2 < h2){
  	*f_ij = f_ij_soft;
  	*pot_ij = pot_ij_soft;
	}*/

  /* Get long-range correction */
  /*const float u_lr = r * r_s_inv;
  float corr_f_lr, corr_pot_lr;
  long_grav_eval(u_lr, &corr_f_lr, &corr_pot_lr);
  *f_ij *= corr_f_lr;
  *pot_ij *= corr_pot_lr;*/
}

//PP FULL INTERACTIONS
__device__ void grav_pp_full(int* active, float dim_0, float dim_1, float dim_2, float *h_i, float *h_j, float *mass_j_arr, float r_s_inv, const float *x_i, const float *x_j, const float *y_i, const float *y_j, const float *z_i, const float *z_j, float *a_x_i, float *a_y_i, float *a_z_i, float *pot_i, const int gcount_i, const int gcount_padded_j, const int periodic, int ci_active, int cj_active, int symmetric, int max_r_decision){

    int t = blockIdx.x*blockDim.x +threadIdx.x;
    int T = blockDim.x*gridDim.x;
    int s = blockIdx.y*blockDim.y +threadIdx.y;
    int S = blockDim.y*gridDim.y;

    for (int pid = t; pid < gcount_i; pid+=T) {

    //Local accumulators for the acceleration and potential
    float a_x = 0.f, a_y = 0.f, a_z = 0.f, pot = 0.f;

    // Loop over every particle in the other cell.
    for (int pjd = s; pjd < gcount_padded_j; pjd+=S) {

      float mass_j = mass_j_arr[pjd];

      // Compute the pairwise distance.
      float dx = x_j[pjd] - x_i[pid];
      float dy = y_j[pjd] - y_i[pid];
      float dz = z_j[pjd] - z_i[pid];

      // Correct for periodic BCs
      dx = nearestf1(dx, dim_0);
      dy = nearestf1(dy, dim_1);
      dz = nearestf1(dz, dim_2);

      const float r2 = dx * dx + dy * dy + dz * dz;

      // Pick the maximal softening length of i and j
      const float h = max(h_i[pid], h_j[pjd]);
      const float h2 = h * h;
      const float h_inv = 1.f / h;
      const float h_inv_3 = h_inv * h_inv * h_inv;

      // Interact!
      float f_ij, pot_ij;
      iact_grav_pp_full(r2, h2, h_inv, h_inv_3, mass_j, &f_ij, &pot_ij);

      // Store it back
      a_x += f_ij * dx;
      a_y += f_ij * dy;
      a_z += f_ij * dz;
      pot += pot_ij;
    }

    // Store everything back in cache
    //accounting for all 4 possibilities of whether treating cell i or j and whether periodic or not
    atomicAdd(&a_x_i[pid], a_x*active[pid]*ci_active*abs(periodic-1) +  a_x*active[pid]*cj_active*symmetric*abs(periodic-1) + a_x*active[pid]*ci_active*periodic*max_r_decision + a_x*active[pid]*cj_active*symmetric*periodic*max_r_decision);
    atomicAdd(&a_y_i[pid], a_y*active[pid]*ci_active*abs(periodic-1) +  a_y*active[pid]*cj_active*symmetric*abs(periodic-1) + a_y*active[pid]*ci_active*periodic*max_r_decision + a_y*active[pid]*cj_active*symmetric*periodic*max_r_decision);
    atomicAdd(&a_z_i[pid], a_z*active[pid]*ci_active*abs(periodic-1) +  a_z*active[pid]*cj_active*symmetric*abs(periodic-1) + a_z*active[pid]*ci_active*periodic*max_r_decision + a_z*active[pid]*cj_active*symmetric*periodic*max_r_decision);
    atomicAdd(&pot_i[pid], pot*active[pid]*ci_active*abs(periodic-1) +  pot*active[pid]*cj_active*symmetric*abs(periodic-1) + pot*active[pid]*ci_active*periodic*max_r_decision + pot*active[pid]*cj_active*symmetric*periodic*max_r_decision);
  }
}

//PP TRUNCATED INTERACTIONS
__device__ void grav_pp_truncated(int* active, float dim_0, float dim_1, float dim_2, float *h_i, float *h_j, float *mass_j_arr, const float r_s_inv, const float *x_i, const float *x_j, const float *y_i, const float *y_j, const float *z_i, const float *z_j, float *a_x_i, float *a_y_i, float *a_z_i, float *pot_i, const int gcount_i, const int gcount_padded_j, const int periodic, int ci_active, int cj_active, int symmetric, int max_r_decision){

    int t = blockIdx.x*blockDim.x +threadIdx.x;
    int T = blockDim.x*gridDim.x;
    int s = blockIdx.y*blockDim.y +threadIdx.y;
    int S = blockDim.y*gridDim.y;

  /* Loop over all particles in ci... */
  for (int pid = t; pid < gcount_i; pid+=T){

    /* Local accumulators for the acceleration and potential */
    float a_x = 0.f, a_y = 0.f, a_z = 0.f, pot = 0.f;

    /* Loop over every particle in the other cell. */
    for (int pjd = s; pjd < gcount_padded_j; pjd+=S){

      const float mass_j = mass_j_arr[pjd];
	
      //Compute the pairwise distance.
      float dx = x_j[pjd] - x_i[pid];
      float dy = y_j[pjd] - y_i[pid];
      float dz = z_j[pjd] - z_i[pid];

      /* Correct for periodic BCs */
      dx = nearestf1(dx, dim_0);
      dy = nearestf1(dy, dim_1);
      dz = nearestf1(dz, dim_2);

      const float r2 = dx * dx + dy * dy + dz * dz;

      /* Pick the maximal softening length of i and j */
      const float h = max(h_i[pid], h_j[pjd]);
      const float h2 = h * h;
      const float h_inv = 1.f / h;
      const float h_inv_3 = h_inv * h_inv * h_inv;

      /* Interact! */
      float f_ij, pot_ij;
      iact_grav_pp_truncated(r2, h2, h_inv, h_inv_3, mass_j, r_s_inv,
                                    &f_ij, &pot_ij);

      /* Store it back */
      a_x += f_ij * dx;
      a_y += f_ij * dy;
      a_z += f_ij * dz;
      pot += pot_ij;
    }

    /* Store everything back in cache */
    // treating both possibilities of whether treating cell i or cell j
    atomicAdd(&a_x_i[pid], a_x*active[pid]*ci_active*periodic*abs(max_r_decision-1) + a_x*active[pid]*cj_active*symmetric*periodic*abs(max_r_decision-1));
    atomicAdd(&a_y_i[pid], a_y*active[pid]*ci_active*periodic*abs(max_r_decision-1) + a_y*active[pid]*cj_active*symmetric*periodic*abs(max_r_decision-1));
    atomicAdd(&a_z_i[pid], a_z*active[pid]*ci_active*periodic*abs(max_r_decision-1) + a_z*active[pid]*cj_active*symmetric*periodic*abs(max_r_decision-1));
    atomicAdd(&pot_i[pid], pot*active[pid]*ci_active*periodic*abs(max_r_decision-1) + pot*active[pid]*cj_active*symmetric*periodic*abs(max_r_decision-1));
  }
}


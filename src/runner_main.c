/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2012 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
 *                    Matthieu Schaller (schaller@strw.leidenuniv.nl)
 *               2015 Peter W. Draper (p.w.draper@durham.ac.uk)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#include <unistd.h>

/* Config parameters. */
#include <config.h>

/* MPI headers. */
#ifdef WITH_MPI
#include <mpi.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif
/* GPU headers */
// #include "hip.h"
#include "hip/hip_runtime.h"
#ifdef __cplusplus
}
#endif

/* This object's header. */
#include "runner.h"

/* Local headers. */
#include "engine.h"
#include "feedback.h"
#include "gpu_malloc.h"
#include "gpu_params.h"
#include "parser.h"
#include "runner_doiact_sinks.h"
#include "scheduler.h"
#include "space_getsid.h"
#include "timers.h"

/* Import the gravity loop functions. */
#include "runner_doiact_grav.h"

/* Import the density loop functions. */
#define FUNCTION density
#define FUNCTION_TASK_LOOP TASK_LOOP_DENSITY
#include "runner_doiact_hydro.h"
#include "runner_doiact_undef.h"

/* Import the gradient loop functions (if required). */
#ifdef EXTRA_HYDRO_LOOP
#define FUNCTION gradient
#define FUNCTION_TASK_LOOP TASK_LOOP_GRADIENT
#include "runner_doiact_hydro.h"
#include "runner_doiact_undef.h"
#endif

/* Import the force loop functions. */
#define FUNCTION force
#define FUNCTION_TASK_LOOP TASK_LOOP_FORCE
#include "runner_doiact_hydro.h"
#include "runner_doiact_undef.h"

/* Import the limiter loop functions. */
#define FUNCTION limiter
#define FUNCTION_TASK_LOOP TASK_LOOP_LIMITER
#include "runner_doiact_limiter.h"
#include "runner_doiact_undef.h"

/* Import the stars density loop functions. */
#define FUNCTION density
#define FUNCTION_TASK_LOOP TASK_LOOP_DENSITY
#include "runner_doiact_stars.h"
#include "runner_doiact_undef.h"

#ifdef EXTRA_STAR_LOOPS

/* Import the stars prepare1 loop functions. */
#define FUNCTION prep1
#define FUNCTION_TASK_LOOP TASK_LOOP_STARS_PREP1
#include "runner_doiact_stars.h"
#include "runner_doiact_undef.h"

/* Import the stars prepare2 loop functions. */
#define FUNCTION prep2
#define FUNCTION_TASK_LOOP TASK_LOOP_STARS_PREP2
#include "runner_doiact_stars.h"
#include "runner_doiact_undef.h"

#endif /* EXTRA_STAR_LOOPS */

/* Import the stars feedback loop functions. */
#define FUNCTION feedback
#define FUNCTION_TASK_LOOP TASK_LOOP_FEEDBACK
#include "runner_doiact_stars.h"
#include "runner_doiact_undef.h"

/* Import the black hole density loop functions. */
#define FUNCTION density
#define FUNCTION_TASK_LOOP TASK_LOOP_DENSITY
#include "runner_doiact_black_holes.h"
#include "runner_doiact_undef.h"

/* Import the black hole feedback loop functions. */
#define FUNCTION swallow
#define FUNCTION_TASK_LOOP TASK_LOOP_SWALLOW
#include "runner_doiact_black_holes.h"
#include "runner_doiact_undef.h"

/* Import the black hole feedback loop functions. */
#define FUNCTION feedback
#define FUNCTION_TASK_LOOP TASK_LOOP_FEEDBACK
#include "runner_doiact_black_holes.h"
#include "runner_doiact_undef.h"

/* Import the RT gradient loop functions */
#define FUNCTION rt_gradient
#define FUNCTION_TASK_LOOP TASK_LOOP_RT_GRADIENT
#include "runner_doiact_hydro.h"
#include "runner_doiact_undef.h"

/* Import the RT transport (force) loop functions. */
#define FUNCTION rt_transport
#define FUNCTION_TASK_LOOP TASK_LOOP_RT_TRANSPORT
#include "runner_doiact_hydro.h"
#include "runner_doiact_undef.h"

extern void self_pp_offload(int periodic, float rmax_i, double min_trunc,
                            int *active_i, const float *x_i, const float *y_i,
                            const float *z_i, float *pot_i, float *a_x_i,
                            float *a_y_i, float *a_z_i, float *mass_i_arr,
                            const float *r_s_inv, float *h_i,
                            const int *gcount_i, const int *gcount_padded_i,
                            int ci_active, float *d_h_i, float *d_mass_i,
                            float *d_x_i, float *d_y_i, float *d_z_i,
                            float *d_a_x_i, float *d_a_y_i, float *d_a_z_i,
                            float *d_pot_i, int *d_active_i);
/**
 * @brief The #runner main thread routine.
 *
 * @param data A pointer to this thread's data.
 */
void *runner_main(void *data) {

  struct runner *r = (struct runner *)data;
  struct engine *e = r->e;
  struct scheduler *sched = &e->sched;

  // TODO: automate method for getting max cell size value
  int max_cell_size = 8000;  // parser_get_opt_param_int(params,
                             // "Scheduler:cell_split_size", space_splitsize);

  /* Main loop. */
  while (1) {

    /* Wait at the barrier. */
    engine_barrier(e);

    /* Can we go home yet? */
    if (e->step_props & engine_step_prop_done) break;

    /* Re-set the pointer to the previous task, as there is none. */
    struct task *t = NULL;
    struct task *prev = NULL;

    /* floats needed for GPU calculations */
    float *h_i;
    float *h_j;
    float *mass_i;
    float *mass_j;
    float *x_i;
    float *x_j;
    float *y_i;
    float *y_j;
    float *z_i;
    float *z_j;
    float *a_x_i;
    float *a_y_i;
    float *a_z_i;
    float *a_x_j;
    float *a_y_j;
    float *a_z_j;
    float *pot_i;
    float *pot_j;
    int *active_i;
    int *active_j;
    float *CoM_i;
    float *CoM_j;
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

    // define number of cells to transfer
    int ncells = 1;  // THIS VERSION ONLY WORKS FOR ONE CELL (which does
                     // somewhat negate the purpose but its getting there...)

    // allocate memory on host
    hipMallocHost((void **)&h_i, ncells * max_cell_size * sizeof(float));
    hipMallocHost((void **)&h_j, ncells * max_cell_size * sizeof(float));
    hipMallocHost((void **)&mass_i, ncells * max_cell_size * sizeof(float));
    hipMallocHost((void **)&mass_j, ncells * max_cell_size * sizeof(float));
    hipMallocHost((void **)&x_i, ncells * max_cell_size * sizeof(float));
    hipMallocHost((void **)&x_j, ncells * max_cell_size * sizeof(float));
    hipMallocHost((void **)&y_i, ncells * max_cell_size * sizeof(float));
    hipMallocHost((void **)&y_j, ncells * max_cell_size * sizeof(float));
    hipMallocHost((void **)&z_i, ncells * max_cell_size * sizeof(float));
    hipMallocHost((void **)&z_j, ncells * max_cell_size * sizeof(float));
    hipMallocHost((void **)&a_x_i, ncells * max_cell_size * sizeof(float));
    hipMallocHost((void **)&a_y_i, ncells * max_cell_size * sizeof(float));
    hipMallocHost((void **)&a_z_i, ncells * max_cell_size * sizeof(float));
    hipMallocHost((void **)&a_x_j, ncells * max_cell_size * sizeof(float));
    hipMallocHost((void **)&a_y_j, ncells * max_cell_size * sizeof(float));
    hipMallocHost((void **)&a_z_j, ncells * max_cell_size * sizeof(float));
    hipMallocHost((void **)&pot_i, ncells * max_cell_size * sizeof(float));
    hipMallocHost((void **)&pot_j, ncells * max_cell_size * sizeof(float));
    hipMallocHost((void **)&active_i, ncells * max_cell_size * sizeof(int));
    hipMallocHost((void **)&active_j, ncells * max_cell_size * sizeof(int));
    hipMallocHost((void **)&CoM_i, ncells * 3 * sizeof(float));
    hipMallocHost((void **)&CoM_j, ncells * 3 * sizeof(float));

    // allocate memory on device
    hipMalloc((void **)&d_h_i, ncells * max_cell_size * sizeof(float));
    hipMalloc((void **)&d_h_j, ncells * max_cell_size * sizeof(float));
    hipMalloc((void **)&d_mass_i, ncells * max_cell_size * sizeof(float));
    hipMalloc((void **)&d_mass_j, ncells * max_cell_size * sizeof(float));
    hipMalloc((void **)&d_x_i, ncells * max_cell_size * sizeof(float));
    hipMalloc((void **)&d_x_j, ncells * max_cell_size * sizeof(float));
    hipMalloc((void **)&d_y_i, ncells * max_cell_size * sizeof(float));
    hipMalloc((void **)&d_y_j, ncells * max_cell_size * sizeof(float));
    hipMalloc((void **)&d_z_i, ncells * max_cell_size * sizeof(float));
    hipMalloc((void **)&d_z_j, ncells * max_cell_size * sizeof(float));
    hipMalloc((void **)&d_a_x_i, ncells * max_cell_size * sizeof(float));
    hipMalloc((void **)&d_a_y_i, ncells * max_cell_size * sizeof(float));
    hipMalloc((void **)&d_a_z_i, ncells * max_cell_size * sizeof(float));
    hipMalloc((void **)&d_a_x_j, ncells * max_cell_size * sizeof(float));
    hipMalloc((void **)&d_a_y_j, ncells * max_cell_size * sizeof(float));
    hipMalloc((void **)&d_a_z_j, ncells * max_cell_size * sizeof(float));
    hipMalloc((void **)&d_pot_i, ncells * max_cell_size * sizeof(float));
    hipMalloc((void **)&d_pot_j, ncells * max_cell_size * sizeof(float));
    hipMalloc((void **)&d_active_i, ncells * max_cell_size * sizeof(int));
    hipMalloc((void **)&d_active_j, ncells * max_cell_size * sizeof(int));
    hipMalloc((void **)&d_CoM_i, ncells * 3 * sizeof(float));
    hipMalloc((void **)&d_CoM_j, ncells * 3 * sizeof(float));

    int pack_count = 0;

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) printf("Error1: %s\n", hipGetErrorString(err));

    /* Loop while there are tasks... */
    while (1) {

      /* If there's no old task, try to get a new one. */
      if (t == NULL) {

        /* Get the task. */
        TIMER_TIC
        t = scheduler_gettask(sched, r->qid, prev);  // from here cell is locked
        TIMER_TOC(timer_gettask);

        /* Did I get anything? */
        if (t == NULL) break;
      }

      /* Get the cells. */
      struct cell *ci = t->ci;
      struct cell *cj = t->cj;

#ifdef SWIFT_DEBUG_TASKS
      /* Mark the thread we run on */
      t->rid = r->cpuid;

      /* And recover the pair direction */
      if (t->type == task_type_pair || t->type == task_type_sub_pair) {
        struct cell *ci_temp = ci;
        struct cell *cj_temp = cj;
        double shift[3];
        t->sid = space_getsid_and_swap_cells(e->s, &ci_temp, &cj_temp, shift);
      } else {
        t->sid = -1;
      }
#endif

#ifdef SWIFT_DEBUG_CHECKS
      /* Check that we haven't scheduled an inactive task */
      t->ti_run = e->ti_current;
      /* Store the task that will be running (for debugging only) */
      r->t = t;
#endif

      const ticks task_beg = getticks();
      /* Different types of tasks... */
      switch (t->type) {
        case task_type_self:
          if (t->subtype == task_subtype_density)
            runner_doself1_branch_density(r, ci);
#ifdef EXTRA_HYDRO_LOOP
          else if (t->subtype == task_subtype_gradient)
            runner_doself1_branch_gradient(r, ci);
#endif
          else if (t->subtype == task_subtype_force)
            runner_doself2_branch_force(r, ci);
          else if (t->subtype == task_subtype_limiter)
            runner_doself1_branch_limiter(r, ci);

          // self grav recursive
          else if (t->subtype == task_subtype_grav) {
            // make long arrays with all the values
            struct gravity_cache *const ci_cache = &r->ci_gravity_cache;
            struct gravity_cache *const cj_cache = &r->cj_gravity_cache;

            // put values into long arrays
            for (int i = 0; i < max_cell_size;
                 i++) {  // change to gcount for cell
              h_i[pack_count * max_cell_size + i] = ci_cache->epsilon[i];
              mass_i[pack_count * max_cell_size + i] = ci_cache->m[i];
              x_i[pack_count * max_cell_size + i] = ci_cache->x[i];
              y_i[pack_count * max_cell_size + i] = ci_cache->y[i];
              z_i[pack_count * max_cell_size + i] = ci_cache->z[i];
              a_x_i[pack_count * max_cell_size + i] = ci_cache->a_x[i];
              a_y_i[pack_count * max_cell_size + i] = ci_cache->a_y[i];
              a_z_i[pack_count * max_cell_size + i] = ci_cache->a_z[i];
              pot_i[pack_count * max_cell_size + i] = ci_cache->pot[i];
              active_i[pack_count * max_cell_size + i] = ci_cache->active[i];
              //CoM_i[pack_count * max_cell_size + i] = ci_cache->active[i];
              // add two arrays for each particle to idenify where cj starts and
              // ends
            }

            pack_count += 1;
            // Here we need to unlock the cell(s)
            // if arrays have been filled
            if (pack_count == ncells) {

              printf("Outbound! GPU: %f CPU: %f \n",
                     a_x_i[(pack_count - 1) * max_cell_size + 1],
                     ci_cache->a_x[1]);

              // now copy all the arrays to the device
              hipMemcpyAsync(d_h_i, h_i, ncells * max_cell_size * sizeof(float),
                             hipMemcpyHostToDevice, 0);
              hipMemcpyAsync(d_h_j, h_j, ncells * max_cell_size * sizeof(float),
                             hipMemcpyHostToDevice, 0);
              hipMemcpyAsync(d_mass_i, mass_i,
                             ncells * max_cell_size * sizeof(float),
                             hipMemcpyHostToDevice, 0);
              hipMemcpyAsync(d_mass_j, mass_j,
                             ncells * max_cell_size * sizeof(float),
                             hipMemcpyHostToDevice, 0);
              hipMemcpyAsync(d_x_i, x_i, ncells * max_cell_size * sizeof(float),
                             hipMemcpyHostToDevice, 0);
              hipMemcpyAsync(d_y_i, y_i, ncells * max_cell_size * sizeof(float),
                             hipMemcpyHostToDevice, 0);
              hipMemcpyAsync(d_z_i, z_i, ncells * max_cell_size * sizeof(float),
                             hipMemcpyHostToDevice, 0);
              hipMemcpyAsync(d_x_j, x_j, ncells * max_cell_size * sizeof(float),
                             hipMemcpyHostToDevice, 0);
              hipMemcpyAsync(d_y_j, y_j, ncells * max_cell_size * sizeof(float),
                             hipMemcpyHostToDevice, 0);
              hipMemcpyAsync(d_z_j, z_j, ncells * max_cell_size * sizeof(float),
                             hipMemcpyHostToDevice, 0);
              hipMemcpyAsync(d_a_x_i, a_x_i,
                             ncells * max_cell_size * sizeof(float),
                             hipMemcpyHostToDevice, 0);
              hipMemcpyAsync(d_a_y_i, a_y_i,
                             ncells * max_cell_size * sizeof(float),
                             hipMemcpyHostToDevice, 0);
              hipMemcpyAsync(d_a_z_i, a_z_i,
                             ncells * max_cell_size * sizeof(float),
                             hipMemcpyHostToDevice, 0);
              hipMemcpyAsync(d_a_x_j, a_x_j,
                             ncells * max_cell_size * sizeof(float),
                             hipMemcpyHostToDevice, 0);
              hipMemcpyAsync(d_a_y_j, a_y_j,
                             ncells * max_cell_size * sizeof(float),
                             hipMemcpyHostToDevice, 0);
              hipMemcpyAsync(d_a_z_j, a_z_j,
                             ncells * max_cell_size * sizeof(float),
                             hipMemcpyHostToDevice, 0);
              hipMemcpyAsync(d_pot_i, pot_i,
                             ncells * max_cell_size * sizeof(float),
                             hipMemcpyHostToDevice, 0);
              hipMemcpyAsync(d_pot_j, pot_j,
                             ncells * max_cell_size * sizeof(float),
                             hipMemcpyHostToDevice, 0);
              hipMemcpyAsync(d_active_i, active_i,
                             ncells * max_cell_size * sizeof(int),
                             hipMemcpyHostToDevice, 0);
              hipMemcpyAsync(d_active_j, active_j,
                             ncells * max_cell_size * sizeof(int),
                             hipMemcpyHostToDevice, 0);
              hipMemcpyAsync(d_CoM_i, CoM_i, ncells * 3 * sizeof(float),
                             hipMemcpyHostToDevice, 0);
              hipMemcpyAsync(d_CoM_j, CoM_j, ncells * 3 * sizeof(float),
                             hipMemcpyHostToDevice, 0);

              hipError_t err = hipGetLastError();
              if (err != hipSuccess)
                printf("Error2: %s\n", hipGetErrorString(err));

              // hipDeviceSynchronize();

              runner_doself_grav_pp_gpu(r, ci, d_h_i, d_mass_i, d_x_i, d_y_i,
                                        d_z_i, d_a_x_i, d_a_y_i, d_a_z_i,
                                        d_pot_i, d_active_i);

              // hipDeviceSynchronize();

              a_x_i[1] = 0.f;
              printf("Reset to 0: %f \n",
                     a_x_i[(pack_count - 1) * max_cell_size + 1]);

              hipMemcpyAsync(a_x_i, d_a_x_i,
                             ncells * max_cell_size * sizeof(float),
                             hipMemcpyDeviceToHost, 0);
              hipMemcpyAsync(a_y_i, d_a_y_i,
                             ncells * max_cell_size * sizeof(float),
                             hipMemcpyDeviceToHost, 0);
              hipMemcpyAsync(a_z_i, d_a_z_i,
                             ncells * max_cell_size * sizeof(float),
                             hipMemcpyDeviceToHost, 0);
              hipMemcpyAsync(a_x_j, d_a_x_j,
                             ncells * max_cell_size * sizeof(float),
                             hipMemcpyDeviceToHost, 0);
              hipMemcpyAsync(a_y_j, d_a_y_j,
                             ncells * max_cell_size * sizeof(float),
                             hipMemcpyDeviceToHost, 0);
              hipMemcpyAsync(a_z_j, d_a_z_j,
                             ncells * max_cell_size * sizeof(float),
                             hipMemcpyDeviceToHost, 0);
              hipMemcpyAsync(pot_i, d_pot_i,
                             ncells * max_cell_size * sizeof(float),
                             hipMemcpyDeviceToHost, 0);
              hipMemcpyAsync(pot_j, d_pot_j,
                             ncells * max_cell_size * sizeof(float),
                             hipMemcpyDeviceToHost, 0);

              hipDeviceSynchronize();
              hipError_t err3 = hipGetLastError();
              if (err != hipSuccess)
                printf("Error4: %s\n", hipGetErrorString(err3));

              printf("Inbound! GPU: %f \n",
                     a_x_i[(pack_count - 1) * max_cell_size + 1]);
              // for(int pack=0; pack<pack_count; pack++){
              // cii = cell_list[pack];
              // same for cjj
              // while (cell_locktree(cii);
              // while (cell_locktree(cjj);)
              // UNPACK
              // unlock cells i and j
              // enqueue_dependencies(); //Line 3296 in Abou repo
              ///}
              // reset counter for next pack
              pack_count = 0;
            }
          } else if (t->subtype == task_subtype_external_grav)
            runner_do_grav_external(r, ci, 1);
          else if (t->subtype == task_subtype_stars_density)
            runner_doself_branch_stars_density(r, ci);
#ifdef EXTRA_STAR_LOOPS
          else if (t->subtype == task_subtype_stars_prep1)
            runner_doself_branch_stars_prep1(r, ci);
          else if (t->subtype == task_subtype_stars_prep2)
            runner_doself_branch_stars_prep2(r, ci);
#endif
          else if (t->subtype == task_subtype_stars_feedback)
            runner_doself_branch_stars_feedback(r, ci);
          else if (t->subtype == task_subtype_bh_density)
            runner_doself_branch_bh_density(r, ci);
          else if (t->subtype == task_subtype_bh_swallow)
            runner_doself_branch_bh_swallow(r, ci);
          else if (t->subtype == task_subtype_do_gas_swallow)
            runner_do_gas_swallow_self(r, ci, 1);
          else if (t->subtype == task_subtype_do_bh_swallow)
            runner_do_bh_swallow_self(r, ci, 1);
          else if (t->subtype == task_subtype_bh_feedback)
            runner_doself_branch_bh_feedback(r, ci);
          else if (t->subtype == task_subtype_rt_gradient)
            runner_doself1_branch_rt_gradient(r, ci);
          else if (t->subtype == task_subtype_rt_transport)
            runner_doself2_branch_rt_transport(r, ci);
          else if (t->subtype == task_subtype_sink_swallow)
            runner_doself_branch_sinks_swallow(r, ci);
          else if (t->subtype == task_subtype_sink_do_gas_swallow)
            runner_do_sinks_gas_swallow_self(r, ci, 1);
          else if (t->subtype == task_subtype_sink_do_sink_swallow)
            runner_do_sinks_sink_swallow_self(r, ci, 1);
          else
            error("Unknown/invalid task subtype (%s).",
                  subtaskID_names[t->subtype]);
          break;

        case task_type_pair:
          if (t->subtype == task_subtype_density)
            runner_dopair1_branch_density(r, ci, cj);
#ifdef EXTRA_HYDRO_LOOP
          else if (t->subtype == task_subtype_gradient)
            runner_dopair1_branch_gradient(r, ci, cj);
#endif
          else if (t->subtype == task_subtype_force)
            runner_dopair2_branch_force(r, ci, cj);
          else if (t->subtype == task_subtype_limiter)
            runner_dopair1_branch_limiter(r, ci, cj);
          else if (t->subtype == task_subtype_grav) {
            /*//pseudo memcpy function
            //make long arrays with all the values
            struct gravity_cache *const ci_cache = &r->ci_gravity_cache;
            struct gravity_cache *const cj_cache = &r->cj_gravity_cache;

            h_i[pack_count*max_cell_size] = ci_cache->epsilon;
            h_j[pack_count*max_cell_size] = cj_cache->epsilon;
            mass_i[pack_count*max_cell_size] = ci_cache->m;
            mass_j[pack_count*max_cell_size] = cj_cache->m;
            x_i[pack_count*max_cell_size] = ci_cache->x;
            x_j[pack_count*max_cell_size] = cj_cache->x;
            y_i[pack_count*max_cell_size] = ci_cache->y;
            y_j[pack_count*max_cell_size] = cj_cache->y;
            z_i[pack_count*max_cell_size] = ci_cache->z;
            z_j[pack_count*max_cell_size] = cj_cache->z;
            a_x_i[pack_count*max_cell_size] = ci_cache->a_x;
            a_x_j[pack_count*max_cell_size] = cj_cache->a_x;
            a_y_i[pack_count*max_cell_size] = ci_cache->a_y;
            a_y_j[pack_count*max_cell_size] = cj_cache->a_y;
            a_z_i[pack_count*max_cell_size] = ci_cache->a_z;
            a_z_j[pack_count*max_cell_size] = cj_cache->a_z;
            pot_i[pack_count*max_cell_size] = ci_cache->pot;
            pot_j[pack_count*max_cell_size] = cj_cache->pot;
            active_i[pack_count*max_cell_size] = ci_cache->active;
            active_j[pack_count*max_cell_size] = cj_cache->active;
            CoM_i[pack_count*max_cell_size] = ci_cache->active;
            CoM_j[pack_count*max_cell_size] = cj_cache->active;

            pack_count += 1;

            if (pack_count == ncells){


            //need to memcpy final values to device when read
            */

            runner_dopair_recursive_grav(
                r, ci, cj, 1, d_h_i, d_h_j, d_mass_i, d_mass_j, d_x_i, d_x_j,
                d_y_i, d_y_j, d_z_i, d_z_j, d_a_x_i, d_a_y_i, d_a_z_i, d_a_x_j,
                d_a_y_j, d_a_z_j, d_pot_i, d_pot_j, d_active_i, d_active_j,
                d_CoM_i, d_CoM_j);
          } else if (t->subtype == task_subtype_stars_density)
            runner_dopair_branch_stars_density(r, ci, cj);
#ifdef EXTRA_STAR_LOOPS
          else if (t->subtype == task_subtype_stars_prep1)
            runner_dopair_branch_stars_prep1(r, ci, cj);
          else if (t->subtype == task_subtype_stars_prep2)
            runner_dopair_branch_stars_prep2(r, ci, cj);
#endif
          else if (t->subtype == task_subtype_stars_feedback)
            runner_dopair_branch_stars_feedback(r, ci, cj);
          else if (t->subtype == task_subtype_bh_density)
            runner_dopair_branch_bh_density(r, ci, cj);
          else if (t->subtype == task_subtype_bh_swallow)
            runner_dopair_branch_bh_swallow(r, ci, cj);
          else if (t->subtype == task_subtype_do_gas_swallow)
            runner_do_gas_swallow_pair(r, ci, cj, 1);
          else if (t->subtype == task_subtype_do_bh_swallow)
            runner_do_bh_swallow_pair(r, ci, cj, 1);
          else if (t->subtype == task_subtype_bh_feedback)
            runner_dopair_branch_bh_feedback(r, ci, cj);
          else if (t->subtype == task_subtype_rt_gradient)
            runner_dopair1_branch_rt_gradient(r, ci, cj);
          else if (t->subtype == task_subtype_rt_transport)
            runner_dopair2_branch_rt_transport(r, ci, cj);
          else if (t->subtype == task_subtype_sink_swallow)
            runner_dopair_branch_sinks_swallow(r, ci, cj);
          else if (t->subtype == task_subtype_sink_do_gas_swallow)
            runner_do_sinks_gas_swallow_pair(r, ci, cj, 1);
          else if (t->subtype == task_subtype_sink_do_sink_swallow)
            runner_do_sinks_sink_swallow_pair(r, ci, cj, 1);
          else
            error("Unknown/invalid task subtype (%s/%s).",
                  taskID_names[t->type], subtaskID_names[t->subtype]);
          break;

        case task_type_sub_self:
          if (t->subtype == task_subtype_density)
            runner_dosub_self1_density(r, ci, 1);
#ifdef EXTRA_HYDRO_LOOP
          else if (t->subtype == task_subtype_gradient)
            runner_dosub_self1_gradient(r, ci, 1);
#endif
          else if (t->subtype == task_subtype_force)
            runner_dosub_self2_force(r, ci, 1);
          else if (t->subtype == task_subtype_limiter)
            runner_dosub_self1_limiter(r, ci, 1);
          else if (t->subtype == task_subtype_stars_density)
            runner_dosub_self_stars_density(r, ci, 1);
#ifdef EXTRA_STAR_LOOPS
          else if (t->subtype == task_subtype_stars_prep1)
            runner_dosub_self_stars_prep1(r, ci, 1);
          else if (t->subtype == task_subtype_stars_prep2)
            runner_dosub_self_stars_prep2(r, ci, 1);
#endif
          else if (t->subtype == task_subtype_stars_feedback)
            runner_dosub_self_stars_feedback(r, ci, 1);
          else if (t->subtype == task_subtype_bh_density)
            runner_dosub_self_bh_density(r, ci, 1);
          else if (t->subtype == task_subtype_bh_swallow)
            runner_dosub_self_bh_swallow(r, ci, 1);
          else if (t->subtype == task_subtype_do_gas_swallow)
            runner_do_gas_swallow_self(r, ci, 1);
          else if (t->subtype == task_subtype_do_bh_swallow)
            runner_do_bh_swallow_self(r, ci, 1);
          else if (t->subtype == task_subtype_bh_feedback)
            runner_dosub_self_bh_feedback(r, ci, 1);
          else if (t->subtype == task_subtype_rt_gradient)
            runner_dosub_self1_rt_gradient(r, ci, 1);
          else if (t->subtype == task_subtype_rt_transport)
            runner_dosub_self2_rt_transport(r, ci, 1);
          else if (t->subtype == task_subtype_sink_swallow)
            runner_dosub_self_sinks_swallow(r, ci, 1);
          else if (t->subtype == task_subtype_sink_do_gas_swallow)
            runner_do_sinks_gas_swallow_self(r, ci, 1);
          else if (t->subtype == task_subtype_sink_do_sink_swallow)
            runner_do_sinks_sink_swallow_self(r, ci, 1);
          else
            error("Unknown/invalid task subtype (%s/%s).",
                  taskID_names[t->type], subtaskID_names[t->subtype]);
          break;

        case task_type_sub_pair:
          if (t->subtype == task_subtype_density)
            runner_dosub_pair1_density(r, ci, cj, 1);
#ifdef EXTRA_HYDRO_LOOP
          else if (t->subtype == task_subtype_gradient)
            runner_dosub_pair1_gradient(r, ci, cj, 1);
#endif
          else if (t->subtype == task_subtype_force)
            runner_dosub_pair2_force(r, ci, cj, 1);
          else if (t->subtype == task_subtype_limiter)
            runner_dosub_pair1_limiter(r, ci, cj, 1);
          else if (t->subtype == task_subtype_stars_density)
            runner_dosub_pair_stars_density(r, ci, cj, 1);
#ifdef EXTRA_STAR_LOOPS
          else if (t->subtype == task_subtype_stars_prep1)
            runner_dosub_pair_stars_prep1(r, ci, cj, 1);
          else if (t->subtype == task_subtype_stars_prep2)
            runner_dosub_pair_stars_prep2(r, ci, cj, 1);
#endif
          else if (t->subtype == task_subtype_stars_feedback)
            runner_dosub_pair_stars_feedback(r, ci, cj, 1);
          else if (t->subtype == task_subtype_bh_density)
            runner_dosub_pair_bh_density(r, ci, cj, 1);
          else if (t->subtype == task_subtype_bh_swallow)
            runner_dosub_pair_bh_swallow(r, ci, cj, 1);
          else if (t->subtype == task_subtype_do_gas_swallow)
            runner_do_gas_swallow_pair(r, ci, cj, 1);
          else if (t->subtype == task_subtype_do_bh_swallow)
            runner_do_bh_swallow_pair(r, ci, cj, 1);
          else if (t->subtype == task_subtype_bh_feedback)
            runner_dosub_pair_bh_feedback(r, ci, cj, 1);
          else if (t->subtype == task_subtype_rt_gradient)
            runner_dosub_pair1_rt_gradient(r, ci, cj, 1);
          else if (t->subtype == task_subtype_rt_transport)
            runner_dosub_pair2_rt_transport(r, ci, cj, 1);
          else if (t->subtype == task_subtype_sink_swallow)
            runner_dosub_pair_sinks_swallow(r, ci, cj, 1);
          else if (t->subtype == task_subtype_sink_do_gas_swallow)
            runner_do_sinks_gas_swallow_pair(r, ci, cj, 1);
          else if (t->subtype == task_subtype_sink_do_sink_swallow)
            runner_do_sinks_sink_swallow_pair(r, ci, cj, 1);
          else
            error("Unknown/invalid task subtype (%s/%s).",
                  taskID_names[t->type], subtaskID_names[t->subtype]);
          break;

        case task_type_sort:
          /* Cleanup only if any of the indices went stale. */
          runner_do_hydro_sort(
              r, ci, t->flags,
              ci->hydro.dx_max_sort_old > space_maxreldx * ci->dmin,
              cell_get_flag(ci, cell_flag_rt_requests_sort), 1);
          /* Reset the sort flags as our work here is done. */
          t->flags = 0;
          break;
        case task_type_rt_sort:
          /* Cleanup only if any of the indices went stale.
           * NOTE: we check whether we reset the sort flags when the
           * recv tasks are running. Cells without an RT recv task
           * don't have rt_sort tasks. */
          runner_do_hydro_sort(
              r, ci, t->flags,
              ci->hydro.dx_max_sort_old > space_maxreldx * ci->dmin, 1, 1);
          /* Reset the sort flags as our work here is done. */
          t->flags = 0;
          break;
        case task_type_stars_sort:
          /* Cleanup only if any of the indices went stale. */
          runner_do_stars_sort(
              r, ci, t->flags,
              ci->stars.dx_max_sort_old > space_maxreldx * ci->dmin, 1);
          /* Reset the sort flags as our work here is done. */
          t->flags = 0;
          break;
        case task_type_init_grav:
          runner_do_init_grav(r, ci, 1);
          break;
        case task_type_ghost:
          runner_do_ghost(r, ci, 1);
          break;
#ifdef EXTRA_HYDRO_LOOP
        case task_type_extra_ghost:
          runner_do_extra_ghost(r, ci, 1);
          break;
#endif
        case task_type_stars_ghost:
          runner_do_stars_ghost(r, ci, 1);
          break;
        case task_type_bh_density_ghost:
          runner_do_black_holes_density_ghost(r, ci, 1);
          break;
        case task_type_bh_swallow_ghost3:
          runner_do_black_holes_swallow_ghost(r, ci, 1);
          break;
        case task_type_drift_part:
          runner_do_drift_part(r, ci, 1);
          break;
        case task_type_drift_spart:
          runner_do_drift_spart(r, ci, 1);
          break;
        case task_type_drift_sink:
          runner_do_drift_sink(r, ci, 1);
          break;
        case task_type_drift_bpart:
          runner_do_drift_bpart(r, ci, 1);
          break;
        case task_type_drift_gpart:
          runner_do_drift_gpart(r, ci, 1);
          break;
        case task_type_kick1:
          runner_do_kick1(r, ci, 1);
          break;
        case task_type_kick2:
          runner_do_kick2(r, ci, 1);
          break;
        case task_type_end_hydro_force:
          runner_do_end_hydro_force(r, ci, 1);
          break;
        case task_type_end_grav_force:
          runner_do_end_grav_force(r, ci, 1);
          break;
        case task_type_csds:
          runner_do_csds(r, ci, 1);
          break;
        case task_type_timestep:
          runner_do_timestep(r, ci, 1);
          break;
        case task_type_timestep_limiter:
          runner_do_limiter(r, ci, 0, 1);
          break;
        case task_type_timestep_sync:
          runner_do_sync(r, ci, 0, 1);
          break;
        case task_type_collect:
          runner_do_timestep_collect(r, ci, 1);
          break;
        case task_type_rt_collect_times:
          runner_do_collect_rt_times(r, ci, 1);
          break;
#ifdef WITH_MPI
        case task_type_send:
          if (t->subtype == task_subtype_tend) {
            free(t->buff);
          } else if (t->subtype == task_subtype_sf_counts) {
            free(t->buff);
          } else if (t->subtype == task_subtype_part_swallow) {
            free(t->buff);
          } else if (t->subtype == task_subtype_bpart_merger) {
            free(t->buff);
          } else if (t->subtype == task_subtype_limiter) {
            free(t->buff);
          }
          break;
        case task_type_recv:
          if (t->subtype == task_subtype_tend) {
            cell_unpack_end_step(ci, (struct pcell_step *)t->buff);
            free(t->buff);
          } else if (t->subtype == task_subtype_sf_counts) {
            cell_unpack_sf_counts(ci, (struct pcell_sf *)t->buff);
            cell_clear_stars_sort_flags(ci, /*clear_unused_flags=*/0);
            free(t->buff);
          } else if (t->subtype == task_subtype_xv) {
            runner_do_recv_part(r, ci, 1, 1);
          } else if (t->subtype == task_subtype_rho) {
            runner_do_recv_part(r, ci, 0, 1);
          } else if (t->subtype == task_subtype_gradient) {
            runner_do_recv_part(r, ci, 0, 1);
          } else if (t->subtype == task_subtype_rt_gradient) {
            runner_do_recv_part(r, ci, 2, 1);
          } else if (t->subtype == task_subtype_rt_transport) {
            runner_do_recv_part(r, ci, -1, 1);
          } else if (t->subtype == task_subtype_part_swallow) {
            cell_unpack_part_swallow(ci,
                                     (struct black_holes_part_data *)t->buff);
            free(t->buff);
          } else if (t->subtype == task_subtype_bpart_merger) {
            cell_unpack_bpart_swallow(ci,
                                      (struct black_holes_bpart_data *)t->buff);
            free(t->buff);
          } else if (t->subtype == task_subtype_limiter) {
            /* Nothing to do here. Unpacking done in a separate task */
          } else if (t->subtype == task_subtype_gpart) {
            runner_do_recv_gpart(r, ci, 1);
          } else if (t->subtype == task_subtype_spart_density) {
            runner_do_recv_spart(r, ci, 1, 1);
          } else if (t->subtype == task_subtype_part_prep1) {
            runner_do_recv_part(r, ci, 0, 1);
          } else if (t->subtype == task_subtype_spart_prep2) {
            runner_do_recv_spart(r, ci, 0, 1);
          } else if (t->subtype == task_subtype_bpart_rho) {
            runner_do_recv_bpart(r, ci, 1, 1);
          } else if (t->subtype == task_subtype_bpart_feedback) {
            runner_do_recv_bpart(r, ci, 0, 1);
          } else {
            error("Unknown/invalid task subtype (%d).", t->subtype);
          }
          break;

        case task_type_pack:
          runner_do_pack_limiter(r, ci, &t->buff, 1);
          task_get_unique_dependent(t)->buff = t->buff;
          break;
        case task_type_unpack:
          runner_do_unpack_limiter(r, ci, t->buff, 1);
          break;
#endif
        case task_type_grav_down:
          runner_do_grav_down(r, t->ci, 1);
          break;
        case task_type_grav_long_range:
          runner_do_grav_long_range(r, t->ci, 1);
          break;
        case task_type_grav_mm:
          runner_dopair_grav_mm_progenies(r, t->flags, t->ci, t->cj);
          break;
        case task_type_cooling:
          runner_do_cooling(r, t->ci, 1);
          break;
        case task_type_star_formation:
          runner_do_star_formation(r, t->ci, 1);
          break;
        case task_type_star_formation_sink:
          runner_do_star_formation_sink(r, t->ci, 1);
          break;
        case task_type_stars_resort:
          runner_do_stars_resort(r, t->ci, 1);
          break;
        case task_type_sink_formation:
          runner_do_sink_formation(r, t->ci);
          break;
        case task_type_fof_self:
          runner_do_fof_search_self(r, t->ci, 1);
          break;
        case task_type_fof_pair:
          runner_do_fof_search_pair(r, t->ci, t->cj, 1);
          break;
        case task_type_fof_attach_self:
          runner_do_fof_attach_self(r, t->ci, 1);
          break;
        case task_type_fof_attach_pair:
          runner_do_fof_attach_pair(r, t->ci, t->cj, 1);
          break;
        case task_type_neutrino_weight:
          runner_do_neutrino_weighting(r, ci, 1);
          break;
        case task_type_rt_ghost1:
          runner_do_rt_ghost1(r, t->ci, 1);
          break;
        case task_type_rt_ghost2:
          runner_do_rt_ghost2(r, t->ci, 1);
          break;
        case task_type_rt_tchem:
          runner_do_rt_tchem(r, t->ci, 1);
          break;
        case task_type_rt_advance_cell_time:
          runner_do_rt_advance_cell_time(r, t->ci, 1);
          break;
        default:
          error("Unknown/invalid task type (%d).", t->type);
      }
      r->active_time += (getticks() - task_beg);

/* Mark that we have run this task on these cells */
#ifdef SWIFT_DEBUG_CHECKS
      if (ci != NULL) {
        ci->tasks_executed[t->type]++;
        ci->subtasks_executed[t->subtype]++;
      }
      if (cj != NULL) {
        cj->tasks_executed[t->type]++;
        cj->subtasks_executed[t->subtype]++;
      }

      /* This runner is not doing a task anymore */
      r->t = NULL;
#endif

      /* We're done with this task, see if we get a next one. */
      prev = t;
      // Here we need an if statement that schecks if iI am a gravity task
      /*if(t->subtype == task_subtype_grav && t->type == t->type_self){
        t=NULL;
      }
      else{*/
      t = scheduler_done(sched,
                         t);  // This will unlock my deps and unleash hell!
      //}
    } /* main loop. */
    hipFree(d_h_i);
    hipFree(d_h_j);
    hipFree(d_mass_i);
    hipFree(d_mass_j);
    hipFree(d_x_i);
    hipFree(d_x_j);
    hipFree(d_y_i);
    hipFree(d_y_j);
    hipFree(d_z_i);
    hipFree(d_z_j);
    hipFree(d_a_x_i);
    hipFree(d_a_y_i);
    hipFree(d_a_z_i);
    hipFree(d_a_x_j);
    hipFree(d_a_y_j);
    hipFree(d_a_z_j);
    hipFree(d_pot_i);
    hipFree(d_pot_j);
    hipFree(d_active_i);
    hipFree(d_active_j);
    hipFree(d_CoM_i);
    hipFree(d_CoM_j);
  }
  /* Be kind, rewind. */
  return NULL;
}

ticks runner_get_active_time(const struct runner *restrict r) {
  return r->active_time;
}

void runner_reset_active_time(struct runner *restrict r) { r->active_time = 0; }

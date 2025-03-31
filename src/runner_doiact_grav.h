/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2013 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
 *               2016 Matthieu Schaller (schaller@strw.leidenuniv.nl)
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
#ifndef SWIFT_RUNNER_DOIACT_GRAV_H
#define SWIFT_RUNNER_DOIACT_GRAV_H

#include <config.h>

struct runner;
struct cell;

void runner_do_grav_down(struct runner *r, struct cell *c, int timer);

void runner_dopair_grav_pp(struct runner *r, struct cell *ci, struct cell *cj,
                           const int symmetric, const int allow_mpole, float *d_h_i, float *d_h_j, float *d_mass_i, float *d_mass_j, float *d_x_i, float *d_x_j, float *d_y_i, float *d_y_j, float *d_z_i, float *d_z_j, float *d_a_x_i, float *d_a_y_i, float *d_a_z_i, float *d_a_x_j, float *d_a_y_j, float *d_a_z_j, float *d_pot_i, float *d_pot_j, int *d_active_i, int *d_active_j, float *d_CoM_i, float *d_CoM_j);

void runner_doself_recursive_grav(struct runner *r, struct cell *c,
                                  int gettimer, float *d_h_i, float *d_h_j, float *d_mass_i, float *d_mass_j, float *d_x_i, float *d_x_j, float *d_y_i, float *d_y_j, float *d_z_i, float *d_z_j, float *d_a_x_i, float *d_a_y_i, float *d_a_z_i, float *d_a_x_j, float *d_a_y_j, float *d_a_z_j, float *d_pot_i, float *d_pot_j, int *d_active_i, int *d_active_j, float *d_CoM_i, float *d_CoM_j);

void runner_dopair_recursive_grav(struct runner *r, struct cell *ci,
                                  struct cell *cj, int gettimer, float *d_h_i, float *d_h_j, float *d_mass_i, float *d_mass_j, float *d_x_i, float *d_x_j, float *d_y_i, float *d_y_j, float *d_z_i, float *d_z_j, float *d_a_x_i, float *d_a_y_i, float *d_a_z_i, float *d_a_x_j, float *d_a_y_j, float *d_a_z_j, float *d_pot_i, float *d_pot_j, int *d_active_i, int *d_active_j, float *d_CoM_i, float *d_CoM_j);

void runner_dopair_grav_mm_progenies(struct runner *r, const long long flags,
                                     struct cell *restrict ci,
                                     struct cell *restrict cj);

void runner_do_grav_long_range(struct runner *r, struct cell *ci, int timer);

/* Internal functions (for unit tests and debugging) */

void runner_doself_grav_pp(struct runner *r, struct cell *c, float *d_h_i, float *d_mass_i, float *d_x_i, float *d_y_i, float *d_z_i, float *d_a_x_i, float *d_a_y_i, float *d_a_z_i, float *d_pot_i,int *d_active_i);

void runner_dopair_grav_pp(struct runner *r, struct cell *ci, struct cell *cj,
                           const int symmetric, const int allow_mpole, float *d_h_i, float *d_h_j, float *d_mass_i, float *d_mass_j, float *d_x_i, float *d_x_j, float *d_y_i, float *d_y_j, float *d_z_i, float *d_z_j, float *d_a_x_i, float *d_a_y_i, float *d_a_z_i, float *d_a_x_j, float *d_a_y_j, float *d_a_z_j, float *d_pot_i, float *d_pot_j, int *d_active_i, int *d_active_j, float *d_CoM_i, float *d_CoM_j);

#endif /* SWIFT_RUNNER_DOIACT_GRAV_H */

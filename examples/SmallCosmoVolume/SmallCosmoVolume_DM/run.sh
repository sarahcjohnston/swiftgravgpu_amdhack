#!/bin/bash
#SBATCH --ntasks 1 # The number of cores you need...
#SBATCH -J newGPU #Give it something meaningful.
#SBATCH -o output.%J.out
#SBATCH -e error.%J.err
#SBATCH -p cosma8-shm #or some other partition, e.g. cosma, cosma6, etc.
#SBATCH -A dp004
#SBATCH --constrain=gpu
#SBATCH -t 24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=sarah.c.johnston@durham.ac.uk

module purge
module load nvhpc
module load intel_comp/2024.2.0 compiler-rt tbb compiler mpi
module load ucx/1.13.0rc2
module load parallel_hdf5/1.14.4
module load fftw/3.3.10
module load parmetis
module load gsl/2.5
module load sundials/5.1.0_c8_single 
module load Healpix/3.82


# Run SWIFT
../../../swift --cosmology --self-gravity --threads=32 small_cosmo_volume_dm.yml 2>&1 | tee output.log

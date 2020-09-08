#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Frontera CLX nodes
#
#   *** MPI Job in Normal Queue ***
# 
# Last revised: 20 May 2019
#
# Notes:
#
#   -- Launch this script by executing
#      "sbatch clx.mpi.slurm" on a Frontera login node.
#
#   -- Use ibrun to launch MPI codes on TACC systems.
#      Do NOT use mpirun or mpiexec.
#
#   -- Max recommended MPI ranks per CLX node: 56
#      (start small, increase gradually).
#
#   -- If you're running out of memory, try running
#      fewer tasks per node to give each task more memory.
#
#----------------------------------------------------

#SBATCH -J heat1d           # Job name
#SBATCH -o heat1d.o%j       # Name of stdout output file
#SBATCH -e heat1d.e%j       # Name of stderr error file
#SBATCH -p development          # Queue (partition) name
#SBATCH -N 2               # Total # of nodes 
#SBATCH -n 16              # Total # of mpi tasks
#SBATCH -t 00:30:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH --mail-user=yxbao@utexas.edu

# Any other commands must follow all #SBATCH directives...
module list
pwd
date

# Launch MPI code... 
mpiexec -n $SLURM_NTASKS python3 heat1d_mpi.py 

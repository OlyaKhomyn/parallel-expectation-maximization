#!/bin/bash

#PBS -l select=1:ncpus=4:mem=2gb

#PBS -l walltime=03:00:00

#PBS -q short_cpuQ

module load mpich-3.2
# arguments are executable N D K iter filepath
mpiexec -n 4 parallelRun 800 3 4 100 "data/N800_K4_D3.csv"
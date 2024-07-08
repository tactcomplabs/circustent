#!/bin/bash
#SBATCH --job-name=CT_CUDA
#SBATCH --p toreador
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gpus-per-task=1
#SBATCH --exclusive

source BUILD.sh
nv_a100/add_kernels/add_kernels.sh
nv_a100/all_kernels/all_kernels.sh
nv_a100/parallelism/parallelism.sh
nv_a100/problem_size/problem_size.sh




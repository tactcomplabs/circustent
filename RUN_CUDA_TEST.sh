#!/bin/bash

export CT_ROOT=/home/mibeebe/ct/circustent_cuda
export CT_BUILD=$CT_ROOT/build
export EXE=$CT_BUILD/src/CircusTent/circustent

# $EXE --help
# echo

$EXE --bench RAND_ADD -m 1030560875 -p 1 -i 10000000 --blocks 1000 --threads 1000

$EXE --bench RAND_CAS -m 1030560875 -p 1 -i 10000000 --blocks 1000 --threads 1000

$EXE --bench SG_ADD -m 1030560875 -p 1 -i 10000000 --blocks 1000 --threads 1000

$EXE --bench SG_CAS -m 1030560875 -p 1 -i 10000000 --blocks 1000 --threads 1000

$EXE --bench GATHER_ADD -m 1030560875 -p 1 -i 10000000 --blocks 1000 --threads 1000

$EXE --bench GATHER_CAS -m 1030560875 -p 1 -i 10000000 --blocks 1000 --threads 1000

$EXE --bench SCATTER_ADD -m 1030560875 -p 1 -i 10000000 --blocks 1000 --threads 1000

$EXE --bench SCATTER_CAS -m 1030560875 -p 1 -i 10000000 --blocks 1000 --threads 1000

$EXE --bench PTRCHASE_ADD -m 1030560875 -p 1 -i 10000000 --blocks 1000 --threads 1000

$EXE --bench PTRCHASE_CAS -m 1030560875 -p 1 -i 10000000 --blocks 1000 --threads 1000

$EXE --bench STRIDE1_ADD -m 1030560875 -p 1 -i 10000000 --blocks 1000 --threads 1000s

$EXE --bench STRIDE1_CAS -m 1030560875 -p 1 -i 10000000 --blocks 1000 --threads 1000

$EXE --bench STRIDEN_ADD -m 1030560875 -p 1 -i 10000000 --stride 3 --blocks 1000 --threads 1000

$EXE --bench STRIDEN_CAS -m 1030560875 -p 1 -i 10000000 --stride 3 --blocks 1000 --threads 1000

$EXE --bench CENTRAL_ADD -m 1030560875 -p 1 -i 10000000 --blocks 1000 --threads 1000

$EXE --bench CENTRAL_CAS -m 1030560875 -p 1 -i 10000000 --blocks 1000 --threads 1000


# CUDA
# echo "---------------------"
# echo "       CUDA"
# echo "---------------------"
# $EXE --bench STRIDE1_ADD -m 1030560875 -p 1 -i 10000000 --blocks 10 --threads 10
# $EXE --bench GATHER_ADD  -m 1030560875 -p 1 -i 10000000 --blocks 10 --threads 10
# $EXE --bench SCATTER_ADD -m 1030560875 -p 1 -i 10000000 --blocks 10 --threads 10

# OpenMP and Pthreads
# $EXE --bench STRIDE1_ADD -m 1030560875 -p 1 -i 10000000
# $EXE --bench GATHER_ADD  -m 1030560875 -p 1 -i 10000000
# $EXE --bench SCATTER_ADD -m 1030560875 -p 1 -i 10000000


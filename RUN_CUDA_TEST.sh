#!/bin/bash

export CT_ROOT=/home/mibeebe/ct/circustent_dev_main
export CT_BUILD=$CT_ROOT/build
export EXE=$CT_BUILD/src/CircusTent/circustent

# $EXE --help
# echo

# CUDA
# $EXE --bench RAND_ADD -m 1030560875 -p 1 -i 10000000 --blocks 100 --threads 1000

# $EXE --bench RAND_CAS -m 1030560875 -p 1 -i 10000000 --blocks 100 --threads 1000

# $EXE --bench SG_ADD -m 1030560875 -p 1 -i 10000000 --blocks 100 --threads 1000

# $EXE --bench SG_CAS -m 1030560875 -p 1 -i 10000000 --blocks 100 --threads 1000

# $EXE --bench GATHER_ADD -m 1030560875 -p 1 -i 10000000 --blocks 100 --threads 1000

# $EXE --bench GATHER_CAS -m 1030560875 -p 1 -i 10000000 --blocks 100 --threads 1000

# $EXE --bench SCATTER_ADD -m 1030560875 -p 1 -i 10000000 --blocks 100 --threads 1000

# $EXE --bench SCATTER_CAS -m 1030560875 -p 1 -i 10000000 --blocks 100 --threads 1000

# $EXE --bench PTRCHASE_ADD -m 1030560875 -p 1 -i 10000000 --blocks 100 --threads 1000

# $EXE --bench PTRCHASE_CAS -m 1030560875 -p 1 -i 10000000 --blocks 100 --threads 1000

# $EXE --bench STRIDE1_ADD -m 1030560875 -p 1 -i 10000000 --blocks 100 --threads 1000

# $EXE --bench STRIDE1_CAS -m 1030560875 -p 1 -i 10000000 --blocks 100 --threads 1000

# $EXE --bench STRIDEN_ADD -m 1030560875 -p 1 -i 10000000 --stride 3 --blocks 100 --threads 1000

# $EXE --bench STRIDEN_CAS -m 1030560875 -p 1 -i 10000000 --stride 3 --blocks 100 --threads 1000

# $EXE --bench CENTRAL_ADD -m 1030560875 -p 1 -i 10000000 --blocks 100 --threads 1000

# $EXE --bench CENTRAL_CAS -m 1030560875 -p 1 -i 10000000 --blocks 100 --threads 1000



$EXE --bench RAND_ADD -m 1030560875 -p 1 -i 10000000 --blocks 1 --threads 1

$EXE --bench RAND_CAS -m 1030560875 -p 1 -i 10000000 --blocks 1 --threads 1

$EXE --bench SG_ADD -m 1030560875 -p 1 -i 10000000 --blocks 1 --threads 1

$EXE --bench SG_CAS -m 1030560875 -p 1 -i 10000000 --blocks 1 --threads 1

$EXE --bench GATHER_ADD -m 1030560875 -p 1 -i 10000000 --blocks 1 --threads 1

$EXE --bench GATHER_CAS -m 1030560875 -p 1 -i 10000000 --blocks 1 --threads 1

$EXE --bench SCATTER_ADD -m 1030560875 -p 1 -i 10000000 --blocks 1 --threads 1

$EXE --bench SCATTER_CAS -m 1030560875 -p 1 -i 10000000 --blocks 1 --threads 1

$EXE --bench PTRCHASE_ADD -m 1030560875 -p 1 -i 10000000 --blocks 1 --threads 1

$EXE --bench PTRCHASE_CAS -m 1030560875 -p 1 -i 10000000 --blocks 1 --threads 1

$EXE --bench STRIDE1_ADD -m 1030560875 -p 1 -i 10000000 --blocks 1 --threads 1

$EXE --bench STRIDE1_CAS -m 1030560875 -p 1 -i 10000000 --blocks 1 --threads 1

$EXE --bench STRIDEN_ADD -m 1030560875 -p 1 -i 10000000 --stride 3 --blocks 1 --threads 1

$EXE --bench STRIDEN_CAS -m 1030560875 -p 1 -i 10000000 --stride 3 --blocks 1 --threads 1

$EXE --bench CENTRAL_ADD -m 1030560875 -p 1 -i 10000000 --blocks 1 --threads 1

$EXE --bench CENTRAL_CAS -m 1030560875 -p 1 -i 10000000 --blocks 1 --threads 1







# OMP
# $EXE --bench RAND_ADD -m 1030560875 -p 1 -i 10000000

# $EXE --bench RAND_CAS -m 1030560875 -p 1 -i 10000000

# $EXE --bench SG_ADD -m 1030560875 -p 1 -i 10000000

# $EXE --bench SG_CAS -m 1030560875 -p 1 -i 10000000

# $EXE --bench GATHER_ADD -m 1030560875 -p 1 -i 10000000

# $EXE --bench GATHER_CAS -m 1030560875 -p 1 -i 10000000

# $EXE --bench SCATTER_ADD -m 1030560875 -p 1 -i 10000000

# $EXE --bench SCATTER_CAS -m 1030560875 -p 1 -i 10000000

# $EXE --bench PTRCHASE_ADD -m 1030560875 -p 1 -i 10000000

# $EXE --bench PTRCHASE_CAS -m 1030560875 -p 1 -i 10000000

# $EXE --bench STRIDE1_ADD -m 1030560875 -p 1 -i 10000000

# $EXE --bench STRIDE1_CAS -m 1030560875 -p 1 -i 10000000

# $EXE --bench STRIDEN_ADD -m 1030560875 -p 1 -i 10000000

# $EXE --bench STRIDEN_CAS -m 1030560875 -p 1 -i 10000000

# $EXE --bench CENTRAL_ADD -m 1030560875 -p 1 -i 10000000

# $EXE --bench CENTRAL_CAS -m 1030560875 -p 1 -i 10000000

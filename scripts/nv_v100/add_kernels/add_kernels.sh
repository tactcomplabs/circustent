#!/bin/bash
#
# Script to run all ADD kernels and save output

BENCH="RAND_ADD STRIDE1_ADD PTRCHASE_ADD CENTRAL_ADD SG_ADD SCATTER_ADD GATHER_ADD" 


IMPL="CUDA"

CUDA_FLAGS=""
OMP_FLAGS=""
OACC_FLAGS=""

################################################


TEST_DIR=$(pwd)
OUTPUT_FILE=$TEST_DIR/outputs/$IMPL/circustent_add_kernels_$(date +"%d-%m-%y")_$(date +"%T").txt

cd ../../../
CT_ROOT=$(pwd)
CT_BUILD=$CT_ROOT/build
EXE=$CT_BUILD/src/CircusTent/circustent
cd $TEST_DIR

echo $OUTPUT_FILE


$EXE --bench RAND_ADD -m 32089730048 -i 500000 --blocks 2 --threads 1024
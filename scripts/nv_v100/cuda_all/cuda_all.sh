#!/bin/bash
#
# Script to run all ADD kernels and save output

HLINE="============================================================="

#######################################################################
#           Set the implementation that you wish to run                      
#######################################################################
BACKENDS=("CUDA" "OpenMP" "OpenACC")

IMPL="CUDA"

#######################################################################
#                 Edit the run run configuration
#######################################################################
BLOCKS=16
THREADS=128
MEM_SIZE=32089730048
ITERS=500000

# MEM_SIZE=16044865024
# ITERS=250000

#######################################################################
#                   Set the GPU clock frequency
#######################################################################
# TODO: set GPU clock frequency

#######################################################################
#     Set $COMPILE to true if you want to run the build script
#######################################################################
COMPILE=
if [[ $COMPILE == true ]]; then
    cd ../../
    source BUILD.sh
    cd nv_v100/add_kernels
fi

#######################################################################
#                      Create output file  
#######################################################################
TEST_DIR=$(pwd)
OUTPUT_FILE=$TEST_DIR/outputs/$IMPL/add_kernels_$(date +"%d-%m-%y")_$(date +"%T").txt
if [[ -f $OUTPUT_FILE ]] ; then
    rm $OUTPUT_FILE
    touch $OUTPUT_FILE
else
    touch $OUTPUT_FILE
fi

#######################################################################
#             Specify path to the circustent executable 
#######################################################################
cd ../../../
CT_ROOT=$(pwd)
CT_BUILD=$CT_ROOT/build
EXE=$CT_BUILD/src/CircusTent/circustent
cd $TEST_DIR


#######################################################################
#                         Run each all CUDA kernels
#######################################################################
BENCH="RAND_ADD RAND_CAS STRIDE1_ADD STRIDE1_CAS STRIDEN_ADD STRIDEN_CAS PTRCHASE_ADD PTRCHASE_CAS CENTRAL_ADD CENTRAL_CAS SG_ADD SG_CAS SCATTER_ADD SCATTER_CAS GATHER_ADD GATHER_CAS"
for B in $BENCH; do

    echo $HLINE >> $OUTPUT_FILE
    echo "  Running the $B kernel using the $IMPL impl..." >> $OUTPUT_FILE
    echo "                  $(date +"%d-%m-%y") AT $(date +"%T") " >> $OUTPUT_FILE
    echo $HLINE >> $OUTPUT_FILE

    echo "Running with $BLOCKS blocks, $THREADS threads per block" >> $OUTPUT_FILE
    $EXE --bench $B -m $MEM_SIZE -i $ITERS --blocks $BLOCKS --threads $THREADS >> $OUTPUT_FILE
    echo >> $OUTPUT_FILE
    echo >> $OUTPUT_FILE
done
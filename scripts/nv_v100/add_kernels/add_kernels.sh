#!/bin/bash
#
# Script to run all ADD kernels and save output

HLINE="============================================================="

BACKENDS="CUDA OpenMP OpenACC"

# Set subscript to 0 for CUDA, 1 for OpenMP, 2 for OpenACC
# IMPL=$BACKENDS[0]
IMPL="OpenMP"

BENCH="RAND_ADD STRIDE1_ADD PTRCHASE_ADD CENTRAL_ADD SG_ADD SCATTER_ADD GATHER_ADD" 

# CUDA_FLAGS=""
# OMP_FLAGS=""
# OACC_FLAGS=""

MEM_SIZE=32089730048
ITERS=500000

# TODO: add in these equivalents for OpenMP and OpenACC
BLOCKS=16
THREADS=128

export OMP_NUM_THREADS=128
# export OMP_PLACES=

#######################################################################

TEST_DIR=$(pwd)
OUTPUT_FILE=$TEST_DIR/outputs/$IMPL/add_kernels_$(date +"%d-%m-%y")_$(date +"%T").txt
if [[ -f $OUTPUT_FILE ]] ; then
    rm $OUTPUT_FILE
    touch $OUTPUT_FILE
else
    touch $OUTPUT_FILE
fi

cd ../../../
CT_ROOT=$(pwd)
CT_BUILD=$CT_ROOT/build
EXE=$CT_BUILD/src/CircusTent/circustent
cd $TEST_DIR

for B in $BENCH; do
    echo $HLINE >> $OUTPUT_FILE
    echo "  Running the $B kernel using the $IMPL impl..." >> $OUTPUT_FILE
    echo "            $(date +"%d-%m-%y") AT $(date +"%T") " >> $OUTPUT_FILE
    echo $HLINE >> $OUTPUT_FILE

    if [[ $IMPL == "CUDA" ]]; then    
        # FIXME: if I decide to use the "XXX_FLAGS" var then change this
        $EXE --bench $B -m $MEM_SIZE -i $ITERS --blocks $BLOCKS --threads $THREADS >> $OUTPUT_FILE
        echo >> $OUTPUT_FILE
        echo >> $OUTPUT_FILE
    elif [[ $IML == "OpenMP" ]]; then
        # FIXME: Is OpenMP teams == CUDA blocks? -p is the teams
        $EXE --bench $B -m $MEM_SIZE -p $BLOCKS -i $ITERS >> $OUTPUT_FILE        
        echo >> $OUTPUT_FILE
        echo >> $OUTPUT_FILE
    elif [[ $IMPL == "OpenACC" ]]; then
        $EXE --bench $B -m $MEM_SIZE -p $BLOCKS -i $ITERS >> $OUTPUT_FILE        
        echo >> $OUTPUT_FILE
        echo >> $OUTPUT_FILE 
    fi
done
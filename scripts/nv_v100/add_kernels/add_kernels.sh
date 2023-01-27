#!/bin/bash
#
# Script to run all ADD kernels and save output

HLINE="============================================================="

BACKENDS="CUDA OpenMP OpenACC"

# Set subscript to 0 for CUDA, 1 for OpenMP, 2 for OpenACC
# IMPL=$BACKENDS[0]
IMPL="OpenACC"

#######################################################################

if [[ "$IMPL" == "CUDA" ]]; then
    BLOCKS=16
    THREADS=128
elif [[ "$IMPL" == "OpenMP" ]]; then
    export OMP_DEFAULT_DEVICE=
    export OMP_NUM_THREADS=128
elif [[ "$IMPL" == "OpenACC" ]]; then
    export ACC_DEVICE_TYPE=nvidia
    export ACC_DEVICE_NUM=0
fi


#######################################################################

MEM_SIZE=32089730048
ITERS=500000

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


#######################################################################

BENCH="RAND_ADD STRIDE1_ADD PTRCHASE_ADD CENTRAL_ADD SG_ADD SCATTER_ADD GATHER_ADD" 
for B in $BENCH; do
    echo $HLINE >> $OUTPUT_FILE
    echo "  Running the $B kernel using the $IMPL impl..." >> $OUTPUT_FILE
    echo "                  $(date +"%d-%m-%y") AT $(date +"%T") " >> $OUTPUT_FILE
    echo $HLINE >> $OUTPUT_FILE

    if [[ "$IMPL" == "CUDA" ]]; then    
        $EXE --bench $B -m $MEM_SIZE -i $ITERS --blocks $BLOCKS --threads $THREADS >> $OUTPUT_FILE
        echo >> $OUTPUT_FILE
        echo >> $OUTPUT_FILE
    elif [[ "$IMPL" == "OpenMP" ]]; then
        # FIXME: Is OpenMP teams == CUDA blocks? -p is the teams
        $EXE --bench $B -m $MEM_SIZE -p $BLOCKS -i $ITERS >> $OUTPUT_FILE        
        echo >> $OUTPUT_FILE
        echo >> $OUTPUT_FILE
    elif [[ "$IMPL" == "OpenACC" ]]; then
        # FIXME: Is OpenACC gangs == CUDA blocks? -p is the teams 
        $EXE --bench $B -m $MEM_SIZE -p $BLOCKS -i $ITERS >> $OUTPUT_FILE        
        echo >> $OUTPUT_FILE
        echo >> $OUTPUT_FILE 
    fi
done
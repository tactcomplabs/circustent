#!/bin/bash
#
# Script to run all ADD kernels and save output

HLINE="============================================================="

#######################################################################
#           Set the implementation that you wish to run                      
#######################################################################
BACKENDS=("CUDA" "OpenMP" "OpenACC")

# IMPL=$BACKENDS[3]
IMPL="OpenACC"

#######################################################################
#               Set implementation-specific env variables
#######################################################################
if [[ "$IMPL" == "OpenMP" ]]; then
    BLOCKS=16
    export OMP_DEFAULT_DEVICE=0
    export OMP_NUM_THREADS=128
elif [[ "$IMPL" == "OpenACC" ]]; then
    export ACC_DEVICE_TYPE=nvidia
    export ACC_DEVICE_NUM=0
fi

#######################################################################
#                 Edit the run run configuration
#######################################################################
BLOCKS=16
THREADS=128
MEM_SIZE=32089730048
ITERS=500000

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
#                         Run each add kernel
#######################################################################
BENCH="RAND_ADD STRIDE1_ADD PTRCHASE_ADD CENTRAL_ADD SG_ADD SCATTER_ADD GATHER_ADD" 
for B in $BENCH; do
    echo $HLINE | tee $OUTPUT_FILE
    echo "  Running the $B kernel using the $IMPL impl..." | tee $OUTPUT_FILE
    echo "                  $(date +"%d-%m-%y") AT $(date +"%T") " | tee $OUTPUT_FILE
    echo $HLINE | tee $OUTPUT_FILE

    if [[ "$IMPL" == "CUDA" ]]; then    
        $EXE --bench $B -m $MEM_SIZE -i $ITERS --blocks $BLOCKS --threads $THREADS | tee $OUTPUT_FILE
        echo | tee $OUTPUT_FILE
        echo | tee $OUTPUT_FILE
    elif [[ "$IMPL" == "OpenMP" ]]; then
        # FIXME: Is OpenMP teams == CUDA blocks? -p is the teams
        $EXE --bench $B -m $MEM_SIZE -p $BLOCKS -i $ITERS | tee $OUTPUT_FILE        
        echo | tee $OUTPUT_FILE
        echo | tee $OUTPUT_FILE
    elif [[ "$IMPL" == "OpenACC" ]]; then
        # FIXME: Is OpenACC gangs == CUDA blocks? -p is the teams 
        $EXE --bench $B -m $MEM_SIZE -p $BLOCKS -i $ITERS | tee $OUTPUT_FILE        
        echo | tee $OUTPUT_FILE
        echo | tee $OUTPUT_FILE 
    fi
done
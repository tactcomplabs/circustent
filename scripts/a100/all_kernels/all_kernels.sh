#!/bin/bash
#
# Script to run all ADD kernels and save output

HHLINE="============================================================="
HLINE="------------------------------------------------------"

#######################################################################
#           Set the implementation that you wish to run                      
#######################################################################
BACKENDS="CUDA OpenMP"
IMPL=$1
if [[ -z "$IMPL" ]]; then
  echo "Error: IMPL argument is required."
  echo "Usage: $0 <IMPL> [blocks] [threads] [iters] [mem_size]"
  exit 1
fi

#######################################################################
#                 Edit the run run configuration
#######################################################################
BLOCKS=${2:-32}
THREADS=${3:-256}
ITERS=${4:-500000}
MEM_SIZE=${5:-32089730048}

#######################################################################
#                      Create output file  
#######################################################################
TEST_DIR=$(pwd)
OUTPUT_FILE=$TEST_DIR/outputs/$IMPL/all_kernels_$(date +"%d-%m-%y")_$(date +"%T").txt
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
# BENCH="RAND_ADD RAND_CAS STRIDE1_ADD STRIDE1_CAS STRIDEN_ADD STRIDEN_CAS CENTRAL_ADD CENTRAL_CAS SG_ADD SG_CAS SCATTER_ADD SCATTER_CAS GATHER_ADD GATHER_CAS"
BENCH="STRIDEN_ADD STRIDEN_CAS"
for i in {1..5}; do
  echo $HHLINE >> $OUTPUT_FILE
  echo "        ITERATION $i" >> $OUTPUT_FILE
  echo $HHLINE >> $OUTPUT_FILE
  for B in $BENCH; do
    echo $HLINE >> $OUTPUT_FILE
    echo "  Running the $B kernel using the $IMPL impl..." >> $OUTPUT_FILE
    echo "                  $(date +"%d-%m-%y") AT $(date +"%T") " >> $OUTPUT_FILE
    echo $HLINE >> $OUTPUT_FILE

    if [[ "$IMPL" == "CUDA" ]]; then
      echo "Running with $BLOCKS blocks, $THREADS threads per block" >> $OUTPUT_FILE
      if [[ $BENCH == "STRIDEN_ADD" || $BENCH == "STRIDEN_CAS" ]]; then
        $EXE --bench $B -m $MEM_SIZE -i $ITERS --blocks $BLOCKS --threads $THREADS --stride 10 >> $OUTPUT_FILE
      else
        $EXE --bench $B -m $MEM_SIZE -i $ITERS --blocks $BLOCKS --threads $THREADS >> $OUTPUT_FILE
      fi
      echo >> $OUTPUT_FILE
      echo >> $OUTPUT_FILE
    elif [[ "$IMPL" == "OpenMP" ]]; then
      export OMP_NUM_TEAMS=$BLOCKS
      export OMP_NUM_THREADS=$THREADS
      echo "Running with $BLOCKS teams, $THREADS threads per team" >> $OUTPUT_FILE
      $EXE --bench $B -m $MEM_SIZE -p $BLOCKS -i $ITERS >> $OUTPUT_FILE        
      echo >> $OUTPUT_FILE
      echo >> $OUTPUT_FILE
    elif [[ "$IMPL" == "OpenACC" ]]; then
      $EXE --bench $B -m $MEM_SIZE -p $BLOCKS -i $ITERS >> $OUTPUT_FILE        
      echo >> $OUTPUT_FILE
      echo >> $OUTPUT_FILE 
    fi
  done
  echo "$i iteration complete"
done

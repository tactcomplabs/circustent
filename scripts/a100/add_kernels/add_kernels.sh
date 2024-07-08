#!/bin/bash

HHLINE="============================================================="
HLINE="------------------------------------------------------"

# --- Set the implementation that you wish to run
BACKENDS="CUDA OpenMP"
IMPL=$1
if [[ -z "$IMPL" ]]; then
  echo "Error: IMPL argument is required."
  echo "Usage: $0 <IMPL> [blocks] [threads] [iters] [mem_size]"
  exit 1
fi

# --- Set $COMPILE to true if you want to run the build script
COMPILE=false
if [[ $COMPILE == true ]]; then
  cd ../../
  source BUILD.sh
  cd nv_a100/add_kernels
fi

# --- Edit the run configuration
BLOCKS=${2:-32}
THREADS=${3:-256}
ITERS=${4:-500000}
MEM_SIZE=${5:-40000076736}

# --- Set implementation-specific env variables
if [[ "$IMPL" == "OpenMP" ]]; then
  export OMP_TARGET_OFFLOAD=MANDATORY
  export OMP_DEFAULT_DEVICE=0  # Set to the correct device ID for your A100
  export OMP_NUM_TEAMS=$BLOCKS
  export OMP_NUM_THREADS=$THREADS
  export LIBOMPTARGET_DEBUG=1  # Enable OpenMP runtime debugging
  export NVCOMPILER_ACC_DEBUG=1
  export NVCOMPILER_ACC_NOTIFY=1
  echo "Set OpenMP environment variables: OMP_TARGET_OFFLOAD=$OMP_TARGET_OFFLOAD, OMP_DEFAULT_DEVICE=$OMP_DEFAULT_DEVICE, OMP_NUM_TEAMS=$OMP_NUM_TEAMS, OMP_NUM_THREADS=$OMP_NUM_THREADS, LIBOMPTARGET_DEBUG=$LIBOMPTARGET_DEBUG"
elif [[ "$IMPL" == "OpenACC" ]]; then
  export NVCOMPILER_ACC_NOTIFY=1
  export NV_ACC_TIME=1
  export ACC_DEVICE_TYPE=nvidia
  export ACC_DEVICE_NUM=0  # Set to the correct device ID for your A100
  export ACC_NUM_GANGS=$BLOCKS
  export ACC_NUM_WORKERS=$THREADS
  echo "Set OpenACC environment variables: NVCOMPILER_ACC_NOTIFY=$NVCOMPILER_ACC_NOTIFY, NV_ACC_TIME=$NV_ACC_TIME, ACC_DEVICE_TYPE=$ACC_DEVICE_TYPE, ACC_DEVICE_NUM=$ACC_DEVICE_NUM, ACC_NUM_GANGS=$ACC_NUM_GANGS, ACC_NUM_WORKERS=$ACC_NUM_WORKERS"
fi

# --- Create output file
TEST_DIR=$(pwd)
OUTPUT_FILE=$TEST_DIR/outputs/$IMPL/add_kernels_$(date +"%d-%m-%y")_$(date +"%T").txt
if [[ -f $OUTPUT_FILE ]] ; then
  rm $OUTPUT_FILE
  touch $OUTPUT_FILE
else
  touch $OUTPUT_FILE
fi

# --- Specify path to the circustent executable
cd ../../../
CT_ROOT=$(pwd)
CT_BUILD=$CT_ROOT/build
EXE=$CT_BUILD/src/CircusTent/circustent
cd $TEST_DIR

# --- Run each add kernel
BENCH="RAND_ADD"
#BENCH="RAND_ADD STRIDE1_ADD STRIDEN_ADD CENTRAL_ADD SG_ADD SCATTER_ADD GATHER_ADD"
for i in {1..5}; do
  echo $HHLINE >> $OUTPUT_FILE
  echo "                      ITERATION $i" >> $OUTPUT_FILE
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
      export OMP_NUM_THREADS=$THREADS
      echo "Running with $BLOCKS teams, $THREADS threads per team" >> $OUTPUT_FILE
      echo "Executing: $EXE --bench $B -m $MEM_SIZE -i $ITERS --pes $BLOCKS" >> $OUTPUT_FILE
      if [[ $BENCH == "STRIDEN_ADD" || $BENCH == "STRIDEN_CAS" ]]; then
        $EXE --bench $B -m $MEM_SIZE -i $ITERS --pes $BLOCKS --stride 10 >> $OUTPUT_FILE
      else
        $EXE --bench $B -m $MEM_SIZE -i $ITERS --pes $BLOCKS >> $OUTPUT_FILE
      fi
      echo "Execution finished for $B" >> $OUTPUT_FILE
      echo >> $OUTPUT_FILE
    elif [[ "$IMPL" == "OpenACC" ]]; then
      echo "Executing: $EXE --bench $B -m $MEM_SIZE -p $BLOCKS -i $ITERS" >> $OUTPUT_FILE
      $EXE --bench $B -m $MEM_SIZE -p $BLOCKS -i $ITERS >> $OUTPUT_FILE
      echo >> $OUTPUT_FILE
      echo >> $OUTPUT_FILE
    fi
  done
  echo "$i iteration complete"
done


#!/bin/bash

# Set the common parameters
IMPL="OpenMP"
ITERS=500000
MEM_SIZE=40000076736

# Define the configurations
CONFIGURATIONS=(
  "16 256"
  "32 256"
  "64 256"
  "128 256"
)

# Directory of the main script
MAIN_SCRIPT="add_kernels.sh"

# Run the main script with each configuration
for config in "${CONFIGURATIONS[@]}"; do
  read -r BLOCKS THREADS <<< "$config"
  echo "Running configuration: Blocks=$BLOCKS, Threads=$THREADS"
  bash "$MAIN_SCRIPT" $IMPL $BLOCKS $THREADS $ITERS $MEM_SIZE
done





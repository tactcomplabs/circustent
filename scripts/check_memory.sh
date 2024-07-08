#!/bin/bash

# Get the total GPU memory in MiB
total_memory_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)

# Convert MiB to bytes (1 MiB = 1024 * 1024 bytes)
total_memory_bytes=$(($total_memory_mib * 1024 * 1024))

# Print the total memory in bytes
echo "Total GPU Memory: $total_memory_bytes bytes"


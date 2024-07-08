#!/bin/bash

while true
do
  gpu_utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr -d ' %')

  if [[ "$gpu_utilization" =~ ^[0-9]+$ ]] && [ "$gpu_utilization" -gt 0 ]
  then
    echo "Current time: $(date +%H:%M:%S)"
    echo "GPU utilization: ${gpu_utilization}%"
  fi

  sleep 1
done


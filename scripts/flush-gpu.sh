#!/bin/bash

echo "WARNING: This will kill all processes using NVIDIA GPUs, potentially terminating running jobs."
read -p "Are you sure you want to proceed? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    exit 0
fi

PIDS=$(sudo fuser -v /dev/nvidia* 2>/dev/null | awk '{print $3}' | sort -u)
if [ -z "$PIDS" ]; then
    echo "No processes found using NVIDIA GPUs."
else
    echo "Killing processes: $PIDS"
    echo "$PIDS" | xargs sudo kill -9
    echo "GPU memory flushed."
fi
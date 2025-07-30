#!/bin/bash

# Variables
VENV_DIR=.venv

# Check for venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found. Please run 'make setup' first."
    exit 1
fi

# Run with env vars and pass all args
echo "Running main training script (legacy; consider using 'train' instead)..."
HYDRA_FULL_ERROR=1 CUDA_LAUNCH_BLOCKING=1 uv run python train.py "$@"
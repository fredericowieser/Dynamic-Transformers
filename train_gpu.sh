#!/bin/bash
#SBATCH --job-name=bigt_dev
#SBATCH --partition=agent-xlong
#SBATCH --gres=gpu:2
#SBATCH --output=slurm_out/%j.out
#SBATCH --time=5-00:00:00

# This script is for running a large-scale training job on a GPU server with SLURM.
# It handles environment setup and runs the default training configuration.
#
# To submit this job, run:
# sbatch train_gpu.sh
#
# To monitor the job, use:
# squeue -u $USER
#
# To see the output, check the slurm-*.out file that will be created in this directory.

echo "--- Setting up environment for GPU training ---"

# Set OMP_NUM_THREADS to 1 for efficiency with torch
export OMP_NUM_THREADS=1

# Install uv (if not already installed)
if ! command -v uv &> /dev/null
then
    echo "uv could not be found, installing it now..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the environment to add uv to the PATH for the current session
    source "$HOME/.local/bin/env"
fi

# Create a .venv local virtual environment (if it doesn't exist)
if [ ! -d ".venv" ]
then
    echo "Creating virtual environment..."
# The uv venv command creates a virtual environment in the current directory.
    uv venv
fi

# Install the repo dependencies
echo "Installing dependencies with uv..."
export UV_HTTP_TIMEOUT=600
uv sync

# Activate venv so that `python` uses the project's venv instead of system python
echo "Activating virtual environment..."
source .venv/bin/activate

echo "--- Starting training run on GPU ---"
# Set Hydra to show full stack traces
export HYDRA_FULL_ERROR=1
# Enable blocking CUDA launches to pinpoint asynchronous errors
export CUDA_LAUNCH_BLOCKING=1

# Run training with the default configuration.
# We explicitly set num_processes=2 to match our SLURM gres=gpu:2 allocation.
accelerate launch --num_processes 2 train.py \
    logging.wandb.enabled=true

echo "--- GPU training run finished ---"

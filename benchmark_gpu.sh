#!/bin/bash --login
#SBATCH --partition=agentS-xlong
#SBATCH --gres=gpu:h200:1
#SBATCH --job-name=bench_models
#SBATCH --time=1-00:00:00

# This script is a SLURM wrapper for bench.py, designed to run model benchmarking jobs.
#
# To submit this job, run:
# sbatch benchmark_gpu.sh <bench.py arguments, e.g., --model_path /path/to/model --tasks general>
#
# To monitor the job, use:
# squeue -u $USER
#
# To see the output, check the slurm-*.out file that will be created in this directory.

echo "--- Setting up environment for Benchmarking ---"

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
    uv venv
fi

# Install the repo dependencies
echo "Installing dependencies with uv..."
export UV_HTTP_TIMEOUT=600
uv sync

# Activate venv so that `python` uses the project's venv instead of system python
echo "Activating virtual environment..."
source .venv/bin/activate

echo "--- Starting benchmarking run ---"

# Pass all arguments directly to bench.py
python bench.py "$@"

echo "--- Benchmarking run finished ---"
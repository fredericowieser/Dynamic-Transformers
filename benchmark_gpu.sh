#!/bin/bash
#SBATCH --job-name=bench_transformers
#SBATCH --partition=agent-xlong
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_out/%j.out
#SBATCH --time=1-00:00:00

# This script runs the performance_benchmark.py script to fill in the model variant performance table.
# It uses randomly initialized models and tests across multiple sequence lengths.
#
# To submit this job, run:
# sbatch benchmark_gpu.sh
#
# To monitor the job, use:
# squeue -u $USER

echo "--- Setting up environment for Benchmarking ---"

# Set OMP_NUM_THREADS to 1 for efficiency with torch
export OMP_NUM_THREADS=1

# Install uv (if not already installed)
if ! command -v uv &> /dev/null
then
    echo "uv could not be found, installing it now..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

# Create and sync virtual environment
if [ ! -d ".venv" ]
then
    echo "Creating virtual environment..."
    uv venv
fi

echo "Installing dependencies with uv..."
export UV_HTTP_TIMEOUT=600
uv sync

# Activate venv
echo "Activating virtual environment..."
source .venv/bin/activate

echo "--- Starting benchmarking run ---"

# Define sequence lengths as requested
SEQ_LENGTHS="1024,2048,4096,8192,16384,32768"
MODEL_SIZE="0.5B"

# Execute the performance benchmark script
# It will iterate through Dense, MoD, SDT (Causal), and STT (Causal)
# and print the Markdown table to the output.
python performance_benchmark.py \
    --model_size $MODEL_SIZE \
    --sequence_lengths $SEQ_LENGTHS \
    --batch_size 1 \
    --num_runs 5 \
    --num_warmup_runs 2

echo "--- Benchmarking run finished ---"

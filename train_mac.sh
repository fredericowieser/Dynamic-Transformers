#!/bin/bash

# This script is for running a small-scale training job on a macOS laptop.
# It handles environment setup and uses the 'laptop.yaml' configuration,
# which is set up for CPU/MPS training with a tiny model and dataset.

echo "--- Setting up environment for macOS training ---"

# Set OMP_NUM_THREADS to 1 for efficiency with torch
export OMP_NUM_THREADS=1

# Install uv (if not already installed)
if ! command -v uv &> /dev/null
then
    echo "uv could not be found, installing it now..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Create a .venv local virtual environment (if it doesn't exist)
if [ ! -d ".venv" ]
then
    echo "Creating virtual environment..."
    uv venv
fi

# Install the repo dependencies
echo "Installing dependencies with uv..."
uv sync

# Activate venv so that `python` uses the project's venv instead of system python
echo "Activating virtual environment..."
source .venv/bin/activate

echo "--- Starting training run on Mac ---"
# Run training with the laptop-specific configuration
# This uses a small model and dataset suitable for local debugging.
python train.py --config-name=laptop

echo "--- Mac training run finished ---"

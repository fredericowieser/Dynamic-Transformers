#!/usr/bin/env bash

# setup.sh
# This script sets up the development environment for the Dynamic-Transformer project.

echo "--- Starting Project Setup ---"

# --- 1. Set Environment Variables for Better Debugging ---
# HYDRA_FULL_ERROR=1: Prevents Hydra from truncating stack traces.
# CUDA_LAUNCH_BLOCKING=1: Makes CUDA operations synchronous for clearer error messages.
echo "Setting environment variables for debugging (HYDRA_FULL_ERROR=1, CUDA_LAUNCH_BLOCKING=1)..."
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
echo "Done."
echo ""


# --- 2. Check for and Install 'uv' ---
echo "Checking for 'uv', the Python package manager..."
if ! command -v uv &> /dev/null; then
    echo "'uv' not found. Installing it now via astral.sh..."
    if curl -LsSf https://astral.sh/uv/install.sh | sh; then
        # Add uv to the PATH for the current shell session
        export PATH="$HOME/.local/bin:$PATH"
        echo "'uv' installed successfully."
    else
        echo "Error: Failed to install 'uv'. Please try installing it manually."
        echo "See: https://github.com/astral-sh/uv"
        exit 1
    fi
else
    echo "'uv' is already installed."
fi
echo ""


# --- 3. Create Virtual Environment ---
VENV_DIR=".venv"
PYTHON_VERSION="3.10"
echo "Creating a virtual environment at './${VENV_DIR}' with Python >=${PYTHON_VERSION}..."
if uv venv --python ${PYTHON_VERSION} ${VENV_DIR}; then
    echo "Virtual environment created successfully."
else
    echo "Error: Failed to create the virtual environment."
    echo "Please ensure you have Python ${PYTHON_VERSION} or newer installed and accessible."
    exit 1
fi
echo ""


# --- 4. Install Project Dependencies ---
echo "Installing dependencies from 'pyproject.toml' into the virtual environment..."
# Use 'uv run' to execute the pip install command within the new venv
if uv pip sync pyproject.toml; then
    echo "All project dependencies installed successfully."
else
    echo "Error: Failed to install dependencies."
    echo "Please check the 'pyproject.toml' file and network connection."
    exit 1
fi
echo ""


# --- 5. Final Instructions ---
echo "------------------------------------------------------------------"
echo "Setup Complete!"
echo "------------------------------------------------------------------"
echo ""
echo "Next Steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source ${VENV_DIR}/bin/activate"
echo ""
echo "2. (Optional) Log in to Weights & Biases for experiment tracking:"
echo "   uv run wandb login"
echo ""
echo "--- How to Run Project Scripts ---"
echo ""
echo "Use 'uv run python <script_name>.py -- [arguments]'"
echo ""
echo "  To Train a model:"
echo "      uv run python train.py --config-name=your_config_here"
echo ""
echo "  To Evaluate a trained model:"
echo "      uv run python eval.py --model_path /path/to/your/model --is_instruct"
echo ""
echo "  To Run Inference with a model:"
echo "      uv run python infer.py /path/to/your/model --prompt \"Your prompt here\" --print_gates"
echo ""
echo "------------------------------------------------------------------"
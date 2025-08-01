#!/usr/bin/env bash

# setup.sh
# This script sets up the development environment for the Dynamic-Transformer project.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting Project Setup ---"

# --- 1. Set Environment Variables for Better Debugging ---
# HYDRA_FULL_ERROR=1: Prevents Hydra from truncating stack traces.
# CUDA_LAUNCH_BLOCKING=1: Makes CUDA operations synchronous for clearer error messages.
echo "Setting environment variables..."
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
echo "Done."
echo ""


# --- 2. Check for and Install 'uv' ---
echo "Checking for 'uv', the Python package manager..."
if ! command -v uv &> /dev/null; then
    echo "'uv' not found. Installing it now via astral.sh..."
    if curl -LsSf https://astral.sh/uv/install.sh | sh; then
        # Add uv to the PATH for the current shell session
        # Use a more generic path that uv typically installs to
        export PATH="$HOME/.cargo/bin:$PATH"
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
PYTHON_VERSION="3.10" # Specify your desired Python version explicitly
echo "Creating a virtual environment at './${VENV_DIR}' with Python >=${PYTHON_VERSION}..."
# Remove existing venv to ensure a clean slate
rm -rf "${VENV_DIR}"

# Use --python for strict versioning; if 3.10 not found, uv will error
if uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"; then
    echo "Virtual environment created successfully."
else
    echo "Error: Failed to create the virtual environment."
    echo "Please ensure you have Python ${PYTHON_VERSION} or newer installed and accessible."
    exit 1
fi
echo ""


# --- 4. Install Project Dependencies ---
echo "Installing dependencies from 'pyproject.toml' into the virtual environment..."
# Use 'uv pip sync' to sync the venv with pyproject.toml, including all extras.
# This is generally preferred over `uv pip install -e .` for a full sync.
if uv pip sync pyproject.toml --all-extras; then # <-- Ensured --all-extras here
    echo "All project dependencies installed successfully."
else
    echo "Error: Failed to install dependencies."
    echo "Please check the 'pyproject.toml' file and network connection."
    exit 1
fi
echo ""


# --- 5. Dynamically set LD_LIBRARY_PATH for uv-cached CUDA libraries ---
echo "Setting LD_LIBRARY_PATH for uv-cached CUDA libraries..."
UV_CUDA_LIB_PATHS=""

# Common NVIDIA library directory patterns within uv's cache
# These patterns target where uv typically extracts the native libraries.
LIB_TYPE_PATTERNS=(
    "cuda/lib" "cudnn/lib" "cublas/lib" "cufft/lib" "curand/lib"
    "cusolver/lib" "cusparse/lib" "cusparselt/lib" "cupti/lib" "nccl/lib"
    "nvjitlink/lib" # Added based on common NVIDIA package names
)

# Search within uv's cache for relevant library directories
# Increased maxdepth to ensure deeper 'lib' directories are found
for lib_pattern in "${LIB_TYPE_PATTERNS[@]}"; do
    for lib_dir in $(find "${HOME}/.cache/uv/archive-v0/" -maxdepth 5 -type d -path "*/${lib_pattern}" 2>/dev/null | sort -u); do # <-- Changed maxdepth here
        if ls "${lib_dir}"/*.so* &>/dev/null; then # Check if the directory actually contains .so files
            if [[ ! ":${UV_CUDA_LIB_PATHS}:" == *":${lib_dir}:"* ]]; then
                if [ -z "${UV_CUDA_LIB_PATHS}" ]; then
                    UV_CUDA_LIB_PATHS="${lib_dir}"
                else
                    UV_CUDA_LIB_PATHS="${lib_dir}:${UV_CUDA_LIB_PATHS}"
                fi
            fi
        fi
    done
done

# Check if UV_CUDA_LIB_PATHS is empty or not
if [ -n "${UV_CUDA_LIB_PATHS}" ]; then
    # Prepend the collected uv paths to LD_LIBRARY_PATH
    # This ensures uv's versions are found before system versions
    export LD_LIBRARY_PATH="${UV_CUDA_LIB_PATHS}:${LD_LIBRARY_PATH}"
    echo "LD_LIBRARY_PATH set to: ${LD_LIBRARY_PATH}"
else
    echo "No specific uv-cached CUDA library paths found or needed. This might be normal for CPU-only setups."
fi
echo "Done setting LD_LIBRARY_PATH."

echo ""
# --- 6. Final Instructions ---
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
#!/bin/bash

# Variables
PYTHON_VERSION=3.10
VENV_DIR=.venv

# Ensure pyproject.toml exists
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found."
    exit 1
fi

# Add local bin to PATH
export PATH="$HOME/.local/bin:$PATH"

# Check for uv and install if missing
echo "Checking for uv..."
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "uv installed successfully."
fi

# Create venv
echo "Creating virtual environment with Python $PYTHON_VERSION..."
uv venv --python $PYTHON_VERSION
echo "Virtual environment created at $VENV_DIR/"

# Activate and install dependencies/logins in a subshell
echo "Activating venv and installing dependencies..."
bash -c "
    source $VENV_DIR/bin/activate
    uv pip sync pyproject.toml
    echo 'Dependencies installed.'
    echo 'Running wandb login (enter API key if prompted)...'
    wandb login || echo 'Warning: wandb login failed or skipped.'
    echo 'Running huggingface_cli login (enter token if prompted)...'
    huggingface_cli login || echo 'Warning: huggingface_cli login failed or skipped.'
"

echo ""
echo "Setup complete."
echo ""
echo "Next steps:"
echo "1. Activate the environment: source $VENV_DIR/bin/activate"
echo "Tip: If 'uv' isn't found in your terminal after setup, run..."
echo "source $HOME/.local/bin/env"
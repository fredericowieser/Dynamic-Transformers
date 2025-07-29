# Makefile

# --- Variables ---
# Define the Python version to use for the virtual environment
PYTHON_VERSION := 3.10
VENV_DIR := .venv

# --- Targets ---

.PHONY: setup help run train eval infer flush-gpu clean

setup: pyproject.toml # Ensure pyproject.toml exists before creating venv
	@export PATH="$(HOME)/.local/bin:$$PATH"; \
	echo "Checking for uv..."; \
	if ! command -v uv >/dev/null 2>&1; then \
		echo "uv not found, installing..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "uv installed successfully."; \
	fi; \
	echo "Creating virtual environment with Python $(PYTHON_VERSION)..."; \
	uv venv --python $(PYTHON_VERSION); \
	echo "Virtual environment created at $(VENV_DIR)/"; \
	echo "Activating venv and installing dependencies..."; \
	bash -c "source $(VENV_DIR)/bin/activate && \
		uv pip sync pyproject.toml && \
		echo 'Dependencies installed.' && \
		echo 'Running wandb login (enter API key if prompted)...' && \
		wandb login || echo 'Warning: wandb login failed or skipped.' && \
		echo 'Running huggingface_cli login (enter token if prompted)...' && \
		huggingface_cli login || echo 'Warning: huggingface_cli login failed or skipped.'"; \
	echo ""; \
	echo "Setup complete."; \
	echo ""; \
	echo "Next steps:"; \
	echo "1. Activate the environment: source $(VENV_DIR)/bin/activate"; \
	echo "Tip: If 'uv' isn't found in your terminal after setup, run..."; \
	echo "source $$HOME/.local/bin/env"

help:
	@echo "Available targets:"
	@echo "  setup       Sets up the virtual environment, installs dependencies, and handles logins."
	@echo "  train       Runs the training script with full stack traces (pass args like 'data=sft_mix')."
	@echo "  eval        Runs the evaluation script with full stack traces (pass args like '--model_path=outputs/my_model')."
	@echo "  infer       Runs the inference/chat script with full stack traces (pass args like '--prompt=\"Hello\"')."
	@echo "  flush-gpu   Flushes GPU memory by killing processes using NVIDIA devices (requires sudo)."
	@echo "  run         Legacy: Runs main training script with full stack traces (use 'train' instead)."
	@echo "  clean       Removes virtual environment and __pycache__ folders."
	@echo "  help        Displays this help message."
	@echo ""
	@echo "Environment notes: All Python runs enable HYDRA_FULL_ERROR=1 and CUDA_LAUNCH_BLOCKING=1 for full stack traces."
	@echo "After running 'make setup', remember to activate the environment manually:"
	@echo "  source .venv/bin/activate"
	@echo "Tip: Pass arguments to targets like 'make train data=sft_mix'."

# Capture all arguments passed to targets
ARGS := $(filter-out $@,$(MAKECMDGOALS))
# Prevent Make from trying to interpret ARGS as targets
$(eval $(ARGS):;@:)

run:
	@echo "Running main training script (legacy; consider using 'train' instead)..."
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Error: Virtual environment not found. Please run 'make setup' first."; \
		exit 1; \
	fi
	@HYDRA_FULL_ERROR=1 CUDA_LAUNCH_BLOCKING=1 uv run python train.py $(ARGS)

train:
	@echo "Running training script with full stack traces..."
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Error: Virtual environment not found. Please run 'make setup' first."; \
		exit 1; \
	fi
	@HYDRA_FULL_ERROR=1 CUDA_LAUNCH_BLOCKING=1 uv run python train.py $(ARGS)

eval:
	@echo "Running evaluation script with full stack traces..."
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Error: Virtual environment not found. Please run 'make setup' first."; \
		exit 1; \
	fi
	@HYDRA_FULL_ERROR=1 CUDA_LAUNCH_BLOCKING=1 uv run python eval.py $(ARGS)

infer:
	@echo "Running inference/chat script with full stack traces..."
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Error: Virtual environment not found. Please run 'make setup' first."; \
		exit 1; \
	fi
	@HYDRA_FULL_ERROR=1 CUDA_LAUNCH_BLOCKING=1 uv run python infer.py $(ARGS)

flush-gpu:
	@echo "WARNING: This will kill all processes using NVIDIA GPUs, potentially terminating running jobs."
	@read -p "Are you sure you want to proceed? (y/n): " confirm && [ $$confirm = "y" ] || exit 0; \
	PIDS=$$(sudo fuser -v /dev/nvidia* 2>/dev/null | awk '{print $$3}' | sort -u); \
	if [ -z "$$PIDS" ]; then \
		echo "No processes found using NVIDIA GPUs."; \
	else \
		echo "Killing processes: $$PIDS"; \
		echo "$$PIDS" | xargs sudo kill -9; \
		echo "GPU memory flushed."; \
	fi

clean:
	@echo "Removing virtual environment: $(VENV_DIR)/"
	@rm -rf $(VENV_DIR)
	@echo "Removing all __pycache__ folders..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Cleanup complete."
# Makefile

# --- Variables ---
# Define the Python version to use for the virtual environment
PYTHON_VERSION := 3.10
VENV_DIR := .venv

# --- Targets ---

.PHONY: setup help

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
	echo "Virtual environment created at $(VENV_DIR)/"
	@echo ""
	@echo "Next steps:"
	@echo "1. Activate the environment: source $(VENV_DIR)/bin/activate"
	@echo "2. Install dependencies: uv pip sync pyproject.toml"
	@echo "3. Login to wandb: wandb login"
	@echo ""
	@echo "Tip: If 'uv' isn't found in your terminal after setup, run..."
	@echo "source $$HOME/.local/bin/env"

help:
	@echo "Available targets:"
	@echo "  setup       Sets up the virtual environment and creates pyproject.toml if it doesn't exist."
	@echo "  help        Displays this help message."
	@echo ""
	@echo "After running 'make setup', remember to activate the environment manually:"
	@echo "  source .venv/bin/activate"
	@echo "Then install dependencies: uv pip sync pyproject.toml"
	@echo "And login to wandb: wandb login"

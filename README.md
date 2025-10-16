# Dynamic Transformers

This repository provides a research framework for implementing, training, and evaluating several dynamic transformer architectures. The primary goal is to compare these dynamic models against a standard transformer baseline, focusing on computational efficiency and performance.

The framework is built using PyTorch, Hugging Face `transformers` and `accelerate` for efficient, distributed training. Configuration is managed by `hydra`, allowing for flexible and reproducible experiments.

## Features

- **Multiple Architectures**: Implements several dynamic transformer models alongside a standard baseline:
    - **Standard Transformer**: A conventional transformer based on the `Qwen2` architecture.
    - **Mixture-of-Depths (MoD)**: Allows tokens to bypass layers based on a routing mechanism.
    - **Sparse Dynamic Transformer (SDT)**: Uses a "decision" layer and a "dynamic" layer, where tokens are routed to the dynamic layer based on a surprise-based mechanism.
    - **Sparse Transition Transformer (STT)**: Uses a transition network to predict the next state of a token and routes tokens to a transformer block based on prediction error.
- **Flexible Configuration**: Uses `hydra` for managing all experiment parameters, from model architecture and data sources to optimizer settings.
- **Efficient Training**: Leverages Hugging Face `accelerate` for seamless multi-GPU and mixed-precision training.
- **Versatile Data Handling**: Supports mixing multiple Hugging Face datasets for pre-training or fine-tuning.
- **Integrated Evaluation**: Includes a script to run benchmarks on trained models using the `lm-evaluation-harness`.
- **Experiment Tracking**: Integrates with Weights & Biases (W&B) for logging metrics, losses, and model checkpoints.

## Codebase Structure

The project is organized into several key directories:

```
.
├── config/                 # Hydra configuration files
│   ├── default.yaml        # Base configuration for all parameters
│   └── laptop.yaml         # Overrides for local debugging
├── src/
│   ├── data/               # Data loading and processing modules
│   │   ├── base_dataset.py
│   │   ├── mixed_dataset.py  # Combines multiple datasets
│   │   └── ...
│   ├── models/             # Model implementations
│   │   ├── base/           # Shared components for dynamic models
│   │   ├── mod/            # Mixture-of-Depths (MoD) model
│   │   ├── sdt/            # Sparse Dynamic Transformer (SDT) model
│   │   ├── standard/       # Standard transformer baseline
│   │   └── stt/            # Sparse Transition Transformer (STT) model
│   └── training/           # Training utilities
│       ├── utils.py        # Optimizer setup, checkpointing, etc.
│       └── eval_utils.py   # Adapter for lm-evaluation-harness
├── train.py                # Main script for training models
└── bench.py                # Script for running lm-eval benchmarks
```

### Key Files

- **`train.py`**: The main entry point for launching a training run. It handles parsing the `hydra` config, setting up the model, data, optimizer, and running the training loop with `accelerate`.
- **`bench.py`**: A script to evaluate a trained model checkpoint on standard NLP benchmarks using `lm-eval`. It loads a saved model and runs the specified task suite.
- **`config/default.yaml`**: The central configuration file. It defines all default parameters for the model, data, training, and system settings. Experiments can be defined by overriding these parameters.
- **`src/models/`**: This directory contains the core logic for the different transformer architectures. Each model has its own subdirectory and inherits from `BaseForCausalLM` in `src/models/base/causal_lm.py`.
- **`src/data/mixed_dataset.py`**: This class is responsible for loading, processing, and concatenating multiple datasets from the Hugging Face Hub, as defined in the `data.dataset_configs` section of the configuration.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/dynamic-transformers.git
    cd dynamic-transformers
    ```

2.  **Create a virtual environment and install dependencies:**
    This project uses `uv` for package management. You can install the necessary packages from `pyproject.toml`. The main dependencies are:
    - `torch`
    - `transformers`
    - `accelerate`
    - `hydra-core`
    - `omegaconf`
    - `datasets`
    - `wandb`
    - `lm-eval`
    - `peft`

    You can install them using `uv`:
    ```bash
    uv pip install -r requirements.txt 
    # Or if you have the pyproject.toml with dependencies listed
    uv pip install .
    ```

## Usage

The framework is designed to be run from the command line using `train.py` and `bench.py`.

### Configuration

All aspects of a run are controlled by `hydra` configuration files located in the `config` directory. The main file is `config/default.yaml`.

To create a new experiment, you can either modify `default.yaml` or, more cleanly, create a new configuration file (e.g., `config/my_experiment.yaml`) that inherits from the default and overrides specific parameters.

For example, to change the model type and learning rate, your `my_experiment.yaml` might look like this:
```yaml
# config/my_experiment.yaml
defaults:
  - default

model:
  type: sdt # Switch to the SDT model

training:
  optimizer:
    lr: 5.0e-5
```

### Training

To start a training run, execute `train.py`. You can specify an experiment config or override parameters directly from the command line.

**Examples:**

- **Run the default configuration (as defined in `config/default.yaml`):**
  ```bash
  python train.py
  ```

- **Run a specific experiment configuration:**
  ```bash
  python train.py --config-name=my_experiment
  ```

- **Override parameters from the command line:**
  ```bash
  # Train an STT model with a different batch size and number of steps
  python train.py model.type=stt data.batch_size=4 training.max_steps=5000

  # Train a model from scratch for debugging on a laptop
  python train.py --config-name=laptop
  ```

Training artifacts, logs, and model checkpoints will be saved to the directory specified by `run.output_dir` in the configuration, which defaults to `outputs/RUN_NAME`.

### Evaluation

After training, a final model is saved in Hugging Face format. You can evaluate this model on various benchmarks using `bench.py`.

**Command:**
```bash
python bench.py --model_path <path_to_saved_model> --tasks <task_suite>
```

**Arguments:**
- `--model_path`: Path to the directory containing the saved model (e.g., `outputs/my-run-name/final_model`).
- `--tasks`: A comma-separated list of tasks or task suites to run. Available suites are `general`, `math`, `code`, and `quick_test`.
- `--batch_size`: The batch size for evaluation.

**Example:**
```bash
python bench.py --model_path outputs/experiment-stt-2025-10-16_10-00-00-0.5B/final_model --tasks general,math
```

The script will print a summary table of the results and save a detailed JSON file in the model directory.
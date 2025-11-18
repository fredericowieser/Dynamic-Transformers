# Subjective Depth & Timescale Transformers

This repository contains the code for "Subjective Depth & Timescale Transformers," a research project implementing and evaluating dynamic transformer architectures. It includes two novel models, the Subjective Depth Transformer (SDT) and the Subjective Timescale Transformer (STT), which leverage Bayesian surprise signals to dynamically route computation, learning where and when to compute.

The framework is built on PyTorch, Hugging Face `transformers`, and `accelerate`, with configuration managed by `hydra`.

## Setup

Clone the repository. The training scripts will handle the rest of the environment setup automatically.
```bash
git clone https://github.com/your-username/dynamic-transformers.git
cd dynamic-transformers
```

## Training & Evaluation

The easiest way to run training is using the provided scripts. They will automatically set up a Python virtual environment with `uv`, install dependencies, and launch the training run. Evaluation is automatically performed at the end of training, with results saved to `eval_results.json` in the model's output directory.

### On a macOS Laptop (CPU/MPS)

For local development and debugging, use `train_mac.sh`. This script runs a small-scale training job using the `laptop.yaml` configuration.

```bash
chmod +x train_mac.sh
./train_mac.sh
```

### On a GPU Server (with SLURM)

For full-scale training on a SLURM cluster, use `train_gpu.sh`. This script submits a job to the cluster and uses `accelerate` for distributed training.

```bash
chmod +x train_gpu.sh
sbatch train_gpu.sh
```

You can customize the run by editing the script or by setting environment variables. For example, to log the run to Weights & Biases, set the `WANDB_RUN` variable:

```bash
WANDB_RUN=my-awesome-experiment sbatch train_gpu.sh
```

The model type (e.g., `stt`, `sdt`, `mod`) and other parameters can be modified directly within the `train_gpu.sh` script.

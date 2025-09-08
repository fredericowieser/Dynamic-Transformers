# Dynamic Transformers (DTF)

This repository contains the official implementation for the MSc thesis, **"Dynamic Transformers"**. It provides the code to train, evaluate, and run inference with the proposed Dynamic Transformer (DTF) architecture, a conditional computation model inspired by predictive coding.

## Abstract

The standard Transformer's uniform application of computation to all tokens creates a scalability bottleneck. While methods like Mixture-of-Depths (MoD) offer a solution by processing a subset of tokens based on a learned "importance" score, we propose a more principled approach. The **Dynamic Transformer (DTF)** is a novel architecture inspired by computational neuroscience, specifically Variational Predictive Routing (VPR). DTF uses a surprise-based gating mechanism to conditionally allocate compute, routing tokens based on a context-dependent measure of information gain.

This repository provides code to adapt a pre-trained Qwen2.5-0.5B model to the DTF architecture and compare it against a re-implemented MoD baseline under matched compute capacity ($\gamma=0.5$). Our results show that DTF achieves a small but consistent validation loss advantage over MoD, suggesting a more effective inductive bias for routing.

## Architecture Overview

The DTF replaces the standard uniform stack of Transformer layers with alternating **Decision** and **Dynamic** layers. The Decision Layer computes a posterior state (via a standard TF block) and a prior prediction (via a lightweight PriorFFN). The Dynamic Layer's Predictive Router uses these signals to gate which tokens are processed by a second TF block.

 
*A high-level comparison of the standard Transformer (left) and the DTF architecture (right).*

## Features

- **Model Implementation**: `DynamicQwenForCausalLM` with support for both **DTF** and **MoD** architectures.
- **Training Script**: A robust training script using `accelerate` for distributed training, `hydra` for configuration, and supporting LoRA for parameter-efficient fine-tuning.
- **Evaluation Script**: Integrated with `lm-eval` for standardized benchmarking.
- **Inference Script**: A simple script to run generation with a trained model.
- **Hugging Face Hub Uploader**: A utility to generate a model card and upload models to the Hub.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/dynamic-transformers.git
    cd dynamic-transformers
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -e ".[dev]"
    ```
    *The `-e` flag installs the project in editable mode.*

## Quickstart

All scripts are configured using [Hydra](https://hydra.cc/). You can override any configuration setting from the command line.

### 1. Training a Model

Use the `train.py` script to fine-tune a model. You can select the dynamic architecture by overriding `model`.

**Train a DTF model:**
```bash
python train.py model=dtf run.name=dtf-qwen-0.5b-run1
```

**Train a MoD baseline:**
```bash
python train.py model=mod run.name=mod-qwen-0.5b-run1
```
- Checkpoints will be saved to the `outputs/` directory, structured by date and run name.
- Training progress is logged to Weights & Biases if `logging.wandb.enabled=true`.

### 2. Evaluating a Model

Use the `evaluate.py` script to run benchmarks on a trained checkpoint using the `lm-evaluation-harness`.

```bash
python evaluate.py \
    --model_path outputs/your-run-name/final_model \
    --tasks general
```
- The `--tasks` argument can be a comma-separated list of tasks or pre-defined suites (`general`, `math`, `code`, `quick_test`).
- Results are saved as a JSON file in the model's directory.

### 3. Running Inference

Use the `inference.py` script for text generation.

```bash
python inference.py \
    "path/to/your/model_checkpoint" \
    "The capital of the United Kingdom is" \
    --max_new_tokens 50
```
- **Note:** The script automatically disables the KV cache for dynamic models (`DTF`, `MoD`), as it is incompatible with their routing mechanism.

### 4. Uploading to Hugging Face Hub

The `upload_to_hub.py` script generates a model card and uploads your model, tokenizer, and custom code to the Hub.

```bash
# First, log in to your Hugging Face account
huggingface-cli login

# Then, run the upload script
python upload_to_hub.py \
    path/to/your/model_checkpoint \
    your-new-repo-name \
    --hf_username YOUR_HF_USERNAME \
    --eval_results path/to/eval_results.json
```

## Results Summary

Our key finding is that the DTF architecture consistently achieves a lower validation loss than the MoD baseline at a matched compute capacity ($\gamma=0.5$). This suggests that the surprise-based, model-comparison gating provides a more effective inductive bias for routing than a context-independent importance score.

However, both dynamic models underperform the dense baseline on several downstream benchmarks in our limited transfer-learning setting. This is an expected trade-off due to reduced per-token computation and highlights challenges in adapting pre-trained models to conditional computation. For a detailed analysis, please see Chapter 5 of the [thesis](link-to-thesis.pdf).

## Citation

If you find this work useful in your research, please consider citing the thesis:

```bibtex
@mastersthesis{wieser2025dynamic,
  author  = {Wieser, Frederico Luis},
  title   = {Dynamic Transformers},
  school  = {University College London},
  year    = {2025},
  month   = {September},
  address = {London, UK},
  note    = {MSc Computational Statistics and Machine Learning}
}
```
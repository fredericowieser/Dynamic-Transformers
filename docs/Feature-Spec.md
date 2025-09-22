# Spec.md: Dynamic Transformers - Feature Specification

This document outlines the key features, configurations, and training specifications of the Dynamic Transformers project. It is intended to serve as a blueprint for a rewrite, ensuring that all existing functionality related to training, logging, and model configuration is maintained.

---

## 1. Experiment Tracking & Logging (Weights & Biases) 📊

The project integrates with **Weights & Biases (wandb)** for comprehensive experiment tracking. This is enabled via a configuration flag (`logging.wandb.enabled=true`).

The following metrics are logged to wandb during training runs:

### Standard Training Metrics:
* **Training Loss**: The primary language modeling loss on the training data.
* **Training Perplexity**: Perplexity on the training data, derived from the loss.
* **Validation Loss**: The language modeling loss evaluated on the validation set, providing a measure of generalization.

### Custom Dynamic Transformer (DTF) Metrics:
To analyze the behavior of the dynamic routing mechanism, several internal states and parameters are tracked and logged. These metrics are averaged across all Dynamic Layers in the model before logging.

* **`prior_loss`**: The Mean Squared Error (MSE) between the `PriorFFN`'s prediction and the actual output of the main Transformer block. This measures how well the lightweight prior is approximating the full block's computation.
* **Gating Signal Components**: These track the activation levels of the different parts of the routing decision.
    * **`S_CE`**: The mean activation of the **Expected Change** criterion. This signal is prediction-based and rises as the PriorFFN becomes more accurate.
    * **`S_CU`**: The mean activation of the **Unexpected Change** criterion. This signal is novelty-based and is typically higher early in training before the model has learned the data dynamics.
    * **`G_cont`**: The final combined gating signal, representing the output of the probabilistic OR between `S_CE` and `S_CU`. This value determines which tokens are selected for processing.
* **Learnable Router Parameters**: The evolution of the four learnable parameters within the Predictive Router is tracked to observe how the routing policy adapts over training.
    * **`β_ce`**: The inverse temperature for the Expected Change criterion.
    * **`β_cu`**: The inverse temperature for the Unexpected Change criterion.
    * **`o_ce`**: The learnable offset for the Expected Change criterion.
    * **`m_cu`**: The learnable multiplier for the Unexpected Change criterion.

---

## 2. Optimizer & Scheduler Configuration ⚙️

### Optimizer
The training process uses the **AdamW** optimizer, a standard choice for training Transformers that incorporates decoupled weight decay.
* **Weight Decay**: 0.01.

### Differential Learning Rates
To ensure training stability and preserve pre-trained knowledge, a **differential learning rate** strategy is employed across three distinct parameter groups.

| Parameter Group | Learning Rate | Rationale |
| :--- | :--- | :--- |
| **Base Model** | `1.0e-5` | A conservative rate for the original Transformer block weights to prevent catastrophic forgetting of pre-trained knowledge. |
| **PriorFFN** | `1.0e-3` | A moderately higher rate to encourage the network to quickly learn its predictive function. |
| **Predictive Router**| `1.0e-2` | The highest learning rate, allowing the four learnable gating parameters to adapt rapidly to the fine-tuning data. |

### Learning Rate Scheduler
A **cosine decay learning rate schedule** is used for all parameter groups. This is preceded by a **linear warm-up phase** that covers the first **1%** of the total training steps.

---

## 3. Training & Dataset Specification 📚

### Training Hyperparameters
* **Global Random Seed**: 42.
* **Sequence Length**: 1024 tokens.
* **Per-Device Batch Size**: 16.
* **Gradient Accumulation Steps**: 64.
* **Effective Batch Size**: 1024 (16 per-device * 64 accumulation steps).

### Fine-Tuning Dataset Mixture
The models are fine-tuned on a mixed-domain corpus constructed from several publicly available datasets on the Hugging Face Hub.

| Dataset Name | Ratio | Tokens (M) | Description |
| :--- | :--- | :--- | :--- |
| `wikitext-103-raw-v1` | 1.0 | 103 | Encyclopedic text |
| `cnn_dailymail` (3.0.0) | 0.2 | 43 | High-quality news |
| `storytracer/US-PD-Books`| 0.5 | 650 | Classic literature |
| `HuggingFaceTB/cosmopedia`| 0.1 | 2500 | Synthetic textbooks |
| `sciq` | 1.0 | 3 | Factual science QA |
| `codeparrot-clean-valid` | 1.0 | ~18 | Cleaned Python code |
| `roneneldan/TinyStories` | 1.0 | 525 | Simple causality stories |

---

## 4. Parameter-Efficient Adaptation & Layer Freezing (LoRA) 🧊

The project includes a parameter-efficient fine-tuning (PEFT) option using **Low-Rank Adaptation (LoRA)**.

### LoRA Configuration
* **Rank (`r`)**: 16.
* **Alpha (`α`)**: 32.
* **Dropout**: 0.05.
* **Learning Rate**: When using LoRA, the learning rate for the adapters is set to `1.0e-4` to accelerate convergence.

### Layer Application & Freezing Strategy
The LoRA implementation follows a specific strategy for applying adapters and freezing layers:
* **Frozen Layers**: The weights of the **pre-trained base model are frozen**, which is the core principle of LoRA for memory efficiency.
* **Adapted Layers**: LoRA adapters are applied comprehensively to all **linear projection layers** within the base Transformer blocks. This includes the `query`, `key`, `value`, `output`, and `MLP` layers.
* **Fully Fine-Tuned Layers**: Crucially, the newly introduced dynamic components—the **Predictive Router** and the **PriorFFN**—are **exempted from LoRA**. These components are **fully fine-tuned** (i.e., all their parameters are updated) to ensure they have sufficient capacity to learn the routing policy from scratch.
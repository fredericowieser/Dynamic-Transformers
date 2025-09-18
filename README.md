# Dynamic Transformer

Modular implementation of DTF (Dynamic Transformer) and MoD (Mixture-of-Depths) architectures with multi-GPU support.

## Features

- **DTF**: Surprise-based routing using predictive coding principles
- **MoD**: Importance-based top-k token selection
- **Multi-GPU training** via Accelerate
- **Multi-dataset support** with weighted mixing
- **Hydra configuration** for flexible experimentation
- **Flash Attention 2** support
- **WandB integration** for experiment tracking

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── config/
│   └── base.yaml        # Hydra configuration
├── src/
│   ├── models/
│   │   ├── base/        # Base classes
│   │   ├── dtf/         # DTF implementation
│   │   └── mod/         # MoD implementation
│   └── data/            # Dataset utilities
└── train.py             # Training script
```

## Usage

### Training DTF Model

```bash
python train.py model_type=dtf model.capacity_gamma=0.5
```

### Training MoD Model

```bash
python train.py model_type=mod model.capacity_gamma=0.5
```

### Multi-GPU Training

```bash
accelerate launch --multi_gpu train.py model_type=dtf
```

## Training Examples

### Training from Scratch

#### 50M Parameter Model
Train a small 50M parameter DTF model from scratch on WikiText:

```bash
python train.py \
    model_type=dtf \
    model.base_model_name="Qwen/Qwen2.5-0.5B" \
    model.num_hidden_layers=6 \
    model.hidden_size=512 \
    model.intermediate_size=1536 \
    model.num_attention_heads=8 \
    model.capacity_gamma=0.5 \
    data.dataset_name=wikitext \
    training.num_epochs=10 \
    training.batch_size=8 \
    training.gradient_accumulation_steps=4 \
    training.lr=1e-3 \
    training.warmup_steps=1000
```

#### 0.5B Parameter Model
Train the full Qwen2.5-0.5B model from scratch with DTF:

```bash
python train.py \
    model_type=dtf \
    model.base_model_name="Qwen/Qwen2.5-0.5B" \
    model.capacity_gamma=0.5 \
    model.beta_ce_init=-0.3 \
    model.beta_cu_init=-0.6 \
    data.mixed=true \
    data.dataset_names='["wikitext","openwebtext","pile-subset"]' \
    data.dataset_weights='[0.2,0.5,0.3]' \
    training.num_epochs=3 \
    training.batch_size=4 \
    training.gradient_accumulation_steps=8 \
    training.lr=5e-4 \
    training.warmup_steps=2000
```

### Qwen2.5 Family Models

#### Qwen2.5-1.5B
```bash
accelerate launch --multi_gpu train.py \
    model_type=dtf \
    model.base_model_name="Qwen/Qwen2.5-1.5B" \
    model.capacity_gamma=0.5 \
    training.batch_size=2 \
    training.gradient_accumulation_steps=16 \
    training.lr=2e-4
```

#### Qwen2.5-3B
```bash
accelerate launch --multi_gpu --num_processes=4 train.py \
    model_type=dtf \
    model.base_model_name="Qwen/Qwen2.5-3B" \
    model.capacity_gamma=0.25 \
    training.batch_size=1 \
    training.gradient_accumulation_steps=32 \
    training.lr=1e-4 \
    training.use_flash_attention=true
```

#### Qwen2.5-7B
```bash
accelerate launch --multi_gpu --num_processes=8 train.py \
    model_type=dtf \
    model.base_model_name="Qwen/Qwen2.5-7B" \
    model.capacity_gamma=0.125 \
    training.batch_size=1 \
    training.gradient_accumulation_steps=64 \
    training.lr=5e-5 \
    training.use_flash_attention=true \
    training.gradient_checkpointing=true
```

### Dataset Configurations

#### Simple: Single Dataset (WikiText)
```bash
python train.py \
    model_type=dtf \
    data.dataset_name=wikitext \
    data.mixed=false
```

#### Medium: Two Datasets
```bash
python train.py \
    model_type=dtf \
    data.mixed=true \
    data.dataset_names='["wikitext","openwebtext"]' \
    data.dataset_weights='[0.3,0.7]'
```

#### Complex: Full Dataset Mix
```bash
python train.py \
    model_type=dtf \
    data.mixed=true \
    data.dataset_names='["wikitext","openwebtext","pile-subset","c4-subset","redpajama-subset"]' \
    data.dataset_weights='[0.1,0.3,0.2,0.2,0.2]' \
    data.max_length=2048
```

### Custom Configuration Examples

#### Low Resource Training
For limited GPU memory:
```bash
python train.py \
    model_type=dtf \
    model.base_model_name="Qwen/Qwen2.5-0.5B" \
    model.capacity_gamma=0.25 \
    data.dataset_name=wikitext \
    training.batch_size=1 \
    training.gradient_accumulation_steps=16 \
    training.gradient_checkpointing=true \
    training.mixed_precision="fp16"
```

#### High Performance Training
For maximum speed with sufficient resources:
```bash
accelerate launch --multi_gpu train.py \
    model_type=dtf \
    model.base_model_name="Qwen/Qwen2.5-1.5B" \
    training.batch_size=16 \
    training.gradient_accumulation_steps=1 \
    training.use_flash_attention=true \
    training.mixed_precision="bf16" \
    training.compile_model=true
```

### Custom Configuration File

Create a custom config file `config/my_training.yaml`:

```yaml
model_type: "dtf"
model:
  base_model_name: "Qwen/Qwen2.5-0.5B"
  capacity_gamma: 0.5
  beta_ce_init: -0.3
  beta_cu_init: -0.6
  prior_loss_weight: 0.05

data:
  dataset_name: "wikitext"
  max_length: 1024
  mixed: false

training:
  num_epochs: 5
  batch_size: 4
  gradient_accumulation_steps: 8
  lr: 5e-4
  warmup_steps: 1000
  lr_multipliers:
    base_model: 1.0
    router: 10.0
    prior: 10.0
```

Then run:
```bash
python train.py --config-name=my_training
```

## Model Architecture

Both DTF and MoD models inherit from `BaseDynamicCausalLM` which provides:
- Common interface for dynamic models
- Automatic weight loading from pretrained models
- Unified loss computation

### DTF Components
- **DTFRouter**: Surprise-based routing with CE and CU criteria
- **DTFDecisionLayer**: Computes original, posterior, and prior states
- **DTFDynamicLayer**: Processes selected tokens based on routing
- **PriorFFN**: Lightweight network for prior prediction

### MoD Components
- **MoDRouter**: Learned importance scoring
- **MoDLayer**: Top-k token selection and processing

## Configuration

Key parameters in `config/base.yaml`:

```yaml
model_type: "dtf"              # Model selection
model:
  capacity_gamma: 0.5          # Fraction of tokens to process

  # DTF-specific
  beta_ce_init: -0.3           # CE criterion temperature
  beta_cu_init: -0.6           # CU criterion temperature
  prior_loss_weight: 0.05      # Auxiliary loss weight

training:
  num_epochs: 3
  gradient_accumulation_steps: 8
  lr_multipliers:              # Component-specific LRs
    base_model: 1.0
    router: 10.0
    prior: 10.0
```

## Adding New Models

To add a new dynamic model:

1. Create a new folder in `src/models/`
2. Implement router and layers
3. Create model class inheriting from `BaseDynamicCausalLM`
4. Register in `train.py`:

```python
model_classes = {
    "dtf": DTFForCausalLM,
    "mod": MoDForCausalLM,
    "new_model": NewModelForCausalLM,  # Add here
}
```

## Requirements

See `pyproject.toml` for dependencies. Main requirements:
- PyTorch >= 2.0
- Transformers >= 4.30
- Accelerate
- Hydra
- OmegaConf
- WandB (optional)
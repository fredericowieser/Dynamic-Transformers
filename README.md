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

### Custom Configuration

```bash
python train.py \
    model_type=dtf \
    model.capacity_gamma=0.25 \
    data.dataset_name=openwebtext \
    training.lr=5e-5 \
    training.num_epochs=5
```

### Mixed Datasets

```yaml
# In config or via command line
data:
  mixed: true
  dataset_names: ["wikitext", "openwebtext", "pile-subset"]
  dataset_weights: [0.3, 0.5, 0.2]
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
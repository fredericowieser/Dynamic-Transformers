# Mac Training Guide for Dynamic Transformer

Optimized setup for training ~50M parameter Dynamic Transformer models on Mac.

## Quick Start

### 1. Install Dependencies

```bash
pip install torch transformers accelerate datasets hydra-core omegaconf wandb tqdm
```

### 2. Quick Test Run

```bash
# Quick test (100 steps, ~2 minutes)
python train_mac.py --config-name=quick_test

# Or use the launch script
./run_mac.sh dtf quick
```

### 3. Full Training

```bash
# DTF model with sane defaults
./run_mac.sh dtf full

# MoD model
./run_mac.sh mod full

# Or directly with Hydra
python train_mac.py --config-name=mac_training model_type=dtf
```

## Training Configurations

### Available Configs

- `quick_test.yaml` - 100 steps test run with WikiText
- `mac_training.yaml` - Full config with mixed datasets
- `base.yaml` - Standard multi-GPU config

### Model Sizes

| Config | Parameters | Hidden Size | Layers | Attention Heads |
|--------|-----------|------------|--------|-----------------|
| small_50m | ~50M | 512 | 12 | 8 |
| base (Qwen2.5-0.5B) | 494M | 896 | 24 | 14 |

### Training Modes

Using `run_mac.sh`:

```bash
./run_mac.sh [model_type] [mode]

# Modes:
# - quick: 1000 steps (~10 min on M1)
# - short: 5000 steps (~45 min on M1)
# - full: 3 epochs (~2-3 hours on M1)
```

## Data Configurations

### Single Dataset (Quick Testing)

```yaml
# config/data/wikitext.yaml
data:
  dataset_name: "wikitext"
  dataset_config: "wikitext-2-raw-v1"
  block_size: 512
  batch_size: 8
```

### Mixed Datasets (Production)

```yaml
# config/data/mixed_training.yaml
data:
  mixed: true
  dataset_names:
    - "tatsu-lab/alpaca"
    - "databricks/databricks-dolly-15k"
    - "c4"
    - "wikipedia"
  dataset_weights: [0.3, 0.3, 0.2, 0.2]
```

## Mac-Specific Optimizations

### Memory Management

- Gradient accumulation: 16 steps (effective batch = 128)
- Gradient checkpointing enabled
- Float32 precision (more stable on Mac)
- Smaller prior network (25% of FFN size)

### Performance Settings

- MPS (Metal Performance Shaders) auto-detection
- CPU fallback if MPS unavailable
- Optimized data loading (4 workers)
- No memory pinning (not supported on Mac)

### Training Hyperparameters

Optimized for 50M model on Mac:

```yaml
training:
  optimizer:
    lr: 3e-4  # Higher LR for small model
    weight_decay: 0.1
    scheduler: "cosine_with_restarts"

  lr_multipliers:
    base_model: 1.0
    router: 5.0  # Less aggressive than default
    prior: 5.0
```

## Monitoring

### Console Output

```
🚀 Dynamic Transformer Mac Training
📚 Epoch 1/3
Training: 100%|████| 1000/1000 [10:23<00:00, 1.60it/s, loss=2.345, lr=3e-4]
📊 Step 500: val_loss = 2.123
💾 Saving best model (val_loss: 2.123)
```

### WandB Integration

```bash
# Enable WandB logging
python train_mac.py logging.wandb.enabled=true logging.wandb.entity=YOUR_ENTITY
```

## Custom Training

### Override Any Parameter

```bash
python train_mac.py \
    model_type=dtf \
    model.capacity_gamma=0.25 \
    model.hidden_size=768 \
    training.optimizer.lr=1e-4 \
    data.batch_size=16
```

### Add New Dataset

1. Create config in `config/data/`:
```yaml
# config/data/custom.yaml
data:
  dataset_name: "your-dataset"
  block_size: 1024
  batch_size: 4
```

2. Train with:
```bash
python train_mac.py --config-name=mac_training data=custom
```

## Expected Performance

On Apple M1/M2:

| Model | Dataset | Steps | Time | Val Loss |
|-------|---------|-------|------|----------|
| DTF 50M | WikiText-2 | 1000 | ~10 min | 2.8-3.2 |
| DTF 50M | Mixed | 5000 | ~45 min | 2.2-2.5 |
| MoD 50M | WikiText-2 | 1000 | ~10 min | 3.0-3.4 |

## Troubleshooting

### MPS Issues

If you encounter MPS errors:
```bash
# Force CPU usage
python train_mac.py system.device=cpu
```

### Memory Issues

Reduce batch size or increase gradient accumulation:
```bash
python train_mac.py \
    data.batch_size=4 \
    training.gradient_accumulation_steps=32
```

### Slow Training

- Ensure you're using MPS (check for "✅ Using Mac GPU" message)
- Reduce `num_workers` if CPU-bound
- Use smaller `block_size` for faster iterations

## Model Usage After Training

```python
from src.models import DTFForCausalLM
from transformers import AutoTokenizer

# Load trained model
model = DTFForCausalLM.from_pretrained("outputs/mac_dtf_20240101_120000/best_model")
tokenizer = AutoTokenizer.from_pretrained("outputs/mac_dtf_20240101_120000/best_model")

# Generate text
inputs = tokenizer("Once upon a time", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```
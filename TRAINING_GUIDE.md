# Dynamic Transformer Training Guide

## Overview

This guide explains how to train Dynamic Transformer (DTF) and Mixture of Depths (MoD) models on both CUDA and Metal (Mac) systems. The training infrastructure supports both training from scratch and transfer learning from pretrained models like Qwen2.5.

## Quick Start

### 1. Test Your Setup
```bash
python3 test_setup.py
```

### 2. Quick Training Test (WikiText, 1000 steps)
```bash
./run_training.sh quick dtf
```

### 3. Train from Scratch
```bash
./run_training.sh scratch dtf
```

### 4. Transfer Learning from Qwen-0.5B
```bash
./run_training.sh transfer dtf
```

## Architecture Support

### DTF (Dynamic Transformer)
- Implements surprise-based conditional computation
- Uses Decision Layers to compute original, posterior, and prior states
- Routes tokens based on "expected" and "unexpected" criteria
- Processes ~50% of tokens (configurable via `capacity_gamma`)

### MoD (Mixture of Depths)
- Engineering-driven approach with simple Top-K routing
- Lightweight router selects important tokens
- Remaining tokens bypass via residual connection

## Device-Specific Optimizations

### CUDA Systems
- **Precision**: BFloat16 automatic mixed precision
- **Attention**: Flash Attention 2 (when available)
- **Compilation**: Optional torch.compile for performance
- **Memory**: Gradient checkpointing enabled

### Metal (Mac) Systems
- **Precision**: Float32 for stability
- **Attention**: Eager attention (native PyTorch)
- **Memory**: Gradient checkpointing enabled
- **Compatibility**: MPS-compatible operations (topk instead of kthvalue)

## Configuration Files

### Main Configurations
- `config/training.yaml` - Base training configuration
- `config/training_from_scratch.yaml` - Random initialization training
- `config/training_transfer.yaml` - Transfer learning from pretrained
- `config/mac_training.yaml` - Mac-specific settings

### Data Configurations
- `config/data/wikitext.yaml` - WikiText-2 for quick testing
- `config/data/mixed_training.yaml` - Diverse dataset mixture

### Model Configurations
- `config/model/small_50m.yaml` - ~50M parameter model

## Training Scripts

### Universal Script
```bash
./run_training.sh [MODE] [MODEL_TYPE] [CONFIG]
```

**Modes:**
- `quick` - Quick test with WikiText (1000 steps)
- `scratch` - Train from scratch
- `transfer` - Transfer learning
- `custom` - Use custom config file

**Model Types:**
- `dtf` - Dynamic Transformer
- `mod` - Mixture of Depths

### Examples

#### Quick Test on Mac
```bash
./run_mac_fixed.sh dtf quick
```

#### Custom Configuration
```bash
python3 train.py --config-name=my_config \
    model_type=dtf \
    training.max_steps=10000 \
    training.optimizer.lr=1e-4
```

#### Override Parameters
```bash
python3 train.py \
    model_type=dtf \
    data=wikitext \
    training.from_scratch=true \
    training.num_epochs=5 \
    system.device=mps
```

## Key Parameters

### Training Parameters
- `training.max_steps`: Maximum training steps (-1 for unlimited)
- `training.gradient_accumulation_steps`: Gradient accumulation (default: 16)
- `training.optimizer.lr`: Base learning rate (default: 3e-4)
- `training.from_scratch`: Train from random init (default: false)

### Model Parameters
- `model.model.num_hidden_layers`: Number of transformer layers
- `model.model.hidden_size`: Hidden dimension
- `model.model.capacity_gamma`: Token routing capacity (0.5 = 50%)

### DTF-Specific Parameters
- `beta_ce_init`: Expected criterion temperature
- `beta_cu_init`: Unexpected criterion temperature
- `cu_detection_multiplier_init`: Moving average multiplier
- `prior_ffn_intermediate_size_factor`: Prior network size ratio

## Monitoring Training

### Weights & Biases
Enable by setting `logging.wandb.enabled=true`:
```bash
python3 train.py logging.wandb.enabled=true \
    logging.wandb.project=my-project
```

### Console Output
- Training loss every 50 steps
- Validation loss at eval intervals
- Learning rate and gradient norms

### Checkpoints
Models are saved to `outputs/[model]_[mode]_[timestamp]/`:
- `best_model/` - Best validation loss
- `checkpoint-N/` - Regular checkpoints
- `final_model/` - Final model

## Troubleshooting

### Mac/MPS Issues
- If you encounter MPS operation errors, the code automatically falls back to CPU-compatible operations
- Use Float32 precision for stability
- Disable Flash Attention

### Memory Issues
- Reduce `batch_size` in data config
- Increase `gradient_accumulation_steps`
- Enable `gradient_checkpointing`

### Performance
- CUDA: Enable BF16 and Flash Attention
- Mac: Keep batch sizes small, use gradient accumulation
- Both: Use torch.compile on CUDA for speed boost

## Advanced Usage

### Custom Dataset
Create a new data config:
```yaml
data:
  mixed: false
  dataset_name: "your-dataset"
  block_size: 1024
  batch_size: 4
```

### Multi-GPU Training
The code uses single-GPU by default. For multi-GPU:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py
```

### Hyperparameter Sweeps
Use Hydra's sweep functionality:
```bash
python3 train.py -m training.optimizer.lr=1e-4,3e-4,1e-3
```

## Model Sizes

### From Scratch (~30M params)
- 8 layers, 384 hidden, 1024 intermediate
- Trains quickly for experimentation

### Transfer Learning (0.5B params)
- Full Qwen2.5-0.5B architecture
- Modified with DTF/MoD layers
- Best quality but slower

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA 11.8+ (for CUDA systems)
- macOS 13+ (for Metal)

Install dependencies:
```bash
pip install torch transformers hydra-core omegaconf datasets tqdm wandb
```

## Citation

If you use this code, please cite the Dynamic Transformer paper and acknowledge the implementation.
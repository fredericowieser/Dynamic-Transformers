# Dynamic Transformer

Professional implementation of three transformer architectures: Standard Transformer (Qwen2.5), Dynamic Transformer (DTF), and Mixture of Depths (MoD) with unified training infrastructure.

## 🚀 Quick Start

```bash
# Clone and setup
git clone <repository>
cd Dynamic-Transformer

# Quick test with DTF (1000 steps)
./run_training.sh quick dtf

# Quick test with MoD
./run_training.sh quick mod

# Quick test with Standard Transformer
./run_training.sh quick standard
```

## 🏗️ Architecture Overview

This repository implements three transformer variants:

- **Standard Transformer**: Baseline Qwen2.5 architecture with RMSNorm, SwiGLU, and RoPE
- **Dynamic Transformer (DTF)**: Surprise-based routing using predictive coding principles
- **Mixture of Depths (MoD)**: Learned importance scoring with top-k token selection

All models share a unified base architecture and training pipeline.

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- See `pyproject.toml` for complete dependencies

## 📁 Project Structure

```
Dynamic-Transformer/
├── README.md                    # This file
├── DTF-Spec.md                 # DTF architecture specification
├── MoD-Spec.md                 # MoD architecture specification
├── Qwen-Spec.md                # Qwen2.5 architecture specification
├── TRAINING_GUIDE.md           # Detailed training guide
├── run_training.sh             # Universal training script
├── train.py                    # Main training entry point
├── config/                     # Hydra configurations
│   ├── train.yaml              # Base training config
│   ├── dtf_scratch.yaml        # DTF from scratch
│   ├── dtf_transfer.yaml       # DTF transfer learning
│   ├── mod_scratch.yaml        # MoD from scratch
│   ├── mod_transfer.yaml       # MoD transfer learning
│   ├── standard_scratch.yaml   # Standard from scratch
│   └── standard_transfer.yaml  # Standard transfer learning
└── src/                        # Source code
    ├── models/                 # Model implementations
    │   ├── base/               # Shared base classes
    │   ├── standard/           # Standard Transformer
    │   ├── dtf/                # Dynamic Transformer
    │   └── mod/                # Mixture of Depths
    ├── training/               # Training utilities
    └── data/                   # Dataset utilities
```

## 🎯 Usage

### Universal Training Script

The `run_training.sh` script provides a unified interface for all training modes:

```bash
./run_training.sh [MODE] [MODEL_TYPE] [CONFIG]

# Modes:
#   quick      - Quick test (1000 steps)
#   scratch    - Train from scratch
#   transfer   - Transfer learning from Qwen2.5
#   custom     - Use custom config file

# Model Types:
#   dtf        - Dynamic Transformer
#   mod        - Mixture of Depths
#   standard   - Standard Transformer
```

### Training Examples

#### Quick Testing
```bash
# Test DTF for 1000 steps
./run_training.sh quick dtf

# Test MoD for 1000 steps
./run_training.sh quick mod

# Test Standard Transformer for 1000 steps
./run_training.sh quick standard
```

#### Full Training
```bash
# Train DTF from scratch (10 epochs)
./run_training.sh scratch dtf

# Train MoD with transfer learning (5 epochs)
./run_training.sh transfer mod

# Train Standard Transformer from scratch
./run_training.sh scratch standard
```

#### Direct Training with Hydra
```bash
# DTF from scratch
python train.py --config-name=dtf_scratch

# MoD transfer learning
python train.py --config-name=mod_transfer

# Override parameters
python train.py --config-name=dtf_scratch training.num_epochs=5 training.batch_size=4
```

### Custom Configuration

Create a custom config file in `config/`:

```yaml
# config/my_experiment.yaml
defaults:
  - train
  - _self_

model:
  type: dtf
  size: 0.5B

training:
  from_scratch: true
  num_epochs: 5
  optimizer:
    lr: 5e-4

data:
  batch_size: 16
```

Then run:
```bash
./run_training.sh custom dtf my_experiment
```

## 🏛️ Model Architectures

### Standard Transformer
- **Base**: Qwen2.5 architecture (RMSNorm, SwiGLU, RoPE, GQA)
- **Use Case**: Baseline comparison and standard language modeling
- **Details**: See [Qwen-Spec.md](Qwen-Spec.md)

### Dynamic Transformer (DTF)
- **Innovation**: Surprise-based routing using predictive coding
- **Key Components**:
  - Decision layers compute original, posterior, and prior states
  - Dynamic layers process selected tokens based on routing scores
  - Surprise metrics (CE/CU) determine computational allocation
- **Efficiency**: ~12.5% of tokens processed per layer
- **Details**: See [DTF-Spec.md](DTF-Spec.md)

### Mixture of Depths (MoD)
- **Innovation**: Learned importance scoring for token selection
- **Key Components**:
  - Router networks compute token importance scores
  - Top-k selection chooses most important tokens
  - Auxiliary load balancing loss
- **Efficiency**: ~12.5% of tokens processed per layer
- **Details**: See [MoD-Spec.md](MoD-Spec.md)

## 🎛️ Configuration

### Model Parameters
- `model.type`: Architecture type (`standard`/`dtf`/`mod`)
- `model.size`: Model size (`0.5B`/`1.5B`/`3B`)
- `training.from_scratch`: Train from scratch vs transfer learning

### DTF-Specific Parameters
- `dtf_capacity`: Fraction of tokens to process (default: 0.125)
- `beta_ce_init`: Expected change temperature (default: -0.5)
- `beta_cu_init`: Unexpected change temperature (default: -0.8)

### MoD-Specific Parameters
- `mod_capacity`: Fraction of tokens to process (default: 0.125)
- `mod_aux_loss_weight`: Load balancing weight (default: 0.01)

## 💾 Platform Support

The training script automatically detects and optimizes for:

- **CUDA GPUs**: BF16, AMP, Flash Attention
- **Apple Silicon**: Metal Performance Shaders (MPS), FP32
- **CPU**: Fallback with appropriate settings

## 📊 Monitoring

Training progress is logged with:
- Loss metrics and gradients
- Routing statistics (tokens selected/processed)
- Model efficiency metrics
- Hardware utilization

## 🔧 Development

### Adding New Models

1. Create model implementation in `src/models/new_model/`
2. Inherit from `BaseDynamicModel` or implement standard interface
3. Create config files: `new_model_scratch.yaml`, `new_model_transfer.yaml`
4. Register in training utilities

### Model Interface

All models follow a unified interface:
```python
class MyModel(BaseDynamicModel):
    def forward(self, input_ids, **kwargs):
        # Return: CausalLMOutputWithPast
        pass
```

## 📚 Documentation

- **[DTF-Spec.md](DTF-Spec.md)**: Dynamic Transformer architecture details
- **[MoD-Spec.md](MoD-Spec.md)**: Mixture of Depths architecture details
- **[Qwen-Spec.md](Qwen-Spec.md)**: Qwen2.5 baseline architecture details
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)**: Comprehensive training guide

## 🚀 Performance

All models are optimized for efficiency:

- **Standard Transformer**: Full computational baseline
- **DTF**: ~7-8x computational savings with comparable performance
- **MoD**: ~7-8x computational savings with learned routing

Memory usage scales with model size:
- **0.5B models**: ~3-4GB GPU memory
- **1.5B models**: ~8-10GB GPU memory
- **3B models**: ~16-20GB GPU memory

## 📄 License

See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Follow existing code style and architecture patterns
4. Add tests for new functionality
5. Submit pull request

## 📞 Support

For issues and questions:
1. Check existing documentation
2. Review configuration files
3. Open GitHub issue with:
   - Error logs
   - System information
   - Reproduction steps
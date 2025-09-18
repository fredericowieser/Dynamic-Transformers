#!/bin/bash

echo "🧪 Testing TDTF Training Pipeline"
echo "=================================="

# Create a quick test training config
cat > /tmp/tdtf_test_config.yaml << 'EOF'
defaults:
  - _self_

model:
  type: "tdtf"
  size: "0.5B"

data:
  dataset_name: "wikitext"
  dataset_config: "wikitext-2-raw-v1"  # Smaller dataset
  split: "train"
  max_length: 128  # Shorter sequences
  batch_size: 4    # Smaller batches
  shuffle: true
  num_workers: 0   # No multiprocessing for compatibility

training:
  num_epochs: 1
  max_steps: 5     # Just a few steps to test
  gradient_accumulation_steps: 1
  gradient_clip_val: 1.0
  eval_interval: 5
  eval_samples: 10
  from_scratch: true  # Test from scratch initialization

  optimizer:
    lr: 1e-4
    weight_decay: 0.01
    adam_beta1: 0.9
    adam_beta2: 0.999
    adam_epsilon: 1e-8
    scheduler: "cosine"
    warmup_ratio: 0.0  # No warmup for quick test

system:
  output_dir: "outputs/test_tdtf_quick"
  compile: false
  num_workers: 0
  seed: 42

logging:
  log_interval: 1
  save_interval: 5
  eval_interval: 5
EOF

# Run the training
python train.py --config-path=/tmp --config-name=tdtf_test_config

echo "Training test completed!"
#!/bin/bash

# Mac Training Launch Script
# Easy-to-use script for training Dynamic Transformer models on Mac

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Dynamic Transformer Mac Training${NC}"
echo "=================================="

# Parse arguments
MODEL_TYPE=${1:-dtf}
MODE=${2:-quick}

# Display configuration
echo -e "${YELLOW}Configuration:${NC}"
echo "  Model Type: $MODEL_TYPE"
echo "  Training Mode: $MODE"

# Set training parameters based on mode
case $MODE in
  quick)
    echo -e "${GREEN}⚡ Quick test mode (1000 steps)${NC}"
    MAX_STEPS=1000
    EVAL_INTERVAL=100
    SAVE_INTERVAL=500
    ;;

  short)
    echo -e "${GREEN}📖 Short training (5000 steps)${NC}"
    MAX_STEPS=5000
    EVAL_INTERVAL=250
    SAVE_INTERVAL=1000
    ;;

  full)
    echo -e "${GREEN}📚 Full training (3 epochs)${NC}"
    MAX_STEPS=-1
    EVAL_INTERVAL=500
    SAVE_INTERVAL=2000
    ;;

  *)
    echo -e "${RED}Unknown mode: $MODE${NC}"
    echo "Usage: ./run_mac.sh [model_type] [mode]"
    echo "  model_type: dtf or mod (default: dtf)"
    echo "  mode: quick, short, or full (default: quick)"
    exit 1
    ;;
esac

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 not found!${NC}"
    exit 1
fi

# Check for required packages
echo -e "${YELLOW}Checking dependencies...${NC}"
python3 -c "import torch, transformers, hydra, datasets" 2>/dev/null || {
    echo -e "${RED}Missing dependencies! Install with:${NC}"
    echo "pip install -r requirements.txt"
    exit 1
}

# Create output directory
OUTPUT_DIR="outputs/mac_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# Run training
echo -e "${GREEN}Starting training...${NC}"
echo "Output directory: $OUTPUT_DIR"
echo ""

python3 train_mac.py \
    model_type=$MODEL_TYPE \
    training.max_steps=$MAX_STEPS \
    training.eval_interval=$EVAL_INTERVAL \
    training.save_interval=$SAVE_INTERVAL \
    system.output_dir=$OUTPUT_DIR \
    hydra.run.dir=$OUTPUT_DIR

echo ""
echo -e "${GREEN}✅ Training complete!${NC}"
echo "Models saved to: $OUTPUT_DIR"
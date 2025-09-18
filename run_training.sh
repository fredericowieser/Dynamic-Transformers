#!/bin/bash

# Universal Training Launch Script for Dynamic Transformer
# Works on both Mac (Metal) and CUDA systems

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}╔═══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Dynamic Transformer Training         ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════╝${NC}"
echo ""

# Parse arguments
MODE=${1:-quick}
MODEL_TYPE=${2:-dtf}
CONFIG=${3:-default}

# Function to print usage
usage() {
    echo "Usage: $0 [MODE] [MODEL_TYPE] [CONFIG]"
    echo ""
    echo "MODE options:"
    echo "  quick        - Quick test with WikiText (1000 steps)"
    echo "  scratch      - Train from scratch with WikiText"
    echo "  transfer     - Transfer learning from Qwen-0.5B"
    echo "  custom       - Use custom config file"
    echo ""
    echo "MODEL_TYPE options:"
    echo "  dtf          - Dynamic Transformer (default)"
    echo "  mod          - Mixture of Depths"
    echo "  standard     - Standard Transformer (Qwen2.5)"
    echo ""
    echo "CONFIG options:"
    echo "  default      - Use default config for the mode"
    echo "  <path>       - Path to custom config file"
    echo ""
    echo "Examples:"
    echo "  $0 quick dtf           # Quick test with DTF"
    echo "  $0 scratch mod         # Train MoD from scratch"
    echo "  $0 transfer dtf        # Transfer learning with DTF"
    echo "  $0 custom dtf my.yaml  # Use custom config"
    exit 1
}

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found!${NC}"
    echo "Please install Python 3.8+ to continue."
    exit 1
fi

# Detect available hardware
echo -e "${YELLOW}🔍 Detecting hardware...${NC}"
DEVICE_TYPE="cpu"
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE_TYPE="cuda"
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    echo -e "${GREEN}✅ CUDA GPU detected: $GPU_NAME${NC}"
elif python3 -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    DEVICE_TYPE="mps"
    echo -e "${GREEN}✅ Apple Silicon GPU detected (Metal)${NC}"
else
    echo -e "${YELLOW}⚠️ No GPU detected, will use CPU${NC}"
fi

# Check dependencies
echo -e "${YELLOW}📦 Checking dependencies...${NC}"
MISSING_DEPS=""

for pkg in torch transformers hydra-core datasets tqdm wandb; do
    if ! python3 -c "import ${pkg//-/_}" 2>/dev/null; then
        MISSING_DEPS="$MISSING_DEPS $pkg"
    fi
done

if [ ! -z "$MISSING_DEPS" ]; then
    echo -e "${YELLOW}📥 Installing missing dependencies:$MISSING_DEPS${NC}"
    pip install $MISSING_DEPS
fi

# Set configuration based on mode and model type
case $MODE in
    quick)
        echo -e "${GREEN}⚡ Quick test mode${NC}"
        CONFIG_NAME="${MODEL_TYPE}_scratch"
        EXTRA_ARGS="training.num_epochs=1 training.max_steps=1000 training.eval_interval=100 training.save_interval=500"
        ;;

    scratch)
        echo -e "${GREEN}🔨 Training from scratch${NC}"
        CONFIG_NAME="${MODEL_TYPE}_scratch"
        EXTRA_ARGS=""
        ;;

    transfer)
        echo -e "${GREEN}🎓 Transfer learning mode${NC}"
        CONFIG_NAME="${MODEL_TYPE}_transfer"
        EXTRA_ARGS=""
        ;;

    custom)
        if [ "$CONFIG" == "default" ]; then
            echo -e "${RED}❌ Custom mode requires a config file path${NC}"
            usage
        fi
        echo -e "${GREEN}⚙️ Using custom config: $CONFIG${NC}"
        CONFIG_NAME="$CONFIG"
        EXTRA_ARGS=""
        ;;

    *)
        echo -e "${RED}❌ Unknown mode: $MODE${NC}"
        usage
        ;;
esac

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/${MODEL_TYPE}_${MODE}_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

# Display configuration
echo ""
echo -e "${BLUE}Configuration Summary:${NC}"
echo "  Mode: $MODE"
echo "  Model Type: $MODEL_TYPE"
echo "  Device: $DEVICE_TYPE"
echo "  Output Directory: $OUTPUT_DIR"
echo ""

# Confirmation prompt for non-quick modes
if [ "$MODE" != "quick" ]; then
    read -p "Ready to start training? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Training cancelled."
        exit 1
    fi
fi

# Launch training
echo -e "${GREEN}🚀 Starting training...${NC}"
echo ""

# Set environment variables based on device
if [ "$DEVICE_TYPE" == "cuda" ]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    echo "Using CUDA device: $CUDA_VISIBLE_DEVICES"
fi

# Run training
python3 train.py \
    --config-name="$CONFIG_NAME" \
    system.output_dir="$OUTPUT_DIR" \
    $EXTRA_ARGS

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Training completed successfully!${NC}"
    echo -e "Models saved to: ${BLUE}$OUTPUT_DIR${NC}"

    # Show model files
    echo ""
    echo "Model checkpoints:"
    ls -lh $OUTPUT_DIR/*/pytorch_model.bin 2>/dev/null || ls -lh $OUTPUT_DIR/*/*.safetensors 2>/dev/null || echo "  No model files found yet"
else
    echo ""
    echo -e "${RED}❌ Training failed!${NC}"
    echo "Check the logs above for error details."
    exit 1
fi
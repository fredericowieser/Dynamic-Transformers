#!/bin/bash
# test_pipeline.sh - End-to-End Pipeline test for Causal and Non-Causal routing

echo "================================================================"
echo " Starting End-to-End Testing Pipeline"
echo "================================================================"

set -e # Exit immediately if a command exits with a non-zero status

# Set threads
export OMP_NUM_THREADS=1

# Prepare environment
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

if [ ! -d ".venv" ]; then
    uv venv
fi
uv sync
source .venv/bin/activate

MODELS=("mod" "sdt" "stt")
SIZE="0.5B"
SEQ_LENGTHS="1024,2048,4096,8192,16384,32768"

for MODEL in "${MODELS[@]}"; do
    echo "================================================================"
    echo " Testing Pipeline for Model Type: $MODEL"
    echo "================================================================"

    # 1. Finetune model (50 steps)
    echo "--> [1/4] Finetuning $MODEL for 50 steps..."
    python train.py --config-name test_pipeline model.type=$MODEL

    # Find the output directory (latest created experiment directory for this model)
    LATEST_DIR=$(ls -td outputs/test-pipeline-${MODEL}-* | head -1)
    MODEL_PATH="${LATEST_DIR}/final_model"

    if [ ! -d "$MODEL_PATH" ]; then
        echo "Error: Finetuned model not found at $MODEL_PATH"
        exit 1
    fi

    # 2. Evaluate Non-Causal Router
    echo "--> [2/4] Evaluating $MODEL (Non-Causal Router)..."
    python bench.py --model_path $MODEL_PATH --tasks general --batch_size 8

    # 3. Evaluate Causal Router
    echo "--> [3/4] Evaluating $MODEL (Causal Router)..."
    python bench.py --model_path $MODEL_PATH --tasks general --batch_size 8 --use_causal_router

    # 4. Latency Benchmarking
    echo "--> [4/4] Hardware Latency Benchmarking ($MODEL)..."
    echo "    - Non-Causal Latency:"
    python performance_benchmark.py --model_size $SIZE --sequence_lengths $SEQ_LENGTHS --batch_size 1 --model_types "standard,$MODEL"
    echo "    - Causal Latency:"
    python performance_benchmark.py --model_size $SIZE --sequence_lengths $SEQ_LENGTHS --batch_size 1 --model_types "$MODEL" --use_causal_router
done

echo "================================================================"
echo " All Tests and Benchmarks for All Models Completed"
echo "================================================================"

#!/bin/bash
# test_pipeline.sh - Enhanced End-to-End Pipeline test

echo "================================================================"
echo " Starting End-to-End Testing Pipeline (Multi-GPU Optimized)"
echo "================================================================"

set -e # Exit immediately if a command exits with a non-zero status

# Prepare environment
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Please run 'uv sync' first."
    exit 1
fi
source .venv/bin/activate

# Configuration for Test run
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
TOTAL_BATCH_SIZE=64
PER_DEVICE_BATCH_SIZE=$((TOTAL_BATCH_SIZE / (NUM_GPUS > 0 ? NUM_GPUS : 1)))
ACCUMULATE_GRAD_BATCHES=1
SEQ_LENGTHS="1024,2048,4096,8192,16384,32768"
MODELS=("mod" "sdt" "stt")
SIZE="0.5B"

# Set threads and CUDA config
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

for MODEL in "${MODELS[@]}"; do
    echo "================================================================"
    echo " Testing Pipeline for Model Type: $MODEL"
    echo "================================================================"

    # 1. Finetune model (Multi-GPU if available)
    echo "--> [1/4] Finetuning $MODEL for 50 steps..."
    # Auto-batch sizing is enabled in train.py, so we don't need to force batch_size here.
    # We let it calculate based on available VRAM to reach total batch 64.
    if [ "$NUM_GPUS" -gt 1 ]; then
        accelerate launch --num_processes $NUM_GPUS train.py \
            --config-name test_pipeline \
            model.type=$MODEL \
            model.stt.use_g_threshold_selection=false
    else
        python3 train.py --config-name test_pipeline \
            model.type=$MODEL \
            model.stt.use_g_threshold_selection=false
    fi

    # Find the output directory (latest created experiment directory for this model)
    LATEST_DIR=$(ls -td outputs/test-pipeline-${MODEL}-* | head -1)
    MODEL_PATH="${LATEST_DIR}/final_model"

    if [ ! -d "$MODEL_PATH" ]; then
        echo "Error: Finetuned model not found at $MODEL_PATH"
        exit 1
    fi

    # 2. Evaluate Non-Causal Router (Parallelized across GPUs)
    echo "--> [2/4] Evaluating $MODEL (Non-Causal Router)..."
    python bench.py --model_path $MODEL_PATH --tasks general --batch_size 16 --output_dir "$LATEST_DIR" --limit 10

    # 3. Evaluate Causal Router (Parallelized across GPUs)
    echo "--> [3/4] Evaluating $MODEL (Causal Router)..."
    python bench.py --model_path $MODEL_PATH --tasks general --batch_size 16 --use_causal_router --output_dir "$LATEST_DIR" --limit 10

    # 4. Latency Benchmarking (Logs per sequence length)
    echo "--> [4/4] Hardware Latency Benchmarking ($MODEL)..."
    echo "    - Non-Causal Latency:"
    python performance_benchmark.py --model_size $SIZE --sequence_lengths $SEQ_LENGTHS --batch_size 1 --model_types "standard,$MODEL" --output_path "${LATEST_DIR}/latency_non_causal.json"
    echo "    - Causal Latency:"
    python performance_benchmark.py --model_size $SIZE --sequence_lengths $SEQ_LENGTHS --batch_size 1 --model_types "$MODEL" --use_causal_router --output_path "${LATEST_DIR}/latency_causal.json"
done

echo "================================================================"
echo " All Tests and Benchmarks for All Models Completed"
echo "================================================================"

#!/bin/bash
# run_scaling_experiments.sh - Multi-GPU Scaling Experiments (0.5B to 7B)

set -e # Exit immediately if a command exits with a non-zero status

echo "================================================================"
echo " Starting Scaling Experiments (Multi-GPU Job)"
echo "================================================================"

# Prepare environment
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Please run 'uv sync' first."
    exit 1
fi
source .venv/bin/activate

# Configuration
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
TOTAL_BATCH_SIZE=64
PER_DEVICE_BATCH_SIZE=$((TOTAL_BATCH_SIZE / (NUM_GPUS > 0 ? NUM_GPUS : 1)))
ACCUMULATE_GRAD_BATCHES=1
SEQ_LENGTHS="1024,2048,4096,8192,16384,32768"
MODELS=("mod" "sdt" "stt")
SIZES=("0.5B" "1.5B" "3B" "7B")

# Set threads
export OMP_NUM_THREADS=1

for SIZE in "${SIZES[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo "================================================================"
        echo " STARTING EXPERIMENT: Model=$MODEL, Size=$SIZE"
        echo "================================================================"

        # 1. Train Model
        echo "--> [1/3] Training $MODEL-$SIZE..."
        if [ "$NUM_GPUS" -gt 1 ]; then
            accelerate launch --num_processes $NUM_GPUS train.py \
                --config-name $SIZE \
                model.type=$MODEL \
                data.batch_size=$PER_DEVICE_BATCH_SIZE \
                training.accumulate_grad_batches=$ACCUMULATE_GRAD_BATCHES \
                model.stt.use_g_threshold_selection=false \
                logging.wandb.enabled=true \
                run.run_final_evaluation=false
        else
            python train.py \
                --config-name $SIZE \
                model.type=$MODEL \
                data.batch_size=$PER_DEVICE_BATCH_SIZE \
                training.accumulate_grad_batches=$ACCUMULATE_GRAD_BATCHES \
                model.stt.use_g_threshold_selection=false \
                logging.wandb.enabled=true \
                run.run_final_evaluation=false
        fi

        # Find the experiment output directory
        LATEST_DIR=$(ls -td outputs/experiment-${MODEL}-*-${SIZE} | head -1)
        MODEL_PATH="${LATEST_DIR}/final_model"

        if [ ! -d "$MODEL_PATH" ]; then
            echo "Error: Training failed, model not found at $MODEL_PATH"
            exit 1
        fi

        # 2. Sequential Evaluation (Immediate Logging after each task)
        echo "--> [2/3] Evaluating $MODEL-$SIZE (Non-Causal & Causal)..."
        
        # Non-Causal Router Evaluation (Parallelized across GPUs for speed)
        python bench.py --model_path $MODEL_PATH --tasks general --batch_size 16 --output_dir "$LATEST_DIR"
        
        # Causal Router Evaluation (Parallelized across GPUs for speed)
        python bench.py --model_path $MODEL_PATH --tasks general --batch_size 16 --use_causal_router --output_dir "$LATEST_DIR"

        # 3. Latency Benchmarking (Logs per sequence length)
        echo "--> [3/3] Latency Benchmarking $MODEL-$SIZE..."
        
        # Non-Causal Latency
        python performance_benchmark.py --model_size $SIZE --sequence_lengths $SEQ_LENGTHS --batch_size 1 --model_types "standard,$MODEL" --output_path "${LATEST_DIR}/latency_non_causal.json"
        
        # Causal Latency
        python performance_benchmark.py --model_size $SIZE --sequence_lengths $SEQ_LENGTHS --batch_size 1 --model_types "$MODEL" --use_causal_router --output_path "${LATEST_DIR}/latency_causal.json"
        
        echo "================================================================"
        echo " COMPLETED EXPERIMENT: Model=$MODEL, Size=$SIZE"
        echo "================================================================"
    done
done

echo "================================================================"
echo " ALL SCALING EXPERIMENTS FINISHED SUCCESSFULLY"
echo "================================================================"

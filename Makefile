# Makefile

# --- Variables ---
# (No variables needed here anymore—handled in scripts)

# --- Targets ---

.PHONY: setup help run train eval infer flush-gpu clean

setup:
    @./scripts/setup.sh

help:
    @echo "Available targets:"
    @echo "  setup       Sets up the virtual environment, installs dependencies, and handles logins."
    @echo "  train       Runs the training script with full stack traces (pass args like 'data=sft_mix')."
    @echo "  eval        Runs the evaluation script with full stack traces (pass args like '--model_path=outputs/my_model')."
    @echo "  infer       Runs the inference/chat script with full stack traces (pass args like '--prompt=\"Hello\"')."
    @echo "  flush-gpu   Flushes GPU memory by killing processes using NVIDIA devices (requires sudo)."
    @echo "  run         Legacy: Runs main training script with full stack traces (use 'train' instead)."
    @echo "  clean       Removes virtual environment and __pycache__ folders."
    @echo "  help        Displays this help message."
    @echo ""
    @echo "Environment notes: All Python runs enable HYDRA_FULL_ERROR=1 and CUDA_LAUNCH_BLOCKING=1 for full stack traces."
    @echo "Tip: Pass arguments to targets like 'make train data=sft_mix'."

# Capture all arguments passed to targets (excluding the target name itself)
ARGS := $(filter-out $@,$(MAKECMDGOALS))
# Prevent Make from trying to interpret ARGS as targets
$(eval $(ARGS):;@:)

run:
    @./scripts/run.sh $(ARGS)

train:
    @./scripts/train.sh $(ARGS)

eval:
    @./scripts/eval.sh $(ARGS)

infer:
    @./scripts/infer.sh $(ARGS)

flush-gpu:
    @./scripts/flush-gpu.sh

clean:
    @./scripts/clean.sh
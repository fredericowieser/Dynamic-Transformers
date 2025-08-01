import logging
import os
import sys
import glob

import torch
from transformers import AutoTokenizer, AutoConfig

# Assume src is directly importable because we added its parent directory to sys.path in evaluate.py
try:
    from src.models.d_llama_config import DynamicLlamaConfig
    from src.models.d_llama_causal_lm import DynamicLlamaForCausalLM
except ImportError:
    logging.error(
        "Could not import DynamicLlamaConfig or DynamicLlamaForCausalLM from src.models.d_llama_causal_lm."
        "Please ensure your project structure is correct and 'src' is in your PYTHONPATH."
    )
    sys.exit(1)

log = logging.getLogger(__name__)


def load_weights(model, model_dir, device):
    """
    Load model weights from:
      - one or more .safetensors shards
      - or a single pytorch_model.safetensors
      - or a single pytorch_model.bin
    This utility is taken from your `inference.py` script.
    """
    state_dict = {}
    
    # Try safetensors first
    try:
        from safetensors.torch import load_file as safe_load
    except ImportError:
        safe_load = None
        log.warning("safetensors not installed. Will only try to load .bin files.")

    if safe_load:
        # sharded safetensors: model-*.safetensors
        shard_paths = sorted(glob.glob(os.path.join(model_dir, "model-*.safetensors")))
        if shard_paths:
            log.info(f"Found {len(shard_paths)} safetensors shards. Loading...")
            for shard in shard_paths:
                sd_part = safe_load(shard, device=device)
                state_dict.update(sd_part)
            model.load_state_dict(state_dict, strict=False)
            return

        # single-file pytorch_model.safetensors
        single_safetensors = os.path.join(model_dir, "pytorch_model.safetensors")
        if os.path.isfile(single_safetensors):
            log.info(f"Found single safetensors file: {single_safetensors}. Loading...")
            sd = safe_load(single_safetensors, device=device)
            model.load_state_dict(sd, strict=False)
            return

    # single-file pytorch_model.bin
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.isfile(bin_path):
        log.info(f"Found single PyTorch bin file: {bin_path}. Loading...")
        sd = torch.load(bin_path, map_location=device)
        model.load_state_dict(sd, strict=False)
        return

    raise FileNotFoundError(
        f"No weights found in {model_dir}. "
        "Expected sharded '*.safetensors' or 'pytorch_model.safetensors' or 'pytorch_model.bin'."
    )


def load_dynamic_llama_model_and_tokenizer(
    model_path: str,
    device: str,
    is_instruct: bool = False,
    dynamic_k: float = None,
    ce_bias: float = None,
    gate_warmup_iters: int = None,
    token_wise: bool = None,
):
    """
    Loads DynamicLlamaForCausalLM and its tokenizer, specifically handling its configuration.
    It overrides DynamicLlama-specific config parameters if provided.
    """
    # 1. Load DynamicLlamaConfig first to get default values from config.json
    config = DynamicLlamaConfig.from_pretrained(model_path)
    log.info(f"Loaded base DynamicLlamaConfig: {config}")

    # Apply standard fixes to config (from your original load_model_and_tokenizer)
    if isinstance(config.pad_token_id, (list, tuple)):
        config.pad_token_id = int(config.pad_token_id[0])
        log.info(f"Fixed pad_token_id from list/tuple to {config.pad_token_id}")

    if (
        hasattr(config, "rope_scaling")
        and isinstance(config.rope_scaling, dict)
        and "type" not in config.rope_scaling
    ):
        config.rope_scaling = None
        log.info("Fixed rope_scaling due to missing 'type' key.")

    # 2. Override DynamicLlama specific parameters in the config based on arguments
    # These values are crucial for DynamicLlamaForCausalLM's __init__
    if dynamic_k is not None:
        config.dynamic_k = dynamic_k
        log.info(f"Overriding config.dynamic_k to {dynamic_k}")
    # Else, DynamicLlamaForCausalLM expects it to be set (raises ValueError if None)
    elif not hasattr(config, "dynamic_k") or config.dynamic_k is None:
        raise ValueError(
            "dynamic_k must be provided via config.json or --dynamic_k argument."
        )

    if ce_bias is not None:
        config.ce_bias = ce_bias
        log.info(f"Overriding config.ce_bias to {ce_bias}")
    elif not hasattr(config, "ce_bias") or config.ce_bias is None:
        raise ValueError(
            "ce_bias must be provided via config.json or --ce_bias argument."
        )

    if gate_warmup_iters is not None:
        config.gate_warmup_iters = gate_warmup_iters
        log.info(f"Overriding config.gate_warmup_iters to {gate_warmup_iters}")
    elif not hasattr(config, "gate_warmup_iters") or config.gate_warmup_iters is None:
        # Default as per your DynamicLlamaDecoderLayer default (0) if not set.
        config.gate_warmup_iters = 0 
        log.info("gate_warmup_iters not explicitly set, defaulting to 0.")

    if token_wise is not None:
        config.token_wise = token_wise
        log.info(f"Overriding config.token_wise to {token_wise}")
    elif not hasattr(config, "token_wise") or config.token_wise is None:
        # Default as per your DynamicLlamaDecoderLayer default (True) if not set.
        config.token_wise = True
        log.info("token_wise not explicitly set, defaulting to True.")

    # Ensure prior_loss_weight is present, though it's used by trainer, not model init logic
    if not hasattr(config, "prior_loss_weight") or config.prior_loss_weight is None:
        config.prior_loss_weight = 0.0 # Safe default
        log.info("prior_loss_weight not explicitly set, defaulting to 0.0.")


    # 3. Instantiate the DynamicLlamaForCausalLM with the prepared config
    # We instantiate directly and then load weights, bypassing AutoModelForCausalLM.from_pretrained
    # to ensure our custom config is fully respected.
    model = DynamicLlamaForCausalLM(config=config)
    
    # Apply device map if cuda
    if device == "cuda":
        # Using device_map="auto" during instantiation is generally better
        # if the model directly supported it for custom model, but since we init manually
        # and then load weights, we'll need to handle device manually.
        # This will put the entire model on CUDA.
        # Accelerate is good for device_map="auto" but that's for HF models directly
        # For custom, you often just move it to device.
        model.to(device) 
    else:
        model.to(device)

    # 4. Load weights using the utility function from your inference script
    log.info(f"Attempting to load weights into model from {model_path}...")
    load_weights(model, model_path, device)
    log.info("Model weights loaded.")

    model.eval()

    # 5. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if isinstance(tokenizer.pad_token_id, (list, tuple)):
        tokenizer.pad_token_id = int(tokenizer.pad_token_id[0])
    if tokenizer.pad_token_id is None:
        # Fallback for models without explicit pad_token_id
        # Use EOS token ID, a common practice
        tokenizer.pad_token_id = tokenizer.eos_token_id
        log.info(f"tokenizer.pad_token_id not found, using eos_token_id: {tokenizer.eos_token_id}")

    model.config.pad_token_id = tokenizer.pad_token_id
    log.info(f"Final model.config.pad_token_id set to {model.config.pad_token_id}")

    return model, tokenizer
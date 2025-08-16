from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

class DynamicQwenConfig(Qwen2Config):
    """
    Extends QwenConfig with dynamic gating parameters:
      - dynamic_k:           float in (0,1], fraction of tokens to always compute
      - ce_bias:             float ≥ 0, bias for the “expected change” criterion
      - gate_warmup_iters:   int ≥ 0, number of initial steps to warm up CU gating
    """
    model_type = "dynamic_qwen"

    def __init__(self, **kwargs):
        # pop our new args, leave rest to base QwenConfig
        dynamic_k = kwargs.pop("dynamic_k", None)
        ce_bias = kwargs.pop("ce_bias", None)
        gate_warmup_iters = kwargs.pop("gate_warmup_iters", None)

        super().__init__(**kwargs)

        # required dynamic-gating parameters
        # --- START OF CHANGE ---
        # Removed strict check here. These will be set to None if not explicitly passed
        # during a standard initialization (e.g., from_pretrained).
        # The actual validation will now happen in the trainer/model __init__.
        # --- END OF CHANGE ---

        self.dynamic_k = dynamic_k
        self.ce_bias = ce_bias
        self.gate_warmup_iters = gate_warmup_iters
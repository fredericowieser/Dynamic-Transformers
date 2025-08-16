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
        if dynamic_k is None or ce_bias is None or gate_warmup_iters is None:
            missing = [n for n,v in [
                ("dynamic_k", dynamic_k),
                ("ce_bias", ce_bias),
                ("gate_warmup_iters", gate_warmup_iters),
            ] if v is None]
            raise ValueError(f"Missing DynamicQwenConfig args: {missing}")

        self.dynamic_k = float(dynamic_k)
        self.ce_bias = float(ce_bias)
        self.gate_warmup_iters = int(gate_warmup_iters)
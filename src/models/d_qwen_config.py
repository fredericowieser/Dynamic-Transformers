# src/models/d_qwen_config.py

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

class DynamicQwenConfig(Qwen2Config):
    """
    Extends Qwen2Config with parameters for the dynamic gating mechanism
    (VPR-based routing) and the new hierarchical layer structure.
    """
    model_type = "dynamic_qwen"

    def __init__(self, **kwargs):
        # Pop our new args, leaving the rest to the base Qwen2Config
        # New dynamic gating parameters for VPRRouter (including initial values for learnable parameters)
        self.capacity_gamma = kwargs.pop("capacity_gamma", 1.0) # float in (0,1], fraction of tokens to compute (like MoD's k)
        self.beta_ce_init = kwargs.pop("beta_ce_init", 1.0) # Initial beta for CE sigmoid
        self.beta_cu_init = kwargs.pop("beta_cu_init", 1.0) # Initial beta for CU sigmoid
        self.cu_detection_multiplier_init = kwargs.pop("cu_detection_multiplier_init", 1.0) # Initial multiplier for CU norm
        self.ce_criterion_offset_init = kwargs.pop("ce_criterion_offset_init", 0.0) # Offset for CE criterion (replaces old ce_bias concept)

        self.token_wise_gating = kwargs.pop("token_wise_gating", True) # Whether to apply gating per-token or per-batch
        self.moving_average_window_size = kwargs.pop("moving_average_window_size", 100) # Window size for D_st moving average in CU

        # New Prior FFN parameter
        self.prior_ffn_intermediate_size_factor = kwargs.pop("prior_ffn_intermediate_size_factor", 2.0)

        # New training control parameter: allows freezing main transformer blocks
        self.freeze_main_transformer_blocks = kwargs.pop("freeze_main_transformer_blocks", False)

        super().__init__(**kwargs)

        # Assign popped parameters to self after super().__init__ call
        # This ensures they are properly set on the config object for access later
        self.capacity_gamma = self.capacity_gamma
        self.beta_ce_init = self.beta_ce_init
        self.beta_cu_init = self.beta_cu_init
        self.cu_detection_multiplier_init = self.cu_detection_multiplier_init
        self.ce_criterion_offset_init = self.ce_criterion_offset_init
        self.token_wise_gating = self.token_wise_gating
        self.moving_average_window_size = self.moving_average_window_size
        self.prior_ffn_intermediate_size_factor = self.prior_ffn_intermediate_size_factor
        self.freeze_main_transformer_blocks = self.freeze_main_transformer_blocks

        # Removed all LoRA specific parameters as per the new requirements.
        # Removed old dynamic_k, ce_bias, gate_warmup_iters.
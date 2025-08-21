from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


class DynamicQwenConfig(Qwen2Config):
    """
    Extends Qwen2Config with parameters for dynamic computation, supporting
    both VPR (Variational Predictive Routing) and MoD (Mixture-of-Depths)
    architectures.
    """

    model_type = "dynamic_qwen"

    def __init__(self, **kwargs):
        # --- Architecture Control ---
        self.dynamic_architecture = kwargs.pop("dynamic_architecture", "vpr")

        # --- Shared Dynamic Compute Params ---
        self.capacity_gamma = kwargs.pop("capacity_gamma", 1.0)

        # --- VPR Specific Params ---
        self.beta_ce_init = kwargs.pop("beta_ce_init", 1.0)
        self.beta_cu_init = kwargs.pop("beta_cu_init", 1.0)
        self.cu_detection_multiplier_init = kwargs.pop("cu_detection_multiplier_init", 1.0)
        self.ce_criterion_offset_init = kwargs.pop("ce_criterion_offset_init", 0.0)
        self.token_wise_gating = kwargs.pop("token_wise_gating", True)
        self.moving_average_window_size = kwargs.pop("moving_average_window_size", 100)
        self.prior_ffn_intermediate_size_factor = kwargs.pop(
            "prior_ffn_intermediate_size_factor", 2.0
        )

        # --- General Training Control ---
        self.freeze_main_transformer_blocks = kwargs.pop(
            "freeze_main_transformer_blocks", False
        )

        super().__init__(**kwargs)
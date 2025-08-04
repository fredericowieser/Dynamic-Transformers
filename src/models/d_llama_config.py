from transformers.models.llama.configuration_llama import LlamaConfig


class DynamicLlamaConfig(LlamaConfig):
    model_type = "dynamic_llama"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialised to None must be set explicitly (Hyrda)
        self.dynamic_k = kwargs.pop("dynamic_k", None)
        self.ce_bias = kwargs.pop("ce_bias", None)
        self.gate_warmup_iters = kwargs.pop("gate_warmup_iters", None)
        self.token_wise = kwargs.pop("token_wise", None)
        self.prior_loss_weight = kwargs.pop("prior_loss_weight", None)

        # LoRA and prior FFN initialization parameters
        self.enable_lora_main_path = kwargs.pop("enable_lora_main_path", False)
        self.enable_lora_prior_ffn = kwargs.pop("enable_lora_prior_ffn", False)
        self.lora_r = kwargs.pop("lora_r", 8)
        self.lora_alpha = kwargs.pop("lora_alpha", 16)
        self.lora_dropout = kwargs.pop("lora_dropout", 0.05)
        self.lora_bias = kwargs.pop("lora_bias", "none")
        self.lora_target_modules_main = kwargs.pop(
            "lora_target_modules_main",
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        self.lora_target_modules_prior_ffn = kwargs.pop(
            "lora_target_modules_prior_ffn", ["w1", "w2", "w3"]
        )
        self.init_prior_from_mlp = kwargs.pop("init_prior_from_mlp", False)

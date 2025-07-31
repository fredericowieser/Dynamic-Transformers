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

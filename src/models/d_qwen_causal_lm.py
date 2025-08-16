import logging
from transformers import Qwen2ForCausalLM
from transformers import AutoConfig
from src.models.d_qwen_config import DynamicQwenConfig
from src.models.d_qwen_layers import patch_qwen_layers
# from src.utils.qwen_config_utils import fix_pad_token_id, fix_rope_scaling

logger = logging.getLogger(__name__)

class DynamicQwenForCausalLM(Qwen2ForCausalLM):
    """
    QwenForCausalLM with dynamic, token-wise gating.
    We patch each decoder layer to be DynamicQwenDecoderLayer,
    and inject gating args (dynamic_k, ce_bias, gate_warmup_iters).
    """
    config_class = DynamicQwenConfig

    def __init__(self, config: DynamicQwenConfig):
        super().__init__(config)
        # Apply any config fixes (e.g. rotary scaling, pad token)
        # self.config = fix_pad_token_id(fix_rope_scaling(self.config))

        # Patch the decoder layers for dynamic gating
        patch_qwen_layers(self)

        # Expose gating params on the model
        self._dynamic_k = config.dynamic_k
        self._ce_bias = config.ce_bias
        self._gate_warmup_iters = config.gate_warmup_iters

    @property
    def dynamic_k(self):
        return self._dynamic_k

    @dynamic_k.setter
    def dynamic_k(self, v: float):
        self._dynamic_k = v
        logger.info(f"Set dynamic_k = {v}")

    @property
    def ce_bias(self):
        return self._ce_bias

    @ce_bias.setter
    def ce_bias(self, v: float):
        self._ce_bias = v
        logger.info(f"Set ce_bias = {v}")

    @property
    def gate_warmup_iters(self):
        return self._gate_warmup_iters

    @gate_warmup_iters.setter
    def gate_warmup_iters(self, v: int):
        self._gate_warmup_iters = v
        logger.info(f"Set gate_warmup_iters = {v}")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        labels=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        **kwargs,
    ):
        # Inject gating arguments into each layer
        kwargs["dynamic_k"] = self.dynamic_k
        kwargs["ce_bias"] = self.ce_bias
        kwargs["gate_warmup_iters"] = self.gate_warmup_iters

        # Optionally pass current step as well (if tracking in Trainer)
        if "current_iter" not in kwargs:
            # fallback: use past_key_values length or zero
            kwargs["current_iter"] = (
                getattr(self, "cur_iteration", 0)
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Load dynamic config first
        config = DynamicQwenConfig.from_pretrained(pretrained_model_name_or_path)
        # Load base model with that config
        model = super().from_pretrained(
            pretrained_model_name_or_path, config=config, *model_args, **kwargs
        )
        # Re-apply any post-load fixes
        # model.config = fix_pad_token_id(fix_rope_scaling(model.config))
        # Re-patch layers
        patch_qwen_layers(model)
        return model
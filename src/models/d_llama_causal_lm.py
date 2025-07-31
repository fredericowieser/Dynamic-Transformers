import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.utils.llamam_config_utils import fix_pad_token_id, fix_rope_scaling

from .d_llama_config import DynamicLlamaConfig
from .d_llama_layers import DynamicLlamaDecoderLayer


class DynamicLlamaForCausalLM(LlamaForCausalLM):
    config_class = DynamicLlamaConfig

    def __init__(self, config: DynamicLlamaConfig):
        config = fix_rope_scaling(config)
        config = fix_pad_token_id(config)

        required_params = ["dynamic_k", "ce_bias", "gate_warmup_iters", "token_wise"]
        for param in required_params:
            if getattr(config, param) is None:
                raise ValueError(f"{param} must be set in the config.")

        super().__init__(config)

        self._modify_model_architecture()

        self._log_gates = False
        self._last_gate_means = None

    def _modify_model_architecture(self):
        new_layers = torch.nn.ModuleList()
        for i, layer in enumerate(self.model.layers):
            custom_layer = DynamicLlamaDecoderLayer(self.config, i)
            custom_layer.load_state_dict(layer.state_dict(), strict=False)
            new_layers.append(custom_layer)
        self.model.layers = new_layers

    def set_dynamic_k(self, k: float):
        self.dynamic_k = float(k)

    def set_gate_warmup_iters(self, iters: int):
        self.gate_warmup_iters = iters

    def set_ce_bias(self, bias: float):
        self.ce_bias = bias

    def _prepare_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs
    ) -> dict:
        """
        Prepares inputs: Generates position_ids if missing, creates 4D causal + padding mask.
        Robustness: Validates shapes, handles edge cases (e.g., no mask, empty seq).
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        dtype = input_ids.dtype  # Use input dtype for masks

        # Generate position_ids if not provided
        if position_ids is None:
            log.debug("Generating default position_ids")
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

        # Validate shapes
        if position_ids.shape != (batch_size, seq_len):
            raise ValueError(f"position_ids shape {position_ids.shape} does not match input_ids {input_ids.shape}")

        # Prepare 4D attention mask
        causal_mask_base = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device)
        causal_mask_base = torch.triu(causal_mask_base, diagonal=1)

        if attention_mask is not None:
            if attention_mask.dim() != 2 or attention_mask.shape != (batch_size, seq_len):
                raise ValueError(f"attention_mask must be (batch_size, seq_len), got {attention_mask.shape}")
            expanded_padding_mask = (1 - attention_mask).bool().to(dtype).masked_fill_(
                (1 - attention_mask).bool(), torch.finfo(dtype).min
            ).unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len)
            attention_mask_4d = causal_mask_base.unsqueeze(0).unsqueeze(0) + expanded_padding_mask
            attention_mask_4d = attention_mask_4d.expand(batch_size, 1, seq_len, seq_len)
        else:
            log.debug("No attention_mask provided; using pure causal mask")
            attention_mask_4d = causal_mask_base.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask_4d,
            "position_ids": position_ids,
            **kwargs  # Pass through other kwargs
        }

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        # first get the normal prepared inputs
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            **kwargs,
        )
        # now inject your dynamic params
        model_inputs["dynamic_k"] = self.dynamic_k
        model_inputs["gate_warmup_iters"] = getattr(self, "gate_warmup_iters", 0)
        model_inputs["ce_bias"] = getattr(self, "ce_bias", 0.0)
        return model_inputs

    def enable_gate_logging(self, flag: bool = True):
        self._log_gates = flag
        self._last_gate_means = None

    def get_last_gate_means(self):
        """
        Returns a list of per-layer mean gate values from the *most
        recent* forward / generate call - or None if logging disabled.
        """
        return self._last_gate_means

    def forward(self, *args, **kwargs):
        # Pop dynamic params (fall back to config; raise if unset)
        dynamic_k = kwargs.pop("dynamic_k", self.config.dynamic_k)
        if dynamic_k is None:
            raise ValueError("dynamic_k must be provided via config or kwargs.")
        
        gate_warmup_iters = kwargs.pop("gate_warmup_iters", self.config.gate_warmup_iters)
        if gate_warmup_iters is None:
            raise ValueError("gate_warmup_iters must be provided via config or kwargs.")
        
        ce_bias = kwargs.pop("ce_bias", self.config.ce_bias)
        if ce_bias is None:
            raise ValueError("ce_bias must be provided via config or kwargs.")
        
        current_iter = kwargs.pop("current_iter", 0)
        return_metrics = kwargs.pop("return_metrics", self.training)  # Default to training mode

        # Prepare inputs (handles position_ids, mask)
        prepared = self._prepare_inputs(input_ids, **kwargs)
        hidden_states = self.model.embed_tokens(prepared["input_ids"])
        attention_mask = prepared["attention_mask"]
        position_ids = prepared["position_ids"]

        # Temporarily inject params into config
        original_dynamic_k = self.config.dynamic_k
        original_gate_warmup_iters = self.config.gate_warmup_iters
        original_ce_bias = self.config.ce_bias
        self.config.dynamic_k = dynamic_k
        self.config.gate_warmup_iters = gate_warmup_iters
        self.config.ce_bias = ce_bias

        if self._log_gates:
            self._gate_means_tmp = []
            def _collect(_, __, outputs):
                gate_vec = outputs[-1]
                self._gate_means_tmp.append(gate_vec.mean().item())
                return outputs
            hooks = [l.register_forward_hook(_collect) for l in self.model.layers]
        else:
            hooks = []

        try:
            prior_losses, gate_vecs, ce_proportions, cu_proportions = [], [], [], []

            for layer in self.model.layers:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    current_iter=current_iter,
                    gate_warmup_iters=gate_warmup_iters,
                    dynamic_k=dynamic_k,
                    ce_bias=ce_bias,
                )
                hidden_states = layer_outputs[0]
                if return_metrics:
                    ce_proportions.append(layer_outputs[-4])
                    cu_proportions.append(layer_outputs[-3])
                    prior_losses.append(layer_outputs[-2])
                    gate_vecs.append(layer_outputs[-1])

            hidden_states = self.model.norm(hidden_states)
            logits = self.lm_head(hidden_states)

            if return_metrics:
                avg_prior_loss = torch.stack(prior_losses).mean() if prior_losses else torch.tensor(0.0, device=logits.device)
                return (logits, avg_prior_loss, gate_vecs, ce_proportions, cu_proportions)
            else:
                # Standard HF output for inference
                return CausalLMOutputWithPast(logits=logits)
        except Exception as e:
            log.error(f"Forward pass failed: {str(e)}")
            raise
        finally:
            for h in hooks:
                h.remove()
            if self._log_gates:
                self._last_gate_means = self._gate_means_tmp
            self.config.dynamic_k = original_dynamic_k
            self.config.gate_warmup_iters = original_gate_warmup_iters
            self.config.ce_bias = original_ce_bias

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = DynamicLlamaConfig.from_pretrained(pretrained_model_name_or_path)
        model = super().from_pretrained(
            pretrained_model_name_or_path, config=config, *model_args, **kwargs
        )
        if not isinstance(model, cls):
            model = cls(config)
            model.load_state_dict(
                torch.load(pretrained_model_name_or_path + "/pytorch_model.bin"),
                strict=False,
            )
        return model

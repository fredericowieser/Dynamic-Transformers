# src/models/d_qwen_causal_lm.py

import logging
import torch
from transformers import Qwen2ForCausalLM
# Import the specific CausalLMOutput class for Qwen2, or a generic one
from transformers.modeling_outputs import CausalLMOutputWithPast # Renamed Qwen2CausalLMOutputWithPast for clarity
from src.models.d_qwen_config import DynamicQwenConfig
from src.models.d_qwen_layers import patch_qwen_layers

logger = logging.getLogger(__name__)

class DynamicQwenForCausalLM(Qwen2ForCausalLM):
    config_class = DynamicQwenConfig

    def __init__(self, config: DynamicQwenConfig):
        super().__init__(config)
        # Required parameters check (now done in DynamicQwenConfig.__init__ with flexible loading,
        # but the model instance itself needs these as attributes)
        # This aligns with Llama version: check happens in DynamicLlamaForCausalLM __init__
        required_params = ["dynamic_k", "ce_bias", "gate_warmup_iters"]
        for param in required_params:
            if getattr(config, param) is None:
                raise ValueError(f"{param} must be set in the config.")

        self._dynamic_k = config.dynamic_k
        self._ce_bias = config.ce_bias
        self._gate_warmup_iters = config.gate_warmup_iters
        
        # LoRA parameters are also stored as model attributes for easy access, similar to Llama
        self.enable_lora_main_path = getattr(config, "enable_lora_main_path", False)
        self.enable_lora_prior_ffn = getattr(config, "enable_lora_prior_ffn", False)
        self.init_prior_from_mlp = getattr(config, "init_prior_from_mlp", False)


        # Now patch the layers, which replaces the default Qwen2DecoderLayers with Dynamic ones
        # and re-initializes them with the original weights.
        patch_qwen_layers(self)

        self._log_gates = False
        self._last_gate_means = None

    # Properties (no change)
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

    def enable_gate_logging(self, flag: bool = True):
        self._log_gates = flag
        self._last_gate_means = None

    def get_last_gate_means(self):
        return self._last_gate_means
    
    # New helper method to prepare inputs, similar to Llama
    def _prepare_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> dict:
        """
        Generates position_ids if missing, creates 4D causal + padding mask.
        Robustness: Validates shapes, handles edge cases (e.g., no mask, empty seq).
        (Copied largely from DynamicLlamaForCausalLM)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        mask_dtype = torch.float32  # Use float for masks to support -inf

        # Generate position_ids if not provided
        if position_ids is None:
            logger.debug(f"Generating default position_ids for seq_len={seq_len}")
            position_ids = (
                torch.arange(seq_len, dtype=torch.long, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        # Validate shapes
        if position_ids.shape != (batch_size, seq_len):
            raise ValueError(
                f"position_ids shape {position_ids.shape} does not match input_ids {input_ids.shape}"
            )

        # Prepare 4D attention mask
        # Qwen2 attention mask creation might be slightly different.
        # This part assumes a standard causal mask approach.
        causal_mask_base = torch.full(
            (seq_len, seq_len),
            torch.finfo(mask_dtype).min, # Use float.min for masked out values
            dtype=mask_dtype,
            device=device,
        )
        causal_mask_base = torch.triu(causal_mask_base, diagonal=1)

        if attention_mask is not None:
            if attention_mask.dim() != 2 or attention_mask.shape != (
                batch_size,
                seq_len,
            ):
                raise ValueError(
                    f"attention_mask must be (batch_size, seq_len), got {attention_mask.shape}"
                )
            # Expand padding mask to 4D
            expanded_padding_mask = (
                (1 - attention_mask)
                .bool()
                .to(mask_dtype)
                .masked_fill_((1 - attention_mask).bool(), torch.finfo(mask_dtype).min)
                .unsqueeze(1) # For num_heads (broadcastable)
                .unsqueeze(1) # For query_length (broadcastable)
            )  # (batch_size, 1, 1, seq_len)
            attention_mask_4d = (
                causal_mask_base.unsqueeze(0).unsqueeze(0) + expanded_padding_mask
            )
            attention_mask_4d = attention_mask_4d.expand(
                batch_size, 1, seq_len, seq_len
            ) # Expand to (B, 1, Q, K)
        else:
            logger.debug("No attention_mask provided; using pure causal mask")
            attention_mask_4d = (
                causal_mask_base.unsqueeze(0)
                .unsqueeze(0)
                .expand(batch_size, 1, seq_len, seq_len)
            ) # Expand to (B, 1, Q, K)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask_4d,
            "position_ids": position_ids,
            **kwargs,
        }

    # Override prepare_inputs_for_generation to inject dynamic params for generation
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            **kwargs,
        )
        model_inputs["dynamic_k"] = self.dynamic_k
        model_inputs["gate_warmup_iters"] = getattr(self, "gate_warmup_iters", 0)
        model_inputs["ce_bias"] = getattr(self, "ce_bias", 0.0)
        # Note: current_iter is typically managed by a trainer or incremented during generation loop
        # For simplicity, we might default it to 0 or derive from past_key_values length for generation.
        model_inputs["current_iter"] = kwargs.get("current_iter", 0) # Generation likely starts at iter 0 or length of previous sequence
        return model_inputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        labels=None, # Labels for CE loss, handled by trainer
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False, # We get hidden states from layers, but don't explicitly pass this down for Qwen2Model
        return_dict=True,
        **kwargs, # Accept extra kwargs, for current_iter and return_metrics
    ):
        if input_ids is None:
            raise ValueError("input_ids must be provided.")

        # Pop dynamic params (fall back to model attributes which are set from config)
        dynamic_k = kwargs.pop("dynamic_k", self.dynamic_k)
        if dynamic_k is None:
            raise ValueError("dynamic_k must be provided via config or kwargs.")

        gate_warmup_iters = kwargs.pop(
            "gate_warmup_iters", self.gate_warmup_iters
        )
        if gate_warmup_iters is None:
            raise ValueError("gate_warmup_iters must be provided via config or kwargs.")

        ce_bias = kwargs.pop("ce_bias", self.ce_bias)
        if ce_bias is None:
            raise ValueError("ce_bias must be provided via config or kwargs.")

        current_iter = kwargs.pop("current_iter", 0)
        return_metrics = kwargs.pop(
            "return_metrics", self.training
        )  # Default to training mode

        # Prepare inputs (handles position_ids, 4D attention mask)
        prepared = self._prepare_inputs(input_ids, attention_mask, position_ids)
        hidden_states = self.model.embed_tokens(prepared["input_ids"])
        attention_mask = prepared["attention_mask"] # This is now the 4D mask
        position_ids = prepared["position_ids"]

        # Temporarily inject params into config for layers to pick up (if they access config)
        # Store original values to restore them later.
        original_dynamic_k = self.config.dynamic_k
        original_gate_warmup_iters = self.config.gate_warmup_iters
        original_ce_bias = self.config.ce_bias
        self.config.dynamic_k = dynamic_k
        self.config.gate_warmup_iters = gate_warmup_iters
        self.config.ce_bias = ce_bias

        if self._log_gates:
            self._gate_means_tmp = []
            def _collect(_, __, outputs):
                # outputs[-1] is gate_vec_for_stats from DynamicQwenDecoderLayer
                gate_vec = outputs[-1]
                self._gate_means_tmp.append(gate_vec.mean().item())
                return outputs
            hooks = [l.register_forward_hook(_collect) for l in self.model.layers]
        else:
            hooks = []

        try:
            prior_losses, gate_vecs, ce_proportions, cu_proportions = [], [], [], []
            all_self_attns = [] if output_attentions else None
            all_past_key_values = [] if use_cache else None

            # Manual loop through layers, similar to DynamicLlamaForCausalLM
            for layer_idx, layer in enumerate(self.model.layers):
                # Pass all necessary args and the dynamic params to the layer
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values[layer_idx] if past_key_values else None,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    # Pass dynamic params
                    current_iter=current_iter,
                    dynamic_k=dynamic_k,
                    ce_bias=ce_bias,
                    gate_warmup_iters=gate_warmup_iters,
                )
                hidden_states = layer_outputs[0] # Updated hidden states

                if output_attentions:
                    all_self_attns.append(layer_outputs[1]) # Qwen2DecoderLayer output is (hidden_states, attns, past_kv) or (hidden_states, attns)
                if use_cache:
                    # If use_cache, past_key_values will be at index 2 if output_attentions is True, else index 1
                    past_kv_index = 2 if output_attentions else 1
                    current_past_key_value = layer_outputs[past_kv_index]
                    if all_past_key_values is None:
                        all_past_key_values = (current_past_key_value,)
                    else:
                        all_past_key_values = all_past_key_values + (current_past_key_value,)

                if return_metrics:
                    # DynamicQwenDecoderLayer returns: (hidden_states_final, ..., avg_ce_proportion, avg_cu_proportion, prior_loss, gate_vec_for_stats)
                    # The indices below are relative to the *end* of the tuple for custom metrics
                    ce_proportions.append(layer_outputs[-4])
                    cu_proportions.append(layer_outputs[-3])
                    prior_losses.append(layer_outputs[-2])
                    gate_vecs.append(layer_outputs[-1])

            hidden_states = self.model.norm(hidden_states) # Final normalization (Qwen's self.model.norm)
            logits = self.lm_head(hidden_states) # Language model head

            if return_metrics:
                avg_prior_loss = (
                    torch.stack(prior_losses).mean()
                    if prior_losses
                    else torch.tensor(0.0, device=logits.device)
                )
                # Return tuple as expected by DynamicQwenTrainer's _calculate_loss
                return (
                    logits,
                    avg_prior_loss,
                    gate_vecs, # List of gate_vec_for_stats (B,T) or (B,) per layer
                    ce_proportions, # List of scalar means per layer
                    cu_proportions, # List of scalar means per layer
                )
            else:
                # Standard Hugging Face CausalLMOutputWithPast for inference
                # This needs proper handling of past_key_values and attentions
                return CausalLMOutputWithPast(
                    logits=logits,
                    past_key_values=all_past_key_values,
                    attentions=all_self_attns,
                    # No output_hidden_states, because Qwen2ForCausalLM doesn't expose this as direct output in its top-level forward
                )
        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            raise
        finally:
            # Clean up hooks and restore config values
            for h in hooks:
                h.remove()
            if self._log_gates:
                self._last_gate_means = self._gate_means_tmp
            self.config.dynamic_k = original_dynamic_k
            self.config.gate_warmup_iters = original_gate_warmup_iters
            self.config.ce_bias = original_ce_bias

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Load the base DynamicQwenConfig first
        config = DynamicQwenConfig.from_pretrained(pretrained_model_name_or_path)

        # Apply kwargs overrides to the config for dynamic params and LoRA, similar to Llama
        config.dynamic_k = kwargs.pop("dynamic_k", getattr(config, "dynamic_k", None))
        config.ce_bias = kwargs.pop("ce_bias", getattr(config, "ce_bias", None))
        config.gate_warmup_iters = kwargs.pop("gate_warmup_iters", getattr(config, "gate_warmup_iters", None))

        config.enable_lora_main_path = kwargs.pop("enable_lora_main_path", getattr(config, "enable_lora_main_path", False))
        config.enable_lora_prior_ffn = kwargs.pop("enable_lora_prior_ffn", getattr(config, "enable_lora_prior_ffn", False))
        config.lora_r = kwargs.pop("lora_r", getattr(config, "lora_r", 8))
        config.lora_alpha = kwargs.pop("lora_alpha", getattr(config, "lora_alpha", 16))
        config.lora_dropout = kwargs.pop("lora_dropout", getattr(config, "lora_dropout", 0.05))
        config.lora_bias = kwargs.pop("lora_bias", getattr(config, "lora_bias", "none"))
        config.lora_target_modules_main = kwargs.pop("lora_target_modules_main", getattr(config, "lora_target_modules_main", []))
        config.lora_target_modules_prior_ffn = kwargs.pop("lora_target_modules_prior_ffn", getattr(config, "lora_target_modules_prior_ffn", []))
        config.init_prior_from_mlp = kwargs.pop("init_prior_from_mlp", getattr(config, "init_prior_from_mlp", False))
        
        # Load the *base* Qwen2ForCausalLM model with our extended config.
        # This will load the standard Qwen2DecoderLayers.
        model = super().from_pretrained(
            pretrained_model_name_or_path, config=config, *model_args, **kwargs
        )

        # Then, patch the layers. This will iterate through the loaded
        # Qwen2DecoderLayers, extract their state_dicts, and replace them
        # with our DynamicQwenDecoderLayer instances initialized with those weights.
        patched_model = patch_qwen_layers(model) # patch_qwen_layers returns the modified model.

        # Ensure dynamic parameters are set on the instance (as @property setters are used)
        # These are now attributes of the patched_model instance
        patched_model._dynamic_k = config.dynamic_k
        patched_model._ce_bias = config.ce_bias
        patched_model._gate_warmup_iters = config.gate_warmup_iters
        patched_model.enable_lora_main_path = config.enable_lora_main_path
        patched_model.enable_lora_prior_ffn = config.enable_lora_prior_ffn
        patched_model.init_prior_from_mlp = config.init_prior_from_mlp

        return patched_model
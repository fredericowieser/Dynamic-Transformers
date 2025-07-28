# src/models/dynamic_llama_causal.py
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from .dynamic_llama import (
    DynamicLlamaBlockWiseDecoderLayer,
    DynamicLlamaTokenWiseDecoderLayer,
)

class DynamicLlamaForCausalLM(LlamaForCausalLM):
    """
    Llama-3 causal-LM whose decoder layers are replaced by the
    Dynamic* layers that contain the extra prior-FFN + gate.

    Extra features:
      • .dynamic_k (float) – gate hyper-parameter
      • .set_dynamic_k(k)
      • .enable_gate_logging(bool)  – store gate means per layer
      • .get_last_gate_means()      – list[float] for most recent fwd
    """

    def __init__(self, config):
        super().__init__(config)

        # hyper-params
        self.dynamic_k = float(getattr(config, "dynamic_k", 0.9))
        self.token_wise = bool(getattr(config, "token_wise", True))

        # logging flag & buffer
        self._log_gates = False
        self._last_gate_means = None

        # swap decoder layers
        custom_cls = (
            DynamicLlamaTokenWiseDecoderLayer
            if self.token_wise
            else DynamicLlamaBlockWiseDecoderLayer
        )
        new_layers = nn.ModuleList()
        for i, old in enumerate(self.model.layers):
            new = custom_cls(self.config, i)
            new.load_state_dict(old.state_dict(), strict=False)
            new_layers.append(new)
        self.model.layers = new_layers

    def set_dynamic_k(self, k: float):
        self.dynamic_k = float(k)

    def enable_gate_logging(self, flag: bool = True):
        self._log_gates = flag
        self._last_gate_means = None

    def get_last_gate_means(self):
        """
        Returns a list of per-layer mean gate values from the *most
        recent* forward / generate call – or None if logging disabled.
        """
        return self._last_gate_means


    def forward(self, *args, **kwargs):
        # The decoder layers already look up dynamic_k from self.model_cfg,
        # so we simply make it available as an attribute:
        self.model_cfg = type("Cfg", (), {"dynamic_k": self.dynamic_k})

        # Clear buffer if we want to log this pass
        if self._log_gates:
            self._gate_means_tmp = []

            # hook – called inside every decoder layer
            def _collect(_, __, outputs):
                gate_vec = outputs[-1]  # (B,) or (B,T) mean already computed
                self._gate_means_tmp.append(gate_vec.mean().item())
                return outputs

            hooks = [
                l.register_forward_hook(_collect) for l in self.model.layers
            ]
        else:
            hooks = []

        try:
            out = super().forward(*args, **kwargs)
        finally:
            # remove hooks even if model.forward raised
            for h in hooks:
                h.remove()

        if self._log_gates:
            self._last_gate_means = self._gate_means_tmp

        return out
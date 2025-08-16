import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from src.models.d_qwen_fnn import FeedForward

class DynamicQwenDecoderLayer(nn.Module):
    """
    Wraps a QwenDecoderLayer, adding dynamic token-wise gating:
      - Computes posterior via base_layer
      - Predicts a "prior" via FeedForward on the original input
      - Gates between posterior vs. input based on CE/CU criteria
    """
    def __init__(self, base_layer: Qwen2DecoderLayer, config):
        super().__init__()
        self.base_layer = base_layer
        self.config = config
        # small FFN for prior prediction
        self.prior_ffn = FeedForward(config)

    def forward(self,
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                use_cache=False,
                output_attentions=False,
                **kwargs):
        # pop gating args (fall back to config)
        current_iter = kwargs.pop("current_iter", 0)
        dynamic_k = kwargs.pop("dynamic_k", self.config.dynamic_k)
        ce_bias = kwargs.pop("ce_bias", self.config.ce_bias)
        warmup = kwargs.pop("gate_warmup_iters",
                            self.config.gate_warmup_iters)

        # original input to block
        x0 = hidden_states

        # run the base Qwen layer (gets new hidden + other outputs)
        outputs = self.base_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # hidden_states_out is first element
        hs = outputs[0]

        # compute prior prediction from original input
        prior = self.prior_ffn(x0)

        # compute per-token mse losses
        d_st = F.mse_loss(hs, x0, reduction="none").mean(-1)  # (B,T)
        d_ch = F.mse_loss(hs, prior, reduction="none").mean(-1)  # (B,T)

        # CE: expected change criterion
        CE = d_st > (d_ch - ce_bias)
        # CU: unexpected (surprise) criterion
        mean_dst = d_st.detach().mean()
        CU = d_st > (dynamic_k * mean_dst)

        # warmup: only CU or only CE first?
        if current_iter < warmup:
            gate = CE.float()  # bias toward CE during warmup
        else:
            gate = (CE | CU).float()

        # mix hidden states
        gate = gate.unsqueeze(-1)  # (B,T,1)
        mixed = gate * hs + (1.0 - gate) * x0

        # replace the hidden state in outputs tuple
        outputs = (mixed, ) + outputs[1:]
        return outputs

def patch_qwen_layers(model):
    """
    Replace each QwenDecoderLayer in model with DynamicQwenDecoderLayer.
    Call this after loading the pretrained model and setting model.config.
    """
    for i, layer in enumerate(model.qwen.layers):
        dynamic = DynamicQwenDecoderLayer(layer, model.config)
        model.qwen.layers[i] = dynamic
    return model
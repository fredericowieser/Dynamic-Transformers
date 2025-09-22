import torch
import torch.nn as nn

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make it a parameter of the model
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from original RoPE: x.shape[-1] is not necessarily self.dim * 2
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_dim]
        # position_ids: [bs, seq_len]
        seq_len = x.shape[-2]

        # Recompute cos/sin if sequence length exceeds cache
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len)

        return (
            self.cos_cached[:, :, position_ids, :].to(x.dtype),
            self.sin_cached[:, :, position_ids, :].to(x.dtype),
        )

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [bs, num_attention_heads, seq_len, head_dim]
    # cos, sin: [1, 1, seq_len, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

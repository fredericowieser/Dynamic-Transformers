import torch
import torch.nn as nn
import torch.nn.functional as F


class DTFRouter(nn.Module):
    """DTF surprise-based token routing."""

    def __init__(self, config):
        super().__init__()
        self.capacity = config.capacity_gamma
        self.beta_ce = nn.Parameter(torch.tensor(config.beta_ce_init))
        self.beta_cu = nn.Parameter(torch.tensor(config.beta_cu_init))
        self.cu_mult = nn.Parameter(torch.tensor(config.cu_detection_multiplier_init))
        self.ce_offset = nn.Parameter(torch.tensor(config.ce_criterion_offset_init))

    def compute_surprise(self, orig, post, prior):
        """Compute surprise metrics."""
        d_st = F.mse_loss(post, orig, reduction="none").mean(-1)
        d_ch = F.mse_loss(post, prior, reduction="none").mean(-1)
        return d_st, d_ch

    def compute_moving_average(self, values, window=100):
        """Compute causal moving average."""
        padded = F.pad(values, (window-1, 0), mode="replicate")
        windows = padded.unfold(-1, window, 1)
        return windows.mean(-1)

    def forward(self, original, posterior, prior):
        """Compute routing decision."""
        # Surprise metrics
        d_st, d_ch = self.compute_surprise(original, posterior, prior)

        # Moving average for unexpected criterion
        ma_d_st = self.compute_moving_average(d_st.detach())

        # Gating criteria
        ce_val = d_st - (d_ch - torch.log(self.ce_offset + 1e-10))
        cu_val = d_st - (self.cu_mult * ma_d_st)

        # Soft gates
        s_ce = torch.sigmoid(F.softplus(self.beta_ce) * ce_val)
        s_cu = torch.sigmoid(F.softplus(self.beta_cu) * cu_val)

        # Combined signal
        signal = s_ce + s_cu - (s_ce * s_cu)

        # Top-k selection
        k = int(self.capacity * signal.numel())
        threshold = signal.flatten().kthvalue(signal.numel() - k + 1)[0] if k < signal.numel() else -float('inf')
        mask = (signal >= threshold).float()

        return mask, signal, {"s_ce": s_ce, "s_cu": s_cu}
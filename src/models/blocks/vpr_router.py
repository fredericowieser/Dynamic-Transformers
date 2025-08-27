import logging

import torch
import torch.nn.functional as F
from torch import nn

log = logging.getLogger(__name__)


class VPRRouter(nn.Module):
    """
    Implements the Variational Predictive Routing (VPR) logic to make
    per-token or per-batch routing decisions within a Transformer layer.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.token_wise_gating = getattr(config, "token_wise_gating", True)
        self.moving_average_window_size = getattr(
            config, "moving_average_window_size", 100
        )
        # beta_ce
        if getattr(config, "learn_beta_ce", False):
            self.beta_ce = nn.Parameter(torch.tensor(config.beta_ce_init, dtype=torch.float32))
        else:
            self.register_buffer('beta_ce', torch.tensor(config.beta_ce_init, dtype=torch.float32))

        # beta_cu
        if getattr(config, "learn_beta_cu", False):
            self.beta_cu = nn.Parameter(torch.tensor(config.beta_cu_init, dtype=torch.float32))
        else:
            self.register_buffer('beta_cu', torch.tensor(config.beta_cu_init, dtype=torch.float32))

        # cu_detection_multiplier
        if getattr(config, "learn_cu_multiplier", False):
            self.cu_detection_multiplier = nn.Parameter(torch.tensor(config.cu_detection_multiplier_init, dtype=torch.float32))
        else:
            self.register_buffer('cu_detection_multiplier', torch.tensor(config.cu_detection_multiplier_init, dtype=torch.float32))

        # ce_criterion_offset
        if getattr(config, "learn_ce_offset", False):
            self.ce_criterion_offset = nn.Parameter(torch.tensor(config.ce_criterion_offset_init, dtype=torch.float32))
        else:
            self.register_buffer('ce_criterion_offset', torch.tensor(config.ce_criterion_offset_init, dtype=torch.float32))
            
        log.info(f"VPRRouter Layer {self.layer_idx} Parameter Trainability:")
        log.info(f"  - learn_beta_ce: {getattr(config, 'learn_beta_ce', False)}")
        log.info(f"  - learn_beta_cu: {getattr(config, 'learn_beta_cu', False)}")
        log.info(f"  - learn_cu_multiplier: {getattr(config, 'learn_cu_multiplier', False)}")
        log.info(f"  - learn_ce_offset: {getattr(config, 'learn_ce_offset', False)}")
    
    @property
    def current_beta_ce(self):
        return self.beta_ce.item()

    @property
    def current_beta_cu(self):
        return self.beta_cu.item()

    @property
    def current_cu_detection_multiplier(self):
        return self.cu_detection_multiplier.item()

    @property
    def current_ce_criterion_offset(self):
        return self.ce_criterion_offset.item()

    def _calculate_moving_average(self, d_st_values: torch.Tensor) -> torch.Tensor:
        """
        Calculates the causal moving average for d_st values.
        """
        if self.moving_average_window_size <= 0:
            return d_st_values.mean(dim=-1, keepdim=True).expand_as(d_st_values)

        padded_d_st = F.pad(
            d_st_values, (self.moving_average_window_size - 1, 0), mode="replicate"
        )
        windows = padded_d_st.unfold(
            dimension=-1, size=self.moving_average_window_size, step=1
        )
        return windows.mean(dim=-1)

    def forward(
        self,
        original_input_to_block: torch.Tensor,
        posterior_full_path_output: torch.Tensor,
        prior_hidden_states: torch.Tensor,
        capacity_gamma: float,
        is_training: bool = True,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, float, float, float, float,
    ]:
        d_st_tok = F.mse_loss(
            posterior_full_path_output,
            original_input_to_block,
            reduction="none",
        ).mean(-1)
        d_ch_tok = F.mse_loss(
            posterior_full_path_output,
            prior_hidden_states,
            reduction="none",
        ).mean(-1)

        ce_criterion_offset_val = self.ce_criterion_offset

        if self.token_wise_gating:
            CE_val = d_st_tok - (d_ch_tok - torch.log(ce_criterion_offset_val + 1e-10))
            ma_d_st_tok = self._calculate_moving_average(d_st_tok.detach())
            CU_val = d_st_tok - (self.cu_detection_multiplier * ma_d_st_tok)
        else:
            mean_d_st = d_st_tok.mean(dim=-1, keepdim=True)
            mean_d_ch = d_ch_tok.mean(dim=-1, keepdim=True)
            CE_val = mean_d_st - (mean_d_ch - torch.log(ce_criterion_offset_val + 1e-10))
            CU_val = mean_d_st - (self.cu_detection_multiplier * mean_d_st.detach())

        S_CE = torch.sigmoid(self.beta_ce * CE_val)
        S_CU = torch.sigmoid(self.beta_cu * CU_val)

        combined_gating_signal_continuous = S_CE + S_CU - (S_CE * S_CU)

        def get_stats(tensor):
            return {
                "mean": tensor.mean(),
                "std": tensor.std(),
                "min": tensor.min(),
                "max": tensor.max(),
            }

        s_ce_stats = get_stats(S_CE)
        s_cu_stats = get_stats(S_CU)
        g_cont_stats = get_stats(combined_gating_signal_continuous)

        if capacity_gamma >= 1.0:
            threshold = -torch.finfo(combined_gating_signal_continuous.dtype).max
        else:
            if self.token_wise_gating:
                flat_g_signal = combined_gating_signal_continuous.flatten()
                threshold = torch.quantile(flat_g_signal.float(), (1.0 - capacity_gamma))
            else:
                threshold = torch.quantile(
                    combined_gating_signal_continuous.float, (1.0 - capacity_gamma), dim=0
                )

        gate_vec_binary = (combined_gating_signal_continuous >= threshold).float()

        avg_ce_proportion = S_CE.mean()
        avg_cu_proportion = S_CU.mean()

        return (
            gate_vec_binary,
            s_ce_stats,
            s_cu_stats,
            g_cont_stats,
            d_st_tok, # TODO: Remove if not needed
            d_ch_tok, # TODO: Remove if not needed
            combined_gating_signal_continuous,
            self.current_beta_ce,
            self.current_beta_cu,
            self.current_cu_detection_multiplier,
            self.current_ce_criterion_offset,
        )
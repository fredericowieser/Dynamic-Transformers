import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

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
        self.moving_average_window_size = getattr(config, "moving_average_window_size", 100)

        # Learnable parameters for the sigmoid activations and criteria.
        # Initial values for these will be set via config and then learned.
        self.beta_ce = nn.Parameter(torch.tensor(config.beta_ce_init, dtype=torch.float32))
        self.beta_cu = nn.Parameter(torch.tensor(config.beta_cu_init, dtype=torch.float32))

        # Initial 'change' multiplier for CU, will be learned.
        self.cu_detection_multiplier = nn.Parameter(torch.tensor(config.cu_detection_multiplier_init, dtype=torch.float32))

    def _calculate_moving_average(self, d_st_values: torch.Tensor) -> torch.Tensor:
        """
        Calculates the causal moving average for d_st values for each token
        within its sequence.
        d_st_values: (B, T)
        Returns: (B, T)
        """
        if self.moving_average_window_size <= 0:
            # If window size is 0 or less, just return the mean of the current d_st batch
            # This is a fallback and generally less desired for "moving average"
            return d_st_values.mean(dim=-1, keepdim=True).expand_as(d_st_values)

        # Pad the sequence on the left to handle window for early tokens
        # We need to compute an average up to the current token, so pad to the left
        padded_d_st = F.pad(d_st_values, (self.moving_average_window_size - 1, 0), mode='replicate')

        # Use unfold to create windows.
        # (B, T_padded) -> (B, T, window_size)
        windows = padded_d_st.unfold(dimension=-1, size=self.moving_average_window_size, step=1)

        # FIX: Average across the window_size dimension (last dimension)
        ma_d_st = windows.mean(dim=-1) # (B, T)

        return ma_d_st


    def forward(
        self,
        original_input_to_block: torch.Tensor, # Z^{n-1}
        posterior_full_path_output: torch.Tensor, # H^{D_n}_{trans} (from MLP output)
        prior_hidden_states: torch.Tensor, # H^{D_n}_{prior} (from Prior FFN output)
        capacity_gamma: float, # This is the preset gamma (k in MoD) from config
        is_training: bool = True, # To enable deterministic routing in inference
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the routing decision based on VPR criteria.

        Args:
            original_input_to_block (torch.Tensor): The hidden states input to the Decision Layer.
                Shape: (batch_size, sequence_length, hidden_size)
            posterior_full_path_output (torch.Tensor): The output of the full Transformer computation
                (Attention + MLP) in the Decision Layer.
                Shape: (batch_size, sequence_length, hidden_size)
            prior_hidden_states (torch.Tensor): The predicted prior hidden states from the
                Prior FFN in the Decision Layer.
                Shape: (batch_size, sequence_length, hidden_size)
            capacity_gamma (float): A float in (0, 1] representing the fraction of tokens
                to route through the dynamic computation. Acts as the capacity in MoD.
            is_training (bool): Flag indicating if the model is in training mode.
                                Used for deterministic routing during inference.

        Returns:
            tuple: (gate_vec_final, avg_ce_proportion, avg_cu_proportion, d_st_tok, d_ch_tok, combined_gating_signal)
            - gate_vec_final (torch.Tensor): The (B, T) or (B,) tensor of routing decisions (0 or 1).
            - avg_ce_proportion (torch.Tensor): Scalar mean of CE activations.
            - avg_cu_proportion (torch.Tensor): Scalar mean of CU activations.
            - d_st_tok (torch.Tensor): Per-token static surprise (B, T).
            - d_ch_tok (torch.Tensor): Per-token change surprise (B, T).
            - combined_gating_signal (torch.Tensor): Per-token continuous gating signal (B, T).
        """
        # Calculate per-token MSE losses
        # MSE is averaged over the last dimension (hidden_size)
        d_st_tok = F.mse_loss(posterior_full_path_output, original_input_to_block, reduction="none").mean(-1)  # (B, T)
        d_ch_tok = F.mse_loss(posterior_full_path_output, prior_hidden_states, reduction="none").mean(-1)  # (B, T)

        # Let's use `ce_criterion_offset_init` from config as the offset.
        ce_criterion_offset = self.config.ce_criterion_offset_init

        if self.token_wise_gating:
            # Per-token calculation
            CE_val = d_st_tok - (d_ch_tok - ce_criterion_offset) # (B, T)
            ma_d_st_tok = self._calculate_moving_average(d_st_tok.detach()) # Detach for stable moving average.
            CU_val = d_st_tok - (self.cu_detection_multiplier * ma_d_st_tok) # (B, T)
        else:
            # Per-batch/sequence calculation (mean over tokens)
            mean_d_st = d_st_tok.mean(dim=-1, keepdim=True) # (B, 1)
            mean_d_ch = d_ch_tok.mean(dim=-1, keepdim=True) # (B, 1)

            CE_val = mean_d_st - (mean_d_ch - ce_criterion_offset) # (B, 1)
            CU_val = mean_d_st - (self.cu_detection_multiplier * mean_d_st.detach()) # (B, 1)

        # Sigmoid gating scores
        S_CE = torch.sigmoid(self.beta_ce * CE_val)
        S_CU = torch.sigmoid(self.beta_cu * CU_val)

        # Combined Gating Signal (logical OR equivalent)
        combined_gating_signal = S_CE + S_CU - (S_CE * S_CU) # (B, T) or (B, 1)

        # Determine the dynamic threshold (capacity_gamma is the percentile)
        if capacity_gamma >= 1.0: # If gamma is 1.0, process all tokens.
            threshold = -torch.finfo(combined_gating_signal.dtype).max # Always pass
        else:
            # Calculate the threshold that lets capacity_gamma fraction of tokens pass
            if self.token_wise_gating:
                # Flatten all tokens in batch to find global threshold
                flat_g_signal = combined_gating_signal.flatten()
                threshold = torch.quantile(flat_g_signal, (1.0 - capacity_gamma))
            else:
                # Compute quantile per batch (for the batch_wise decision)
                # If combined_gating_signal is (B, 1), quantile along dim=0 gives a scalar (1,)
                threshold = torch.quantile(combined_gating_signal, (1.0 - capacity_gamma), dim=0)
                # Ensure threshold can broadcast correctly if it collapses a dimension
                if threshold.ndim < combined_gating_signal.ndim:
                    threshold = threshold.view(1,) if self.token_wise_gating else threshold.view(-1, 1, 1)

        # Final routing decision (binary gate: 1 if processed, 0 if skipped)
        if is_training:
            # During training, use continuous signal for gradient flow, but make it binary for gate_vec_final
            gate_vec_final = (combined_gating_signal >= threshold).float()
        else:
            # During inference, apply a hard threshold for deterministic routing
            gate_vec_final = (combined_gating_signal >= threshold).float()


        # Metrics for logging
        avg_ce_proportion = S_CE.mean() # Average probability of expected event
        avg_cu_proportion = S_CU.mean() # Average probability of unexpected event

        return (
            gate_vec_final,
            avg_ce_proportion,
            avg_cu_proportion,
            d_st_tok,
            d_ch_tok,
            combined_gating_signal,
        )
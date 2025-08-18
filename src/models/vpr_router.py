import torch
import torch.nn.functional as F
import logging
from torch import nn # Keep nn imported if you use other nn.Modules like nn.Linear, etc.

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
        self.beta_ce = nn.Parameter(torch.tensor(config.beta_ce_init, dtype=torch.float32))
        self.beta_cu = nn.Parameter(torch.tensor(config.beta_cu_init, dtype=torch.float32))

        # --- START OF CHANGE ---
        # Make cu_detection_multiplier non-learnable
        # It's now a direct attribute, initialized from config, not an nn.Parameter
        self.cu_detection_multiplier_val = config.cu_detection_multiplier_init
        # --- END OF CHANGE ---

        # Learnable offset for CE criterion
        self.ce_criterion_offset = nn.Parameter(torch.tensor(config.ce_criterion_offset_init, dtype=torch.float32))


    # Add properties to expose the current learnable parameter values
    @property
    def current_beta_ce(self):
        return self.beta_ce.item()

    @property
    def current_beta_cu(self):
        return self.beta_cu.item()

    # --- START OF CHANGE ---
    # Update property to reflect non-learnable nature
    @property
    def current_cu_detection_multiplier(self):
        return self.cu_detection_multiplier_val # Directly return the fixed value
    # --- END OF CHANGE ---

    @property
    def current_ce_criterion_offset(self):
        return self.ce_criterion_offset.item()


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

        # Average across the window_size dimension (last dimension)
        ma_d_st = windows.mean(dim=-1) # (B, T)

        return ma_d_st


    def forward(
        self,
        original_input_to_block: torch.Tensor, # Z^{n-1}
        posterior_full_path_output: torch.Tensor, # H^{D_n}_{trans} (from MLP output)
        prior_hidden_states: torch.Tensor, # H^{D_n}_{prior} (from Prior FFN output)
        capacity_gamma: float, # This is the preset gamma (k in MoD) from config
        is_training: bool = True, # To enable deterministic routing in inference
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float, float, float]:
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
            tuple:
            - gate_vec_binary (torch.Tensor): The (B, T) or (B,) tensor of binary routing decisions (0 or 1).
            - avg_ce_proportion (torch.Tensor): Scalar mean of CE activations.
            - avg_cu_proportion (torch.Tensor): Scalar mean of CU activations.
            - d_st_tok (torch.Tensor): Per-token static surprise (B, T).
            - d_ch_tok (torch.Tensor): Per-token change surprise (B, T).
            - combined_gating_signal_continuous (torch.Tensor): Per-token continuous gating signal (B, T).
            - current_beta_ce (float): Current value of learnable beta_ce.
            - current_beta_cu (float): Current value of learnable beta_cu.
            - current_cu_detection_multiplier (float): Current value of non-learnable cu_detection_multiplier.
            - current_ce_criterion_offset (float): Current value of learnable ce_criterion_offset.
        """
        # Calculate per-token MSE losses
        # MSE is averaged over the last dimension (hidden_size)
        d_st_tok = F.mse_loss(posterior_full_path_output, original_input_to_block, reduction="none").mean(-1)  # (B, T)
        d_ch_tok = F.mse_loss(posterior_full_path_output, prior_hidden_states, reduction="none").mean(-1)  # (B, T)

        # Use `self.ce_criterion_offset` (now a learnable parameter)
        ce_criterion_offset_val = self.ce_criterion_offset

        if self.token_wise_gating:
            # Per-token calculation
            CE_val = d_st_tok - (d_ch_tok - ce_criterion_offset_val) # (B, T)
            ma_d_st_tok = self._calculate_moving_average(d_st_tok.detach()) # Detach for stable moving average.
            # --- START OF CHANGE ---
            CU_val = d_st_tok - (self.cu_detection_multiplier_val * ma_d_st_tok) # Use the fixed value
            # --- END OF CHANGE ---
        else:
            # Per-batch/sequence calculation (mean over tokens)
            mean_d_st = d_st_tok.mean(dim=-1, keepdim=True) # (B, 1)
            mean_d_ch = d_ch_tok.mean(dim=-1, keepdim=True) # (B, 1)

            CE_val = mean_d_st - (mean_d_ch - ce_criterion_offset_val) # (B, 1)
            # --- START OF CHANGE ---
            CU_val = mean_d_st - (self.cu_detection_multiplier_val * mean_d_st.detach()) # Use the fixed value
            # --- END OF CHANGE ---

        # Sigmoid gating scores
        S_CE = torch.sigmoid(self.beta_ce * CE_val)
        S_CU = torch.sigmoid(self.beta_cu * CU_val)

        # Combined Gating Signal (logical OR equivalent), this is the continuous signal
        combined_gating_signal_continuous = S_CE + S_CU - (S_CE * S_CU) # (B, T) or (B, 1)

        # Determine the dynamic threshold (capacity_gamma is the percentile)
        if capacity_gamma >= 1.0: # If gamma is 1.0, process all tokens.
            threshold = -torch.finfo(combined_gating_signal_continuous.dtype).max # Always pass
        else:
            # Calculate the threshold that lets capacity_gamma fraction of tokens pass
            if self.token_wise_gating:
                # Flatten all tokens in batch to find global threshold
                flat_g_signal = combined_gating_signal_continuous.flatten()
                threshold = torch.quantile(flat_g_signal, (1.0 - capacity_gamma))
            else:
                # Compute quantile per batch (for the batch_wise decision)
                threshold = torch.quantile(combined_gating_signal_continuous, (1.0 - capacity_gamma), dim=0)

        # Final binary routing decision (0 or 1) based on threshold
        gate_vec_binary = (combined_gating_signal_continuous >= threshold).float()

        # Metrics for logging
        avg_ce_proportion = S_CE.mean() # Average probability of expected event
        avg_cu_proportion = S_CU.mean() # Average probability of unexpected event

        return (
            gate_vec_binary,
            avg_ce_proportion,
            avg_cu_proportion,
            d_st_tok,
            d_ch_tok,
            combined_gating_signal_continuous,
            self.current_beta_ce, # Current value of learnable beta_ce.
            self.current_beta_cu, # Current value of learnable beta_cu.
            self.current_cu_detection_multiplier, # Current value of non-learnable cu_detection_multiplier.
            self.current_ce_criterion_offset, # Current value of learnable ce_criterion_offset.
        )
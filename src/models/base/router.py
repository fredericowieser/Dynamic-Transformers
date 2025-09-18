import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class BaseRouter(nn.Module, ABC):
    """Base router for token selection shared by MoD and DTF."""

    def __init__(self, capacity: float = 0.125):
        super().__init__()
        self.capacity = capacity

    @abstractmethod
    def compute_routing_scores(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute routing scores for each token.

        Args:
            hidden_states: Input tensor [B, T, D]
            **kwargs: Additional arguments for specific routers

        Returns:
            Routing scores [B, T]
        """
        pass

    def select_tokens(
        self,
        scores: torch.Tensor,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select top-k tokens based on routing scores.

        Args:
            scores: Routing scores [B, T]
            hidden_states: Input tensor [B, T, D]

        Returns:
            selected_states: Selected token states [B*k, D]
            batch_idx: Batch indices for scattering [B*k]
            token_idx: Token indices [B*k]
            selected_scores: Scores of selected tokens [B*k]
        """
        B, T, D = hidden_states.shape
        k = max(1, int(T * self.capacity))

        # Select top-k tokens
        topk_vals, topk_idx = scores.topk(k, dim=-1)  # [B, k]

        # Create batch indices
        batch_idx = torch.arange(B, device=scores.device).unsqueeze(1)  # [B, 1]
        batch_idx = batch_idx.expand(-1, k)  # [B, k]

        # Gather selected tokens
        selected_states = hidden_states[batch_idx, topk_idx]  # [B, k, D]
        selected_states = selected_states.reshape(-1, D)  # [B*k, D]

        # Flatten indices for scattering
        batch_idx_flat = batch_idx.reshape(-1)  # [B*k]
        token_idx_flat = topk_idx.reshape(-1)  # [B*k]
        selected_scores_flat = topk_vals.reshape(-1)  # [B*k]

        return selected_states, batch_idx_flat, token_idx_flat, selected_scores_flat

    def scatter_tokens(
        self,
        processed_tokens: torch.Tensor,
        original_states: torch.Tensor,
        batch_idx: torch.Tensor,
        token_idx: torch.Tensor
    ) -> torch.Tensor:
        """Scatter processed tokens back to original positions.

        Args:
            processed_tokens: Processed token states [B*k, D]
            original_states: Original states to update [B, T, D]
            batch_idx: Batch indices [B*k]
            token_idx: Token indices [B*k]

        Returns:
            Updated states with processed tokens [B, T, D]
        """
        B, T, D = original_states.shape

        # Initialize output with original states
        output = original_states.clone()

        # Scatter back to original positions using flat indexing
        output[batch_idx, token_idx] = processed_tokens

        return output
import logging
from collections import deque

import torch

log = logging.getLogger(__name__)
ROLLING_WINDOW_SIZE = 100


class GateLogger:
    """
    Handles logging and rolling statistics for dynamic gate activations.
    """

    def __init__(self, num_layers: int):
        self.per_layer_gate_activation_rolling_history = [
            {
                "mean": deque(maxlen=ROLLING_WINDOW_SIZE),
                "std": deque(maxlen=ROLLING_WINDOW_SIZE),
            }
            for _ in range(num_layers)
        ]

    def update_rolling_history(self, per_layer_gate_stats: list[dict]):
        """Updates the rolling history with new stats from a training step."""
        for i, stats in enumerate(per_layer_gate_stats):
            # Ensure stats are tensors before calling .item()
            mean_val = stats["mean"].item() if isinstance(stats["mean"], torch.Tensor) else stats["mean"]
            std_val = stats["std"].item() if isinstance(stats["std"], torch.Tensor) else stats["std"]
            
            hist = self.per_layer_gate_activation_rolling_history[i]
            hist["mean"].append(mean_val)
            hist["std"].append(std_val)

    def log_rolling_history(self, global_step: int, log_interval: int):
        """Logs the current rolling average statistics to the console."""
        if global_step > 0 and global_step % log_interval == 0:
            lines = [
                f"--- Per-Layer Gate Activations (Rolling Avg over last {ROLLING_WINDOW_SIZE} steps) ---"
            ]
            for i, history in enumerate(self.per_layer_gate_activation_rolling_history):
                if history["mean"]:
                    rolling_mean = sum(history["mean"]) / len(history["mean"])
                    # Calculate std dev from the list of stds for a sense of variance
                    rolling_std_of_means = torch.tensor(list(history["mean"])).std().item()

                    lines.append(
                        f"  Layer {i:02d}: Mean Activation = {rolling_mean:.3f} (Std of Means = {rolling_std_of_means:.3f})"
                    )
            log.info("\n".join(lines))


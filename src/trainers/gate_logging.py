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
                "std":  deque(maxlen=ROLLING_WINDOW_SIZE),
            }
            for _ in range(num_layers)
        ]

    def update_rolling_history(self, per_layer_gate_stats: list[dict]):
        for i, stats in enumerate(per_layer_gate_stats):
            hist = self.per_layer_gate_activation_rolling_history[i]
            hist["mean"].append(stats["mean"].item())
            hist["std"].append(stats["std"].item())

    def log_rolling_history(self, global_step: int, log_interval: int):
        if global_step % log_interval != 0:
            return
        lines = [
            (
                f"--- Per-Layer Gate Activations (Training, "
                f"Rolling Avg over last {ROLLING_WINDOW_SIZE} steps) ---"
            )
        ]
        for i, history in enumerate(self.per_layer_gate_activation_rolling_history):
            if history["mean"]:
                rolling_mean = sum(history["mean"]) / len(history["mean"])
                rolling_std = (
                    torch.tensor(list(history["std"])).std().item()
                    if len(history["std"]) > 1
                    else 0.0
                )
                lines.append(
                    f"  Layer {i}: Mean = {rolling_mean:.3f}, "
                    f"Std = {rolling_std:.3f}"
                )
        log.info("\n".join(lines))

    @staticmethod
    def log_gate_metrics(
        module,
        prefix: str,
        overall_gate_activation_mean: torch.Tensor,
        per_layer_gate_stats: list[dict[str, torch.Tensor]],
        on_step: bool,
        on_epoch: bool,
    ):
        """
        Logs overall and per-layer gate activation metrics via Lightning's
        `module.log`.
        """
        module.log(
            f"{prefix}_dynamic_model/overall_gate_activation_mean",
            overall_gate_activation_mean,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
        )
        for i, stats in enumerate(per_layer_gate_stats):
            module.log(
                f"{prefix}_dynamic_layer/gate_mean/layer_{i}",
                stats["mean"],
                on_step=on_step,
                on_epoch=on_epoch,
            )
            module.log(
                f"{prefix}_dynamic_layer/gate_std/layer_{i}",
                stats["std"],
                on_step=on_step,
                on_epoch=on_epoch,
            )
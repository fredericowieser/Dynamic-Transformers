"""
Generates and saves comparison plots for VPR (Variational Predictive Routing)
metrics from multiple W&B runs.

Reads data from the _dynamic_metrics.csv files exported by download_wandb_run.py.
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# --- 1. CONFIGURE YOUR VPR RUNS HERE ---
# The key is the name that will appear in the plot's legend.
# The value is the path to the '_dynamic_metrics.csv' file.
VPR_RUNS_TO_PLOT = {
    "Dynamic (gamma=0.5)": "wandb_exports/PRETRAINTESTqwen2.50.5Bvprpretrain_mix20250828_132910gamma0.5_dynamic_metrics.csv",
    # Add other VPR runs here
    # "VPR (gamma=0.7)": "path/to/your/vpr_gamma0.7_dynamic_metrics.csv",
}

# --- 2. SCRIPT CONFIGURATION ---
OUTPUT_DIR = Path("./plots")


def plot_gating_signals(run_data: dict):
    """
    Generates a single plot combining the min, mean, and max of all gating signals.
    """
    plot_title = "Gating Signal Analysis"
    log.info(f"Generating plot for: {plot_title}")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 9))

    signal_colors = {
        "S_CE": "royalblue",
        "S_CU": "darkorange",
        "G_cont": "green",
    }

    for run_name, filepath in run_data.items():
        try:
            df = pd.read_csv(filepath)
            for signal, color in signal_colors.items():
                mean_col, min_col, max_col = f"{signal}_mean", f"{signal}_min", f"{signal}_max"
                
                if all(c in df.columns for c in [mean_col, min_col, max_col]):
                    # Plot the mean as a solid line
                    mean_df = df[["_step", mean_col]].dropna()
                    ax.plot(mean_df["_step"], mean_df[mean_col], color=color, linewidth=2.5, label=f"{run_name} - {signal} (mean)")

                    # Plot min and max as dashed lines
                    min_df = df[["_step", min_col]].dropna()
                    max_df = df[["_step", max_col]].dropna()
                    ax.plot(min_df["_step"], min_df[min_col], color=color, linestyle='--', linewidth=1.5, alpha=0.8)
                    ax.plot(max_df["_step"], max_df[max_col], color=color, linestyle='--', linewidth=1.5, alpha=0.8)
                else:
                    log.warning(f"One or more columns for signal '{signal}' not found in {filepath}. Skipping.")

        except FileNotFoundError:
            log.error(f"File not found: {filepath}. Skipping this run for this plot.")
        except Exception as e:
            log.error(f"Could not process {filepath}. Error: {e}")

    ax.set_title(plot_title, fontsize=18, weight='bold')
    ax.set_xlabel("Training Step", fontsize=14)
    ax.set_ylabel("Signal Value", fontsize=14)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1) # Set y-axis to be from 0 to 1

    output_path = OUTPUT_DIR / f"{plot_title.replace(' ', '_').lower()}.pdf"
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"✅ Plot saved to: {output_path}")


def plot_router_parameters(run_data: dict):
    """
    Generates a 2x2 quadrant plot for the learned router parameters.
    """
    plot_title = "Learned Router Parameters"
    log.info(f"Generating plot for: {plot_title}")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    axes = axes.flatten()

    params = {
        "beta_ce_mean": "β_CE (mean)",
        "beta_cu_mean": "β_CU (mean)",
        "cu_multiplier_mean": "CU Multiplier (mean)",
        "ce_offset_mean": "CE Offset (mean)",
    }
    
    # Assign colors per run
    run_colors = plt.cm.viridis(np.linspace(0, 0.85, len(run_data)))

    for run_idx, (run_name, filepath) in enumerate(run_data.items()):
        try:
            df = pd.read_csv(filepath)
            for ax_idx, (param_col, param_title) in enumerate(params.items()):
                ax = axes[ax_idx]
                if param_col in df.columns:
                    clean_df = df[["_step", param_col]].dropna()
                    ax.plot(
                        clean_df["_step"],
                        clean_df[param_col],
                        linewidth=2,
                        label=run_name,
                        color=run_colors[run_idx]
                    )
                ax.set_title(param_title, fontsize=14)
                ax.tick_params(axis='y', labelsize=10)

        except FileNotFoundError:
            log.error(f"File not found: {filepath}. Skipping this run for this plot.")
        except Exception as e:
            log.error(f"Could not process {filepath}. Error: {e}")

    fig.suptitle(plot_title, fontsize=20, weight='bold')
    for ax in axes:
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_xlim(left=0)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=12, bbox_to_anchor=(0.95, 0.95))
    
    fig.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle

    output_path = OUTPUT_DIR / f"{plot_title.replace(' ', '_').lower()}.pdf"
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"✅ Plot saved to: {output_path}")


def main():
    """Main function to orchestrate the VPR plotting process."""
    if not VPR_RUNS_TO_PLOT:
        log.warning("The 'VPR_RUNS_TO_PLOT' dictionary is empty. Nothing to plot.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_gating_signals(VPR_RUNS_TO_PLOT)
    plot_router_parameters(VPR_RUNS_TO_PLOT)
    
    log.info("✨ All VPR plots generated successfully.")


if __name__ == "__main__":
    main()


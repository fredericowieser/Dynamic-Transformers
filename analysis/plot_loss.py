"""
Generates and saves publication-quality comparison plots for training and
validation metrics from multiple W&B runs, comparing different model architectures.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# --- 1. CONFIGURE YOUR RUNS HERE ---
# Add the runs you want to plot to this dictionary.
# The key is the name that will appear in the plot's legend.
# The value is the path to the '_common_metrics.csv' file.
RUNS_TO_PLOT = {
    "Dynamic (gamma=0.5)": "wandb_exports/PRETRAINTESTqwen2.50.5Bvprpretrain_mix20250828_132910gamma0.5_common_metrics.csv",
    "MoD (gamma=0.5)": "wandb_exports/PRETRAINTESTqwen2.50.5Bmodpretrain_mix20250828_140245gamma0.5_common_metrics.csv",
    # Add more runs here, for example:
    # "VPR (gamma=0.7)": "path/to/your/vpr_gamma0.7_common_metrics.csv",
    # "MoD (gamma=0.7)": "path/to/your/mod_gamma0.7_common_metrics.csv",
}

# --- 2. SCRIPT CONFIGURATION ---
OUTPUT_DIR = Path("./plots_for_paper_final")
METRICS_TO_PLOT = {
    "train/loss": "Training Loss",
    "train/perplexity": "Training Perplexity",
    "val/loss": "Validation Loss",
}

def assign_colors(run_names):
    """Assigns distinct colors to runs based on their architecture."""
    colors = {}
    # Use a count to handle multiple runs of the same type gracefully
    dyn_count = sum(1 for name in run_names if "dynamic" in name.lower() or "vpr" in name.lower())
    mod_count = sum(1 for name in run_names if "mod" in name.lower())

    # Use perceptually distinct colormaps
    dyn_colors = plt.cm.Reds(np.linspace(0.5, 0.9, dyn_count or 1))
    mod_colors = plt.cm.Blues(np.linspace(0.5, 0.9, mod_count or 1))

    dyn_idx, mod_idx = 0, 0
    for name in run_names:
        name_lower = name.lower()
        if "dynamic" in name_lower or "vpr" in name_lower:
            colors[name] = dyn_colors[dyn_idx]
            dyn_idx += 1
        elif "mod" in name_lower:
            colors[name] = mod_colors[mod_idx]
            mod_idx += 1
        else:
            colors[name] = "gray"  # Fallback for unidentified runs
    return colors


def create_plot(metric_key: str, plot_title: str, run_data: dict, colors: dict):
    """Generates and saves a single, publication-quality plot for a given metric."""
    log.info(f"Generating plot for: {plot_title}")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    for run_name, filepath in run_data.items():
        try:
            df = pd.read_csv(filepath)
            if metric_key in df.columns:
                clean_df = df[["_step", metric_key]].dropna()
                run_color = colors.get(run_name, "gray")

                if metric_key == "val/loss":
                    # For validation, plot a clean line with markers
                    ax.plot(
                        clean_df["_step"], clean_df[metric_key], marker='x',
                        linestyle='-', color=run_color, markersize=10,
                        linewidth=3.5, label=run_name
                    )
                else:
                    # For training metrics, plot EMA over a faint raw data line
                    ax.plot(
                        clean_df["_step"], clean_df[metric_key],
                        color=run_color, linewidth=1.5, alpha=0.2
                    )
                    ema = clean_df[metric_key].ewm(span=15, adjust=False).mean()
                    ax.plot(
                        clean_df["_step"], ema, color=run_color,
                        linewidth=3.5, label=run_name
                    )
            else:
                log.warning(f"Metric '{metric_key}' not found in {filepath}. Skipping.")
        except FileNotFoundError:
            log.error(f"File not found: {filepath}. Skipping this run.")
        except Exception as e:
            log.error(f"Could not process {filepath}. Error: {e}")

    ax.set_title(f"Architecture Comparison: {plot_title}", fontsize=24, weight='bold')
    ax.set_xlabel("Training Step", fontsize=20)
    ax.set_ylabel(plot_title, fontsize=20)
    ax.legend(fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlim(left=0)

    # Use a logarithmic scale for loss and perplexity plots
    if "perplexity" in metric_key.lower() or "loss" in metric_key.lower():
        ax.set_yscale("log")
        ax.set_ylabel(f"{plot_title} (Log Scale)", fontsize=20)

    output_filename = f"architecture_comparison_{plot_title.replace(' ', '_').lower()}.pdf"
    output_path = OUTPUT_DIR / output_filename
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"✅ Plot saved to: {output_path}")


def main():
    """Main function to orchestrate the plotting process."""
    if not RUNS_TO_PLOT:
        log.warning("The 'RUNS_TO_PLOT' dictionary is empty. Nothing to plot.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    colors = assign_colors(RUNS_TO_PLOT.keys())

    for metric_key, plot_title in METRICS_TO_PLOT.items():
        create_plot(metric_key, plot_title, RUNS_TO_PLOT, colors)
    
    log.info("✨ All architecture comparison plots generated successfully! ✨")


if __name__ == "__main__":
    main()
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

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# Configure runs to plot - key is legend name, value is CSV path
RUNS_TO_PLOT = {
    "Dynamic (gamma=0.5)": "wandb_exports/PRETRAINTESTqwen2.50.5Bvprpretrain_mix20250828_132910gamma0.5_common_metrics.csv",
    "MoD (gamma=0.5)": "wandb_exports/PRETRAINTESTqwen2.50.5Bmodpretrain_mix20250828_140245gamma0.5_common_metrics.csv",
}

# Output configuration
OUTPUT_DIR = Path("./plots_for_paper_final")
METRICS_TO_PLOT = {
    "train/loss": "Training Loss",
    "train/perplexity": "Training Perplexity",
    "val/loss": "Validation Loss",
}

def assign_colors(run_names):
    """Assigns distinct colors to runs based on their architecture."""
    colors = {}
    # Handle multiple runs of the same architecture
    dyn_count = sum(1 for name in run_names if "dynamic" in name.lower() or "vpr" in name.lower())
    mod_count = sum(1 for name in run_names if "mod" in name.lower())

    # Assign distinct colors by architecture
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
            colors[name] = "gray"
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
                    # Validation metrics with markers
                    ax.plot(
                        clean_df["_step"], clean_df[metric_key], marker='x',
                        linestyle='-', color=run_color, markersize=10,
                        linewidth=3.5, label=run_name
                    )
                else:
                    # Training metrics with EMA smoothing
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

    # Use log scale for loss and perplexity
    if "perplexity" in metric_key.lower() or "loss" in metric_key.lower():
        ax.set_yscale("log")
        ax.set_ylabel(f"{plot_title} (Log Scale)", fontsize=20)

    output_filename_pdf = f"architecture_comparison_{plot_title.replace(' ', '_').lower()}.pdf"
    output_path_pdf = OUTPUT_DIR / output_filename_pdf
    fig.savefig(output_path_pdf, format="pdf", bbox_inches="tight")

    output_filename_svg = f"architecture_comparison_{plot_title.replace(' ', '_').lower()}.svg"
    output_path_svg = OUTPUT_DIR / output_filename_svg
    fig.savefig(output_path_svg, format="svg", bbox_inches="tight")

    plt.close(fig)
    log.info(f"✅ Plots saved to: {output_path_pdf} and {output_path_svg}")


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
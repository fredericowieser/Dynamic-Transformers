"""
Generates and saves a comparison plot for specific training losses 
from different model runs.
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

# Configure metrics to plot
# Format: { "Legend Name": ("file_path.csv", "column_name") }
METRICS_TO_PLOT = {
    "STT (TPN Loss)": (
        "wandb_exports/experimentstt20250923_0400350.5B_dynamic_metrics.csv",
        "train/loss/stt_tpn_loss_unscaled"
    ),
    "SDT (PriorFFN Loss)": (
        "wandb_exports/PRETRAINTESTqwen2.50.5Bvprpretrain_mix20250828_132910gamma0.5_common_metrics.csv",
        "train/prior_loss"
    ),
}

# Output configuration
OUTPUT_DIR = Path("./specific_plots")
PLOT_TITLE = "Comparison of Specific Training Losses"

def create_plot(metrics_to_plot: dict, plot_title: str):
    """Generates and saves a single plot for the specified metrics."""
    log.info(f"Generating plot for: {plot_title}")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = {"STT (TPN Loss)": "green", "SDT (PriorFFN Loss)": "red"}

    for run_name, (filepath, metric_key) in metrics_to_plot.items():
        try:
            df = pd.read_csv(filepath)
            if metric_key in df.columns:
                clean_df = df[["_step", metric_key]].dropna()
                proportional_step = clean_df["_step"] / clean_df["_step"].max()
                run_color = colors.get(run_name, "gray")

                # Plot raw data with transparency
                ax.plot(
                    proportional_step, clean_df[metric_key],
                    color=run_color, linewidth=0.5, alpha=0.3
                )
                # Plot EMA for a smoother line
                ema = clean_df[metric_key].ewm(span=50, adjust=False).mean()
                ax.plot(
                    proportional_step, ema, color=run_color,
                    linewidth=2.0, alpha=0.8, label=run_name
                )
            else:
                log.warning(f"Metric '{metric_key}' not found in {filepath}. Skipping.")
        except FileNotFoundError:
            log.error(f"File not found: {filepath}. Skipping this run.")
        except Exception as e:
            log.error(f"Could not process {filepath}. Error: {e}")

    ax.set_title(plot_title, fontsize=24, weight='bold')
    ax.set_xlabel("Proportion of Training", fontsize=20)
    ax.set_ylabel("Loss", fontsize=20)
    ax.legend(fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlim(left=0, right=1)

    output_filename_pdf = f"{plot_title.replace(' ', '_').lower()}.pdf"
    output_path_pdf = OUTPUT_DIR / output_filename_pdf
    fig.savefig(output_path_pdf, format="pdf", bbox_inches="tight")

    output_filename_svg = f"{plot_title.replace(' ', '_').lower()}.svg"
    output_path_svg = OUTPUT_DIR / output_filename_svg
    fig.savefig(output_path_svg, format="svg", bbox_inches="tight")

    plt.close(fig)
    log.info(f"✅ Plots saved to: {output_path_pdf} and {output_path_svg}")


def main():
    """Main function to orchestrate the plotting process."""
    if not METRICS_TO_PLOT:
        log.warning("The 'METRICS_TO_PLOT' dictionary is empty. Nothing to plot.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    create_plot(METRICS_TO_PLOT, PLOT_TITLE)
    
    log.info("✨ Plot generated successfully! ✨")


if __name__ == "__main__":
    main()

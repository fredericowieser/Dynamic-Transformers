"""
Generates plots for router statistics from a W&B run.
"""

import logging
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from scipy.stats import linregress

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# Configuration
DATA_FILE = "wandb_exports/experimentstt20250923_1727080.5B_dynamic_metrics.csv"
OUTPUT_DIR = Path("./router_stats_plots")

def plot_selected_tokens_proportion(data_file: str, output_dir: Path):
    """Plots the selected tokens proportion for each layer."""
    log.info("Generating plot for selected tokens proportion...")
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        log.error(f"File not found: {data_file}")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Find all relevant columns
    layer_cols = [col for col in df.columns if "extra/router_stats/stt/layer_" in col and "/selected_tokens_proportion" in col]
    
    # Extract layer numbers and sort
    layers = sorted([int(re.search(r"layer_(\d+)", col).group(1)) for col in layer_cols])
    
    # Create a color map from blue to red
    colors = plt.cm.get_cmap('coolwarm', len(layers))

    for i, layer_num in enumerate(layers):
        col_name = f"extra/router_stats/stt/layer_{layer_num}/selected_tokens_proportion"
        clean_df = df[["_step", col_name]].dropna()
        proportional_step = clean_df["_step"] / clean_df["_step"].max()
        ema = clean_df[col_name].ewm(span=15, adjust=False).mean()
        
        ax.plot(proportional_step, ema, color=colors(i), label=f"Layer {layer_num}")

    ax.set_title("Proportion of Selected Tokens per Layer (EMA-15)", fontsize=24, weight='bold')
    ax.set_xlabel("Proportion of Training", fontsize=20)
    ax.set_ylabel("Selected Tokens Proportion", fontsize=20)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlim(left=0, right=1)

    output_path_pdf = output_dir / "selected_tokens_proportion.pdf"
    output_path_svg = output_dir / "selected_tokens_proportion.svg"
    fig.savefig(output_path_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(output_path_svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    log.info(f"✅ Plot saved to {output_path_pdf} and {output_path_svg}")

def plot_inferred_selected_scatter(data_file: str, output_dir: Path):
    """Creates a scatter plot of the final inferred selected value vs. layer number."""
    log.info("Generating scatter plot for final inferred selected values...")
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        log.error(f"File not found: {data_file}")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    layer_nums = []
    final_values = []

    # Find all relevant columns
    layer_cols = [col for col in df.columns if "extra/val_router_stats/stt/layer_" in col and "/inferred_selected" in col]
    
    for col_name in layer_cols:
        layer_num = int(re.search(r"layer_(\d+)", col_name).group(1))
        final_value = df[col_name].dropna().iloc[-1] if not df[col_name].dropna().empty else None
        if final_value is not None:
            layer_nums.append(layer_num)
            final_values.append(final_value)

    ax.scatter(layer_nums, final_values, s=100, marker='x')

    # Add a trend line and regression stats
    if len(layer_nums) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(layer_nums, final_values)
        r_squared = r_value**2
        
        # Create the regression line
        x_vals = np.array(layer_nums)
        y_vals = intercept + slope * x_vals
        ax.plot(x_vals, y_vals, color='red', linestyle='-', label=f"y = {slope:.2f}x + {intercept:.2f}")

        # Add stats to the plot
        stats_text = f"$R^2 = {r_squared:.2f}$\n$p = {p_value:.2f}$"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_title("Final Inferred Selected Value vs. Layer Number", fontsize=24, weight='bold')
    ax.set_xlabel("Layer Number", fontsize=20)
    ax.set_ylabel("Final Inferred Selected Value", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(fontsize=12)

    output_path_pdf = output_dir / "inferred_selected_scatter.pdf"
    output_path_svg = output_dir / "inferred_selected_scatter.svg"
    fig.savefig(output_path_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(output_path_svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    log.info(f"✅ Scatter plot saved to {output_path_pdf} and {output_path_svg}")

def main():
    """Main function to orchestrate the plotting process."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_selected_tokens_proportion(DATA_FILE, OUTPUT_DIR)
    plot_inferred_selected_scatter(DATA_FILE, OUTPUT_DIR)
    log.info("✨ All router stats plots generated successfully! ✨")

if __name__ == "__main__":
    main()
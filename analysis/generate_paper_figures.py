"""
Generates all figures for the ICLR paper.
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

# Output configuration
OUTPUT_DIR = Path("./paper_figures")


# --- Scatter Plot of Inferred Selected Values ---
ROUTER_STATS_DATA_FILE = "wandb_exports/experimentstt20250923_1727080.5B_dynamic_metrics.csv"

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

    # Find all relevant columns and sort them by layer number
    layer_cols_with_num = []
    for col in df.columns:
        if "extra/val_router_stats/stt/layer_" in col and "/inferred_selected" in col:
            match = re.search(r"layer_(\d+)", col)
            if match:
                layer_cols_with_num.append((int(match.group(1)), col))
    layer_cols_with_num.sort()

    for layer_num, col_name in layer_cols_with_num:
        final_value = df[col_name].dropna().iloc[-1] if not df[col_name].dropna().empty else None
        if final_value is not None:
            layer_nums.append(layer_num)
            final_values.append(final_value)

    ax.scatter(layer_nums, final_values, s=200, marker='x')
    ax.plot(layer_nums, final_values, linestyle='--', alpha=0.5, label="Layer Path")

    # Add a trend line and regression stats
    if len(layer_nums) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(layer_nums, final_values)
        r_squared = r_value**2
        
        # Create the regression line
        x_vals = np.array(layer_nums)
        y_vals = intercept + slope * x_vals
        ax.plot(x_vals, y_vals, color='red', linestyle='-', linewidth=2.5, label=f"y = {slope:.2f}x + {intercept:.2f}")

        # Add stats to the plot
        stats_text = f"$R^2 = {r_squared:.2f}$\n$p = {p_value:.2f}$"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=20,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_title("Learnt Dynamic Capacities per Layer", fontsize=30, weight='bold')
    ax.set_xlabel("Layer Number", fontsize=26)
    ax.tick_params(axis='both', which='major', labelsize=26)
    ax.legend(fontsize=18, fancybox=True, frameon=True, shadow=True)
    ax.set_xlim(left=0)

    output_path_pdf = output_dir / "inferred_selected_scatter.pdf"
    output_path_svg = output_dir / "inferred_selected_scatter.svg"
    fig.savefig(output_path_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(output_path_svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    log.info(f"✅ Scatter plot saved to {output_path_pdf} and {output_path_svg}")


# --- Architecture Comparison Plot ---
ARCH_RUNS_TO_PLOT = {
    "SDT": "wandb_exports/PRETRAINTESTqwen2.50.5Bvprpretrain_mix20250828_132910gamma0.5_common_metrics.csv",
    "MoD": "wandb_exports/PRETRAINTESTqwen2.50.5Bmodpretrain_mix20250828_140245gamma0.5_common_metrics.csv",
    "STT": "wandb_exports/experimentstt20250923_0400350.5B_common_metrics.csv",
}

def assign_arch_colors(run_names):
    """Assigns distinct colors to runs based on their architecture."""
    colors = {}
    sdt_count = sum(1 for name in run_names if "sdt" in name.lower())
    mod_count = sum(1 for name in run_names if "mod" in name.lower())
    stt_count = sum(1 for name in run_names if "stt" in name.lower())

    sdt_colors = plt.cm.Reds(np.linspace(0.7, 0.9, sdt_count or 1))
    mod_colors = plt.cm.Blues(np.linspace(0.7, 0.9, mod_count or 1))
    stt_colors = plt.cm.Greens(np.linspace(0.7, 0.9, stt_count or 1))

    sdt_idx, mod_idx, stt_idx = 0, 0, 0
    for name in run_names:
        name_lower = name.lower()
        if "sdt" in name_lower:
            colors[name] = sdt_colors[sdt_idx]
            sdt_idx += 1
        elif "mod" in name_lower:
            colors[name] = mod_colors[mod_idx]
            mod_idx += 1
        elif "stt" in name_lower:
            colors[name] = stt_colors[stt_idx]
            stt_idx += 1
        else:
            colors[name] = "gray"
    return colors

def plot_architecture_comparison(run_data: dict, output_dir: Path):
    """Generates and saves a publication-quality plot for training loss."""
    metric_key = "train/loss"
    plot_title = "Top-K Architecture Comparison"
    log.info(f"Generating plot for: {plot_title}")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = assign_arch_colors(run_data.keys())

    for run_name, filepath in run_data.items():
        try:
            df = pd.read_csv(filepath)
            if metric_key in df.columns:
                clean_df = df[["_step", metric_key]].dropna()
                proportional_step = clean_df["_step"] / clean_df["_step"].max()
                run_color = colors.get(run_name, "gray")

                # Simulate log scale for y-axis
                ema = np.log10(clean_df[metric_key].ewm(span=15, adjust=False).mean())
                ax.plot(
                    proportional_step, ema, color=run_color,
                    linewidth=2.0, alpha=0.7, label=run_name
                )
            else:
                log.warning(f"Metric '{metric_key}' not found in {filepath}. Skipping.")
        except FileNotFoundError:
            log.error(f"File not found: {filepath}. Skipping this run.")
        except Exception as e:
            log.error(f"Could not process {filepath}. Error: {e}")

    ax.set_title(plot_title, fontsize=30, weight='bold')
    ax.set_xlabel("Proportion of Training", fontsize=26)
    ax.set_ylabel("Training Loss (Log Scale)", fontsize=26, labelpad=30)
    ax.tick_params(axis='x', which='major', labelsize=26)
    ax.set_yticklabels([])
    ax.set_yticks([])

    # Add horizontal grid lines with labels for simulated log scale
    for y in [2, 3, 6, 10]:
        ax.axhline(y=np.log10(y), color='#d3d3d3', linestyle='-', linewidth=0.8, zorder=1)
        ax.text(0, np.log10(y), str(y), transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=26)

    # Add vertical grid lines
    for x in [0.2, 0.4, 0.6, 0.8]:
        ax.axvline(x=x, color='#d3d3d3', linestyle='-', linewidth=0.8, zorder=1)

    ax.legend(fontsize=24, fancybox=True, frameon=True, shadow=True)
    ax.set_xlim(left=0, right=1)

    output_filename_pdf = "architecture_comparison_training_loss.pdf"
    output_path_pdf = output_dir / output_filename_pdf
    fig.savefig(output_path_pdf, format="pdf", bbox_inches="tight")

    output_filename_svg = "architecture_comparison_training_loss.svg"
    output_path_svg = output_dir / output_filename_svg
    fig.savefig(output_path_svg, format="svg", bbox_inches="tight")

    plt.close(fig)
    log.info(f"✅ Architecture comparison plot saved to: {output_path_pdf} and {output_path_svg}")


# --- Combined Losses Plot ---
PRIOR_ABLATION_RUNS = {
    "SDT Prior Factor = 0.0625": {
        "common": "wandb_exports/PRETRAINTESTqwen2.50.5Bvprpretrain_mix20250828_132910gamma0.5_common_metrics.csv",
    },
    "SDT Prior Factor = 0.125": {
        "common": "wandb_exports/LARGERPRIORTESTqwen2.50.5Bvprpretrain_mix20250902_031859gamma0.5_common_metrics.csv",
    },
    "SDT Prior Factor = 0.25": {
        "common": "wandb_exports/LARGERPRIORTESTqwen2.50.5Bvprpretrain_mix20250902_140547gamma0.5_common_metrics.csv",
    },
    "SDT Prior Factor = 0.5": {
        "common": "wandb_exports/LARGERPRIORTESTqwen2.50.5Bvprpretrain_mix20250903_005619gamma0.5_common_metrics.csv",
    },
}

SPECIFIC_LOSSES_TO_PLOT = {
    "STT Prior Factor = 0.0625": (
        "wandb_exports/experimentstt20250923_0400350.5B_dynamic_metrics.csv",
        "train/loss/stt_tpn_loss_unscaled"
    ),
}

def assign_combined_colors(run_names):
    """Assigns colors for the combined plot."""
    colors = {}
    prior_palette = plt.cm.get_cmap('viridis', len(PRIOR_ABLATION_RUNS))
    for i, name in enumerate(PRIOR_ABLATION_RUNS.keys()):
        colors[name] = prior_palette(i)
    colors["STT Prior Factor = 0.0625"] = "green"
    return colors

def plot_combined_losses(prior_runs: dict, specific_losses: dict, output_dir: Path):
    """Generates a combined plot of prior network auxiliary loss and specific training losses."""
    plot_title = "Comparison of Change Priors"
    log.info(f"Generating plot for: {plot_title}")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    all_run_names = list(prior_runs.keys()) + list(specific_losses.keys())
    colors = assign_combined_colors(all_run_names)

    # Plot prior factor ablation runs
    for run_name, paths in prior_runs.items():
        filepath = paths.get("common")
        if not filepath: continue
        try:
            df = pd.read_csv(filepath)
            metric_key = "train/prior_loss"
            if metric_key in df.columns:
                clean_df = df[["_step", metric_key]].dropna()
                proportional_step = clean_df["_step"] / clean_df["_step"].max()
                run_color = colors.get(run_name)

                ema = clean_df[metric_key].ewm(span=15, adjust=False).mean()
                ax.plot(proportional_step, ema, color=run_color, linewidth=2.0, label=run_name)
        except Exception as e:
            log.error(f"Could not process {filepath} for {run_name}. Error: {e}")

    # Plot specific losses
    for run_name, (filepath, metric_key) in specific_losses.items():
        try:
            df = pd.read_csv(filepath)
            if metric_key in df.columns:
                clean_df = df[["_step", metric_key]].dropna()
                proportional_step = clean_df["_step"] / clean_df["_step"].max()
                run_color = colors.get(run_name, "gray")

                ema = clean_df[metric_key].ewm(span=15, adjust=False).mean()
                ax.plot(proportional_step, ema, color=run_color, linewidth=2.0, alpha=0.8, label=run_name)
        except FileNotFoundError:
            log.error(f"File not found: {filepath}. Skipping this run.")
        except Exception as e:
            log.error(f"Could not process {filepath}. Error: {e}")

    ax.set_title(plot_title, fontsize=30, weight='bold')
    ax.set_xlabel("Proportion of Training", fontsize=26)
    ax.set_ylabel("Loss", fontsize=26)
    ax.legend(fontsize=18, fancybox=True, frameon=True, shadow=True)
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.set_xlim(left=0, right=1)
    fig.tight_layout()

    output_filename_pdf = "combined_losses_comparison.pdf"
    output_path_pdf = output_dir / output_filename_pdf
    fig.savefig(output_path_pdf, format="pdf", bbox_inches="tight")

    output_filename_svg = "combined_losses_comparison.svg"
    output_path_svg = output_dir / output_filename_svg
    fig.savefig(output_path_svg, format="svg", bbox_inches="tight")

    plt.close(fig)
    log.info(f"✅ Combined losses plot saved to: {output_path_pdf} and {output_path_svg}")

import shutil

def main():
    """Main function to orchestrate the plotting process."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"✨ Generating all paper figures in {OUTPUT_DIR} ✨")

    # Generate the plots
    plot_inferred_selected_scatter(ROUTER_STATS_DATA_FILE, OUTPUT_DIR)
    plot_architecture_comparison(ARCH_RUNS_TO_PLOT, OUTPUT_DIR)
    plot_combined_losses(PRIOR_ABLATION_RUNS, SPECIFIC_LOSSES_TO_PLOT, OUTPUT_DIR)

    log.info("✨ All paper figures generated successfully! ✨")

if __name__ == "__main__":
    main()

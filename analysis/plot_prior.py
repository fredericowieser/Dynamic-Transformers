"""
Generates and saves a complete set of publication-quality comparison plots for a
prior factor ablation study, covering both common training metrics and VPR-specific
dynamic metrics from multiple W&B runs.
"""

import logging
import sys
from pathlib import Path
import re

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


# --- 1. CONFIGURE YOUR RUNS HERE ---
# Add the runs you want to plot to this dictionary.
# The key is the name that will appear in the plot's legend.
# The value is a dictionary containing paths to BOTH metrics files.
RUNS_TO_PLOT = {
    "Prior Factor = 0.0625": {
        "common": "wandb_exports/PRETRAINTESTqwen2.50.5Bvprpretrain_mix20250828_132910gamma0.5_common_metrics.csv",
        "dynamic": "wandb_exports/PRETRAINTESTqwen2.50.5Bvprpretrain_mix20250828_132910gamma0.5_dynamic_metrics.csv",
    },
    "Prior Factor = 0.125": {
        "common": "wandb_exports/LARGERPRIORTESTqwen2.50.5Bvprpretrain_mix20250902_031859gamma0.5_common_metrics.csv",
        "dynamic": "wandb_exports/LARGERPRIORTESTqwen2.50.5Bvprpretrain_mix20250902_031859gamma0.5_dynamic_metrics.csv",
    },
    "Prior Factor = 0.25": {
        "common": "wandb_exports/LARGERPRIORTESTqwen2.50.5Bvprpretrain_mix20250902_140547gamma0.5_common_metrics.csv",
        "dynamic": "wandb_exports/LARGERPRIORTESTqwen2.50.5Bvprpretrain_mix20250902_140547gamma0.5_dynamic_metrics.csv",
    },
    "Prior Factor = 0.5": {
        "common": "wandb_exports/LARGERPRIORTESTqwen2.50.5Bvprpretrain_mix20250903_005619gamma0.5_common_metrics.csv",
        "dynamic": "wandb_exports/LARGERPRIORTESTqwen2.50.5Bvprpretrain_mix20250903_005619gamma0.5_dynamic_metrics.csv",
    },
}

# --- 2. SCRIPT CONFIGURATION ---
OUTPUT_DIR = Path("./plots_for_paper_final")
# Use descriptive titles for the plots
COMMON_METRICS_TO_PLOT = {
    "train/loss": "Training Loss",
    "train/perplexity": "Training Perplexity",
    "val/loss": "Validation Loss",
    "train/prior_loss": "Prior Network Auxiliary Loss",
}

# --- COLOR ASSIGNMENT ---
def assign_colors(run_names):
    """
    Assigns a unique, predictable color to each run, ensuring a consistent
    red, yellow, green, blue progression based on the prior factor.
    """
    def extract_prior_factor(run_name):
        match = re.search(r"=\s*([\d.]+)", run_name)
        return float(match.group(1)) if match else float('inf')

    sorted_names = sorted(run_names, key=extract_prior_factor)
    color_palette = ['#D62728', '#FFBF00', '#2CA02C', '#1F77B4'] # Red, Amber, Green, Blue
    return {name: color for name, color in zip(sorted_names, color_palette)}

# --- PLOTTING FUNCTIONS FOR COMMON METRICS ---

def create_common_metric_plot(metric_key: str, plot_title: str, run_data: dict, colors: dict):
    log.info(f"Generating common metric plot for: {plot_title}")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    for run_name, paths in run_data.items():
        filepath = paths.get("common")
        if not filepath: continue
        try:
            df = pd.read_csv(filepath)
            if metric_key in df.columns:
                clean_df = df[["_step", metric_key]].dropna()
                run_color = colors.get(run_name)

                if metric_key == "val/loss":
                    ax.plot(clean_df["_step"], clean_df[metric_key], marker='x', linestyle='-', color=run_color, markersize=8, linewidth=2.5, label=run_name)
                else:
                    ax.plot(clean_df["_step"], clean_df[metric_key], color=run_color, linewidth=1.5, alpha=0.2)
                    ema = clean_df[metric_key].ewm(span=15, adjust=False).mean()
                    ax.plot(clean_df["_step"], ema, color=run_color, linewidth=3.0, label=run_name)
        except Exception as e:
            log.error(f"Could not process {filepath} for {run_name}. Error: {e}")

    ax.set_title(f"Effect of Prior Factor on {plot_title}", fontsize=24, weight='bold')
    ax.set_xlabel("Training Step", fontsize=20)
    ax.set_ylabel(plot_title, fontsize=20)
    ax.legend(fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlim(left=0)

    if "perplexity" in metric_key.lower() or "loss" in metric_key.lower():
        ax.set_yscale("log")
        ax.set_ylabel(f"{plot_title} (Log Scale)", fontsize=20)

    output_filename = f"prior_factor_{plot_title.replace(' ', '_').lower()}_comparison.pdf"
    output_path = OUTPUT_DIR / output_filename
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"✅ Common metric plot saved to: {output_path}")

# --- PLOTTING FUNCTIONS FOR DYNAMIC (VPR) METRICS ---

def plot_combined_gating_signals_quadrant(run_data: dict, colors: dict):
    log.info("Generating combined quadrant plot for gating signal components...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(22, 14), sharex=True, sharey=True)
    axes = axes.flatten()

    signal_colors = {
        "S_CU": {"line": "#00C0B0", "fill": "#5CFFE4"},      # Dark Blue line, Light Sky Blue fill
        "S_CE": {"line": "#FF00FF", "fill": "#FF77D2"},      # Fuchsia line, Less harsh Light Pink fill
    }
    overlap_color = "#7B21BB" # Indigo

    sorted_run_names = sorted(run_data.keys(), key=lambda name: float(re.search(r"=\s*([\d.]+)", name).group(1)))

    for i, run_name in enumerate(sorted_run_names):
        ax = axes[i]
        paths = run_data[run_name]
        filepath = paths.get("dynamic")
        if not filepath: continue
        try:
            df = pd.read_csv(filepath)
            run_color = colors.get(run_name)

            # Prepare dataframes
            gcont_df = df[["_step", "G_cont_mean", "G_cont_min", "G_cont_max"]].dropna()
            scu_df = df[["_step", "S_CU_mean", "S_CU_min", "S_CU_max"]].dropna()
            sce_df = df[["_step", "S_CE_mean", "S_CE_min", "S_CE_max"]].dropna()

            # Merge to align steps for overlap calculation
            merged_df = pd.merge(scu_df, sce_df, on="_step", suffixes=('_cu', '_ce'))
            overlap_min = np.maximum(merged_df["S_CU_min"], merged_df["S_CE_min"])
            overlap_max = np.minimum(merged_df["S_CU_max"], merged_df["S_CE_max"])

            # Plotting order: backgrounds first, then overlap, then lines
            ax.fill_between(gcont_df["_step"], gcont_df["G_cont_min"], gcont_df["G_cont_max"], color=run_color, alpha=0.25, zorder=0)
            ax.fill_between(merged_df["_step"], merged_df["S_CU_min"], merged_df["S_CU_max"], color=signal_colors["S_CU"]["fill"], alpha=0.4, zorder=1)
            ax.fill_between(merged_df["_step"], merged_df["S_CE_min"], merged_df["S_CE_max"], color=signal_colors["S_CE"]["fill"], alpha=0.5, zorder=2)
            ax.fill_between(merged_df["_step"], overlap_min, overlap_max, where=overlap_max > overlap_min, color=overlap_color, alpha=0.5, zorder=3)
            
            # Plot lines on top
            ax.plot(gcont_df["_step"], gcont_df["G_cont_mean"], color=run_color, linewidth=3.5, zorder=5)
            ax.plot(scu_df["_step"], scu_df["S_CU_mean"], color=signal_colors["S_CU"]["line"], linewidth=3.0, zorder=4)
            ax.plot(sce_df["_step"], sce_df["S_CE_mean"], color=signal_colors["S_CE"]["line"], linewidth=3.0, zorder=4)

            ax.set_title(run_name, fontsize=22)
            ax.tick_params(axis='both', which='major', labelsize=18)
            ax.set_xlim(left=0)
        except Exception as e:
            log.error(f"Could not process {filepath} for {run_name}. Error: {e}")
    
    fig.suptitle("Analysis of Gating Signal Components by Prior Factor", fontsize=28, weight='bold', y=0.98)
    fig.text(0.5, 0.03, 'Training Step', ha='center', va='center', fontsize=22)
    fig.text(0.06, 0.5, 'Signal Activation', ha='center', va='center', rotation='vertical', fontsize=22)
    fig.tight_layout(rect=[0.07, 0.05, 1, 0.95])
    
    output_path = OUTPUT_DIR / "prior_factor_combined_gating_signals.pdf"
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"✅ Combined gating signal plot saved to: {output_path}")


def plot_router_parameters_separately(run_data: dict, colors: dict):
    log.info("Generating separate plots for router parameters...")
    params = {
        "beta_ce_mean": r"Evolution of Gating Temperature ($\beta_{CE}$)",
        "beta_cu_mean": r"Evolution of Gating Temperature ($\beta_{CU}$)",
        "cu_multiplier_mean": r"Evolution of Novelty Multiplier ($m_{CU}$)",
        "ce_offset_mean": r"Evolution of Prediction Offset ($o_{CE}$)",
    }

    for param_col, param_title in params.items():
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for run_name, paths in run_data.items():
            filepath = paths.get("dynamic")
            if not filepath: continue
            try:
                df = pd.read_csv(filepath)
                if param_col in df.columns:
                    clean_df = df[["_step", param_col]].dropna()
                    ax.plot(clean_df["_step"], clean_df[param_col], linewidth=3.5, label=run_name, color=colors.get(run_name))
            except Exception as e:
                log.error(f"Could not process {filepath} for {run_name}. Error: {e}")

        ax.set_title(param_title, fontsize=24, weight='bold')
        ax.set_xlabel("Training Step", fontsize=20)
        ax.set_ylabel("Learned Parameter Value", fontsize=20)
        ax.legend(fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlim(left=0)

        output_filename = f"prior_factor_{param_col}_comparison.pdf"
        output_path = OUTPUT_DIR / output_filename
        fig.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        log.info(f"✅ Router parameter plot saved to: {output_path}")

# --- MAIN ORCHESTRATION ---
def main():
    """Main function to orchestrate the entire plotting process."""
    if not RUNS_TO_PLOT:
        log.warning("The 'RUNS_TO_PLOT' dictionary is empty. Nothing to plot.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    colors = assign_colors(RUNS_TO_PLOT.keys())

    log.info("--- Starting Common Metric Plots ---")
    for metric_key, plot_title in COMMON_METRICS_TO_PLOT.items():
        create_common_metric_plot(metric_key, plot_title, RUNS_TO_PLOT, colors)
    
    log.info("--- Starting Dynamic VPR Metric Plots ---")
    plot_combined_gating_signals_quadrant(RUNS_TO_PLOT, colors)
    plot_router_parameters_separately(RUNS_TO_PLOT, colors)
    
    log.info("✨ All plots for the paper generated successfully! ✨")

if __name__ == "__main__":
    main()
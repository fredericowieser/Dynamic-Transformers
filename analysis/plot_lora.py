"""
Generates and saves a complete set of publication-quality comparison plots for
an ablation study on adaptation methods (Full Finetuning vs. LoRA), covering
both common and dynamic VPR metrics.
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


# --- 1. CONFIGURE YOUR RUNS HERE ---
# Add the runs you want to compare.
RUNS_TO_PLOT = {
    "Finetune (Dynamic)": {
        "common": "wandb_exports/PRETRAINTESTqwen2.50.5Bvprpretrain_mix20250828_132910gamma0.5_common_metrics.csv",
        "dynamic": "wandb_exports/PRETRAINTESTqwen2.50.5Bvprpretrain_mix20250828_132910gamma0.5_dynamic_metrics.csv",
    },
    "LoRA (Dynamic)": {
        "common": "wandb_exports/LORATESTqwen2.50.5Bvprpretrain_mix20250901_150242gamma0.5_common_metrics.csv",
        "dynamic": "wandb_exports/LORATESTqwen2.50.5Bvprpretrain_mix20250901_150242gamma0.5_dynamic_metrics.csv",
    },
}

# --- 2. SCRIPT CONFIGURATION ---
OUTPUT_DIR = Path("./plots_for_paper_final")
COMMON_METRICS_TO_PLOT = {
    "train/loss": "Training Loss",
    "train/perplexity": "Training Perplexity",
    "val/loss": "Validation Loss",
    "train/prior_loss": "Prior Network Auxiliary Loss",
}

# --- COLOR ASSIGNMENT ---
def assign_colors(run_names):
    """Assigns specific colors for the Finetune vs. LoRA comparison."""
    # Specific color mapping for this ablation study
    color_map = {
        "Finetune (Dynamic)": "#D62728",  # A strong red
        "LoRA (Dynamic)": "#FF69B4",     # Hot Pink
    }
    # Fallback for any unexpected run names
    return {name: color_map.get(name, "gray") for name in run_names}

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
                    ax.plot(clean_df["_step"], clean_df[metric_key], color=run_color, linewidth=1.0, alpha=0.25)
                    ema = clean_df[metric_key].ewm(span=15, adjust=False).mean()
                    ax.plot(clean_df["_step"], ema, color=run_color, linewidth=2.5, label=run_name)
        except Exception as e:
            log.error(f"Could not process {filepath} for {run_name}. Error: {e}")

    ax.set_title(f"Adaptation Method Comparison: {plot_title}", fontsize=18, weight='bold')
    ax.set_xlabel("Training Step", fontsize=16)
    ax.set_ylabel(plot_title, fontsize=16)
    ax.legend(fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlim(left=0)

    if "perplexity" in metric_key.lower() or "loss" in metric_key.lower():
        ax.set_yscale("log")
        ax.set_ylabel(f"{plot_title} (Log Scale)", fontsize=16)

    output_filename = f"adaptation_comparison_{plot_title.replace(' ', '_').lower()}.pdf"
    output_path = OUTPUT_DIR / output_filename
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"✅ Common metric plot saved to: {output_path}")

# --- PLOTTING FUNCTIONS FOR DYNAMIC (VPR) METRICS ---

def plot_gating_signals_side_by_side(run_data: dict, colors: dict):
    log.info("Generating side-by-side plots for each gating signal...")
    signals_to_plot = {
        "S_CE": r"Analysis of Expected Event Signal ($S_{CE}$)",
        "S_CU": r"Analysis of Unexpected Event Signal ($S_{CU}$)",
        "G_cont": r"Analysis of Final Gating Signal ($G_{cont}$)",
    }

    for signal_key, signal_title in signals_to_plot.items():
        plt.style.use('seaborn-v0_8-whitegrid')
        # Create a 1x2 grid for side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
        axes = axes.flatten()

        for ax, (run_name, paths) in zip(axes, run_data.items()):
            filepath = paths.get("dynamic")
            if not filepath: continue
            try:
                df = pd.read_csv(filepath)
                mean_col, min_col, max_col = f"{signal_key}_mean", f"{signal_key}_min", f"{signal_key}_max"
                if all(c in df.columns for c in [mean_col, min_col, max_col]):
                    mean_df, min_df, max_df = df[["_step", mean_col]].dropna(), df[["_step", min_col]].dropna(), df[["_step", max_col]].dropna()
                    run_color = colors.get(run_name)
                    ax.plot(mean_df["_step"], mean_df[mean_col], color=run_color, linewidth=2.5)
                    ax.fill_between(mean_df["_step"], min_df[min_col], max_df[max_col], color=run_color, alpha=0.2)
                    ax.set_title(run_name, fontsize=16)
                    ax.tick_params(axis='both', which='major', labelsize=12)
                    ax.set_xlim(left=0)
            except Exception as e:
                log.error(f"Could not process {filepath} for {run_name}. Error: {e}")

        fig.suptitle(signal_title, fontsize=22, weight='bold')
        fig.text(0.5, 0.04, 'Training Step', ha='center', va='center', fontsize=18)
        fig.text(0.06, 0.5, 'Signal Activation', ha='center', va='center', rotation='vertical', fontsize=18)
        fig.tight_layout(rect=[0.07, 0.05, 1, 0.95])
        output_path = OUTPUT_DIR / f"adaptation_comparison_{signal_key}.pdf"
        fig.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        log.info(f"✅ Gating signal plot saved to: {output_path}")

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
                    ax.plot(clean_df["_step"], clean_df[param_col], linewidth=2.5, label=run_name, color=colors.get(run_name))
            except Exception as e:
                log.error(f"Could not process {filepath} for {run_name}. Error: {e}")

        ax.set_title(f"Adaptation Method Comparison: {param_title}", fontsize=18, weight='bold')
        ax.set_xlabel("Training Step", fontsize=16)
        ax.set_ylabel("Learned Parameter Value", fontsize=16)
        ax.legend(fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlim(left=0)

        output_filename = f"adaptation_comparison_{param_col}.pdf"
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
    plot_gating_signals_side_by_side(RUNS_TO_PLOT, colors)
    plot_router_parameters_separately(RUNS_TO_PLOT, colors)
    
    log.info("✨ All adaptation comparison plots generated successfully! ✨")

if __name__ == "__main__":
    main()

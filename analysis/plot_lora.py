"""
Generates and saves a complete set of publication-quality comparison plots for
an ablation study on adaptation methods (Full Finetuning vs. LoRA), covering
both common and dynamic VPR metrics, now with overlap visualization.
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# Configure runs to compare
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

# Output configuration
OUTPUT_DIR = Path("./plots_for_paper_final")
COMMON_METRICS_TO_PLOT = {
    "train/loss": "Training Loss",
    "train/perplexity": "Training Perplexity",
    "val/loss": "Validation Loss",
    "train/prior_loss": "Prior Network Auxiliary Loss",
}

def assign_colors(run_names):
    """Assigns specific colors for the Finetune vs. LoRA comparison."""
    color_map = {
        "Finetune (Dynamic)": "#D62728",  # A strong red
        "LoRA (Dynamic)": "#9467BD",     # A strong purple
    }
    return {name: color_map.get(name, "gray") for name in run_names}


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
                    ax.plot(clean_df["_step"], clean_df[metric_key], marker='x', linestyle='-', color=run_color, markersize=10, linewidth=3.5, label=run_name)
                else:
                    ax.plot(clean_df["_step"], clean_df[metric_key], color=run_color, linewidth=1.5, alpha=0.2)
                    ema = clean_df[metric_key].ewm(span=15, adjust=False).mean()
                    ax.plot(clean_df["_step"], ema, color=run_color, linewidth=3.5, label=run_name)
        except Exception as e:
            log.error(f"Could not process {filepath} for {run_name}. Error: {e}")

    ax.set_title(f"Adaptation Method Comparison: {plot_title}", fontsize=24, weight='bold')
    ax.set_xlabel("Training Step", fontsize=20)
    ax.set_ylabel(plot_title, fontsize=20)
    ax.legend(fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlim(left=0)

    if "perplexity" in metric_key.lower() or "loss" in metric_key.lower():
        ax.set_yscale("log")
        ax.set_ylabel(f"{plot_title} (Log Scale)", fontsize=20)

    output_filename_pdf = f"adaptation_comparison_{plot_title.replace(' ', '_').lower()}.pdf"
    output_path_pdf = OUTPUT_DIR / output_filename_pdf
    fig.savefig(output_path_pdf, format="pdf", bbox_inches="tight")

    output_filename_svg = f"adaptation_comparison_{plot_title.replace(' ', '_').lower()}.svg"
    output_path_svg = OUTPUT_DIR / output_filename_svg
    fig.savefig(output_path_svg, format="svg", bbox_inches="tight")

    plt.close(fig)
    log.info(f"✅ Common metric plots saved to: {output_path_pdf} and {output_path_svg}")


def plot_combined_gating_signals(run_data: dict, colors: dict):
    log.info("Generating combined plot for gating signal components with overlap...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    signal_colors = {
        "S_CU": {"line": "#00C0B0", "fill": "#5CFFE4"},
        "S_CE": {"line": "#FF00FF", "fill": "#FF77D2"},
    }
    overlap_color = "#7B21BB"

    for ax, (run_name, paths) in zip(axes, run_data.items()):
        filepath = paths.get("dynamic")
        if not filepath: continue
        try:
            df = pd.read_csv(filepath)
            run_color = colors.get(run_name)

            # Prepare data for plotting
            gcont_df = df[["_step", "G_cont_mean", "G_cont_min", "G_cont_max"]].dropna()
            scu_df = df[["_step", "S_CU_mean", "S_CU_min", "S_CU_max"]].dropna()
            sce_df = df[["_step", "S_CE_mean", "S_CE_min", "S_CE_max"]].dropna()

            # Calculate signal overlap
            merged_df = pd.merge(scu_df, sce_df, on="_step", suffixes=('_cu', '_ce'))
            overlap_min = np.maximum(merged_df["S_CU_min"], merged_df["S_CE_min"])
            overlap_max = np.minimum(merged_df["S_CU_max"], merged_df["S_CE_max"])

            # Plot backgrounds, overlap, then lines
            ax.fill_between(gcont_df["_step"], gcont_df["G_cont_min"], gcont_df["G_cont_max"], color=run_color, alpha=0.25, zorder=0)
            ax.fill_between(merged_df["_step"], merged_df["S_CU_min"], merged_df["S_CU_max"], color=signal_colors["S_CU"]["fill"], alpha=0.4, zorder=1)
            ax.fill_between(merged_df["_step"], merged_df["S_CE_min"], merged_df["S_CE_max"], color=signal_colors["S_CE"]["fill"], alpha=0.5, zorder=2)
            ax.fill_between(merged_df["_step"], overlap_min, overlap_max, where=overlap_max > overlap_min, color=overlap_color, alpha=0.5, zorder=3)

            # Plot signal lines
            ax.plot(gcont_df["_step"], gcont_df["G_cont_mean"], color=run_color, linewidth=3.5, zorder=5)
            ax.plot(scu_df["_step"], scu_df["S_CU_mean"], color=signal_colors["S_CU"]["line"], linewidth=3.0, zorder=4)
            ax.plot(sce_df["_step"], sce_df["S_CE_mean"], color=signal_colors["S_CE"]["line"], linewidth=3.0, zorder=4)

            ax.set_title(run_name, fontsize=22)
            ax.tick_params(axis='both', which='major', labelsize=18)
            ax.set_xlim(left=0)

        except Exception as e:
            log.error(f"Could not process {filepath} for {run_name}. Error: {e}")

    fig.suptitle("Analysis of Gating Signal Components", fontsize=28, weight='bold', y=0.98)
    fig.text(0.5, 0.02, 'Training Step', ha='center', va='center', fontsize=22)
    fig.text(0.07, 0.5, 'Signal Activation', ha='center', va='center', rotation='vertical', fontsize=22)
    fig.tight_layout(rect=[0.08, 0.04, 1, 0.95])

    output_path_pdf = OUTPUT_DIR / "adaptation_comparison_combined_gating_signals.pdf"
    fig.savefig(output_path_pdf, format="pdf", bbox_inches="tight")

    output_path_svg = OUTPUT_DIR / "adaptation_comparison_combined_gating_signals.svg"
    fig.savefig(output_path_svg, format="svg", bbox_inches="tight")

    plt.close(fig)
    log.info(f"✅ Combined gating signal plots saved to: {output_path_pdf} and {output_path_svg}")


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

        ax.set_title(f"Adaptation Method Comparison: {param_title}", fontsize=24, weight='bold')
        ax.set_xlabel("Training Step", fontsize=20)
        ax.set_ylabel("Learned Parameter Value", fontsize=20)
        ax.legend(fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlim(left=0)

        output_filename_pdf = f"adaptation_comparison_{param_col}.pdf"
        output_path_pdf = OUTPUT_DIR / output_filename_pdf
        fig.savefig(output_path_pdf, format="pdf", bbox_inches="tight")

        output_filename_svg = f"adaptation_comparison_{param_col}.svg"
        output_path_svg = OUTPUT_DIR / output_filename_svg
        fig.savefig(output_path_svg, format="svg", bbox_inches="tight")

        plt.close(fig)
        log.info(f"✅ Router parameter plots saved to: {output_path_pdf} and {output_path_svg}")

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
    plot_combined_gating_signals(RUNS_TO_PLOT, colors)
    plot_router_parameters_separately(RUNS_TO_PLOT, colors)
    
    log.info("✨ All adaptation comparison plots generated successfully! ✨")

if __name__ == "__main__":
    main()
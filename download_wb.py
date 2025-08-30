"""
Downloads historical metric data for a specific Weights & Biases run
and saves it to CSV files for analysis.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import wandb
from wandb.apis.public import Run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def sanitize_filename(name: str) -> str:
    """Removes invalid characters from a string to make it a valid filename."""
    return "".join(c for c in name if c.isalnum() or c in (' ', '.', '_')).rstrip()


def get_run_data(api: wandb.Api, run_path: str) -> Run | None:
    """Safely fetches a run object from the wandb API."""
    try:
        run = api.run(run_path)
        log.info(f"✅ Successfully fetched run: '{run.name}'")
        return run
    except wandb.errors.CommError as e:
        log.error(f"❌ Error fetching run '{run_path}'. Please check the path and your permissions.")
        log.error(f"   Details: {e}")
        return None


def process_and_save_data(run: Run, output_dir: Path):
    """Processes run history into common and dynamic metrics and saves them to CSV."""
    log.info("Downloading metrics history... (this may take a moment for long runs)")
    run_history_df = run.history()
    log.info(f"Found {len(run_history_df)} steps in the run history.")

    # --- Identify Common and Dynamic Metric Columns ---
    # Metrics are logged with prefixes like 'train_vpr_signals/'
    vpr_prefixes = ("train_vpr_signals/", "train_vpr_router/")
    
    common_cols = [
        "_step", "train/loss", "train/lm_loss", "train/perplexity",
        "val/loss", "lr", "train/prior_loss"
    ]
    
    # Filter for existing columns only to avoid errors
    existing_common_cols = [col for col in common_cols if col in run_history_df.columns]
    dynamic_cols = [col for col in run_history_df.columns if col.startswith(vpr_prefixes)]
    
    # --- Create DataFrames ---
    common_df = run_history_df[existing_common_cols].set_index("_step")
    
    # --- Generate Filenames ---
    base_filename = sanitize_filename(run.name)
    common_csv_path = output_dir / f"{base_filename}_common_metrics.csv"
    
    # --- Save Common Metrics ---
    common_df.to_csv(common_csv_path)
    log.info(f"📈 Saved common metrics to: {common_csv_path}")

    # --- Save Dynamic Metrics if they exist ---
    if dynamic_cols:
        dynamic_df = run_history_df[["_step"] + dynamic_cols].set_index("_step")
        # Clean up column names by removing prefixes
        dynamic_df.columns = [
            col.replace("train_vpr_signals/", "").replace("train_vpr_router/", "")
            for col in dynamic_df.columns
        ]
        dynamic_csv_path = output_dir / f"{base_filename}_dynamic_metrics.csv"
        dynamic_df.to_csv(dynamic_csv_path)
        log.info(f"🔬 Saved dynamic metrics to: {dynamic_csv_path}")
    else:
        log.info("No VPR-specific metrics found to save.")


def main():
    """Parses arguments and orchestrates the download process."""
    parser = argparse.ArgumentParser(
        description="Download metrics from a Weights & Biases run to CSV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "run_path",
        type=str,
        help="The full path to the W&B run, in the format 'entity/project/run_id'.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./wandb_exports"),
        help="The directory where CSV files will be saved.",
    )
    args = parser.parse_args()

    # Ensure the output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Initialize W&B API and Fetch Run ---
    api = wandb.Api()
    run = get_run_data(api, args.run_path)

    if run:
        process_and_save_data(run, args.output_dir)
        log.info("✨ Download complete.")


if __name__ == "__main__":
    main()

"""
### ⚙️ How to Use the Script

1.  **Install `pandas`**:
    This script requires the `pandas` library to handle the data.
    ```bash
    pip install pandas
    ```

2.  **Find Your Run Path**:
    You need to provide the script with the path to your run, which consists of three parts: `entity/project/run_id`.
    * Go to your project page on the W&B website.
    * Click on a specific run in the table.
    * The URL will look something like this: `https://wandb.ai/your-entity/your-project/runs/run-id-string`
    * From this, your run path is `your-entity/your-project/run-id-string`.

    

3.  **Run the Script**:
    Execute the script from your terminal, providing the full run path.

    **Example:**
    ```bash
    python download_wb.py "huawei-noahs-ark/Dynamic-Transformers/g758wgvq"
"""

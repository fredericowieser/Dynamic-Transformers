
import pandas as pd
import re
import numpy as np

ROUTER_STATS_DATA_FILE = "wandb_exports/experimentstt20250923_1727080.5B_dynamic_metrics.csv"

def calculate_savings():
    """Calculates the savings based on the final inferred selected values."""
    try:
        df = pd.read_csv(ROUTER_STATS_DATA_FILE)
    except FileNotFoundError:
        print(f"File not found: {ROUTER_STATS_DATA_FILE}")
        return

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
            final_values.append(final_value)

    gamma_bar = np.mean(final_values)
    
    self_attention_savings = (1 - gamma_bar**2) / 2
    memory_savings = (1 - gamma_bar) / 2
    
    print(f"Average processing capacity (gamma_bar): {gamma_bar:.4f}")
    print(f"Self-attention savings: {self_attention_savings:.4f}")
    print(f"Memory savings: {memory_savings:.4f}")

if __name__ == "__main__":
    calculate_savings()

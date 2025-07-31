# eval_main.py
import argparse
from src.eval.config import load_model_config, save_results
from src.eval.utils import load_model_and_tokenizer
from src.eval.benchmarks import run_all_benchmarks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Dynamic-Transformer model.")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--device", default="cuda", help="Device: cpu, cuda")
    parser.add_argument("--max_eval_samples", type=int, default=512, help="Max samples per benchmark")
    parser.add_argument("--is_instruct", action="store_true", help="Model is instruct-tuned")
    parser.add_argument("--ce_bias", type=float, default=None, help="Override CE bias (default from config)")
    parser.add_argument("--dynamic_k", type=float, default=None, help="Override dynamic K (default from config)")
    parser.add_argument("--output_file", default="results.json", help="Output JSON file")
    args = parser.parse_args()

    # Load config and set defaults
    config = load_model_config(args.model_path)
    ce_bias = args.ce_bias if args.ce_bias is not None else config["ce_bias"]
    dynamic_k = args.dynamic_k if args.dynamic_k is not None else config["dynamic_k"]

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device, args.is_instruct)
    # Apply overrides if needed (e.g., to model config)
    if hasattr(model.config, "ce_bias"):
        model.config.ce_bias = ce_bias
    if hasattr(model.config, "dynamic_k"):
        model.config.dynamic_k = dynamic_k

    print(f"Running evaluation with CE bias: {ce_bias}, Dynamic K: {dynamic_k}")
    results = run_all_benchmarks(model, tokenizer, args.max_eval_samples, args.is_instruct)

    # Save full results, including averages
    save_results(args.output_file, results)
    print(f"Evaluation complete. Results saved to {args.output_file}")
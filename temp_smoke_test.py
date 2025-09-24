import argparse
import logging
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoConfig

from src.data.mixed_dataset import MixedDataset
from src.models.mod.model import MoDForCausalLM
from src.models.sdt.model import SDTForCausalLM
from src.models.stt.model import STTForCausalLM
from src.models.standard.model import StandardTransformerForCausalLM

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run a diagnostic smoke test on a model's validation forward pass.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model directory.")
    args = parser.parse_args()

    log.info(f"Starting smoke test for model at: {args.model_path}")

    # --- Load Model ---
    log.info("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    # Force model_type to stt if it's qwen2, to bypass the saving bug for old checkpoints
    if config.model_type == 'qwen2':
        log.warning("Loaded config has model_type 'qwen2'. Overriding to 'stt' for this test.")
        config.model_type = 'stt'

    model_type = getattr(config, "model_type", "standard")
    model_class_map = {
        "standard": StandardTransformerForCausalLM,
        "mod": MoDForCausalLM,
        "sdt": SDTForCausalLM,
        "stt": STTForCausalLM,
    }
    model_class = model_class_map.get(model_type)
    if not model_class:
        raise ValueError(f"Unknown model type '{model_type}' in config.")

    log.info(f"Explicitly loading model class: {model_class.__name__}")
    model = model_class.from_pretrained(
        args.model_path,
        config=config, # Pass the potentially modified config
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval() # Set to evaluation mode

    # --- Load Data ---
    log.info("Loading validation data...")
    # Load the base training config to get dataset parameters
    cfg = OmegaConf.load('config/default.yaml')
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    datamodule = MixedDataset(
        dataset_configs=cfg.data.dataset_configs,
        tokenizer_name=cfg.data.tokenizer_name,
        block_size=cfg.data.block_size,
        batch_size=cfg.data.batch_size,
        validation_split_percentage=cfg.data.validation_split_percentage,
    )
    datamodule.setup()
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    eval_loader = torch.utils.data.DataLoader(
        datamodule.val_dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=cfg.data.batch_size,
    )

    log.info("Fetching one batch from the validation set...")
    try:
        batch = next(iter(eval_loader))
    except StopIteration:
        log.error("Validation data loader is empty. Cannot run test.")
        return

    batch = {k: v.to(device) for k, v in batch.items()}
    log.info(f"Batch loaded to device: {device}. Keys: {batch.keys()}, Shapes: {{k: v.shape for k, v in batch.items()}}")

    # --- Run Forward Pass ---
    log.info("--- Initiating validation forward pass on ALL batches... ---")
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            log.info(f"--- Processing Batch {i+1}/{len(eval_loader)} ---")
            batch = {k: v.to(device) for k, v in batch.items()}

            try:
                outputs = model(**batch)
                loss = outputs.get('loss')
                
                if loss is not None:
                    log.info(f"Batch {i+1} Loss: {loss.item()}")
                    if torch.isinf(loss) or torch.isnan(loss):
                        log.error(f"FATAL: Infinite or NaN loss detected in batch {i+1}. This is the poison pill!")
                        break # Stop after finding the bad batch
                else:
                    log.warning(f"Loss not found in outputs for batch {i+1}.")

            except Exception as e:
                log.error(f"An exception occurred during the forward pass on batch {i+1}: {e}", exc_info=True)
                break

if __name__ == "__main__":
    main()

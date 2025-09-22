import torch
import hydra
from pathlib import Path
from omegaconf import DictConfig
from transformers import AutoTokenizer
from src.training.utils import create_model

@hydra.main(config_path="config", config_name="infer", version_base="1.3")
def main(cfg: DictConfig):
    """Loads a model checkpoint and generates text from user prompts."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load Model
    print(f"Loading model checkpoint from: {cfg.checkpoint_path}")
    model = create_model(
        cfg.model.type,
        cfg.model.size,
        from_scratch=False,
        cfg=cfg
    )
    state_dict = torch.load(Path(cfg.checkpoint_path) / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("\nModel loaded. Enter a prompt to generate text. Type 'exit' to quit.")
    
    while True:
        prompt_text = input("Prompt: ")
        if prompt_text.lower() == 'exit':
            break

        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=cfg.generation.max_new_tokens,
                do_sample=cfg.generation.do_sample,
                temperature=cfg.generation.temperature,
                top_k=cfg.generation.top_k,
                top_p=cfg.generation.top_p,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n--- Generated Text ---")
        print(generated_text)
        print("----------------------\n")

if __name__ == "__main__":
    main()
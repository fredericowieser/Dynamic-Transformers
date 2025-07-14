import torch
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, data_handler, config, logger):
        self.model = model
        self.optimizer = optimizer
        self.data_handler = data_handler
        self.config = config
        self.logger = logger

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.config.training_params.eval_iters)
            for k in range(self.config.training_params.eval_iters):
                X, Y = self.data_handler.get_batch(split)
                # Standard Hugging Face models take 'labels' for loss calculation
                outputs = self.model(X, labels=Y)
                losses[k] = outputs.loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def train(self):
        self.model.train()
        pbar = tqdm(range(self.config.training_params.max_iters))
        for it in pbar:
            # Evaluation and summary logging
            if it % self.config.training_params.eval_interval == 0 or it == self.config.training_params.max_iters - 1:
                losses = self.estimate_loss()
                self.logger.info(f"Step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                self.logger.log_metrics({
                    'loss/train': losses['train'],
                    'loss/val': losses['val']
                }, step=it)

            # Training step
            xb, yb = self.data_handler.get_batch("train")
            
            # The forward pass for a standard HF model
            outputs = self.model(xb, labels=yb)
            loss = outputs.loss
            
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            # Per-step logging
            self.logger.log_metrics({'loss/step': loss.item()}, step=it)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
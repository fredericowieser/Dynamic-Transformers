import torch
import torch.nn.functional as F

def set_seed(seed):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_metrics(model, batch, global_step):
    """
    A helper function to calculate all relevant losses and metrics.
    """
    model_output = model(
        **batch,
        current_iter=global_step,
        return_dict=True
    )
    
    shift_logits = model_output.logits[..., :-1, :].contiguous()
    shift_labels = batch["labels"][..., 1:].contiguous()
    
    lm_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    total_loss = lm_loss

    perplexity = torch.exp(lm_loss)

    metrics = {
        "total_loss": total_loss,
        "lm_loss": lm_loss,
        "perplexity": perplexity,
    }

    # --- START: NEW METRIC EXTRACTION ---
    if hasattr(model_output, 'vpr_metrics') and model_output.vpr_metrics is not None:
        vpr_metrics = model_output.vpr_metrics
        prior_loss = vpr_metrics.pop("prior_loss") # Pop to handle it separately
        
        if prior_loss is not None:
            prior_loss_weight = model.module.config.prior_loss_weight if hasattr(model, 'module') else model.config.prior_loss_weight
            total_loss += prior_loss * prior_loss_weight
        
        # Update metrics dict with the rest of the VPR stats
        metrics["prior_loss"] = prior_loss
        metrics.update(vpr_metrics) 

    # Update total_loss in the metrics dict after potentially adding prior_loss
    metrics["total_loss"] = total_loss

    if hasattr(model_output, 'vpr_metrics') and model_output.vpr_metrics.get("gate_vectors_per_layer"):
        gate_vectors = model_output.vpr_metrics["gate_vectors_per_layer"]
        metrics["overall_gate_activation_mean"] = torch.stack([gv.mean() for gv in gate_vectors]).mean()
        metrics["per_layer_gate_stats"] = [
            {"mean": gv.mean(), "std": gv.std() if gv.numel() > 1 else torch.tensor(0.0)}
            for gv in gate_vectors
        ]
    else:
        metrics["overall_gate_activation_mean"] = torch.tensor(0.0)
        metrics["per_layer_gate_stats"] = []

    vpr_metrics = [
        "avg_ce_proportion", "avg_cu_proportion", "avg_beta_ce", "avg_beta_cu",
        "avg_cu_detection_multiplier", "avg_ce_criterion_offset", "combined_gating_signal_mean"
    ]
    for key in vpr_metrics:
        if hasattr(model_output, key):
            metrics[key] = getattr(model_output, key)

    return metrics

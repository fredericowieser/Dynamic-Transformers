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

    # Dynamic prior loss weighting
    prior_loss = None
    current_prior_loss_weight = 0.0

    if hasattr(model_output, 'vpr_metrics') and model_output.vpr_metrics is not None:
        vpr_metrics = model_output.vpr_metrics
        prior_loss = vpr_metrics.get("prior_loss")

        if prior_loss is not None:
            # Get model config (handle accelerator wrapper)
            config = model.module.config if hasattr(model, 'module') else model.config

            # Calculate weight based on schedule
            schedule_cfg = getattr(config, 'prior_loss_schedule', None)
            if schedule_cfg is not None:
                initial_w = schedule_cfg['initial_weight']
                final_w = schedule_cfg['final_weight']
                decay_steps = schedule_cfg['decay_steps']

                if global_step < decay_steps:
                    progress = global_step / decay_steps
                    current_prior_loss_weight = initial_w - progress * (initial_w - final_w)
                else:
                    current_prior_loss_weight = final_w
            else:
                # Default weight if no schedule specified
                current_prior_loss_weight = 0.1

            total_loss += prior_loss * current_prior_loss_weight
        
        metrics.update(vpr_metrics)

    # Include prior loss in total
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

    metrics["current_prior_loss_weight"] = current_prior_loss_weight

    return metrics

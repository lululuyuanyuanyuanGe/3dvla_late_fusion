import torch
import json
import numpy as np
import logging

logger = logging.getLogger(__name__)

def load_norm_stats(stats_path):
    """
    Loads normalization statistics from a JSON file.
    Expected format: {"action": {"mean": [...], "std": [...]}} or similar LeRobot format.
    """
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    # Handle top-level "norm_stats" wrapper
    if "norm_stats" in stats:
        stats = stats["norm_stats"]

    # Handle different stats formats
    if "action" in stats:
        return stats["action"]
    if "actions" in stats: # Support Libero/LeRobot plural key
        return stats["actions"]
    # Fallback for simple mean/std dict
    return stats

def normalize_action(action, stats):
    """
    Normalizes action using Gaussian statistics (mean/std).
    Args:
        action (np.ndarray or torch.Tensor): Raw action.
        stats (dict): Contains 'mean' and 'std'.
    """
    # Helper to convert list stats to numpy/tensor matches action type
    if isinstance(action, torch.Tensor):
        device = action.device
        mean = torch.tensor(stats['mean'], device=device, dtype=action.dtype)
        std = torch.tensor(stats['std'], device=device, dtype=action.dtype)
    else:
        mean = np.array(stats['mean'], dtype=action.dtype)
        std = np.array(stats['std'], dtype=action.dtype)
        
    # Handle dimension mismatch (e.g. Data=7dim, Stats=14dim)
    # If stats are larger, slice them to match data.
    if mean.shape[-1] > action.shape[-1]:
        # logger.warning(f"Slicing norm stats from {mean.shape[-1]} to {action.shape[-1]}")
        mean = mean[..., :action.shape[-1]]
        std = std[..., :action.shape[-1]]
        
    # eps to avoid division by zero
    std = std + 1e-8
    
    return (action - mean) / std

def unnormalize_action(action, stats):
    """
    Un-normalizes action.
    """
    if isinstance(action, torch.Tensor):
        device = action.device
        mean = torch.tensor(stats['mean'], device=device, dtype=action.dtype)
        std = torch.tensor(stats['std'], device=device, dtype=action.dtype)
    else:
        mean = np.array(stats['mean'], dtype=action.dtype)
        std = np.array(stats['std'], dtype=action.dtype)
        
    if mean.shape[-1] > action.shape[-1]:
        mean = mean[..., :action.shape[-1]]
        std = std[..., :action.shape[-1]]
        
    return action * std + mean

def freeze_backbone(model):
    """
    Freezes VLM backbone, keeping only the Action Expert and its Connectors trainable.
    """
    # 1. Freeze EVERYTHING first
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Unfreeze Action Expert
    # This assumes the Action Expert module is stored at `model.action_expert`
    if hasattr(model, "action_expert") and model.action_expert is not None:
        logger.info("Unfreezing Action Expert...")
        for name, param in model.action_expert.named_parameters():
            param.requires_grad = True
    else:
        logger.warning("Model does not have an 'action_expert' module! Checking for alternatives...")

    # 3. Unfreeze Projectors/Connectors
    # Specifically unfreeze the connector MLP/Adapters if they exist
    if hasattr(model, "geometric_projector"):
        logger.info("Unfreezing geometric_projector...")
        for param in model.geometric_projector.parameters():
            param.requires_grad = True
            
    if hasattr(model, "fusion_projector"):
        logger.info("Unfreezing fusion_projector...")
        for param in model.fusion_projector.parameters():
            param.requires_grad = True

    # Log Trainable Parameters
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )

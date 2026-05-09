"""
FOCAL experiment — particle-aware block masking for I-JEPA.

Standard I-JEPA masks random blocks. FOCAL biases target blocks
toward high-variance (particle-containing) patches so the predictor
must reconstruct occluded particle structure specifically.

Particle detection strategy: patch-level variance.
High variance → likely particle edge or interior.
Works across brightfield, holographic, and fluorescence without labels.
"""
import math
import torch
from torch import Tensor
from dataclasses import dataclass
from typing import List, Tuple

from src.masking import MaskConfig, Masks, _block_to_mask, _sample_block


def _patch_variance_map(images: Tensor, patch_size: int = 16) -> Tensor:
    """
    Compute per-patch variance for a batch of images.
    images: (B, C, H, W)
    returns: (B, N) where N = (H/P)*(W/P), values in [0, 1]
    """
    B, C, H, W = images.shape
    grid = H // patch_size
    # Unfold into patches: (B, C, grid, grid, P, P)
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # (B, C, grid, grid, P*P)
    patches = patches.contiguous().view(B, C, grid, grid, -1)
    # Variance across patch pixels and channels: (B, grid, grid)
    var = patches.var(dim=-1).mean(dim=1)
    # Flatten and normalize to [0, 1] per image
    var = var.view(B, -1)
    vmin = var.min(dim=1, keepdim=True).values
    vmax = var.max(dim=1, keepdim=True).values
    var = (var - vmin) / (vmax - vmin + 1e-8)
    return var  # (B, N)


def _particle_biased_block(
    grid_h: int,
    grid_w: int,
    var_map: Tensor,           # (N,) — patch variances for ONE image
    scale: Tuple[float, float],
    aspect: Tuple[float, float],
    bias_strength: float = 2.0,
    max_attempts: int = 30,
) -> Tensor:
    """
    Sample a block biased toward high-variance (particle) regions.
    Tries multiple candidate blocks, picks the one with highest mean variance.
    Returns (grid_h * grid_w,) bool mask.
    """
    best_mask = None
    best_score = -1.0
    attempts = min(max_attempts, 10 + int(bias_strength * 5))

    for _ in range(attempts):
        row, col, h, w = _sample_block(grid_h, grid_w, scale, aspect, var_map.device)
        m = _block_to_mask(row, col, h, w, grid_h, grid_w, var_map.device)
        if m.sum() == 0:
            continue
        score = var_map[m].mean().item()
        if score > best_score:
            best_score = score
            best_mask = m

    if best_mask is None:
        # Fallback: random block
        row, col, h, w = _sample_block(grid_h, grid_w, scale, aspect, var_map.device)
        best_mask = _block_to_mask(row, col, h, w, grid_h, grid_w, var_map.device)

    return best_mask


def sample_focal_masks(
    images: Tensor,
    cfg: MaskConfig,
    patch_size: int = 16,
    bias_strength: float = 2.0,
) -> Masks:
    """
    FOCAL masking: target blocks biased toward particle regions.
    Context is still a large random block (same as standard I-JEPA).

    images: (B, C, H, W) — batch of training images (used for variance map only)
    """
    B = images.shape[0]
    device = images.device
    grid = images.shape[2] // patch_size
    num_patches = grid * grid

    # Compute variance map for each image in batch
    var_maps = _patch_variance_map(images.detach(), patch_size)  # (B, N)

    # Sample target blocks biased toward particles — per-image for diversity
    # then aggregate by OR for the batch (ensures consistent mask shapes)
    all_target_masks: List[Tensor] = []
    combined_target = torch.zeros(B, num_patches, dtype=torch.bool, device=device)

    for t in range(cfg.num_targets):
        # Sample a block per image biased toward its particles
        per_image_masks = []
        for b in range(B):
            m = _particle_biased_block(
                grid, grid, var_maps[b],
                cfg.target_scale, cfg.target_aspect,
                bias_strength=bias_strength,
            )
            per_image_masks.append(m)
        # Stack: (B, N)
        tgt = torch.stack(per_image_masks, dim=0)

        # For JEPA we need a uniform mask across the batch (predictor sees same positions)
        # Use the batch-mean variance to pick one shared block
        mean_var = var_maps.mean(dim=0)  # (N,)
        shared = _particle_biased_block(
            grid, grid, mean_var,
            cfg.target_scale, cfg.target_aspect,
            bias_strength=bias_strength,
        )
        shared_batch = shared.unsqueeze(0).expand(B, -1)
        all_target_masks.append(shared_batch)
        combined_target |= shared_batch

    # Context: large random block, exclude all target patches
    mean_var = var_maps.mean(dim=0)
    for _attempt in range(20):
        row, col, h, w = _sample_block(grid, grid, cfg.context_scale, (1.0, 1.0), device)
        ctx = _block_to_mask(row, col, h, w, grid, grid, device)
        ctx_clean = ctx & ~combined_target[0]
        if ctx_clean.sum() > 0:
            break
    context_batch = ctx_clean.unsqueeze(0).expand(B, -1)

    return Masks(context=context_batch, targets=all_target_masks)

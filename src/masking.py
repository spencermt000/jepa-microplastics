import math
import torch
from torch import Tensor
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class MaskConfig:
    num_targets: int = 4
    target_scale: Tuple[float, float] = (0.15, 0.2)
    target_aspect: Tuple[float, float] = (0.75, 1.5)
    context_scale: Tuple[float, float] = (0.85, 1.0)


@dataclass
class Masks:
    context: Tensor    # (B, N_patches) bool — patches used as context
    targets: List[Tensor]  # list of (B, N_patches) bool — each target block


def _sample_block(
    grid_h: int,
    grid_w: int,
    scale: Tuple[float, float],
    aspect: Tuple[float, float],
    device,
) -> Tuple[int, int, int, int]:
    """Sample a random block; returns (row, col, h, w) in patch coords."""
    total = grid_h * grid_w
    area = total * (torch.empty(1).uniform_(*scale).item())
    ar = torch.empty(1).uniform_(*aspect).item()
    h = max(1, min(grid_h, int(round(math.sqrt(area * ar)))))
    w = max(1, min(grid_w, int(round(math.sqrt(area / ar)))))
    row = torch.randint(0, max(1, grid_h - h + 1), (1,)).item()
    col = torch.randint(0, max(1, grid_w - w + 1), (1,)).item()
    return int(row), int(col), h, w


def _block_to_mask(row: int, col: int, h: int, w: int, grid_h: int, grid_w: int, device) -> Tensor:
    mask = torch.zeros(grid_h, grid_w, dtype=torch.bool, device=device)
    mask[row:row + h, col:col + w] = True
    return mask.flatten()


def sample_masks(
    batch_size: int,
    num_patches: int,
    cfg: MaskConfig,
    device,
) -> Masks:
    """
    Sample context and target masks for a batch.
    All batch items share the same mask layout (standard I-JEPA practice).
    """
    grid = int(math.sqrt(num_patches))
    assert grid * grid == num_patches, "num_patches must be a perfect square"

    # Sample target blocks (no CLS token — patches only)
    target_masks_list: List[Tensor] = []
    all_target = torch.zeros(num_patches, dtype=torch.bool, device=device)
    for _ in range(cfg.num_targets):
        for _attempt in range(20):
            row, col, h, w = _sample_block(grid, grid, cfg.target_scale, cfg.target_aspect, device)
            m = _block_to_mask(row, col, h, w, grid, grid, device)
            if m.sum() > 0:
                break
        target_masks_list.append(m)
        all_target |= m

    # Context = large random block minus all target patches
    for _attempt in range(20):
        row, col, h, w = _sample_block(grid, grid, cfg.context_scale, (1.0, 1.0), device)
        ctx = _block_to_mask(row, col, h, w, grid, grid, device)
        ctx_clean = ctx & ~all_target
        if ctx_clean.sum() > 0:
            break
    context_mask = ctx_clean

    # Expand to batch
    context_batch = context_mask.unsqueeze(0).expand(batch_size, -1)
    targets_batch = [t.unsqueeze(0).expand(batch_size, -1) for t in target_masks_list]

    return Masks(context=context_batch, targets=targets_batch)


def build_mask_config(cfg: dict) -> MaskConfig:
    return MaskConfig(
        num_targets=cfg.get("num_targets", 4),
        target_scale=tuple(cfg.get("target_scale", [0.15, 0.2])),
        target_aspect=tuple(cfg.get("target_aspect", [0.75, 1.5])),
        context_scale=tuple(cfg.get("context_scale", [0.85, 1.0])),
    )

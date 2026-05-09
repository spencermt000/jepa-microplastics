import torch
import torch.nn as nn
from src.embedding import Embedding


class LinearProbe(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, emb: Embedding) -> torch.Tensor:
        # Use CLS token (position 0) as the image representation
        cls = emb.tokens[:, 0, :]       # (B, D)
        return self.head(cls)


class MLPProbe(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, emb: Embedding) -> torch.Tensor:
        cls = emb.tokens[:, 0, :]
        return self.head(cls)


class MeanPoolProbe(nn.Module):
    """Alternative: mean-pool all patch tokens instead of CLS."""
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, emb: Embedding) -> torch.Tensor:
        patch_tokens = emb.tokens[:, 1:, :]             # drop CLS
        patch_mask = emb.mask[:, 1:].unsqueeze(-1)      # (B, N, 1)
        pooled = (patch_tokens * patch_mask).sum(dim=1) / patch_mask.sum(dim=1).clamp(min=1)
        return self.head(pooled)


def build_probe(cfg: dict, embed_dim: int, num_classes: int):
    probe_type = cfg.get("type", "linear")
    if probe_type == "linear":
        return LinearProbe(embed_dim, num_classes)
    elif probe_type == "mlp":
        return MLPProbe(embed_dim, num_classes, cfg.get("hidden_dim", 512), cfg.get("dropout", 0.0))
    elif probe_type == "meanpool":
        return MeanPoolProbe(embed_dim, num_classes)
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")

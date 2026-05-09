"""
ANCHOR experiment — SimCLR contrastive baseline.

ProjectionHead takes an Embedding → projects CLS token to a
normalized hypersphere for NT-Xent loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.embedding import Embedding


class ProjectionHead(nn.Module):
    """2-layer MLP projection head (SimCLR §3)."""
    def __init__(self, in_dim: int = 192, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, emb: Embedding) -> torch.Tensor:
        cls = emb.tokens[:, 0, :]          # (B, D) — CLS token
        z = self.net(cls)
        return F.normalize(z, dim=-1)      # unit hypersphere


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Normalized temperature-scaled cross-entropy loss (NT-Xent).
    z1, z2: (B, D) L2-normalized projections of two augmented views.
    Positive pair: (z1[i], z2[i]). All other pairs in the batch are negatives.
    """
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                         # (2B, D)
    sim = (z @ z.T) / temperature                          # (2B, 2B)

    # Mask out self-similarity on the diagonal
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))

    # Positive indices: z1[i] pairs with z2[i] at index B+i, and vice versa
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B, device=z.device),
    ])

    return F.cross_entropy(sim, labels)


class SimCLR(nn.Module):
    """
    ANCHOR — SimCLR wrapper around VitTinyEncoder.
    Encodes two augmented views, projects both, computes NT-Xent.
    """
    def __init__(self, encoder, projection_head: ProjectionHead, temperature: float = 0.07):
        super().__init__()
        self.encoder = encoder
        self.projection_head = projection_head
        self.temperature = temperature

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """x1, x2: (B, C, H, W) — two augmented views of the same images."""
        z1 = self.projection_head(self.encoder(x1))
        z2 = self.projection_head(self.encoder(x2))
        loss = nt_xent_loss(z1, z2, self.temperature)
        with torch.no_grad():
            emb_var = self.encoder(x1).tokens[:, 1:, :].var(dim=0).mean().item()
        return loss, emb_var


def build_simclr(encoder_cfg: dict, proj_cfg: dict, temperature: float = 0.07) -> SimCLR:
    from src.encoder import build_encoder
    encoder = build_encoder(encoder_cfg)
    head = ProjectionHead(
        in_dim=encoder_cfg.get("embed_dim", 192),
        hidden_dim=proj_cfg.get("hidden_dim", 512),
        out_dim=proj_cfg.get("out_dim", 128),
    )
    return SimCLR(encoder, head, temperature)

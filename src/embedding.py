from dataclasses import dataclass, field
from typing import Optional
import torch
from torch import Tensor


@dataclass
class Embedding:
    tokens: Tensor        # (B, N, D)
    mask: Tensor          # (B, N) bool — True = token is present/valid
    positions: Tensor     # (B, N) or (B, N, 2) — patch indices or spatial coords
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.tokens.ndim == 3, f"tokens must be (B, N, D), got {self.tokens.shape}"
        assert self.mask.shape == self.tokens.shape[:2], (
            f"mask {self.mask.shape} must match (B, N) = {self.tokens.shape[:2]}"
        )

    @property
    def B(self) -> int:
        return self.tokens.shape[0]

    @property
    def N(self) -> int:
        return self.tokens.shape[1]

    @property
    def D(self) -> int:
        return self.tokens.shape[2]

    def select(self, bool_mask: Tensor) -> "Embedding":
        """Select a subset of tokens per batch item given a (B, N) bool mask."""
        # Pack ragged selection into fixed-length by gathering; caller ensures uniform count.
        B, N_out = bool_mask.shape[0], bool_mask.sum(dim=1)
        assert (N_out == N_out[0]).all(), "select() requires same count across batch"
        n = int(N_out[0].item())
        idx = bool_mask.nonzero(as_tuple=False)          # (total, 2)
        tok = self.tokens[idx[:, 0], idx[:, 1]].view(B, n, self.D)
        pos = self.positions[idx[:, 0], idx[:, 1]].view(B, n, *self.positions.shape[2:])
        msk = torch.ones(B, n, dtype=torch.bool, device=self.tokens.device)
        return Embedding(tokens=tok, mask=msk, positions=pos, metadata=self.metadata)

    def to(self, device) -> "Embedding":
        return Embedding(
            tokens=self.tokens.to(device),
            mask=self.mask.to(device),
            positions=self.positions.to(device),
            metadata=self.metadata,
        )

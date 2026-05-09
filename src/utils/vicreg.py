import torch
import torch.nn.functional as F
from torch import Tensor


def vicreg_loss(
    z: Tensor,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    eps: float = 1e-4,
) -> Tensor:
    """
    VICReg regularization on a batch of embeddings z: (B, D).
    Keeps embeddings from collapsing to a constant.
    Only applied to predicted tokens, not target tokens.
    """
    B, D = z.shape
    z = z - z.mean(dim=0)

    # Variance: each dim should have std > 1
    std = z.std(dim=0)
    var_loss = F.relu(1.0 - std).mean()

    # Covariance: off-diagonal elements should be ~0
    cov = (z.T @ z) / (B - 1)
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    cov_loss = off_diag / D

    return sim_coeff * 0.0 + std_coeff * var_loss + cov_coeff * cov_loss


def vicreg_loss_pair(
    z1: Tensor,
    z2: Tensor,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
) -> Tensor:
    """VICReg loss between two views z1, z2: (B, D)."""
    sim_loss = F.mse_loss(z1, z2)

    def _var_cov(z):
        B, D = z.shape
        z = z - z.mean(0)
        std = z.std(0)
        var_loss = F.relu(1.0 - std).mean()
        cov = (z.T @ z) / (B - 1)
        off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        cov_loss = off_diag / D
        return var_loss, cov_loss

    vl1, cl1 = _var_cov(z1)
    vl2, cl2 = _var_cov(z2)

    return (
        sim_coeff * sim_loss
        + std_coeff * (vl1 + vl2) / 2
        + cov_coeff * (cl1 + cl2) / 2
    )

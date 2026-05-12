import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from src.embedding import Embedding
from src.encoder import VitTinyEncoder, build_encoder
from src.predictor import NarrowPredictor, build_predictor
from src.masking import MaskConfig, Masks, sample_masks, build_mask_config
from src.masking_focal import sample_focal_masks
from src.utils.ema import EMA
from src.utils.vicreg import vicreg_loss


@dataclass
class JEPALoss:
    total: torch.Tensor
    jepa: torch.Tensor
    vicreg: Optional[torch.Tensor]
    embedding_var: float


class JEPA(nn.Module):
    """
    I-JEPA training module.

    online_encoder  — trained by gradient descent
    target_encoder  — EMA of online_encoder (no gradients)
    predictor       — narrow Transformer, also trained by gradient descent
    """

    def __init__(
        self,
        online_encoder: VitTinyEncoder,
        target_encoder: VitTinyEncoder,
        predictor: NarrowPredictor,
        mask_cfg: MaskConfig,
        vicreg_cfg: Optional[dict] = None,
        focal_masking: bool = False,
        focal_bias_strength: float = 2.0,
    ):
        super().__init__()
        self.online_encoder = online_encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.mask_cfg = mask_cfg
        self.ema = EMA(online_encoder, target_encoder)
        self.focal_masking = focal_masking
        self.focal_bias_strength = focal_bias_strength

        self.vicreg_enabled = vicreg_cfg is not None and vicreg_cfg.get("enabled", False)
        self.vicreg_cfg = vicreg_cfg or {}

    @property
    def num_patches(self) -> int:
        return self.online_encoder.num_patches

    def forward(self, images: torch.Tensor) -> JEPALoss:
        """
        images: (B, C, H, W)
        Returns JEPALoss with scalar .total for .backward().
        """
        B, device = images.shape[0], images.device
        if self.focal_masking:
            masks: Masks = sample_focal_masks(images, self.mask_cfg,
                                              bias_strength=self.focal_bias_strength)
        else:
            masks: Masks = sample_masks(B, self.num_patches, self.mask_cfg, device)

        # --- Target encoder (no grad) ---
        with torch.no_grad():
            full_target: Embedding = self.target_encoder(images)
            # Drop CLS token — only patch tokens are prediction targets
            patch_tokens = full_target.tokens[:, 1:, :]   # (B, N, D)
            patch_positions = full_target.positions[:, 1:] # (B, N)

        # --- Online encoder: only sees context patches ---
        full_online: Embedding = self.online_encoder(images)
        ctx_mask_with_cls = torch.cat([
            torch.ones(B, 1, dtype=torch.bool, device=device),
            masks.context,
        ], dim=1)  # prepend True for CLS
        ctx_tokens = full_online.tokens[ctx_mask_with_cls].view(
            B, ctx_mask_with_cls.sum(dim=1)[0].item(), full_online.D
        )
        ctx_positions = full_online.positions[ctx_mask_with_cls].view(
            B, ctx_mask_with_cls.sum(dim=1)[0].item()
        )
        ctx_embedding = Embedding(
            tokens=ctx_tokens,
            mask=torch.ones(B, ctx_tokens.shape[1], dtype=torch.bool, device=device),
            positions=ctx_positions,
        )

        # --- Predict each target block ---
        total_jepa = torch.tensor(0.0, device=device)
        all_preds = []

        for tgt_mask in masks.targets:
            # Target positions (patch indices, 0-based in the patch sequence)
            tgt_idx = tgt_mask[0].nonzero(as_tuple=True)[0]          # (n_tgt,)
            tgt_positions = tgt_idx.unsqueeze(0).expand(B, -1) + 1   # +1 to skip CLS slot

            # Target representations (from frozen target encoder)
            tgt_tokens = patch_tokens[:, tgt_mask[0], :]              # (B, n_tgt, D)

            # Predicted representations
            pred: Embedding = self.predictor(ctx_embedding, tgt_positions)

            loss_block = F.smooth_l1_loss(pred.tokens, tgt_tokens.detach())
            total_jepa = total_jepa + loss_block
            all_preds.append(pred.tokens)

        total_jepa = total_jepa / len(masks.targets)

        # --- Optional VICReg on encoder tokens + predicted tokens ---
        vicreg_term = None
        if self.vicreg_enabled:
            std_coeff = self.vicreg_cfg.get("std_coeff", 25.0)
            cov_coeff = self.vicreg_cfg.get("cov_coeff", 1.0)

            # Predictor outputs — prevents predictor collapse
            flat_preds = torch.cat(all_preds, dim=1).flatten(0, 1)   # (B*N_total, D)
            vr_pred = vicreg_loss(flat_preds, std_coeff=std_coeff, cov_coeff=cov_coeff)

            # Online encoder patch tokens — prevents encoder collapse directly
            enc_tokens = full_online.tokens[:, 1:, :].flatten(0, 1)  # (B*N_patches, D)
            vr_enc = vicreg_loss(enc_tokens, std_coeff=std_coeff, cov_coeff=cov_coeff)

            vicreg_term = vr_pred + vr_enc

        total = total_jepa + (vicreg_term if vicreg_term is not None else 0.0)

        # Collapse monitor: variance of target patch embeddings
        with torch.no_grad():
            emb_var = patch_tokens.var(dim=0).mean().item()

        return JEPALoss(
            total=total,
            jepa=total_jepa,
            vicreg=vicreg_term,
            embedding_var=emb_var,
        )

    @torch.no_grad()
    def ema_step(self, step: int, total_steps: int, momentum_start: float = 0.996):
        self.ema.update_momentum(step, total_steps, momentum_start)
        self.ema.step()


def build_jepa(encoder_cfg: dict, predictor_cfg: dict, masking_cfg: dict, vicreg_cfg: dict) -> JEPA:
    online = build_encoder(encoder_cfg)
    target = build_encoder(encoder_cfg)
    predictor = build_predictor(predictor_cfg, online.patch_embed.proj.out_channels, online.num_patches)
    mask_cfg = build_mask_config(masking_cfg)
    focal = masking_cfg.get("strategy", "random") == "focal"
    bias = masking_cfg.get("bias_strength", 2.0)
    return JEPA(online, target, predictor, mask_cfg, vicreg_cfg,
                focal_masking=focal, focal_bias_strength=bias)

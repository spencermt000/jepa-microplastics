import torch
import torch.nn as nn
from einops import rearrange
from src.embedding import Embedding


class PredictorBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class NarrowPredictor(nn.Module):
    """
    Narrow Transformer predictor (I-JEPA §3.2).

    Takes context Embedding (already encoded by online encoder) + target positions,
    outputs predicted target Embedding in the encoder's latent space.

    Context tokens are projected to predictor dim. Target query tokens are
    learned positional embeddings conditioned on target patch indices.
    """

    def __init__(
        self,
        encoder_dim: int = 192,
        predictor_dim: int = 128,
        num_heads: int = 4,
        depth: int = 4,
        num_patches: int = 196,  # 14×14 for 224/16
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.predictor_dim = predictor_dim

        self.input_proj = nn.Linear(encoder_dim, predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, encoder_dim)

        # Positional embeddings for the predictor's full patch space
        self.predictor_pos_embed = nn.Embedding(num_patches + 1, predictor_dim)  # +1 for CLS

        self.blocks = nn.ModuleList([
            PredictorBlock(predictor_dim, num_heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.predictor_pos_embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, context: Embedding, target_positions: torch.Tensor) -> Embedding:
        """
        context:          Embedding (B, N_ctx, D_enc) — encoded context patches
        target_positions: (B, N_tgt) — patch indices of target tokens to predict
        returns:          Embedding (B, N_tgt, D_enc) — predicted target representations
        """
        B, N_tgt = target_positions.shape

        # Project context tokens into predictor space and add their positional encoding
        ctx_tokens = self.input_proj(context.tokens)                         # (B, N_ctx, P)
        ctx_pos = self.predictor_pos_embed(context.positions)                # (B, N_ctx, P)
        ctx_tokens = ctx_tokens + ctx_pos

        # Build target query tokens: learned pos embed for each target position
        tgt_tokens = self.predictor_pos_embed(target_positions)              # (B, N_tgt, P)

        # Concatenate: context first, then target queries
        tokens = torch.cat([ctx_tokens, tgt_tokens], dim=1)                 # (B, N_ctx+N_tgt, P)

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        # Extract only the target predictions
        pred_tokens = tokens[:, ctx_tokens.shape[1]:, :]                    # (B, N_tgt, P)
        pred_tokens = self.output_proj(pred_tokens)                          # (B, N_tgt, D_enc)

        mask = torch.ones(B, N_tgt, dtype=torch.bool, device=pred_tokens.device)
        return Embedding(tokens=pred_tokens, mask=mask, positions=target_positions)


def build_predictor(cfg: dict, encoder_dim: int, num_patches: int) -> NarrowPredictor:
    return NarrowPredictor(
        encoder_dim=encoder_dim,
        predictor_dim=cfg.get("embed_dim", 128),
        num_heads=cfg.get("num_heads", 4),
        depth=cfg.get("depth", 4),
        num_patches=num_patches,
    )

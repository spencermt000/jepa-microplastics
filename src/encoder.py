import math
import torch
import torch.nn as nn
from einops import rearrange
from src.embedding import Embedding


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return rearrange(self.proj(x), "b d h w -> b (h w) d")


class Attention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.dropout(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x), attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class VitTinyEncoder(nn.Module):
    """ViT-Tiny: 192-dim, 12 layers, 3 heads, patch_size=16 → Embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def num_patches(self) -> int:
        return self.patch_embed.num_patches

    def forward(self, x: torch.Tensor) -> Embedding:
        """x: (B, C, H, W) → Embedding with tokens (B, N+1, D)."""
        B = x.shape[0]
        patches = self.patch_embed(x)                         # (B, N, D)
        N = patches.shape[1]

        cls = self.cls_token.expand(B, -1, -1)                # (B, 1, D)
        tokens = torch.cat([cls, patches], dim=1)             # (B, N+1, D)
        tokens = self.pos_drop(tokens + self.pos_embed)

        for block in self.blocks:
            tokens, _ = block(tokens)
        tokens = self.norm(tokens)

        # positions: 0 = CLS, 1..N = patch indices
        positions = torch.arange(N + 1, device=x.device).unsqueeze(0).expand(B, -1)
        mask = torch.ones(B, N + 1, dtype=torch.bool, device=x.device)

        return Embedding(tokens=tokens, mask=mask, positions=positions)

    def forward_features_with_attn(self, x: torch.Tensor):
        """Returns (Embedding, list_of_attn_maps) for visualization."""
        B = x.shape[0]
        patches = self.patch_embed(x)
        N = patches.shape[1]
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed)

        attn_maps = []
        for block in self.blocks:
            tokens, attn = block(tokens)
            attn_maps.append(attn)
        tokens = self.norm(tokens)

        positions = torch.arange(N + 1, device=x.device).unsqueeze(0).expand(B, -1)
        mask = torch.ones(B, N + 1, dtype=torch.bool, device=x.device)
        return Embedding(tokens=tokens, mask=mask, positions=positions), attn_maps


def build_encoder(cfg: dict) -> VitTinyEncoder:
    return VitTinyEncoder(
        img_size=cfg.get("img_size", 224),
        patch_size=cfg.get("patch_size", 16),
        embed_dim=cfg.get("embed_dim", 192),
        depth=cfg.get("depth", 12),
        num_heads=cfg.get("num_heads", 3),
        dropout=cfg.get("dropout", 0.1),
    )

"""
Visualization: t-SNE/UMAP of embeddings + last-layer attention maps.
Usage:
    python scripts/visualize.py --checkpoint checkpoints/jepa_final.pt --method tsne
    python scripts/visualize.py --checkpoint checkpoints/jepa_final.pt --method umap
    python scripts/visualize.py --checkpoint checkpoints/jepa_final.pt --method attn --n_images 8
"""
import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.encoder import build_encoder
from src.data.dataset import load_metadata, MicroplasticsDataset
from src.data.transforms import eval_transform


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def extract_embeddings(encoder, loader, device, label_col=None):
    encoder.eval()
    feats, labels, paths = [], [], []
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            images, lbl = batch
        else:
            images, lbl = batch, torch.full((batch.shape[0],), -1)
        emb = encoder(images.to(device))
        feats.append(emb.tokens[:, 0, :].cpu().numpy())
        labels.append(lbl.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def plot_embedding(proj, labels, method: str, out_path: str, class_names=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    unique = np.unique(labels[labels >= 0])
    cmap = plt.cm.get_cmap("tab10", len(unique))
    for i, cls in enumerate(unique):
        mask = labels == cls
        name = class_names[cls] if class_names and cls < len(class_names) else str(cls)
        ax.scatter(proj[mask, 0], proj[mask, 1], label=name, alpha=0.6, s=15, color=cmap(i))
    ax.legend(markerscale=2, fontsize=8)
    ax.set_title(f"{method.upper()} of encoder embeddings")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def visualize_attention(encoder, images, out_path: str, patch_size: int = 16):
    encoder.eval()
    with torch.no_grad():
        emb, attn_maps = encoder.forward_features_with_attn(images)
    last_attn = attn_maps[-1]              # (B, heads, N+1, N+1)
    cls_attn = last_attn[:, :, 0, 1:]     # (B, heads, N)
    B, H, N = cls_attn.shape
    grid = int(N ** 0.5)
    cls_attn = cls_attn.mean(dim=1)       # mean over heads: (B, N)
    cls_attn = cls_attn.view(B, grid, grid).cpu().numpy()

    fig, axes = plt.subplots(2, B, figsize=(B * 3, 6))
    for i in range(B):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axes[0, i].imshow(img)
        axes[0, i].axis("off")
        axes[1, i].imshow(cls_attn[i], cmap="hot", interpolation="nearest")
        axes[1, i].axis("off")
    axes[0, 0].set_title("Image", loc="left")
    axes[1, 0].set_title("CLS Attention", loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/probe.yaml")
    parser.add_argument("--metadata", default="processed/metadata.csv")
    parser.add_argument("--label_col", default="label_polymer")
    parser.add_argument("--method", default="tsne", choices=["tsne", "umap", "attn"])
    parser.add_argument("--n_images", type=int, default=8, help="Images for attention viz")
    parser.add_argument("--out_dir", default="results/viz")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = build_encoder(cfg["encoder"])
    ckpt = torch.load(args.checkpoint, map_location=device)
    encoder.load_state_dict(ckpt["online_encoder"])
    encoder = encoder.to(device)

    from src.data.dataset import load_metadata
    metadata = load_metadata(args.metadata)
    labeled = metadata[metadata[args.label_col].notna()].reset_index(drop=True)
    classes = sorted(labeled[args.label_col].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    labeled = labeled.copy()
    labeled["_label_idx"] = labeled[args.label_col].map(class_to_idx)

    if args.method == "attn":
        ds = MicroplasticsDataset(labeled.head(args.n_images), label_col=args.label_col, transform=eval_transform())
        loader = DataLoader(ds, batch_size=args.n_images, shuffle=False)
        images, _ = next(iter(loader))
        visualize_attention(encoder, images.to(device), str(out_dir / "attention.png"))
        return

    loader = DataLoader(
        MicroplasticsDataset(labeled, label_col=args.label_col, transform=eval_transform()),
        batch_size=64, shuffle=False, num_workers=2,
    )
    feats, labels = extract_embeddings(encoder, loader, device)

    if args.method == "tsne":
        from sklearn.manifold import TSNE
        proj = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(feats)
        plot_embedding(proj, labels, "tsne", str(out_dir / "tsne.png"), classes)
    elif args.method == "umap":
        import umap
        proj = umap.UMAP(n_components=2, random_state=42).fit_transform(feats)
        plot_embedding(proj, labels, "umap", str(out_dir / "umap.png"), classes)


if __name__ == "__main__":
    main()

"""
Post-training analysis pipeline for a JEPA checkpoint.

Runs in sequence:
  1. Linear probe  — 5-fold CV on polymer / morphology / composition
  2. Label efficiency — fraction sweep {1%..100%} on polymer, 5 seeds each
  3. Embedding viz — t-SNE, UMAP, and CLS attention maps

All tasks log to MLflow under a single parent run with nested child runs.
Plots saved to --out_dir (default: results/analyze).

Usage:
    python scripts/analyze.py --checkpoint checkpoints/jepa_ep0300.pt
"""
import argparse
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.encoder import build_encoder
from src.probe import build_probe, LinearProbe
from src.utils.metrics import compute_metrics
from src.data.dataset import load_metadata, eval_splits, label_efficiency_split, MicroplasticsDataset
from src.data.transforms import train_probe_transform, eval_transform


LABEL_COLS = ["label_polymer", "label_morphology", "label_composition"]
FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
N_SEEDS = 5


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_encoder(cfg, checkpoint, device):
    encoder = build_encoder(cfg["encoder"])
    ckpt = torch.load(checkpoint, map_location=device)
    encoder.load_state_dict(ckpt["online_encoder"])
    encoder = encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    return encoder


@torch.no_grad()
def extract_features(encoder, loader, device):
    feats, labels = [], []
    for images, lbl in loader:
        emb = encoder(images.to(device))
        feats.append(emb.tokens[:, 0, :].cpu())
        labels.append(lbl)
    return torch.cat(feats), torch.cat(labels)


def _train_linear_probe(probe, train_feats, train_labels, cfg, device):
    train_cfg = cfg["training"]
    opt = torch.optim.SGD(
        probe.parameters(),
        lr=train_cfg["lr"],
        momentum=train_cfg.get("momentum", 0.9),
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=train_cfg["epochs"])
    criterion = nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(train_feats, train_labels)
    loader = DataLoader(ds, batch_size=train_cfg["batch_size"], shuffle=True)
    probe.train()
    for _ in range(train_cfg["epochs"]):
        for f, l in loader:
            logits = probe.head(f.to(device))
            loss = criterion(logits, l.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    probe.eval()
    return probe


# ── Task 1: Linear probe ─────────────────────────────────────────────────────

def run_probe(encoder, cfg, metadata, label_col, device, embed_dim):
    print(f"\n── Probe: {label_col} ──")
    labeled = metadata[metadata[label_col].notna()]
    if len(labeled) == 0:
        print(f"  No labeled data, skipping.")
        return None, None

    fold_accs, fold_f1s = [], []
    with mlflow.start_run(run_name=f"probe-{label_col}", nested=True):
        mlflow.log_params({"label_col": label_col})
        for fold, (train_ds, val_ds) in enumerate(eval_splits(
            metadata, label_col,
            transform_train=train_probe_transform(),
            transform_val=eval_transform(),
        )):
            num_classes = len(train_ds.classes)
            train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=2)
            val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)
            train_feats, train_labels = extract_features(encoder, train_loader, device)
            val_feats, val_labels = extract_features(encoder, val_loader, device)

            probe = build_probe(cfg["probe"], embed_dim, num_classes).to(device)
            probe = _train_linear_probe(probe, train_feats, train_labels, cfg, device)

            with torch.no_grad():
                logits = probe.head(val_feats.to(device)).cpu()
            m = compute_metrics(logits, val_labels, num_classes)
            fold_accs.append(m["accuracy"])
            fold_f1s.append(m["macro_f1"])
            print(f"  fold {fold}: acc={m['accuracy']:.4f}  f1={m['macro_f1']:.4f}")
            mlflow.log_metrics({f"fold{fold}_acc": m["accuracy"], f"fold{fold}_f1": m["macro_f1"]})

        mean_acc = statistics.mean(fold_accs)
        mean_f1 = statistics.mean(fold_f1s)
        print(f"  → mean acc={mean_acc:.4f}  mean f1={mean_f1:.4f}")
        mlflow.log_metrics({"mean_accuracy": mean_acc, "mean_f1": mean_f1})
    return mean_acc, mean_f1


# ── Task 2: Label efficiency ──────────────────────────────────────────────────

def run_label_efficiency(encoder, cfg, metadata, device, embed_dim):
    print("\n── Label Efficiency (polymer) ──")
    results = {}
    with mlflow.start_run(run_name="label-efficiency", nested=True):
        mlflow.log_params({"label_col": "label_polymer", "n_seeds": N_SEEDS, "fractions": str(FRACTIONS)})
        for frac in FRACTIONS:
            accs = []
            for seed in range(N_SEEDS):
                train_ds, val_ds = label_efficiency_split(
                    metadata, "label_polymer", frac, seed=seed,
                    transform_train=train_probe_transform(),
                    transform_val=eval_transform(),
                )
                num_classes = len(train_ds.classes)
                train_feats, train_labels = extract_features(encoder, DataLoader(train_ds, batch_size=64, num_workers=2), device)
                val_feats, val_labels = extract_features(encoder, DataLoader(val_ds, batch_size=64, num_workers=2), device)

                probe = LinearProbe(embed_dim, num_classes).to(device)
                opt = torch.optim.SGD(probe.parameters(), lr=0.1, momentum=0.9)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
                ds = torch.utils.data.TensorDataset(train_feats, train_labels)
                probe.train()
                for _ in range(100):
                    for f, l in DataLoader(ds, batch_size=64, shuffle=True):
                        logits = probe.head(f.to(device))
                        nn.CrossEntropyLoss()(logits, l.to(device)).backward()
                        opt.step(); opt.zero_grad()
                    sched.step()
                probe.eval()
                with torch.no_grad():
                    logits = probe.head(val_feats.to(device)).cpu()
                acc = compute_metrics(logits, val_labels, num_classes)["accuracy"]
                accs.append(acc)
                print(f"  frac={frac:.2f} seed={seed} acc={acc:.4f}")

            mean_acc = statistics.mean(accs)
            std_acc = statistics.stdev(accs) if len(accs) > 1 else 0.0
            results[frac] = (mean_acc, std_acc)
            mlflow.log_metrics({f"acc_{frac}": mean_acc, f"std_{frac}": std_acc})
            print(f"  frac={frac:.2f}: {mean_acc:.4f} ± {std_acc:.4f}")
    return results


# ── Task 3: Visualization ─────────────────────────────────────────────────────

def _save_scatter(proj, labels, method, path, class_names):
    fig, ax = plt.subplots(figsize=(10, 8))
    unique = np.unique(labels[labels >= 0])
    cmap = plt.cm.get_cmap("tab10", len(unique))
    for i, cls in enumerate(unique):
        mask = labels == cls
        name = class_names[cls] if cls < len(class_names) else str(cls)
        ax.scatter(proj[mask, 0], proj[mask, 1], label=name, alpha=0.6, s=15, color=cmap(i))
    ax.legend(markerscale=2, fontsize=8)
    ax.set_title(f"{method} of encoder embeddings")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)


def _save_attention(encoder, images, path):
    with torch.no_grad():
        _, attn_maps = encoder.forward_features_with_attn(images)
    last_attn = attn_maps[-1]
    cls_attn = last_attn[:, :, 0, 1:].mean(dim=1)
    B, N = cls_attn.shape
    grid = int(N ** 0.5)
    cls_attn = cls_attn.view(B, grid, grid).cpu().numpy()
    fig, axes = plt.subplots(2, B, figsize=(B * 3, 6))
    for i in range(B):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axes[0, i].imshow(img); axes[0, i].axis("off")
        axes[1, i].imshow(cls_attn[i], cmap="hot", interpolation="nearest"); axes[1, i].axis("off")
    axes[0, 0].set_title("Image", loc="left")
    axes[1, 0].set_title("CLS Attention", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)


def run_viz(encoder, cfg, metadata, device, out_dir, label_col="label_polymer"):
    print("\n── Embedding Visualization ──")
    labeled = metadata[metadata[label_col].notna()].reset_index(drop=True)
    classes = sorted(labeled[label_col].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    labeled = labeled.copy()
    labeled["_label_idx"] = labeled[label_col].map(class_to_idx)

    loader = DataLoader(
        MicroplasticsDataset(labeled, label_col=label_col, transform=eval_transform()),
        batch_size=64, shuffle=False, num_workers=2,
    )
    feats, labels = [], []
    with torch.no_grad():
        for images, lbl in loader:
            emb = encoder(images.to(device))
            feats.append(emb.tokens[:, 0, :].cpu().numpy())
            labels.append(lbl.numpy())
    feats = np.concatenate(feats)
    labels = np.concatenate(labels)

    with mlflow.start_run(run_name="embedding-viz", nested=True):
        from sklearn.manifold import TSNE
        proj = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(feats)
        tsne_path = out_dir / "tsne.png"
        _save_scatter(proj, labels, "t-SNE", tsne_path, classes)
        mlflow.log_artifact(str(tsne_path))

        import umap as umap_lib
        proj_umap = umap_lib.UMAP(n_components=2, random_state=42).fit_transform(feats)
        umap_path = out_dir / "umap.png"
        _save_scatter(proj_umap, labels, "UMAP", umap_path, classes)
        mlflow.log_artifact(str(umap_path))

        n_attn = 8
        sample_ds = MicroplasticsDataset(labeled.head(n_attn), label_col=label_col, transform=eval_transform())
        images, _ = next(iter(DataLoader(sample_ds, batch_size=n_attn)))
        attn_path = out_dir / "attention.png"
        try:
            _save_attention(encoder, images.to(device), attn_path)
            mlflow.log_artifact(str(attn_path))
        except Exception as e:
            print(f"  Attention viz skipped: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/probe.yaml")
    parser.add_argument("--metadata", default="processed/metadata.csv")
    parser.add_argument("--out_dir", default="results/analyze")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    embed_dim = cfg["encoder"].get("embed_dim", 192)

    mlflow.set_tracking_uri(cfg.get("mlflow", {}).get("tracking_uri", "http://192.168.4.51:5000"))
    mlflow.set_experiment("jepa-microplastics-analysis")

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    encoder = load_encoder(cfg, args.checkpoint, device)
    metadata = load_metadata(args.metadata)

    probe_summary = {}
    with mlflow.start_run(run_name=f"analyze-{Path(args.checkpoint).stem}"):
        mlflow.log_params({"checkpoint": args.checkpoint})

        for label_col in LABEL_COLS:
            acc, f1 = run_probe(encoder, cfg, metadata, label_col, device, embed_dim)
            if acc is not None:
                probe_summary[label_col] = (acc, f1)
                mlflow.log_metrics({f"{label_col}_acc": acc, f"{label_col}_f1": f1})

        le_results = run_label_efficiency(encoder, cfg, metadata, device, embed_dim)
        for frac, (m, s) in le_results.items():
            mlflow.log_metrics({f"le_acc_{frac}": m, f"le_std_{frac}": s})

        run_viz(encoder, cfg, metadata, device, out_dir)

    print("\n" + "=" * 52)
    print("ANALYSIS COMPLETE")
    print("=" * 52)
    print("Linear probe:")
    for col, (acc, f1) in probe_summary.items():
        print(f"  {col}: acc={acc:.4f}  f1={f1:.4f}")
    print("Label efficiency (polymer):")
    for frac, (m, s) in le_results.items():
        print(f"  {frac * 100:5.1f}%: {m:.4f} ± {s:.4f}")
    print(f"\nPlots → {out_dir}/")


if __name__ == "__main__":
    main()

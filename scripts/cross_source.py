"""
DRIFT experiment — cross-source generalization evaluation.

Leave-one-source-out: for each dataset source, train a linear probe
on all OTHER sources, evaluate on the held-out source.
Tests whether representations transfer across microscopy modalities
(brightfield PEESEgroup vs holographic vs Kaggle sets).

Usage:
    python scripts/cross_source.py --checkpoint checkpoints/jepa_final.pt
    python scripts/cross_source.py --checkpoint checkpoints/jepa_final.pt --label_col label_polymer
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
from src.encoder import build_encoder
from src.probe import LinearProbe
from src.utils.metrics import compute_metrics
from src.data.dataset import load_metadata, MicroplasticsDataset
from src.data.transforms import train_probe_transform, eval_transform


def load_cfg(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def extract(encoder, loader, device):
    encoder.eval()
    feats, labels = [], []
    for images, lbl in loader:
        emb = encoder(images.to(device))
        feats.append(emb.tokens[:, 0, :].cpu())
        labels.append(lbl)
    return torch.cat(feats), torch.cat(labels)


def probe_accuracy(train_feats, train_labels, val_feats, val_labels, embed_dim, num_classes, device):
    probe = LinearProbe(embed_dim, num_classes).to(device)
    opt = torch.optim.SGD(probe.parameters(), lr=0.1, momentum=0.9)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    ds = torch.utils.data.TensorDataset(train_feats, train_labels)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    probe.train()
    for _ in range(100):
        for f, l in loader:
            logits = probe.head(f.to(device))
            loss = nn.CrossEntropyLoss()(logits, l.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    probe.eval()
    with torch.no_grad():
        logits = probe.head(val_feats.to(device)).cpu()
    return compute_metrics(logits, val_labels, num_classes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/probe.yaml")
    parser.add_argument("--label_col", default="label_polymer")
    parser.add_argument("--metadata", default="processed/metadata.csv")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim = cfg["encoder"].get("embed_dim", 192)

    encoder = build_encoder(cfg["encoder"])
    ckpt = torch.load(args.checkpoint, map_location=device)
    encoder.load_state_dict(ckpt["online_encoder"])
    encoder = encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)

    metadata = load_metadata(args.metadata)
    labeled = metadata[metadata[args.label_col].notna()].copy()
    sources = sorted(labeled["source"].unique().tolist())

    if len(sources) < 2:
        print(f"Need ≥2 labeled sources. Found: {sources}")
        return

    # Build class index from full labeled set for consistency
    classes = sorted(labeled[args.label_col].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    labeled["_label_idx"] = labeled[args.label_col].map(class_to_idx)
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")
    print(f"Sources: {sources}\n")

    mlflow.set_tracking_uri(cfg.get("mlflow", {}).get("tracking_uri", ""))
    mlflow.set_experiment("jepa-drift-cross-source")

    results = {}
    with mlflow.start_run():
        mlflow.log_params({
            "checkpoint": args.checkpoint,
            "label_col": args.label_col,
            "sources": ",".join(sources),
        })

        for held_out in sources:
            train_df = labeled[labeled["source"] != held_out].reset_index(drop=True)
            val_df   = labeled[labeled["source"] == held_out].reset_index(drop=True)

            # Skip if held-out source has no labeled data or too few classes
            val_classes = val_df[args.label_col].unique()
            if len(train_df) == 0 or len(val_df) == 0:
                print(f"[{held_out}] skipped — insufficient data")
                continue

            print(f"[{held_out}] train={len(train_df)} val={len(val_df)}")

            train_ds = MicroplasticsDataset(train_df, label_col=args.label_col,
                                             transform=train_probe_transform())
            val_ds   = MicroplasticsDataset(val_df,   label_col=args.label_col,
                                             transform=eval_transform())
            # Override class index to use global one for consistent label encoding
            train_ds.class_to_idx = class_to_idx
            val_ds.class_to_idx   = class_to_idx

            train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0)
            val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0)

            print(f"  Extracting features...")
            train_feats, train_labels = extract(encoder, train_loader, device)
            val_feats,   val_labels   = extract(encoder, val_loader,   device)

            metrics = probe_accuracy(
                train_feats, train_labels, val_feats, val_labels,
                embed_dim, num_classes, device
            )
            results[held_out] = metrics
            print(f"  acc={metrics['accuracy']:.4f}  f1={metrics['macro_f1']:.4f}")
            mlflow.log_metrics({
                f"drift_{held_out}_acc": metrics["accuracy"],
                f"drift_{held_out}_f1":  metrics["macro_f1"],
            })

    print("\n=== DRIFT Cross-Source Results ===")
    for src, m in results.items():
        print(f"  held-out={src:15s}  acc={m['accuracy']:.4f}  f1={m['macro_f1']:.4f}")

    if results:
        mean_acc = sum(m["accuracy"] for m in results.values()) / len(results)
        print(f"\n  Mean cross-source accuracy: {mean_acc:.4f}")


if __name__ == "__main__":
    main()

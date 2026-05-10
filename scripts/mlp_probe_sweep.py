"""
MLP probe sweep across JEPA checkpoints ep050–ep300.

Uses a 2-layer MLP head instead of linear probe, to test whether
later checkpoints have better representations that require nonlinear
evaluation to unlock.

Saves results to results/mlp_probe_sweep.csv
"""
import csv, statistics, sys
from pathlib import Path

sys.path.insert(0, "/workspace")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.encoder import build_encoder
from src.utils.metrics import compute_metrics
from src.data.dataset import load_metadata, eval_splits
from src.data.transforms import train_probe_transform, eval_transform
import yaml

CHECKPOINTS = [
    ("ep0050", "checkpoints/jepa_ep0050.pt"),
    ("ep0100", "checkpoints/jepa_ep0100.pt"),
    ("ep0150", "checkpoints/jepa_ep0150.pt"),
    ("ep0200", "checkpoints/jepa_ep0200.pt"),
    ("ep0250", "checkpoints/jepa_ep0250.pt"),
    ("ep0300", "checkpoints/jepa_ep0300.pt"),
]
MLP_EPOCHS = 200
LABEL_COL = "label_polymer"


class MLPProbe(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_cfg():
    with open("configs/probe.yaml") as f:
        return yaml.safe_load(f)


def train_mlp_probe(train_feats, train_labels, embed_dim, num_classes, device):
    probe = MLPProbe(embed_dim, num_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MLP_EPOCHS)
    ds = torch.utils.data.TensorDataset(train_feats, train_labels)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    probe.train()
    for _ in range(MLP_EPOCHS):
        for f, l in loader:
            nn.CrossEntropyLoss()(probe(f.to(device)), l.to(device)).backward()
            opt.step(); opt.zero_grad()
        sched.step()
    probe.eval()
    return probe


@torch.no_grad()
def extract_features(encoder, loader, device):
    feats, labels = [], []
    for images, lbl in loader:
        out = encoder(images.to(device))
        feats.append(out.tokens[:, 0, :].cpu())
        labels.append(lbl)
    return torch.cat(feats), torch.cat(labels)


def run_probe_cv(encoder, embed_dim, metadata, device):
    accs, f1s = [], []
    for fold, (train_ds, val_ds) in enumerate(eval_splits(
        metadata, LABEL_COL,
        transform_train=train_probe_transform(),
        transform_val=eval_transform(),
    )):
        num_classes = len(train_ds.classes)
        train_feats, train_labels = extract_features(
            encoder, DataLoader(train_ds, batch_size=64, num_workers=0), device)
        val_feats, val_labels = extract_features(
            encoder, DataLoader(val_ds, batch_size=64, num_workers=0), device)
        probe = train_mlp_probe(train_feats, train_labels, embed_dim, num_classes, device)
        with torch.no_grad():
            logits = probe(val_feats.to(device)).cpu()
        m = compute_metrics(logits, val_labels, num_classes)
        accs.append(m["accuracy"])
        f1s.append(m["macro_f1"])
        print(f"    fold {fold}: acc={m['accuracy']:.4f}  f1={m['macro_f1']:.4f}")
    return statistics.mean(accs), statistics.stdev(accs), statistics.mean(f1s)


def main():
    cfg = load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim = cfg["encoder"].get("embed_dim", 192)
    print(f"Device: {device}  |  Probe: 2-layer MLP (192→256→num_classes)\n")

    metadata = load_metadata("processed/metadata.csv")

    results = []
    print(f"{'Epoch':<10} {'Mean Acc':>10} {'±Std':>8} {'Mean F1':>10}")
    print("-" * 42)

    for label, ckpt_path in CHECKPOINTS:
        print(f"\n[{label}] Loading {ckpt_path}")
        encoder = build_encoder(cfg["encoder"])
        ckpt = torch.load(ckpt_path, map_location=device)
        encoder.load_state_dict(ckpt["online_encoder"])
        encoder = encoder.to(device).eval()
        for p in encoder.parameters():
            p.requires_grad_(False)

        mean_acc, std_acc, mean_f1 = run_probe_cv(encoder, embed_dim, metadata, device)
        print(f"  {label:<10} {mean_acc:>10.4f} {std_acc:>8.4f} {mean_f1:>10.4f}")
        results.append({"epoch": label, "mean_acc": mean_acc, "std_acc": std_acc, "mean_f1": mean_f1})

    print("\n\n=== JEPA MLP Probe Learning Curve ===")
    print(f"{'Epoch':<10} {'Mean Acc':>10} {'±Std':>8} {'Mean F1':>10}")
    print("-" * 42)
    for r in results:
        print(f"{r['epoch']:<10} {r['mean_acc']:>10.4f} {r['std_acc']:>8.4f} {r['mean_f1']:>10.4f}")

    out = Path("results/mlp_probe_sweep.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "mean_acc", "std_acc", "mean_f1"])
        w.writeheader(); w.writerows(results)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

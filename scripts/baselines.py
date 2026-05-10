"""
Baseline comparison experiments against JEPA-pretrained encoder.

Encoders: random | imagenet | dinov2 | supervised | jepa
Each runs:
  1. Linear probe  — 5-fold CV (same folds/seeds as JEPA evaluation)
  2. Label efficiency — fractions [0.01, 0.05, 0.10, 0.25, 0.50, 1.0], 5 seeds

Results saved to results/baselines/{encoder}_results.csv
Logged to MLflow experiment "jepa-microplastics-baselines"

Usage:
    python scripts/baselines.py --encoder random
    python scripts/baselines.py --encoder imagenet
    python scripts/baselines.py --encoder dinov2
    python scripts/baselines.py --encoder supervised
    python scripts/baselines.py --encoder jepa --checkpoint checkpoints/jepa_ep0300.pt
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import mlflow

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.encoder import build_encoder
from src.utils.metrics import compute_metrics
from src.data.dataset import load_metadata, eval_splits, label_efficiency_split
from src.data.transforms import train_probe_transform, eval_transform

FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
N_SEEDS = 5
LABEL_COL = "label_polymer"
PROBE_EPOCHS = 100
SUP_EPOCHS = 100
SUP_BATCH = 8
SUP_GRAD_ACCUM = 4


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Encoder loading ───────────────────────────────────────────────────────────

def get_frozen_encoder(encoder_type, cfg, checkpoint, device):
    """Returns (forward_fn, embed_dim) where forward_fn(images) → (B, D)."""
    if encoder_type == "jepa":
        model = build_encoder(cfg["encoder"]).to(device)
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt["online_encoder"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        def extract(images):
            return model(images).tokens[:, 0, :]
        return extract, 192

    elif encoder_type == "random":
        model = build_encoder(cfg["encoder"]).to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        def extract(images):
            return model(images).tokens[:, 0, :]
        return extract, 192

    elif encoder_type == "imagenet":
        import timm
        model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0).to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model, 192

    elif encoder_type == "dinov2":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True).to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model, 384

    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(extractor, loader, device):
    feats, labels = [], []
    for images, lbl in loader:
        out = extractor(images.to(device))
        feats.append(out.cpu())
        labels.append(lbl)
    return torch.cat(feats), torch.cat(labels)


# ── Linear probe training ─────────────────────────────────────────────────────

def train_linear_probe(train_feats, train_labels, embed_dim, num_classes, device):
    probe = nn.Linear(embed_dim, num_classes).to(device)
    opt = torch.optim.SGD(probe.parameters(), lr=0.1, momentum=0.9)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PROBE_EPOCHS)
    ds = torch.utils.data.TensorDataset(train_feats, train_labels)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    probe.train()
    for _ in range(PROBE_EPOCHS):
        for f, l in loader:
            nn.CrossEntropyLoss()(probe(f.to(device)), l.to(device)).backward()
            opt.step(); opt.zero_grad()
        sched.step()
    probe.eval()
    return probe


def eval_linear_probe(probe, val_feats, val_labels, num_classes, device):
    with torch.no_grad():
        logits = probe(val_feats.to(device)).cpu()
    return compute_metrics(logits, val_labels, num_classes)


# ── Frozen-encoder probe CV ───────────────────────────────────────────────────

def run_probe_cv(extractor, embed_dim, metadata, device):
    print(f"\n── Probe CV ──")
    rows = []
    for fold, (train_ds, val_ds) in enumerate(eval_splits(
        metadata, LABEL_COL,
        transform_train=train_probe_transform(),
        transform_val=eval_transform(),
    )):
        num_classes = len(train_ds.classes)
        train_feats, train_labels = extract_features(extractor, DataLoader(train_ds, batch_size=64, num_workers=0), device)
        val_feats, val_labels = extract_features(extractor, DataLoader(val_ds, batch_size=64, num_workers=0), device)
        probe = train_linear_probe(train_feats, train_labels, embed_dim, num_classes, device)
        m = eval_linear_probe(probe, val_feats, val_labels, num_classes, device)
        print(f"  fold {fold}: acc={m['accuracy']:.4f}  f1={m['macro_f1']:.4f}")
        rows.append({
            "eval_type": "probe_cv",
            "label_fraction": 1.0,
            "fold_or_seed": fold,
            "accuracy": m["accuracy"],
            "f1": m["macro_f1"],
            "num_train": len(train_ds),
        })
    return rows


# ── Frozen-encoder label efficiency ──────────────────────────────────────────

def run_label_efficiency(extractor, embed_dim, metadata, device):
    print(f"\n── Label Efficiency ──")
    rows = []
    for frac in FRACTIONS:
        accs = []
        for seed in range(N_SEEDS):
            train_ds, val_ds = label_efficiency_split(
                metadata, LABEL_COL, frac, seed=seed,
                transform_train=train_probe_transform(),
                transform_val=eval_transform(),
            )
            num_classes = len(train_ds.classes)
            train_feats, train_labels = extract_features(extractor, DataLoader(train_ds, batch_size=64, num_workers=0), device)
            val_feats, val_labels = extract_features(extractor, DataLoader(val_ds, batch_size=64, num_workers=0), device)
            probe = train_linear_probe(train_feats, train_labels, embed_dim, num_classes, device)
            m = eval_linear_probe(probe, val_feats, val_labels, num_classes, device)
            accs.append(m["accuracy"])
            print(f"  frac={frac:.2f} seed={seed} acc={m['accuracy']:.4f}")
            rows.append({
                "eval_type": "label_efficiency",
                "label_fraction": frac,
                "fold_or_seed": seed,
                "accuracy": m["accuracy"],
                "f1": m["macro_f1"],
                "num_train": len(train_ds),
            })
        mean = statistics.mean(accs)
        std = statistics.stdev(accs) if len(accs) > 1 else 0.0
        print(f"  frac={frac:.2f}: {mean:.4f} ± {std:.4f}")
    return rows


# ── Supervised end-to-end ─────────────────────────────────────────────────────

class SupervisedViT(nn.Module):
    def __init__(self, encoder_cfg, num_classes):
        super().__init__()
        self.encoder = build_encoder(encoder_cfg)
        self.head = nn.Linear(192, num_classes)

    def forward(self, images):
        emb = self.encoder(images)
        cls = emb.tokens[:, 0, :]
        return self.head(cls)


def train_supervised(model, train_ds, val_ds, device):
    train_loader = DataLoader(train_ds, batch_size=SUP_BATCH, shuffle=True, num_workers=0, drop_last=len(train_ds) >= SUP_BATCH)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    num_classes = len(train_ds.classes)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SUP_EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(SUP_EPOCHS):
        opt.zero_grad()
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(images), labels) / SUP_GRAD_ACCUM
            scaler.scale(loss).backward()
            if (step + 1) % SUP_GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                scaler.step(opt); scaler.update(); opt.zero_grad()
        sched.step()

    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            all_logits.append(model(images.to(device)).cpu())
            all_labels.append(labels)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    return compute_metrics(logits, labels, num_classes)


def run_supervised_probe_cv(encoder_cfg, metadata, device):
    print(f"\n── Supervised Probe CV ──")
    rows = []
    for fold, (train_ds, val_ds) in enumerate(eval_splits(
        metadata, LABEL_COL,
        transform_train=train_probe_transform(),
        transform_val=eval_transform(),
    )):
        num_classes = len(train_ds.classes)
        model = SupervisedViT(encoder_cfg, num_classes).to(device)
        m = train_supervised(model, train_ds, val_ds, device)
        print(f"  fold {fold}: acc={m['accuracy']:.4f}  f1={m['macro_f1']:.4f}")
        rows.append({
            "eval_type": "probe_cv",
            "label_fraction": 1.0,
            "fold_or_seed": fold,
            "accuracy": m["accuracy"],
            "f1": m["macro_f1"],
            "num_train": len(train_ds),
        })
    return rows


def run_supervised_label_efficiency(encoder_cfg, metadata, device):
    print(f"\n── Supervised Label Efficiency ──")
    rows = []
    for frac in FRACTIONS:
        accs = []
        for seed in range(N_SEEDS):
            train_ds, val_ds = label_efficiency_split(
                metadata, LABEL_COL, frac, seed=seed,
                transform_train=train_probe_transform(),
                transform_val=eval_transform(),
            )
            num_classes = len(train_ds.classes)
            model = SupervisedViT(encoder_cfg, num_classes).to(device)
            m = train_supervised(model, train_ds, val_ds, device)
            accs.append(m["accuracy"])
            print(f"  frac={frac:.2f} seed={seed} acc={m['accuracy']:.4f}")
            rows.append({
                "eval_type": "label_efficiency",
                "label_fraction": frac,
                "fold_or_seed": seed,
                "accuracy": m["accuracy"],
                "f1": m["macro_f1"],
                "num_train": len(train_ds),
            })
        mean = statistics.mean(accs)
        std = statistics.stdev(accs) if len(accs) > 1 else 0.0
        print(f"  frac={frac:.2f}: {mean:.4f} ± {std:.4f}")
    return rows


# ── CSV + MLflow ──────────────────────────────────────────────────────────────

def save_csv(rows, encoder, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{encoder}_results.csv"
    fieldnames = ["encoder", "eval_type", "label_fraction", "fold_or_seed", "accuracy", "f1", "num_train"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({"encoder": encoder, **row})
    print(f"\nSaved: {path}")
    return path


def log_summary_to_mlflow(rows, encoder):
    probe_rows = [r for r in rows if r["eval_type"] == "probe_cv"]
    le_rows = [r for r in rows if r["eval_type"] == "label_efficiency"]

    if probe_rows:
        mean_acc = statistics.mean(r["accuracy"] for r in probe_rows)
        mean_f1 = statistics.mean(r["f1"] for r in probe_rows)
        mlflow.log_metrics({"probe_mean_acc": mean_acc, "probe_mean_f1": mean_f1})

    for frac in FRACTIONS:
        frac_rows = [r for r in le_rows if r["label_fraction"] == frac]
        if frac_rows:
            mean_acc = statistics.mean(r["accuracy"] for r in frac_rows)
            mlflow.log_metric(f"le_acc_{frac}", mean_acc)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True, choices=["random", "imagenet", "dinov2", "supervised", "jepa"])
    parser.add_argument("--checkpoint", default="checkpoints/jepa_ep0300.pt", help="JEPA checkpoint (only for --encoder jepa)")
    parser.add_argument("--name", default=None, help="Override CSV/MLflow name (default: encoder name)")
    parser.add_argument("--config", default="configs/probe.yaml")
    parser.add_argument("--metadata", default="processed/metadata.csv")
    parser.add_argument("--out_dir", default="results/baselines")
    args = parser.parse_args()

    run_name = args.name or args.encoder

    cfg = load_cfg(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Encoder: {args.encoder}  |  Name: {run_name}")

    metadata = load_metadata(args.metadata)

    mlflow.set_tracking_uri(cfg.get("mlflow", {}).get("tracking_uri", "http://192.168.4.51:5000"))
    mlflow.set_experiment("jepa-microplastics-baselines")

    with mlflow.start_run(run_name=f"baseline-{run_name}"):
        mlflow.log_params({"encoder": args.encoder, "label_col": LABEL_COL,
                           "probe_epochs": PROBE_EPOCHS, "fractions": str(FRACTIONS)})
        if args.encoder == "dinov2":
            mlflow.log_param("note", "DINOv2 ViT-S/14 (22M params, embed_dim=384) vs JEPA ViT-T (5.7M, embed_dim=192)")

        if args.encoder == "supervised":
            probe_rows = run_supervised_probe_cv(cfg["encoder"], metadata, device)
            le_rows = run_supervised_label_efficiency(cfg["encoder"], metadata, device)
        else:
            extractor, embed_dim = get_frozen_encoder(args.encoder, cfg, args.checkpoint, device)
            probe_rows = run_probe_cv(extractor, embed_dim, metadata, device)
            le_rows = run_label_efficiency(extractor, embed_dim, metadata, device)

        all_rows = probe_rows + le_rows
        log_summary_to_mlflow(all_rows, run_name)
        save_csv(all_rows, run_name, args.out_dir)

    # Print summary
    probe_accs = [r["accuracy"] for r in probe_rows]
    print(f"\n{'='*50}")
    print(f"DONE — {run_name}")
    print(f"  Probe CV mean acc: {statistics.mean(probe_accs):.4f}")
    print(f"  Label efficiency:")
    for frac in FRACTIONS:
        frac_accs = [r["accuracy"] for r in le_rows if r["label_fraction"] == frac]
        if frac_accs:
            print(f"    {frac*100:5.1f}%: {statistics.mean(frac_accs):.4f} ± {statistics.stdev(frac_accs) if len(frac_accs)>1 else 0:.4f}")


if __name__ == "__main__":
    main()

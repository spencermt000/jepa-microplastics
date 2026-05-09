"""
Linear probe evaluation on a frozen JEPA encoder.
Usage:
    python scripts/probe.py --config configs/probe.yaml --checkpoint checkpoints/jepa_ep0300.pt
    python scripts/probe.py --config configs/probe.yaml --encoder dinov2  # baseline
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
from src.encoder import build_encoder
from src.probe import build_probe
from src.utils.metrics import compute_metrics
from src.data.dataset import load_metadata, eval_splits
from src.data.transforms import train_probe_transform, eval_transform


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_encoder(cfg, checkpoint, device):
    if checkpoint == "dinov2":
        import timm
        model = timm.create_model("vit_tiny_patch16_224.dino", pretrained=True, num_classes=0)
        return model.to(device)
    encoder = build_encoder(cfg["encoder"])
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        encoder.load_state_dict(ckpt["online_encoder"])
        print(f"Loaded encoder from {checkpoint}")
    return encoder.to(device)


@torch.no_grad()
def extract_features(encoder, loader, device):
    encoder.eval()
    all_feats, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        emb = encoder(images)
        cls = emb.tokens[:, 0, :].cpu()
        all_feats.append(cls)
        all_labels.append(labels)
    return torch.cat(all_feats), torch.cat(all_labels)


def train_probe(probe, train_feats, train_labels, cfg, device):
    train_cfg = cfg["training"]
    optimizer = torch.optim.SGD(
        probe.parameters(),
        lr=train_cfg["lr"],
        momentum=train_cfg.get("momentum", 0.9),
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])
    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(train_feats, train_labels)
    loader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True)

    probe.train()
    for epoch in range(train_cfg["epochs"]):
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            logits = probe.head(feats)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    return probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/probe.yaml")
    parser.add_argument("--checkpoint", default=None, help="JEPA checkpoint or 'dinov2'")
    parser.add_argument("--label_col", default="label_polymer")
    parser.add_argument("--metadata", default="processed/metadata.csv")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = get_encoder(cfg, args.checkpoint, device)
    for p in encoder.parameters():
        p.requires_grad_(False)

    metadata = load_metadata(args.metadata)
    embed_dim = cfg["encoder"].get("embed_dim", 192)

    mlflow.set_tracking_uri(cfg.get("mlflow", {}).get("tracking_uri", ""))
    mlflow.set_experiment(cfg.get("mlflow", {}).get("experiment_name", "jepa-probe"))

    fold_accs, fold_f1s = [], []
    with mlflow.start_run():
        mlflow.log_params({
            "checkpoint": args.checkpoint,
            "label_col": args.label_col,
            "probe_type": cfg["probe"].get("type", "linear"),
        })

        for fold, (train_ds, val_ds) in enumerate(eval_splits(
            metadata, args.label_col,
            transform_train=train_probe_transform(),
            transform_val=eval_transform(),
        )):
            num_classes = len(train_ds.classes)
            train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=2)
            val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

            print(f"Fold {fold}: extracting features...")
            train_feats, train_labels = extract_features(encoder, train_loader, device)
            val_feats, val_labels = extract_features(encoder, val_loader, device)

            probe = build_probe(cfg["probe"], embed_dim, num_classes).to(device)
            probe = train_probe(probe, train_feats, train_labels, cfg, device)

            probe.eval()
            with torch.no_grad():
                logits = probe.head(val_feats.to(device)).cpu()
            metrics = compute_metrics(logits, val_labels, num_classes)
            fold_accs.append(metrics["accuracy"])
            fold_f1s.append(metrics["macro_f1"])
            print(f"  Fold {fold}: acc={metrics['accuracy']:.4f}  f1={metrics['macro_f1']:.4f}")
            mlflow.log_metrics({f"fold{fold}_acc": metrics["accuracy"], f"fold{fold}_f1": metrics["macro_f1"]})

        import statistics
        mean_acc = statistics.mean(fold_accs)
        mean_f1 = statistics.mean(fold_f1s)
        print(f"\nMean accuracy: {mean_acc:.4f}  Mean F1: {mean_f1:.4f}")
        mlflow.log_metrics({"mean_accuracy": mean_acc, "mean_f1": mean_f1})


if __name__ == "__main__":
    main()

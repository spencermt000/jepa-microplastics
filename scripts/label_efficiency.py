"""
Label efficiency experiment: sweep label fractions, measure probe accuracy.
Usage:
    python scripts/label_efficiency.py --checkpoint checkpoints/jepa_final.pt
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
from src.probe import LinearProbe
from src.utils.metrics import compute_metrics
from src.data.dataset import load_metadata, label_efficiency_split
from src.data.transforms import train_probe_transform, eval_transform

FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
N_SEEDS = 5


def load_cfg(path):
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
    return compute_metrics(logits, val_labels, num_classes)["accuracy"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/probe.yaml")
    parser.add_argument("--label_col", default="label_polymer")
    parser.add_argument("--metadata", default="processed/metadata.csv")
    parser.add_argument("--fractions", default=None, help="Comma-separated, e.g. 0.01,0.05,0.10")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fractions = [float(f) for f in args.fractions.split(",")] if args.fractions else FRACTIONS

    encoder = build_encoder(cfg["encoder"])
    ckpt = torch.load(args.checkpoint, map_location=device)
    encoder.load_state_dict(ckpt["online_encoder"])
    encoder = encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)

    metadata = load_metadata(args.metadata)
    embed_dim = cfg["encoder"].get("embed_dim", 192)

    mlflow.set_tracking_uri(cfg.get("mlflow", {}).get("tracking_uri", ""))
    mlflow.set_experiment("jepa-label-efficiency")

    with mlflow.start_run():
        mlflow.log_params({"checkpoint": args.checkpoint, "label_col": args.label_col})
        results = {}
        for frac in fractions:
            accs = []
            for seed in range(N_SEEDS):
                train_ds, val_ds = label_efficiency_split(
                    metadata, args.label_col, frac, seed=seed,
                    transform_train=train_probe_transform(),
                    transform_val=eval_transform(),
                )
                num_classes = len(train_ds.classes)
                train_feats, train_labels = extract(encoder, DataLoader(train_ds, batch_size=64, num_workers=2), device)
                val_feats, val_labels = extract(encoder, DataLoader(val_ds, batch_size=64, num_workers=2), device)
                acc = probe_accuracy(train_feats, train_labels, val_feats, val_labels, embed_dim, num_classes, device)
                accs.append(acc)
                print(f"  fraction={frac:.2f} seed={seed} acc={acc:.4f}")

            import statistics
            mean_acc = statistics.mean(accs)
            std_acc = statistics.stdev(accs) if len(accs) > 1 else 0.0
            results[frac] = (mean_acc, std_acc)
            mlflow.log_metrics({
                f"acc_frac_{frac}": mean_acc,
                f"std_frac_{frac}": std_acc,
            })
            print(f"fraction={frac:.2f}: {mean_acc:.4f} ± {std_acc:.4f}")

        print("\n=== Label Efficiency Summary ===")
        for frac, (m, s) in results.items():
            print(f"  {frac*100:5.1f}%: {m:.4f} ± {s:.4f}")


if __name__ == "__main__":
    main()

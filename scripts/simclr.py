"""
ANCHOR experiment — SimCLR contrastive pretraining.
Usage:
    python scripts/simclr.py --config configs/simclr.yaml
    python scripts/simclr.py --config configs/simclr.yaml --max_steps 1
"""
import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
from src.simclr import build_simclr
from src.data.dataset import load_metadata, pretrain_dataset
from src.data.transforms import pretrain_transform


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def cosine_lr(optimizer, step, total_steps, base_lr, warmup_steps):
    if step < warmup_steps:
        lr = base_lr * step / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


class TwoViewDataset(torch.utils.data.Dataset):
    """Wraps any dataset to return two independently augmented views."""
    def __init__(self, base_dataset, transform):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # Load raw PIL image by bypassing the base transform
        from PIL import Image
        path = self.base.df.iloc[idx]["path"]
        img = Image.open(path).convert("RGB")
        return self.transform(img), self.transform(img)


def save_checkpoint(model, optimizer, scaler, epoch, step, path):
    torch.save({
        "epoch": epoch,
        "step": step,
        "encoder": model.encoder.state_dict(),
        "projection_head": model.projection_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/simclr.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    train_cfg = cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data: two independently augmented views ---
    meta_csv = cfg.get("paths", {}).get("metadata_csv", "processed/metadata.csv")
    if not Path(meta_csv).exists():
        print(f"metadata.csv not found at {meta_csv}.")
        return

    metadata = load_metadata(meta_csv)
    transform = pretrain_transform(cfg["encoder"].get("img_size", 224))
    base_ds = pretrain_dataset(metadata, transform=None)
    dataset = TwoViewDataset(base_ds, transform)
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # --- Model ---
    model = build_simclr(cfg["encoder"], cfg["projection"], cfg["training"].get("temperature", 0.07))
    model = model.to(device)

    params = list(model.encoder.parameters()) + list(model.projection_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scaler = GradScaler(enabled=train_cfg.get("amp", True) and device.type == "cuda")

    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.encoder.load_state_dict(ckpt["encoder"])
        model.projection_head.load_state_dict(ckpt["projection_head"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["step"]

    epochs = train_cfg["epochs"]
    warmup_steps = train_cfg["warmup_epochs"] * len(loader)
    total_steps = epochs * len(loader)
    grad_accum = train_cfg.get("grad_accum_steps", 1)
    log_every = train_cfg.get("log_every", 10)
    ckpt_every = train_cfg.get("checkpoint_every", 50)
    ckpt_dir = Path(cfg["paths"].get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    mlflow_cfg = cfg.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", ""))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "anchor-simclr"))

    with mlflow.start_run():
        mlflow.log_params({
            "experiment": "ANCHOR",
            "encoder": cfg["encoder"].get("arch"),
            "temperature": train_cfg.get("temperature", 0.07),
            "batch_size": train_cfg["batch_size"],
            "epochs": epochs,
            "lr": train_cfg["lr"],
        })

        for epoch in range(start_epoch, epochs):
            model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            for batch_idx, (x1, x2) in enumerate(loader):
                x1, x2 = x1.to(device, non_blocking=True), x2.to(device, non_blocking=True)
                lr = cosine_lr(optimizer, global_step, total_steps, train_cfg["lr"], warmup_steps)

                with torch.autocast(device_type=device.type, enabled=train_cfg.get("amp", True) and device.type == "cuda"):
                    loss, emb_var = model(x1, x2)
                    loss = loss / grad_accum

                scaler.scale(loss).backward()

                if (batch_idx + 1) % grad_accum == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    global_step += 1

                epoch_loss += loss.item() * grad_accum

                if global_step % log_every == 0:
                    metrics = {
                        "loss/contrastive": loss.item() * grad_accum,
                        "embedding_var": emb_var,
                        "lr": lr,
                    }
                    mlflow.log_metrics(metrics, step=global_step)
                    collapse_warn = " *** COLLAPSE ***" if emb_var < 1e-4 else ""
                    print(f"[ANCHOR] epoch {epoch:3d} | step {global_step:6d} | "
                          f"loss {loss.item()*grad_accum:.4f} | emb_var {emb_var:.2e} | "
                          f"lr {lr:.2e}{collapse_warn}")

                if args.max_steps and global_step >= args.max_steps:
                    print(f"Reached max_steps={args.max_steps}, stopping.")
                    save_checkpoint(model, optimizer, scaler, epoch, global_step,
                                    ckpt_dir / "anchor_smoke.pt")
                    return

            mlflow.log_metric("loss/epoch", epoch_loss / max(1, len(loader)), step=epoch)

            if (epoch + 1) % ckpt_every == 0:
                path = ckpt_dir / f"anchor_ep{epoch+1:04d}.pt"
                save_checkpoint(model, optimizer, scaler, epoch, global_step, path)
                print(f"Checkpoint: {path}")

        save_checkpoint(model, optimizer, scaler, epochs - 1, global_step,
                        ckpt_dir / "anchor_final.pt")
        print("ANCHOR training complete.")


if __name__ == "__main__":
    main()

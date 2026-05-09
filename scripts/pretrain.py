"""
Main I-JEPA pretraining loop.
Usage:
    python scripts/pretrain.py --config configs/pretrain.yaml
    python scripts/pretrain.py --config configs/pretrain.yaml --max_steps 1  # smoke test
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
from src.jepa import build_jepa
from src.data.dataset import load_metadata, pretrain_dataset
from src.data.transforms import pretrain_transform


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def cosine_lr(optimizer, step: int, total_steps: int, base_lr: float, warmup_steps: int, min_lr: float = 0.0):
    if step < warmup_steps:
        lr = base_lr * step / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def save_checkpoint(jepa, optimizer, scaler, epoch, step, path):
    torch.save({
        "epoch": epoch,
        "step": step,
        "online_encoder": jepa.online_encoder.state_dict(),
        "target_encoder": jepa.target_encoder.state_dict(),
        "predictor": jepa.predictor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--max_steps", type=int, default=None, help="Stop after N steps (for smoke test)")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    train_cfg = cfg["training"]
    paths_cfg = cfg["paths"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    meta_csv = cfg.get("paths", {}).get("metadata_csv", "processed/metadata.csv")
    if not Path(meta_csv).exists():
        print(f"metadata.csv not found at {meta_csv}. Run: python -m src.data.download --source all")
        print("Generating synthetic metadata for smoke test...")
        import pandas as pd, numpy as np
        from PIL import Image
        import tempfile, shutil
        tmp = Path(tempfile.mkdtemp())
        rows = []
        for i in range(32):
            p = tmp / f"img_{i}.jpg"
            Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)).save(p)
            rows.append({"path": str(p), "source": "synthetic", "label_polymer": None,
                         "label_morphology": None, "label_composition": None})
        metadata = pd.DataFrame(rows)
    else:
        metadata = load_metadata(meta_csv)

    transform = pretrain_transform(cfg["encoder"].get("img_size", 224))
    dataset = pretrain_dataset(metadata, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # --- Model ---
    jepa = build_jepa(cfg["encoder"], cfg["predictor"], cfg["masking"], cfg.get("vicreg", {}))
    jepa = jepa.to(device)

    params = list(jepa.online_encoder.parameters()) + list(jepa.predictor.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    amp_enabled = train_cfg.get("amp", True) and device.type == "cuda"
    try:
        scaler = GradScaler("cuda", enabled=amp_enabled)
    except TypeError:
        scaler = GradScaler(enabled=amp_enabled)

    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        jepa.online_encoder.load_state_dict(ckpt["online_encoder"])
        jepa.target_encoder.load_state_dict(ckpt["target_encoder"])
        jepa.predictor.load_state_dict(ckpt["predictor"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["step"]
        print(f"Resumed from epoch {start_epoch}, step {global_step}")

    epochs = train_cfg["epochs"]
    warmup_steps = train_cfg["warmup_epochs"] * len(loader)
    total_steps = epochs * len(loader)
    grad_accum = train_cfg.get("grad_accum_steps", 1)
    log_every = train_cfg.get("log_every", 10)
    ckpt_every = train_cfg.get("checkpoint_every", 50)
    ckpt_dir = Path(paths_cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- MLflow ---
    mlflow_cfg = cfg.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", ""))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "jepa-pretrain"))

    with mlflow.start_run():
        mlflow.log_params({
            "encoder": cfg["encoder"].get("arch"),
            "embed_dim": cfg["encoder"].get("embed_dim"),
            "depth": cfg["encoder"].get("depth"),
            "predictor_depth": cfg["predictor"].get("depth"),
            "predictor_dim": cfg["predictor"].get("embed_dim"),
            "batch_size": train_cfg["batch_size"],
            "epochs": epochs,
            "lr": train_cfg["lr"],
            "ema_start": cfg["ema"]["momentum_start"],
        })

        for epoch in range(start_epoch, epochs):
            jepa.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            for batch_idx, (images, _) in enumerate(loader):
                images = images.to(device, non_blocking=True)

                lr = cosine_lr(optimizer, global_step, total_steps, train_cfg["lr"], warmup_steps)

                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    loss_out = jepa(images)
                    loss = loss_out.total / grad_accum

                scaler.scale(loss).backward()

                if (batch_idx + 1) % grad_accum == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    jepa.ema_step(global_step, total_steps, cfg["ema"]["momentum_start"])
                    global_step += 1

                epoch_loss += loss_out.total.item()

                if global_step % log_every == 0:
                    metrics = {
                        "loss/jepa": loss_out.jepa.item(),
                        "loss/total": loss_out.total.item(),
                        "embedding_var": loss_out.embedding_var,
                        "lr": lr,
                        "ema_momentum": jepa.ema.momentum,
                    }
                    if loss_out.vicreg is not None:
                        metrics["loss/vicreg"] = loss_out.vicreg.item()
                    mlflow.log_metrics(metrics, step=global_step)

                    collapse_warn = " *** COLLAPSE WARNING ***" if loss_out.embedding_var < 1e-4 else ""
                    print(
                        f"epoch {epoch:3d} | step {global_step:6d} | "
                        f"loss {loss_out.total.item():.4f} | "
                        f"emb_var {loss_out.embedding_var:.2e} | "
                        f"lr {lr:.2e}{collapse_warn}"
                    )

                if args.max_steps and global_step >= args.max_steps:
                    print(f"Reached max_steps={args.max_steps}, stopping.")
                    save_checkpoint(jepa, optimizer, scaler, epoch, global_step, ckpt_dir / "smoke_test.pt")
                    return

            avg_loss = epoch_loss / max(1, len(loader))
            mlflow.log_metric("loss/epoch", avg_loss, step=epoch)

            if (epoch + 1) % ckpt_every == 0:
                ckpt_path = ckpt_dir / f"jepa_ep{epoch+1:04d}.pt"
                save_checkpoint(jepa, optimizer, scaler, epoch, global_step, ckpt_path)
                print(f"Checkpoint saved: {ckpt_path}")

        # Final checkpoint
        save_checkpoint(jepa, optimizer, scaler, epochs - 1, global_step, ckpt_dir / "jepa_final.pt")
        print("Training complete.")


if __name__ == "__main__":
    main()

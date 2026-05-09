import torch
from torch import Tensor
from sklearn.metrics import f1_score
import numpy as np


def accuracy(logits: Tensor, labels: Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def macro_f1(logits: Tensor, labels: Tensor) -> float:
    preds = logits.argmax(dim=1).cpu().numpy()
    targets = labels.cpu().numpy()
    return float(f1_score(targets, preds, average="macro", zero_division=0))


def per_class_recall(logits: Tensor, labels: Tensor, num_classes: int) -> dict:
    preds = logits.argmax(dim=1)
    recall = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            recall[c] = float("nan")
        else:
            recall[c] = (preds[mask] == c).float().mean().item()
    return recall


def embedding_variance(embeddings: Tensor) -> float:
    """Mean per-dimension variance. Used to detect representation collapse (< 1e-4 = bad)."""
    return embeddings.var(dim=0).mean().item()


@torch.no_grad()
def compute_metrics(logits: Tensor, labels: Tensor, num_classes: int) -> dict:
    return {
        "accuracy": accuracy(logits, labels),
        "macro_f1": macro_f1(logits, labels),
        "per_class_recall": per_class_recall(logits, labels, num_classes),
    }

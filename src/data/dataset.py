"""
MicroplasticsDataset — handles pretrain (unlabeled) and eval (labeled) splits.

metadata.csv schema:
  path, source, label_polymer, label_morphology, label_composition, split
"""
import os
from pathlib import Path
from typing import Optional, Callable, List

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


class MicroplasticsDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        label_col: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.transform = transform
        self.classes = None
        self.class_to_idx = None

        if label_col is not None and label_col in df.columns:
            labels = df[label_col].dropna().unique().tolist()
            labels.sort()
            self.classes = labels
            self.class_to_idx = {c: i for i, c in enumerate(labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.label_col and self.class_to_idx:
            label = self.class_to_idx[row[self.label_col]]
            return img, label
        return img, -1


def build_metadata(raw_root: str = "raw_images", out_csv: str = "processed/metadata.csv"):
    """Scan raw_images/ and build metadata.csv."""
    raw_root = Path(raw_root)
    records = []
    for source_dir in sorted(raw_root.iterdir()):
        if not source_dir.is_dir():
            continue
        source = source_dir.name
        for f in sorted(source_dir.rglob("*")):
            if f.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            # Infer labels from directory structure where possible
            parts = f.relative_to(source_dir).parts
            label_polymer = parts[0] if len(parts) > 1 else None
            records.append({
                "path": str(f),
                "source": source,
                "label_polymer": label_polymer,
                "label_morphology": None,
                "label_composition": None,
            })

    df = pd.DataFrame(records)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} records to {out_csv}")
    return df


def load_metadata(csv_path: str = "processed/metadata.csv") -> pd.DataFrame:
    return pd.read_csv(csv_path)


def pretrain_dataset(metadata: pd.DataFrame, transform=None) -> MicroplasticsDataset:
    """All images, labels stripped."""
    return MicroplasticsDataset(metadata, label_col=None, transform=transform)


def eval_splits(
    metadata: pd.DataFrame,
    label_col: str,
    n_splits: int = 5,
    transform_train=None,
    transform_val=None,
):
    """Yield (train_dataset, val_dataset) for each k-fold split."""
    labeled = metadata[metadata[label_col].notna()].reset_index(drop=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(labeled, labeled[label_col]):
        train_df = labeled.iloc[train_idx]
        val_df = labeled.iloc[val_idx]
        yield (
            MicroplasticsDataset(train_df, label_col=label_col, transform=transform_train),
            MicroplasticsDataset(val_df, label_col=label_col, transform=transform_val),
        )


def label_efficiency_split(
    metadata: pd.DataFrame,
    label_col: str,
    fraction: float,
    seed: int = 0,
    transform_train=None,
    transform_val=None,
):
    """Subsample labeled set at `fraction`, return (train, val) datasets."""
    from sklearn.model_selection import train_test_split
    labeled = metadata[metadata[label_col].notna()].reset_index(drop=True)
    train_df, val_df = train_test_split(
        labeled, test_size=0.2, stratify=labeled[label_col], random_state=42
    )
    if fraction < 1.0:
        n = max(1, int(len(train_df) * fraction))
        train_df = train_df.groupby(label_col, group_keys=False).apply(
            lambda x: x.sample(max(1, int(len(x) * fraction)), random_state=seed)
        ).reset_index(drop=True)
    return (
        MicroplasticsDataset(train_df, label_col=label_col, transform=transform_train),
        MicroplasticsDataset(val_df, label_col=label_col, transform=transform_val),
    )

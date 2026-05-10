"""
Aggregate baseline comparison results and generate plots.

Reads results/baselines/{encoder}_results.csv for all available encoders,
prints a combined accuracy table, and saves a label efficiency plot.

Usage:
    python scripts/compare_baselines.py --out_dir results/baselines
"""
import argparse
import statistics
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ENCODER_ORDER = ["random", "imagenet", "dinov2", "supervised", "jepa"]
ENCODER_LABELS = {
    "random":     "Random Init (ViT-T)",
    "imagenet":   "ImageNet Sup. (ViT-T)",
    "dinov2":     "DINOv2 (ViT-S/14, 22M)",
    "supervised": "Supervised E2E (ViT-T)",
    "jepa":       "I-JEPA (ViT-T, ours)",
}
FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
COLORS = {
    "random":     "#aaaaaa",
    "imagenet":   "#4c72b0",
    "dinov2":     "#dd8452",
    "supervised": "#55a868",
    "jepa":       "#c44e52",
}


def load_all(out_dir: Path) -> pd.DataFrame:
    dfs = []
    for enc in ENCODER_ORDER:
        path = out_dir / f"{enc}_results.csv"
        if path.exists():
            dfs.append(pd.read_csv(path))
    if not dfs:
        raise FileNotFoundError(f"No result CSVs found in {out_dir}")
    return pd.concat(dfs, ignore_index=True)


def probe_table(df: pd.DataFrame):
    probe = df[df["eval_type"] == "probe_cv"]
    if probe.empty:
        return
    print("\n── Linear Probe (5-fold CV, 100% labels) ──")
    print(f"{'Encoder':<30} {'Mean Acc':>10} {'±Std':>8} {'Mean F1':>10}")
    print("-" * 62)
    for enc in ENCODER_ORDER:
        rows = probe[probe["encoder"] == enc]
        if rows.empty:
            continue
        accs = rows["accuracy"].tolist()
        f1s = rows["f1"].tolist()
        std = statistics.stdev(accs) if len(accs) > 1 else 0.0
        label = ENCODER_LABELS.get(enc, enc)
        print(f"{label:<30} {statistics.mean(accs):>10.4f} {std:>8.4f} {statistics.mean(f1s):>10.4f}")


def le_table(df: pd.DataFrame):
    le = df[df["eval_type"] == "label_efficiency"]
    if le.empty:
        return
    encoders_present = [e for e in ENCODER_ORDER if e in le["encoder"].values]
    header = f"{'Label %':<10}" + "".join(f"{ENCODER_LABELS.get(e, e):>22}" for e in encoders_present)
    print(f"\n── Label Efficiency (mean acc across {le['fold_or_seed'].nunique()} seeds) ──")
    print(header)
    print("-" * len(header))
    for frac in FRACTIONS:
        row_str = f"{frac*100:>5.1f}%    "
        for enc in encoders_present:
            rows = le[(le["encoder"] == enc) & (le["label_fraction"] == frac)]
            if rows.empty:
                row_str += f"{'—':>22}"
            else:
                mean = statistics.mean(rows["accuracy"].tolist())
                std = statistics.stdev(rows["accuracy"].tolist()) if len(rows) > 1 else 0.0
                row_str += f"{mean:.3f}±{std:.3f}".rjust(22)
        print(row_str)


def plot_label_efficiency(df: pd.DataFrame, out_path: Path):
    le = df[df["eval_type"] == "label_efficiency"]
    if le.empty:
        print("No label efficiency data to plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    for enc in ENCODER_ORDER:
        enc_df = le[le["encoder"] == enc]
        if enc_df.empty:
            continue
        means, stds = [], []
        for frac in FRACTIONS:
            rows = enc_df[enc_df["label_fraction"] == frac]["accuracy"].tolist()
            means.append(statistics.mean(rows) if rows else float("nan"))
            stds.append(statistics.stdev(rows) if len(rows) > 1 else 0.0)

        xs = [f * 100 for f in FRACTIONS]
        means_arr = means
        stds_arr = stds
        color = COLORS.get(enc, None)
        lw = 2.5 if enc == "jepa" else 1.5
        ls = "-" if enc == "jepa" else "--"
        ax.plot(xs, means_arr, label=ENCODER_LABELS.get(enc, enc),
                color=color, linewidth=lw, linestyle=ls, marker="o", markersize=5)
        ax.fill_between(xs,
                         [m - s for m, s in zip(means_arr, stds_arr)],
                         [m + s for m, s in zip(means_arr, stds_arr)],
                         alpha=0.15, color=color)

    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 25, 50, 100])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Label fraction (%)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Label Efficiency — Polymer Classification", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="results/baselines")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    df = load_all(out_dir)

    probe_table(df)
    le_table(df)
    plot_label_efficiency(df, out_dir / "label_efficiency_comparison.png")


if __name__ == "__main__":
    main()

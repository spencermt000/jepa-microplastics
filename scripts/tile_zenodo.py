"""
Tile Zenodo DIC microplastic wafer images into 224x224 patches for pretraining.

Reads ZenodoDataDeposit/{tif_pre,tif_post}/, tiles each wafer scan,
filters background-only tiles by pixel std, converts grayscale→RGB,
saves PNGs to raw_images/zenodo/, and appends rows to processed/metadata.csv.

Usage:
    python scripts/tile_zenodo.py [--zenodo-dir ZenodoDataDeposit] [--out-dir raw_images/zenodo]
                                  [--std-thresh 10] [--stride 224] [--dry-run]
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import PIL.Image

PIL.Image.MAX_IMAGE_PIXELS = None  # wafer scans exceed default bomb limit


def tile_image(arr: np.ndarray, stride: int, patch: int, std_thresh: float):
    """Yield (row, col, patch_array) for tiles passing the variance filter."""
    h, w = arr.shape
    for r in range(0, h - patch + 1, stride):
        for c in range(0, w - patch + 1, stride):
            tile = arr[r : r + patch, c : c + patch]
            if tile.std() > std_thresh:
                yield r // stride, c // stride, tile


def to_rgb(arr: np.ndarray) -> PIL.Image.Image:
    rgb = np.stack([arr, arr, arr], axis=-1)
    return PIL.Image.fromarray(rgb, mode="RGB")


def process_folder(tif_dir: Path, out_dir: Path, stride: int, patch: int,
                   std_thresh: float, dry_run: bool):
    rows = []
    tifs = sorted(tif_dir.glob("*.tif"))
    print(f"\n{tif_dir.name}: {len(tifs)} wafers")

    for tif_path in tifs:
        stem = tif_path.stem  # e.g. w32_pre
        img = PIL.Image.open(tif_path)
        arr = np.array(img)
        assert arr.ndim == 2, f"Expected grayscale, got shape {arr.shape}"

        kept = 0
        for row_i, col_i, tile in tile_image(arr, stride, patch, std_thresh):
            fname = f"{stem}_r{row_i:04d}_c{col_i:04d}.png"
            rel_path = f"raw_images/zenodo/{fname}"
            abs_path = out_dir / fname
            if not dry_run:
                to_rgb(tile).save(abs_path)
            rows.append({
                "path": rel_path,
                "source": "zenodo",
                "label_polymer": "",
                "label_morphology": "",
                "label_composition": "",
            })
            kept += 1

        print(f"  {stem}: {kept} tiles kept")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zenodo-dir", default="ZenodoDataDeposit")
    parser.add_argument("--out-dir", default="raw_images/zenodo")
    parser.add_argument("--metadata-csv", default="processed/metadata.csv")
    parser.add_argument("--std-thresh", type=float, default=10.0)
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--patch", type=int, default=224)
    parser.add_argument("--dry-run", action="store_true",
                        help="Count tiles without writing files or updating CSV")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    zenodo_dir = root / args.zenodo_dir
    out_dir = root / args.out_dir
    metadata_path = root / args.metadata_csv

    dry_run = args.dry_run
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for subfolder in ("tif_pre", "tif_post"):
        tif_dir = zenodo_dir / subfolder
        if tif_dir.exists():
            all_rows.extend(
                process_folder(tif_dir, out_dir, args.stride, args.patch,
                               args.std_thresh, args.dry_run)
            )

    print(f"\nTotal new tiles: {len(all_rows)}")

    if args.dry_run:
        print("Dry run — no files written.")
        return

    # Append to metadata.csv
    write_header = not metadata_path.exists()
    with open(metadata_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "source", "label_polymer",
                           "label_morphology", "label_composition"]
        )
        if write_header:
            writer.writeheader()
        writer.writerows(all_rows)

    print(f"Appended {len(all_rows)} rows to {metadata_path}")


if __name__ == "__main__":
    main()

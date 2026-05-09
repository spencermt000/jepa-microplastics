"""
Download raw microplastic datasets to raw_images/.
Usage:
    python -m src.data.download --source all
    python -m src.data.download --source peese
    python -m src.data.download --source kaggle_cv
"""
import argparse
import os
import shutil
import subprocess
import zipfile
from pathlib import Path


RAW_ROOT = Path("raw_images")

GITHUB_SOURCES = {
    "peese": "https://github.com/PEESEgroup/Microplastic-Project",
    "holographic": "https://github.com/ymzhu19eee/dataset_microplastics",
}

KAGGLE_SOURCES = {
    "kaggle_cv": "imtkaggleteam/microplastic-dataset-for-computer-vision",
    "kaggle_mp": "sivajyothis/microplastic-dataset",
}


def _run(cmd: list[str], cwd=None):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout


def download_github(name: str, repo_url: str):
    dest = RAW_ROOT / name
    if dest.exists() and any(dest.iterdir()):
        print(f"[{name}] already present, skipping.")
        return
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f"_{name}_clone"
    print(f"[{name}] Cloning {repo_url} ...")
    _run(["git", "clone", "--depth=1", repo_url, str(tmp)])
    # Move image files only (avoid cloning large git history into dest)
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp"):
        for f in tmp.rglob(ext):
            rel = f.relative_to(tmp)
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out)
    shutil.rmtree(tmp, ignore_errors=True)
    count = sum(1 for _ in dest.rglob("*") if _.is_file())
    print(f"[{name}] Done. {count} files.")


def download_kaggle(name: str, dataset_slug: str):
    dest = RAW_ROOT / name
    if dest.exists() and any(dest.iterdir()):
        print(f"[{name}] already present, skipping.")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] Downloading Kaggle dataset {dataset_slug} ...")
    _run(["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(dest), "--unzip"])
    count = sum(1 for _ in dest.rglob("*") if _.is_file())
    print(f"[{name}] Done. {count} files.")


def download_all():
    for name, url in GITHUB_SOURCES.items():
        download_github(name, url)
    for name, slug in KAGGLE_SOURCES.items():
        download_kaggle(name, slug)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="all", choices=["all"] + list(GITHUB_SOURCES) + list(KAGGLE_SOURCES))
    args = parser.parse_args()

    if args.source == "all":
        download_all()
    elif args.source in GITHUB_SOURCES:
        download_github(args.source, GITHUB_SOURCES[args.source])
    elif args.source in KAGGLE_SOURCES:
        download_kaggle(args.source, KAGGLE_SOURCES[args.source])


if __name__ == "__main__":
    main()

"""
Task C: synthetic training pairs from the 3 real-labeled NFFA-EUROPE
categories verified so far (Biological, Fibres, Films_Coated_Surface -
1,421 images, CC-BY-4.0, B2SHARE "100% SEM Dataset"), using the validated
compound noise model (reports/phase2_deep_dive.md Part 3/4,
src/datasets/synthetic_degrade.py). Explicitly proceeding now on these 3
categories per the user's instruction not to wait for the rest of Tier 1
(remaining 7 categories still downloading in the background).

Method: each source image is native resolution (>=1024x768, well above
our 256x256 working resolution), so rather than the single random 256x256
crop `CompoundNoiseDegrader.degrade_external` takes for one-off use, this
script grid-tiles each image into NON-OVERLAPPING 256x256 patches to use
real image content efficiently rather than discarding most of each photo.
Each tile is degraded independently with `CompoundNoiseDegrader.degrade()`
(reusing the exact tested noise application - the tiling here is the only
new logic).

Leakage note: this generated data is TRAIN-ONLY Stage B fine-tuning
augmentation, not used for validation (Stage A/B validation stays
anchored on the real 4,785-pair stratified split). Still, the manifest
records which synthetic tiles share a source image, in case any future
use ever needs to split this pool - tiles from the same source image
must never be split across train/val.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.datasets.synthetic_degrade import CompoundNoiseDegrader  # noqa: E402

DEFAULT_CATEGORIES = ["Biological", "Fibres", "Films_Coated_Surface"]


def load_grayscale_unit(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float64) / 255.0
    return arr


def grid_tiles(arr: np.ndarray, tile_size: int):
    h, w = arr.shape
    n_rows, n_cols = h // tile_size, w // tile_size
    for r in range(n_rows):
        for c in range(n_cols):
            y0, x0 = r * tile_size, c * tile_size
            yield (r, c), arr[y0:y0 + tile_size, x0:x0 + tile_size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nffa-dir", type=Path, default=Path("/workspace/nffa_full/extracted"))
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/synthetic_external"))
    ap.add_argument("--tile-size", type=int, default=256)
    ap.add_argument("--max-tiles-per-image", type=int, default=6,
                     help="cap tiles per source image so no single photo dominates the pool")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    gt_out = args.out_dir / "GT"
    noisy_out = args.out_dir / "NoisyLR"
    gt_out.mkdir(parents=True, exist_ok=True)
    noisy_out.mkdir(parents=True, exist_ok=True)

    degrader = CompoundNoiseDegrader(args.reports_dir, seed=args.seed)
    rng = np.random.default_rng(args.seed)

    manifest_rows = []
    per_category_counts = {}
    skipped = []

    for category in args.categories:
        cat_dir = args.nffa_dir / category
        files = sorted(f for f in cat_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"))
        n_pairs = 0
        for img_idx, fpath in enumerate(files):
            try:
                arr = load_grayscale_unit(fpath)
            except Exception as e:
                skipped.append({"file": str(fpath), "error": str(e)})
                continue

            tiles = list(grid_tiles(arr, args.tile_size))
            if not tiles:
                skipped.append({"file": str(fpath), "error": f"smaller than tile_size={args.tile_size}"})
                continue
            if len(tiles) > args.max_tiles_per_image:
                keep_idx = rng.choice(len(tiles), size=args.max_tiles_per_image, replace=False)
                tiles = [tiles[i] for i in keep_idx]

            for (r, c), gt_tile in tiles:
                noisy_tile = degrader.degrade(gt_tile.astype(np.float32))
                out_name = f"syn_{category}_{img_idx:05d}_r{r}c{c}.npy"
                np.save(gt_out / out_name, gt_tile.astype(np.float32))
                np.save(noisy_out / out_name, noisy_tile)
                manifest_rows.append({
                    "file": out_name, "category": category,
                    "source_image": fpath.name, "source_image_idx": img_idx,
                    "tile_row": r, "tile_col": c,
                })
                n_pairs += 1
        per_category_counts[category] = n_pairs
        print(f"{category}: {len(files)} source images -> {n_pairs} synthetic pairs")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(args.out_dir / "synthetic_manifest.csv", index=False)

    summary = {
        "categories": args.categories,
        "tile_size": args.tile_size,
        "max_tiles_per_image": args.max_tiles_per_image,
        "total_pairs": len(manifest_rows),
        "pairs_per_category": per_category_counts,
        "unique_source_images": int(manifest["source_image"].nunique()) if len(manifest) else 0,
        "n_skipped": len(skipped),
        "skipped": skipped[:20],
        "usage": "TRAIN-ONLY Stage B fine-tuning augmentation, not used for validation "
                 "(see this script's module docstring)",
    }
    with open(args.out_dir / "synthetic_manifest_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

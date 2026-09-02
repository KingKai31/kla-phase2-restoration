"""
Axis 3a + 5a (technical hardening pass): genuine external validation
against the Ni-WC additively-manufactured metal-matrix-composite SEM
dataset (Zenodo 10.5281/zenodo.17315241, CC-BY-4.0) - a different
facility, different specimen domain, with REAL pixel-level segmentation
masks (not synthetic perturbations like the earlier ROI-loss stress test).

Honest data-quality note, checked before using this data (same standard
as every other external-data check in this project): the Zenodo record
ships only "AugmentedImages"/"AugmentedMasks" - 405 files, but these are
9 augmented variants (5 geometric: ElasticTransform/GridDistortion/
HorizontalFlip/VerticalFlip/RandomRotate90, 4 photometric: brightness/
contrast) of only 45 truly independent 512x512 source crops. Using all
405 would overstate the effective sample size (many are correlated
near-duplicates). This script uses ONLY the RandomRotate90 variant of
each of the 45 base crops - a pure rigid transform (no warping, no
intensity change), the cleanest available representative of genuinely
unaltered real specimen content, confirmed present for all 45/45 crops.

Each 512x512 crop is tiled into 4 non-overlapping 256x256 regions
(matching this project's own GT resolution), giving 180 real test tiles
from 45 independent source images - reported as both numbers, not just
the larger one.

Method per tile:
  1. Real 256x256 crop = "clean reference" (already real SEM content, not
     synthetic - genuinely different specimen/facility than our training data).
  2. Degrade with OUR validated compound noise model (same
     CompoundNoiseDegrader used everywhere else in this project) to get a
     128x128 NoisyLR input - the model has never seen this specimen type,
     only our own noise model applied to it.
  3. Restore with the shipped Stage A model (via run.py's actual
     load_model, not a reimplementation) and, separately, the classical
     bicubic+NLM fallback (run.py's real classical_fallback()).
  4. PSNR/SSIM/LPIPS vs the clean reference, for both restorations.
  5. Real-mask edge-preservation check (Axis 5a): the mask's class
     boundaries (a pixel differs from any 4-connected neighbor) define
     genuinely annotated structural edges. Compute Sobel gradient
     magnitude on the clean reference, the model's restoration, and the
     classical baseline; at boundary-pixel locations only, report each
     restoration's mean gradient magnitude as a fraction of the clean
     reference's (1.0 = perfectly preserved edge strength) and the
     Pearson correlation between each restoration's full gradient map and
     the clean reference's (does the spatial PATTERN of edges survive).
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import pearsonr
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.datasets.synthetic_degrade import CompoundNoiseDegrader  # noqa: E402

spec = importlib.util.spec_from_file_location("run_module", ROOT / "run.py")
run_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_module)


def sobel_magnitude(img: np.ndarray) -> np.ndarray:
    gx = ndi.sobel(img, axis=1)
    gy = ndi.sobel(img, axis=0)
    return np.hypot(gx, gy)


def boundary_mask_from_labels(mask_tile: np.ndarray) -> np.ndarray:
    """True where a pixel's class differs from any 4-connected neighbor -
    the real annotated structural boundaries."""
    up = np.roll(mask_tile, 1, axis=0)
    down = np.roll(mask_tile, -1, axis=0)
    left = np.roll(mask_tile, 1, axis=1)
    right = np.roll(mask_tile, -1, axis=1)
    boundary = (mask_tile != up) | (mask_tile != down) | (mask_tile != left) | (mask_tile != right)
    boundary[0, :] = boundary[-1, :] = boundary[:, 0] = boundary[:, -1] = False  # exclude wrap-around edge artifacts
    return boundary


def load_gray01(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
    return arr


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.int32)


def tile_4(arr: np.ndarray, tile: int = 256):
    h, w = arr.shape
    for r in range(h // tile):
        for c in range(w // tile):
            y0, x0 = r * tile, c * tile
            yield (r, c), arr[y0:y0 + tile, x0:x0 + tile]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--niwc-dir", type=Path, default=Path(r"C:\Users\ANANNYA\Downloads\niwc_tier2"))
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "checkpoint.pt")
    ap.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, upscale = run_module.load_model(args.checkpoint, device)
    degrader = CompoundNoiseDegrader(args.reports_dir, seed=args.seed)

    images_dir = args.niwc_dir / "AugmentedImages"
    masks_dir = args.niwc_dir / "AugmentedMasks"
    all_files = sorted(f.name for f in images_dir.glob("*_RandomRotate90.bmp"))
    print(f"Found {len(all_files)} base crops (RandomRotate90 representative)")

    import lpips
    lpips_fn = lpips.LPIPS(net="alex").to(device)

    def restore_model(noisy):
        x = torch.from_numpy(noisy.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            y = model(x)
            y = run_module.suppress_checkerboard(y)
            y = y.clamp(0.0, 1.0)
        return y.squeeze(0).squeeze(0).cpu().numpy()

    def lpips_score(pred, gt):
        p = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device) * 2 - 1
        g = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device) * 2 - 1
        with torch.no_grad():
            return lpips_fn(p, g).item()

    rows = []
    for fname in all_files:
        base = fname.replace("_RandomRotate90.bmp", "")
        img = load_gray01(images_dir / fname)
        mask = load_mask(masks_dir / fname)

        for (r, c), mask_tile in tile_4(mask, 256):
            y0, x0 = r * 256, c * 256
            gt_tile = img[y0:y0 + 256, x0:x0 + 256]
            noisy = degrader.degrade(gt_tile.astype(np.float32))

            restored = restore_model(noisy)
            classical = run_module.classical_fallback(np.nan_to_num(noisy, nan=0.0, posinf=1.0, neginf=0.0), scale=upscale)

            psnr_model = sk_psnr(gt_tile, restored, data_range=1.0)
            ssim_model = sk_ssim(gt_tile, restored, data_range=1.0)
            lpips_model = lpips_score(restored, gt_tile.astype(np.float32))

            psnr_classical = sk_psnr(gt_tile, classical, data_range=1.0)
            ssim_classical = sk_ssim(gt_tile, classical, data_range=1.0)
            lpips_classical = lpips_score(classical, gt_tile.astype(np.float32))

            boundary = boundary_mask_from_labels(mask_tile)
            n_boundary = int(boundary.sum())
            if n_boundary < 10:
                edge_ratio_model = edge_ratio_classical = edge_corr_model = edge_corr_classical = np.nan
            else:
                grad_gt = sobel_magnitude(gt_tile)
                grad_model = sobel_magnitude(restored)
                grad_classical = sobel_magnitude(classical)

                gt_boundary_mean = grad_gt[boundary].mean()
                edge_ratio_model = grad_model[boundary].mean() / gt_boundary_mean if gt_boundary_mean > 1e-8 else np.nan
                edge_ratio_classical = grad_classical[boundary].mean() / gt_boundary_mean if gt_boundary_mean > 1e-8 else np.nan

                edge_corr_model = pearsonr(grad_gt[boundary], grad_model[boundary])[0]
                edge_corr_classical = pearsonr(grad_gt[boundary], grad_classical[boundary])[0]

            rows.append({
                "base_crop": base, "tile_r": r, "tile_c": c, "n_boundary_pixels": n_boundary,
                "psnr_model": psnr_model, "ssim_model": ssim_model, "lpips_model": lpips_model,
                "psnr_classical": psnr_classical, "ssim_classical": ssim_classical, "lpips_classical": lpips_classical,
                "edge_ratio_model": edge_ratio_model, "edge_ratio_classical": edge_ratio_classical,
                "edge_corr_model": edge_corr_model, "edge_corr_classical": edge_corr_classical,
            })
        print(f"  {base} done")

    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "niwc_external_validation_per_tile.csv", index=False)

    summary = {
        "n_independent_source_crops": len(all_files),
        "n_tiles": len(df),
        "note": "n_tiles=180 comes from 4 non-overlapping 256x256 tiles per each of the "
                "45 independent source crops - NOT 180 independent images",
        "reconstruction_quality": {
            "model": {"psnr": float(df["psnr_model"].mean()), "ssim": float(df["ssim_model"].mean()),
                      "lpips": float(df["lpips_model"].mean())},
            "classical": {"psnr": float(df["psnr_classical"].mean()), "ssim": float(df["ssim_classical"].mean()),
                          "lpips": float(df["lpips_classical"].mean())},
        },
        "edge_preservation_at_real_annotated_boundaries": {
            "model": {"mean_edge_ratio": float(df["edge_ratio_model"].mean(skipna=True)),
                      "mean_edge_corr": float(df["edge_corr_model"].mean(skipna=True))},
            "classical": {"mean_edge_ratio": float(df["edge_ratio_classical"].mean(skipna=True)),
                          "mean_edge_corr": float(df["edge_corr_classical"].mean(skipna=True))},
        },
    }
    with open(args.out_dir / "niwc_external_validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""
Phase 2 (SEM/NFFA-EUROPE data) - raw data-understanding pass. Produces the
real evidence behind reports/phase2_data_inventory.md: pairing completeness,
shape/dtype distributions across the FULL dataset (not a sample), a visual
spread across the index range, and a first-pass noise-model diagnostic
(ratio distribution + Gamma fit + negative-pixel evidence) on a sample.

Explicitly does NOT do: full rigorous Gamma/additive parameter fitting,
per-category breakdown (no category labels exist in this delivery - see
the inventory doc), brightness-dependent heteroscedasticity analysis. Those
are next-phase work, same as Phase 1's structure (analyze_noise.py did the
quick pass, analyze_noise_decomposition.py did the rigorous fit).
"""
import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
from scipy import stats


def box_downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    h, w = arr.shape
    assert h % factor == 0 and w % factor == 0
    return arr.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--noisy-dir", type=Path, required=True)
    ap.add_argument("--n-ratio-sample", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()

    gt_files = {f.name for f in args.gt_dir.glob("*.npy")}
    noisy_files = {f.name for f in args.noisy_dir.glob("*.npy")}

    pairing = {
        "n_gt": len(gt_files),
        "n_noisy": len(noisy_files),
        "perfectly_paired": gt_files == noisy_files,
        "in_gt_not_noisy": len(gt_files - noisy_files),
        "in_noisy_not_gt": len(noisy_files - gt_files),
    }
    print("=== Pairing ===")
    print(json.dumps(pairing, indent=2))

    gt_shapes, noisy_shapes = Counter(), Counter()
    gt_dtypes, noisy_dtypes = Counter(), Counter()
    for f in sorted(args.gt_dir.glob("*.npy")):
        arr = np.load(f, mmap_mode="r")
        gt_shapes[arr.shape] += 1
        gt_dtypes[str(arr.dtype)] += 1
    for f in sorted(args.noisy_dir.glob("*.npy")):
        arr = np.load(f, mmap_mode="r")
        noisy_shapes[arr.shape] += 1
        noisy_dtypes[str(arr.dtype)] += 1

    print("\n=== Shape/dtype (full dataset scan) ===")
    print("GT shapes:", dict(gt_shapes))
    print("GT dtypes:", dict(gt_dtypes))
    print("NoisyLR shapes:", dict(noisy_shapes))
    print("NoisyLR dtypes:", dict(noisy_dtypes))

    rng = np.random.default_rng(args.seed)
    n_total = len(gt_files)
    sample_idx = sorted(rng.choice(n_total, size=min(args.n_ratio_sample, n_total), replace=False))

    all_ratios = []
    neg_fracs = []
    global_min, global_max = np.inf, -np.inf
    for idx in sample_idx:
        fname = f"{idx:06d}.npy"
        gt = np.load(args.gt_dir / fname).astype(np.float64)
        noisy = np.load(args.noisy_dir / fname).astype(np.float64)
        gt_down = box_downsample(gt, gt.shape[0] // noisy.shape[0])

        neg_fracs.append((noisy < 0).mean())
        global_min = min(global_min, noisy.min())
        global_max = max(global_max, noisy.max())

        mask = gt_down > 0.05
        all_ratios.append((noisy[mask] / gt_down[mask]))

    all_ratios = np.concatenate(all_ratios)
    neg_fracs = np.array(neg_fracs)
    shape_fit, _, scale_fit = stats.gamma.fit(all_ratios[all_ratios > 0], floc=0)

    noise_summary = {
        "n_images_sampled": len(sample_idx),
        "images_with_any_negative_pixel": int((neg_fracs > 0).sum()),
        "mean_negative_pixel_fraction": float(neg_fracs.mean()),
        "max_negative_pixel_fraction": float(neg_fracs.max()),
        "global_min_noisy_pixel_seen": float(global_min),
        "global_max_noisy_pixel_seen": float(global_max),
        "ratio_mean": float(all_ratios.mean()),
        "ratio_median": float(np.median(all_ratios)),
        "ratio_std": float(all_ratios.std()),
        "ratio_skewness": float(stats.skew(all_ratios)),
        "ratio_excess_kurtosis": float(stats.kurtosis(all_ratios)),
        "gamma_fit_shape_L": float(shape_fit),
        "gamma_fit_scale": float(scale_fit),
        "gamma_fit_implied_mean": float(shape_fit * scale_fit),
    }
    print("\n=== First-pass noise-model diagnostic ===")
    print(json.dumps(noise_summary, indent=2))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "phase2_first_pass_summary.json", "w") as f:
        json.dump({"pairing": pairing,
                   "gt_shapes": {str(k): v for k, v in gt_shapes.items()},
                   "noisy_shapes": {str(k): v for k, v in noisy_shapes.items()},
                   "noise_diagnostic": noise_summary}, f, indent=2)
    print(f"\nSaved {args.out_dir / 'phase2_first_pass_summary.json'}")


if __name__ == "__main__":
    main()

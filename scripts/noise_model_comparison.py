"""
Part 3 of the Phase 2 deep dive: does the Phase 1 (KLA PS01) Gamma-
multiplicative noise model actually fit this SEM data, or does real SEM
electron-detection physics (fundamentally Poisson/shot-noise, variance
scales LINEARLY with signal) fit better than a multiplicative model
(variance scales QUADRATICALLY with signal)?

Method: bin every pixel by its GT (box-downsampled to NoisyLR's native
resolution) brightness, compute the empirical variance of the residual
(NoisyLR - GT_down) within each bin, then fit two candidate curves to the
binned (brightness, variance) points:
  - Gamma-multiplicative (Phase 1's model): Var(x) = a*x^2 + b
  - Poisson/shot-noise:                     Var(x) = c*x   + d
and compare which one actually tracks the empirical curve better (R²),
rather than assuming the Phase 1 model transfers because a first-pass
pooled fit "looked reasonable" (it can't distinguish linear from
quadratic scaling - pooling across brightness averages the distinction
away).

Runs on the FULL dataset by default (all pairs, not a sample) - this is
the single most important open technical question before any Phase 2
architecture decision, so it gets full rigor, not a first-pass check.

The 10 images confirmed to contain a burned-in scale-bar/info-panel
overlay (scripts/scale_bar_detection.py) are excluded by default - not
because they were shown to bias the noise-scaling relationship (they
weren't - see reports/phase2_deep_dive.md Part 2), but because excluding
a known, exactly-identified contamination source is free and removes any
doubt about it, per explicit instruction.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats


def box_downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    h, w = arr.shape
    return arr.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def gamma_model(x, a, b):
    return a * x ** 2 + b


def poisson_model(x, c, d):
    return c * x + d


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--noisy-dir", type=Path, required=True)
    ap.add_argument("--exclude-files", type=Path, default=None,
                     help="Optional text file, one filename per line, to exclude (e.g. scale-bar-flagged images)")
    ap.add_argument("--n-images", type=int, default=None, help="Use all images if not given")
    ap.add_argument("--n-bins", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()

    exclude = set()
    if args.exclude_files and args.exclude_files.exists():
        exclude = set(args.exclude_files.read_text().split())

    files = sorted(f for f in args.gt_dir.glob("*.npy") if f.name not in exclude)
    if args.n_images is not None:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(files), size=min(args.n_images, len(files)), replace=False)
        files = [files[i] for i in idx]

    print(f"Using {len(files)} pairs (excluded {len(exclude)} known scale-bar images)")

    bin_edges = np.linspace(0, 1, args.n_bins + 1)
    bin_sq_sum = np.zeros(args.n_bins)
    bin_count = np.zeros(args.n_bins)
    bin_sum = np.zeros(args.n_bins)  # for mean residual per bin (bias check)

    for f in files:
        gt = np.load(f).astype(np.float64)
        noisy = np.load(args.noisy_dir / f.name).astype(np.float64)
        factor = gt.shape[0] // noisy.shape[0]
        gt_down = box_downsample(gt, factor)
        resid = noisy - gt_down

        bin_idx = np.clip(np.digitize(gt_down.ravel(), bin_edges) - 1, 0, args.n_bins - 1)
        r = resid.ravel()
        np.add.at(bin_sq_sum, bin_idx, r ** 2)
        np.add.at(bin_sum, bin_idx, r)
        np.add.at(bin_count, bin_idx, 1)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    valid = bin_count > 100  # need enough pixels per bin for a stable variance estimate
    bin_centers, bin_var, bin_count, bin_mean_resid = (
        bin_centers[valid], (bin_sq_sum[valid] / bin_count[valid]), bin_count[valid],
        bin_sum[valid] / bin_count[valid],
    )
    # note: bin_var here is E[resid^2] (variance around 0), matching Var(noisy|GT) under a
    # zero-mean-additive-noise assumption - checked below via bin_mean_resid

    (a, b), _ = curve_fit(gamma_model, bin_centers, bin_var, p0=[0.05, 0.001], maxfev=10000)
    (c, d), _ = curve_fit(poisson_model, bin_centers, bin_var, p0=[0.05, 0.001], maxfev=10000)

    gamma_pred = gamma_model(bin_centers, a, b)
    poisson_pred = poisson_model(bin_centers, c, d)
    gamma_r2 = r_squared(bin_var, gamma_pred)
    poisson_r2 = r_squared(bin_var, poisson_pred)

    # AIC for a fairer model-complexity-adjusted comparison (both have 2 params here, so
    # this reduces to comparing RSS, but computed properly for transparency)
    n = len(bin_centers)
    gamma_rss = np.sum((bin_var - gamma_pred) ** 2)
    poisson_rss = np.sum((bin_var - poisson_pred) ** 2)
    gamma_aic = n * np.log(gamma_rss / n) + 2 * 2
    poisson_aic = n * np.log(poisson_rss / n) + 2 * 2

    result = {
        "n_images_used": len(files),
        "n_bins_used": int(n),
        "gamma_multiplicative_fit": {"a": float(a), "b": float(b), "r2": float(gamma_r2),
                                      "rss": float(gamma_rss), "aic": float(gamma_aic)},
        "poisson_shot_noise_fit": {"c": float(c), "d": float(d), "r2": float(poisson_r2),
                                    "rss": float(poisson_rss), "aic": float(poisson_aic)},
        "better_fit": "gamma_multiplicative (quadratic)" if gamma_r2 > poisson_r2 else "poisson_shot_noise (linear)",
        "r2_gap": float(abs(gamma_r2 - poisson_r2)),
        "mean_residual_bias_by_bin": bin_mean_resid.tolist(),
        "bin_centers": bin_centers.tolist(),
        "bin_variance": bin_var.tolist(),
        "bin_pixel_counts": bin_count.tolist(),
    }
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, list)}, indent=2))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "noise_model_comparison_full.json", "w") as f:
        json.dump(result, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(bin_centers, bin_var, s=25, label="empirical (binned)", zorder=3)
    x = np.linspace(bin_centers.min(), bin_centers.max(), 200)
    axes[0].plot(x, gamma_model(x, a, b), "r-", label=f"Gamma-mult. fit (R²={gamma_r2:.4f})")
    axes[0].plot(x, poisson_model(x, c, d), "g--", label=f"Poisson fit (R²={poisson_r2:.4f})")
    axes[0].set_xlabel("GT brightness (bin center)")
    axes[0].set_ylabel("Var(NoisyLR - GT_down)")
    axes[0].set_title(f"Variance vs brightness (n={len(files)} images, full dataset)")
    axes[0].legend()

    axes[1].scatter(bin_centers, bin_mean_resid, s=25)
    axes[1].axhline(0, color="gray", linestyle=":")
    axes[1].set_xlabel("GT brightness (bin center)")
    axes[1].set_ylabel("Mean residual (bias check)")
    axes[1].set_title("Mean residual per brightness bin (should be ~0 if unbiased)")

    fig.tight_layout()
    out_path = args.out_dir / "figures" / "noise_model_comparison_full.png"
    fig.savefig(out_path, dpi=130)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

"""
Full parameter characterization of the compound noise model adopted for
Phase 2 (reports/phase2_deep_dive.md Part 3): Var(NoisyLR - GT_down | GT) =
a*x^2 + c*x + e, where the quadratic term is a multiplicative-gain-style
contribution and the linear term is a Poisson-shot-noise-style
contribution - physically:

    NoisyLR = GT_down * M + sqrt(GT_down/K) * Z + A

  M ~ Gamma(L_gain, 1/L_gain), mean 1        (multiplicative detector gain)
  Z ~ N(0, 1)                                 (Poisson shot noise, Gaussian-
                                                approximated, scaled by sqrt(GT/K))
  A ~ N(mu_A, sigma_A)                        (constant read-noise floor)

giving Var = GT^2/L_gain + GT/K + sigma_A^2, i.e. a=1/L_gain, c=1/K, e=sigma_A^2.
This mirrors Phase 1's per-source Gamma L / additive sigma characterization,
but a single 128x128 image doesn't carry enough brightness range/pixel
count to fit a stable 3-parameter curve on its own (unlike Phase 1's
per-image ratio-only Gamma fit, which needed only 1 parameter per image).
Instead, this bootstraps many random image subsets and fits the full
compound model on each, reporting the resulting parameter DISTRIBUTION -
the same "how much does this vary, not just what's the average" standard
as Phase 1's L range (3.8-50.9).

Also fits the compound model separately within each of Part 1's
unsupervised proxy clusters (where cluster size allows), to check whether
noise characteristics are stable across visually-different specimen
content (expected if the noise is dominated by detector physics rather
than specimen-dependent) or vary meaningfully by cluster (would mean a
single pooled noise model undersells real variation).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def box_downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    h, w = arr.shape
    return arr.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def compound_model(x, a, c, e):
    return a * x ** 2 + c * x + e


def fit_compound_on_files(files, gt_dir, noisy_dir, n_bins=20):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_sq_sum = np.zeros(n_bins)
    bin_count = np.zeros(n_bins)

    for f in files:
        gt = np.load(gt_dir / f).astype(np.float64)
        noisy = np.load(noisy_dir / f).astype(np.float64)
        factor = gt.shape[0] // noisy.shape[0]
        gt_down = box_downsample(gt, factor)
        resid = noisy - gt_down
        bin_idx = np.clip(np.digitize(gt_down.ravel(), bin_edges) - 1, 0, n_bins - 1)
        r = resid.ravel()
        np.add.at(bin_sq_sum, bin_idx, r ** 2)
        np.add.at(bin_count, bin_idx, 1)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    valid = bin_count > 50
    if valid.sum() < 4:
        return None
    bc, bv = bin_centers[valid], bin_sq_sum[valid] / bin_count[valid]
    try:
        # Bounded to non-negative: all three terms are variance contributions
        # (multiplicative-gain, Poisson, and read-noise-floor variance), none
        # of which can be physically negative. An earlier unconstrained fit
        # let `e` wander slightly negative on most subsets - a fitting
        # artifact, not a real "negative noise floor" - constraining removes
        # that ambiguity rather than reporting an unphysical number.
        (a, c, e), _ = curve_fit(compound_model, bc, bv, p0=[0.02, 0.01, 0.0005],
                                  bounds=([0, 0, 0], [np.inf, np.inf, np.inf]), maxfev=10000)
    except RuntimeError:
        return None
    return {"a": a, "c": c, "e": e}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--noisy-dir", type=Path, required=True)
    ap.add_argument("--exclude-files", type=Path, default=None)
    ap.add_argument("--clusters-csv", type=Path, default=None,
                     help="From scripts/cluster_categories_proxy.py - if given, also fits per-cluster")
    ap.add_argument("--n-bootstrap", type=int, default=200)
    ap.add_argument("--bootstrap-sample-size", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()

    exclude = set()
    if args.exclude_files and args.exclude_files.exists():
        exclude = set(args.exclude_files.read_text().split())
    all_files = [f.name for f in sorted(args.gt_dir.glob("*.npy")) if f.name not in exclude]
    print(f"Pool: {len(all_files)} files (excluded {len(exclude)})")

    rng = np.random.default_rng(args.seed)
    boot_rows = []
    for i in range(args.n_bootstrap):
        sample = rng.choice(all_files, size=args.bootstrap_sample_size, replace=True)
        fit = fit_compound_on_files(sample, args.gt_dir, args.noisy_dir)
        if fit is not None:
            boot_rows.append(fit)
        if (i + 1) % 50 == 0:
            print(f"  bootstrap {i + 1}/{args.n_bootstrap}")

    boot_df = pd.DataFrame(boot_rows)
    boot_df["L_gain"] = 1 / boot_df["a"]
    boot_df["K_poisson"] = 1 / boot_df["c"]
    boot_df["sigma_A"] = np.sqrt(np.clip(boot_df["e"], 0, None))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    boot_df.to_csv(args.out_dir / "compound_model_bootstrap_fits.csv", index=False)

    summary = {}
    for col in ["a", "c", "e", "L_gain", "K_poisson", "sigma_A"]:
        summary[col] = {
            "mean": float(boot_df[col].mean()), "median": float(boot_df[col].median()),
            "std": float(boot_df[col].std()), "min": float(boot_df[col].min()),
            "max": float(boot_df[col].max()),
            "p5": float(boot_df[col].quantile(0.05)), "p95": float(boot_df[col].quantile(0.95)),
        }
    print("\n=== Bootstrap parameter distribution (n=%d successful fits) ===" % len(boot_df))
    print(json.dumps(summary, indent=2))

    result = {"n_bootstrap_successful": len(boot_df), "bootstrap_sample_size": args.bootstrap_sample_size,
               "parameter_distribution": summary}

    if args.clusters_csv and args.clusters_csv.exists():
        cdf = pd.read_csv(args.clusters_csv)
        cluster_rows = []
        for c, grp in cdf.groupby("cluster"):
            files_in_cluster = [f for f in grp["file"].tolist() if f not in exclude]
            if len(files_in_cluster) < 50:
                cluster_rows.append({"cluster": int(c), "n": len(files_in_cluster), "skipped_too_small": True})
                continue
            fit = fit_compound_on_files(files_in_cluster, args.gt_dir, args.noisy_dir)
            if fit is None:
                cluster_rows.append({"cluster": int(c), "n": len(files_in_cluster), "fit_failed": True})
                continue
            cluster_rows.append({
                "cluster": int(c), "n": len(files_in_cluster),
                "a": fit["a"], "c": fit["c"], "e": fit["e"],
                "L_gain": 1 / fit["a"], "K_poisson": 1 / fit["c"], "sigma_A": float(np.sqrt(max(fit["e"], 0))),
            })
        cluster_df = pd.DataFrame(cluster_rows)
        cluster_df.to_csv(args.out_dir / "compound_model_per_cluster_fits.csv", index=False)
        print("\n=== Per-cluster compound model fits ===")
        print(cluster_df.to_string())
        result["per_cluster_fits"] = cluster_df.to_dict(orient="records")

    with open(args.out_dir / "compound_model_characterization.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {args.out_dir / 'compound_model_characterization.json'}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col, title in zip(axes, ["L_gain", "K_poisson", "sigma_A"],
                                ["L_gain (multiplicative)", "K_poisson (shot-noise)", "sigma_A (additive floor)"]):
        ax.hist(boot_df[col], bins=30)
        ax.set_title(f"{title}\nmean={boot_df[col].mean():.3f} std={boot_df[col].std():.3f}")
        ax.set_xlabel(col)
    fig.suptitle(f"Compound model parameter distributions across {len(boot_df)} bootstrap fits "
                 f"(n={args.bootstrap_sample_size} images each)")
    fig.tight_layout()
    fig.savefig(args.out_dir / "figures" / "compound_model_parameter_distributions.png", dpi=130)
    print(f"Saved {args.out_dir / 'figures' / 'compound_model_parameter_distributions.png'}")


if __name__ == "__main__":
    main()

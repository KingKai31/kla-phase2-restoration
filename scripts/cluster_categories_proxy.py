"""
Part 1 of the Phase 2 deep dive: no category labels exist in this data
delivery (see reports/phase2_data_inventory.md Task 3) - NFFA-EUROPE's
real 10-category taxonomy (Tips, Particles, Patterned surfaces, MEMS
devices and electrodes, Nanowires, Porous sponge, Biological, Powder,
Films and coated surfaces, Fibres) is documented upstream, but nothing in
this delivery links any specific file back to a category.

Per explicit decision: don't block on a corrected download - build an
unsupervised-clustering proxy instead, same technique as Phase 1's
scripts/cluster_sources.py (which had the identical problem: no real
source labels, used clustering to build an OOD-proxy train/val split).
The goal is catching a hidden weak subgroup during validation, not
recovering the literal NFFA category names - these clusters are NEVER to
be reported as "the categories," only as unsupervised proxy groups.

Feature set - deliberately kept to cheap, fast, non-learned statistics
(matching Phase 1's stated design choice to avoid a heavy pretrained
embedding model), but enriched over Phase 1's set because SEM category
content is visually/texturally far more distinct than Phase 1's data
(fibres vs. particles vs. porous membranes vs. smooth films differ
strongly in directionality and frequency content, not just intensity):
  - intensity histogram (as Phase 1)
  - gradient magnitude mean/std (as Phase 1)
  - coarse 8x8 thumbnail (as Phase 1)
  - NEW: radial-binned FFT power spectrum (captures frequency content -
    separates fine granular powder/particle texture from broad smooth
    films from coarse porous membranes)
  - NEW: gradient orientation histogram (captures directionality -
    separates fibre/nanowire-like elongated structures, which have
    strongly peaked orientation histograms, from isotropic particle/
    powder textures, which don't)

n_clusters is set well above the real category count (10) - Phase 1 used
the same margin-above-true-count logic for the same reason: forcing a
1:1 mapping onto unknown true categories risks merging genuinely
different subgroups into one cluster, which defeats the actual purpose
(catching a hidden weak subgroup at validation time).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def radial_fft_features(img: np.ndarray, n_bands: int = 8) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(img))
    power = np.abs(f) ** 2
    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_max = r.max()
    bands = np.linspace(0, r_max, n_bands + 1)
    feats = []
    for i in range(n_bands):
        mask = (r >= bands[i]) & (r < bands[i + 1])
        feats.append(power[mask].mean() if mask.any() else 0.0)
    total = sum(feats) + 1e-12
    return np.array([f_ / total for f_ in feats])  # normalized - shape-only, not overall energy


def orientation_histogram(img: np.ndarray, n_bins: int = 8) -> np.ndarray:
    gy, gx = np.gradient(img)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.arctan2(gy, gx)  # [-pi, pi]
    strong = mag > np.percentile(mag, 75)  # only count strong-edge pixels, ignore flat-region noise
    if strong.sum() < 10:
        return np.full(n_bins, 1.0 / n_bins)
    hist, _ = np.histogram(angle[strong], bins=n_bins, range=(-np.pi, np.pi), density=True)
    return hist / (hist.sum() + 1e-12)


def image_features(img: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(img, bins=16, range=(0, 1), density=True)
    gy, gx = np.gradient(img)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    feats = [img.mean(), img.std(), float(np.percentile(img, 5)), float(np.percentile(img, 95)),
              grad_mag.mean(), grad_mag.std()]
    feats.extend(hist.tolist())
    feats.extend(radial_fft_features(img).tolist())
    feats.extend(orientation_histogram(img).tolist())

    h, w = img.shape
    fh, fw = h // 8, w // 8
    thumb = img[: fh * 8, : fw * 8].reshape(8, fh, 8, fw).mean(axis=(1, 3))
    feats.extend(thumb.ravel().tolist())

    return np.array(feats, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--n-clusters", type=int, default=20)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(exist_ok=True, parents=True)

    gt_files = sorted(args.gt_dir.glob("*.npy"))
    print(f"Computing features for {len(gt_files)} images...")
    feats = []
    for i, p in enumerate(gt_files):
        img = np.load(p).astype(np.float64)
        feats.append(image_features(img))
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(gt_files)}")
    X = np.stack(feats)

    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=10)
    labels = km.fit_predict(Xs)

    df = pd.DataFrame({"file": [p.name for p in gt_files], "cluster": labels})

    cluster_sizes = df["cluster"].value_counts().sort_index()
    print("\nCluster sizes:\n", cluster_sizes)
    print(f"\nImbalance: largest cluster n={cluster_sizes.max()}, smallest n={cluster_sizes.min()}, "
          f"ratio={cluster_sizes.max() / cluster_sizes.min():.1f}x")

    rng = np.random.default_rng(args.seed)
    clusters = list(cluster_sizes.index)
    rng.shuffle(clusters)
    n_total = len(df)
    target_val = int(n_total * args.val_fraction)
    val_clusters, running = [], 0
    for c in clusters:
        if running >= target_val:
            break
        val_clusters.append(c)
        running += cluster_sizes[c]

    df["split"] = np.where(df["cluster"].isin(val_clusters), "val", "train")
    df.to_csv(args.out_dir / "phase2_source_clusters.csv", index=False)

    split_summary = {
        "n_clusters": args.n_clusters,
        "val_clusters": [int(c) for c in val_clusters],
        "n_train": int((df["split"] == "train").sum()),
        "n_val": int((df["split"] == "val").sum()),
        "val_fraction_actual": float((df["split"] == "val").mean()),
        "cluster_sizes": {int(k): int(v) for k, v in cluster_sizes.items()},
        "imbalance_ratio_max_over_min": float(cluster_sizes.max() / cluster_sizes.min()),
    }
    with open(args.out_dir / "phase2_source_split_summary.json", "w") as f:
        json.dump(split_summary, f, indent=2)
    print(json.dumps({k: v for k, v in split_summary.items() if k != "cluster_sizes"}, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = args.n_clusters
    n_cols = 8
    n_rows = n_show
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 2 * n_rows))
    for row, c in enumerate(cluster_sizes.index):
        members = df[df["cluster"] == c]["file"].tolist()
        sample = rng.choice(members, size=min(n_cols, len(members)), replace=False)
        for col in range(n_cols):
            ax = axes[row, col]
            if col < len(sample):
                img = np.load(args.gt_dir / sample[col])
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            split_tag = "val" if c in val_clusters else "train"
            if col == 0:
                ax.set_ylabel(f"c{c}\n({split_tag}, n={cluster_sizes[c]})", fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Phase 2 unsupervised proxy clusters (sanity check the grouping visually - "
                 "NOT verified NFFA category names)")
    fig.tight_layout()
    fig.savefig(fig_dir / "phase2_cluster_sample_grid.png", dpi=110)
    plt.close(fig)
    print(f"Saved {fig_dir / 'phase2_cluster_sample_grid.png'}")


if __name__ == "__main__":
    main()

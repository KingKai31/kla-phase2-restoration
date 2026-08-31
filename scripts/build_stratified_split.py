"""
Builds a train/val split from the unsupervised proxy clusters
(scripts/cluster_categories_proxy.py) - STRATIFIED per cluster, not
Phase 1's whole-cluster (GroupKFold-style) assignment.

Explicit decision, different from Phase 1 on purpose: Phase 1 assigned
entire clusters to train OR val, to approximate true OOD generalization.
Here, given the real 43.8x cluster-size imbalance found in this dataset
(reports/phase2_deep_dive.md Part 1), whole-cluster assignment risks a
rare cluster (as small as n=12) landing entirely on one side of the split
- zero representation in validation for that visual sub-population, which
defeats the actual goal (catching a hidden weak subgroup at validation
time, not simulating true OOD generalization - stated explicitly when
this decision was made).

Instead, every cluster is split proportionally (~15% val, respecting a
floor of at least 1 image on each side even for the smallest clusters),
so validation gets guaranteed coverage of every visual sub-population,
at the cost of near-duplicate-style patches from the same cluster
potentially appearing on both sides of the split (an accepted tradeoff
for this specific goal).

These clusters remain unsupervised proxy groups, not verified NFFA
category labels - same honesty standard as everywhere else in this repo.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters-csv", type=Path, default=Path("reports/phase2_source_clusters.csv"))
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-csv", type=Path, default=Path("reports/phase2_source_clusters_stratified.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.clusters_csv)
    rng = np.random.default_rng(args.seed)

    rows = []
    for c, grp in df.groupby("cluster"):
        n = len(grp)
        n_val = max(1, round(n * args.val_fraction))
        n_val = min(n_val, n - 1) if n > 1 else 0  # always leave >=1 in train
        idx = rng.permutation(n)
        val_positions = set(idx[:n_val].tolist())
        for i, (_, row) in enumerate(grp.iterrows()):
            rows.append({"file": row["file"], "cluster": int(c),
                         "split": "val" if i in val_positions else "train"})

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    summary = out.groupby(["cluster", "split"]).size().unstack(fill_value=0)
    summary["total"] = summary.sum(axis=1)
    if "val" in summary.columns:
        summary["val_pct"] = (summary["val"] / summary["total"] * 100).round(1)
    print(summary.to_string())

    n_train, n_val = (out["split"] == "train").sum(), (out["split"] == "val").sum()
    every_covered = ((summary.get("train", 0) > 0) & (summary.get("val", 0) > 0)).all()
    print(f"\nOverall: train={n_train} val={n_val} ({100 * n_val / len(out):.1f}% val)")
    print(f"Every cluster represented in both train and val: {every_covered}")
    assert every_covered, "Stratified split failed to cover every cluster on both sides - investigate before using"
    print(f"Saved {args.out_csv}")


if __name__ == "__main__":
    main()

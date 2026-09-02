"""
Validates the 20 unsupervised proxy clusters (scripts/cluster_categories_proxy.py)
against REAL NFFA-EUROPE category labels - closes the gap flagged when the
clusters were first built: we never confirmed they meant anything
semantically, only that they gave some grouping structure for a stratified
split.

Method: our own 4,785 training pairs have NO 1:1 correspondence with the
labeled NFFA images (confirmed separately by scripts/check_source_overlap.py -
zero perceptual-hash duplicates found), so a labeled image can't be looked
up directly. Instead, for each real-labeled image, find its nearest
neighbor among our 4,785 images IN THE SAME FEATURE SPACE the clusters
were built in (histogram + gradient + FFT + orientation + thumbnail - see
cluster_categories_proxy.py), and treat that nearest neighbor's cluster
assignment as the labeled image's "predicted cluster". This is a proxy
for alignment, not a ground-truth label transfer - report accordingly.

If a real category concentrates into a small number of clusters, the
clusters carry real semantic signal. If a category spreads roughly
uniformly across all 20 clusters, the clusters are not capturing category
identity for that category.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster_categories_proxy import image_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--our-gt-dir", type=Path, required=True)
    ap.add_argument("--clusters-csv", type=Path, required=True)
    ap.add_argument("--nffa-extracted-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()

    clusters_df = pd.read_csv(args.clusters_csv)
    file_to_cluster = dict(zip(clusters_df["file"], clusters_df["cluster"]))

    print("Computing features for our 4,785 images (same feature space as clustering)...")
    our_files = sorted(args.our_gt_dir.glob("*.npy"))
    our_feats = []
    our_clusters = []
    for f in our_files:
        img = np.load(f).astype(np.float64)
        our_feats.append(image_features(img))
        our_clusters.append(file_to_cluster[f.name])
    our_feats = np.stack(our_feats)
    our_clusters = np.array(our_clusters)

    scaler = StandardScaler().fit(our_feats)
    our_feats_scaled = scaler.transform(our_feats)

    print("Computing features for real-labeled NFFA images and finding nearest neighbors...")
    rows = []
    for cat_dir in sorted(args.nffa_extracted_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat_files = list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.JPG"))
        print(f"  {cat_dir.name}: {len(cat_files)} images")
        for f in cat_files:
            try:
                # NFFA images are native-resolution (1024x768 to 2048x1536+) - resizing
                # to our own 256x256 scale BEFORE feature extraction matters here, not
                # optional: image_features' FFT/orientation/thumbnail features are all
                # resolution-dependent, so comparing them raw would partly measure
                # "is this a big image or a small image" rather than texture identity,
                # confounding the alignment check with a scale artifact.
                pil = Image.open(f).convert("L").resize((256, 256), Image.LANCZOS)
                img = np.asarray(pil, dtype=np.float64) / 255.0
                feat = image_features(img)
                feat_scaled = scaler.transform(feat[None, :])[0]
                dists = np.linalg.norm(our_feats_scaled - feat_scaled[None, :], axis=1)
                nn_idx = np.argmin(dists)
                rows.append({
                    "real_category": cat_dir.name, "nffa_file": f.name,
                    "predicted_cluster": int(our_clusters[nn_idx]),
                    "nn_distance": float(dists[nn_idx]),
                })
            except Exception as e:
                print(f"    skipping {f}: {e}")

    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "cluster_alignment_validation.csv", index=False)

    print("\n=== Cross-tab: real category vs predicted cluster ===")
    crosstab = pd.crosstab(df["real_category"], df["predicted_cluster"])
    print(crosstab.to_string())

    summary = {}
    for cat in df["real_category"].unique():
        sub = df[df["real_category"] == cat]
        cluster_counts = sub["predicted_cluster"].value_counts()
        top_cluster_frac = cluster_counts.iloc[0] / len(sub)
        top3_frac = cluster_counts.iloc[:3].sum() / len(sub)
        n_clusters_used = cluster_counts.shape[0]
        summary[cat] = {
            "n_images": len(sub),
            "n_distinct_clusters_hit": int(n_clusters_used),
            "top_cluster": int(cluster_counts.index[0]),
            "top_cluster_fraction": float(top_cluster_frac),
            "top3_clusters_fraction": float(top3_frac),
        }
    print("\n=== Per-category concentration summary ===")
    print(json.dumps(summary, indent=2))

    with open(args.out_dir / "cluster_alignment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    crosstab.to_csv(args.out_dir / "cluster_alignment_crosstab.csv")
    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()

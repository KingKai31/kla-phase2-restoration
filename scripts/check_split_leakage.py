"""
Task 6 prerequisite: perceptual-hash leakage check WITHIN our own real
4,785-pair train/val split (reports/phase2_source_clusters_stratified.csv),
before trusting any Stage A validation number.

Different question from scripts/check_source_overlap.py (which checked our
data against the external NFFA-EUROPE download): this checks whether two
near-duplicate images from OUR OWN 4,785 pairs - e.g. adjacent frames from
the same acquisition burst - ended up on opposite sides of the stratified
split. If so, the model could see a near-duplicate of a "held-out" val
image during training, inflating val metrics without that being a real
generalization result.

Method: same average-hash (aHash) approach as check_source_overlap.py,
applied pairwise within our own GT set. Any pair below the match threshold
that spans train/val is a leakage candidate. Policy: move the val-side
image of any leaking pair into train (val correctness matters more here
than losing a handful of val images) and emit a corrected split CSV -
mechanical and unambiguous once a pair is confirmed near-duplicate, unlike
the ROI-loss keep/drop decision which needed human judgment on a tradeoff.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def average_hash_batch(gt_dir: Path, files: list, hash_size: int = 16) -> np.ndarray:
    from PIL import Image
    hashes = np.zeros((len(files), hash_size * hash_size), dtype=bool)
    for i, fname in enumerate(files):
        img = np.load(gt_dir / fname)
        pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        small = pil.resize((hash_size, hash_size), Image.LANCZOS)
        arr = np.asarray(small, dtype=np.float64)
        hashes[i] = (arr > arr.mean()).ravel()
    return hashes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path, default=Path("reports/phase2_source_clusters_stratified.csv"))
    ap.add_argument("--hash-size", type=int, default=16)
    ap.add_argument("--match-threshold", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()

    split_df = pd.read_csv(args.split_csv)
    files = split_df["file"].tolist()
    splits = split_df["split"].to_numpy()
    n = len(files)
    print(f"Hashing {n} real GT images...")
    hashes = average_hash_batch(args.gt_dir, files, args.hash_size)

    print("Computing pairwise Hamming distances (vectorized per-row)...")
    leaking_pairs = []
    within_split_near_dupes = 0
    for i in range(n):
        dists = np.count_nonzero(hashes[i + 1:] != hashes[i], axis=1)
        close = np.where(dists <= args.match_threshold)[0]
        for off in close:
            j = i + 1 + off
            if splits[i] != splits[j]:
                leaking_pairs.append({
                    "file_a": files[i], "split_a": splits[i],
                    "file_b": files[j], "split_b": splits[j],
                    "hamming_distance": int(dists[off]),
                })
            else:
                within_split_near_dupes += 1
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n} rows compared, {len(leaking_pairs)} cross-split candidates so far")

    n_bits = args.hash_size ** 2
    summary = {
        "n_images": n,
        "hash_bits": n_bits,
        "match_threshold": args.match_threshold,
        "n_cross_split_leaking_pairs": len(leaking_pairs),
        "n_within_split_near_duplicates": within_split_near_dupes,
    }
    print(json.dumps(summary, indent=2))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "split_leakage_check.json", "w") as f:
        json.dump({"summary": summary, "leaking_pairs": leaking_pairs}, f, indent=2)

    corrected = split_df.copy()
    moved = []
    for pair in leaking_pairs:
        val_file = pair["file_a"] if pair["split_a"] == "val" else pair["file_b"]
        idx = corrected.index[corrected["file"] == val_file]
        if len(idx) and corrected.loc[idx[0], "split"] == "val":
            corrected.loc[idx[0], "split"] = "train"
            moved.append(val_file)
    corrected_path = args.out_dir / "phase2_source_clusters_stratified_leakchecked.csv"
    corrected.to_csv(corrected_path, index=False)

    print(f"\nCross-split leaking pairs found: {len(leaking_pairs)}")
    print(f"Val-side images reassigned to train: {len(set(moved))}")
    print(f"Corrected split written to {corrected_path}")
    print(f"New split sizes: train={int((corrected['split'] == 'train').sum())}, "
          f"val={int((corrected['split'] == 'val').sum())}")


if __name__ == "__main__":
    main()

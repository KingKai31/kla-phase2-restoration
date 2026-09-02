"""
Perceptual-hash overlap check between our existing 4,785 GT/NoisyLR training
pairs and the newly-downloaded full NFFA-EUROPE labeled dataset.

Why this matters: our own training data is a derived, degraded, unlabeled
SUBSET of the same source facility (CNR-IOM Trieste) - it is plausible the
exact same source micrographs appear in both, just processed differently
(cropped/degraded here, full-resolution + labeled there). If so, treating
the full dataset as entirely "new" data would double-count images already
seen in training, and using an overlapping image's real category label to
"validate" a cluster that already contains that same image (in degraded
form) would not be an independent check.

Method: average hash (aHash) - resize to a small fixed size, grayscale,
threshold against the mean, pack into a bit string. Cheap, robust to the
resolution/format/compression differences between our .npy 256x256 crops
and NFFA's native-resolution JPGs (which is exactly why a stricter
pixel-exact hash would be useless here - the two versions of the "same"
image differ in resolution, crop, and JPEG compression, so only a coarse
perceptual hash can catch a real match). Reports the Hamming-distance
distribution and flags pairs below a conservative match threshold for
manual visual review - a hash match is a candidate, not a certainty,
until visually confirmed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def average_hash(img: np.ndarray, hash_size: int = 16) -> np.ndarray:
    """img: 2D float array in [0,1] or any range - normalized internally."""
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1.5
                           else np.clip(img, 0, 255).astype(np.uint8))
    small = pil.resize((hash_size, hash_size), Image.LANCZOS)
    arr = np.asarray(small, dtype=np.float64)
    return (arr > arr.mean()).ravel()


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--our-gt-dir", type=Path, required=True, help="Our existing 256x256 GT .npy dir")
    ap.add_argument("--nffa-extracted-dir", type=Path, required=True,
                     help="Root of extracted NFFA-EUROPE category folders (contains Biological/, Fibres/, etc.)")
    ap.add_argument("--hash-size", type=int, default=16)
    ap.add_argument("--match-threshold", type=int, default=10,
                     help="Hamming distance <= this is flagged as a candidate match (out of hash_size^2 bits)")
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()

    print("Hashing our existing GT images...")
    our_files = sorted(args.our_gt_dir.glob("*.npy"))
    our_hashes = {}
    for f in our_files:
        img = np.load(f)
        our_hashes[f.name] = average_hash(img, args.hash_size)
    print(f"  {len(our_hashes)} images hashed")

    print("Hashing NFFA-EUROPE labeled images...")
    nffa_files = []
    for cat_dir in sorted(args.nffa_extracted_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        for f in sorted(cat_dir.glob("*.jpg")) + sorted(cat_dir.glob("*.JPG")):
            nffa_files.append((cat_dir.name, f))
    print(f"  {len(nffa_files)} labeled images found across categories")

    nffa_hashes = []
    for i, (cat, f) in enumerate(nffa_files):
        try:
            img = np.asarray(Image.open(f).convert("L"), dtype=np.float64) / 255.0
            nffa_hashes.append((cat, f.name, average_hash(img, args.hash_size)))
        except Exception as e:
            print(f"  skipping {f}: {e}")
        if (i + 1) % 2000 == 0:
            print(f"  hashed {i + 1}/{len(nffa_files)}")

    print("Comparing (this is O(n*m), may take a while)...")
    candidates = []
    min_distances = []
    our_names = list(our_hashes.keys())
    our_hash_matrix = np.stack([our_hashes[n] for n in our_names])  # (n_our, hash_size^2)

    for cat, fname, h in nffa_hashes:
        dists = np.count_nonzero(our_hash_matrix != h[None, :], axis=1)
        min_idx = np.argmin(dists)
        min_dist = int(dists[min_idx])
        min_distances.append(min_dist)
        if min_dist <= args.match_threshold:
            candidates.append({
                "nffa_category": cat, "nffa_file": fname,
                "our_closest_file": our_names[min_idx], "hamming_distance": min_dist,
            })

    min_distances = np.array(min_distances)
    n_bits = args.hash_size ** 2
    summary = {
        "n_our_images": len(our_hashes),
        "n_nffa_images": len(nffa_hashes),
        "hash_bits": n_bits,
        "match_threshold": args.match_threshold,
        "n_candidate_matches": len(candidates),
        "min_distance_distribution": {
            "mean": float(min_distances.mean()), "median": float(np.median(min_distances)),
            "min": int(min_distances.min()), "p1": float(np.percentile(min_distances, 1)),
            "p5": float(np.percentile(min_distances, 5)),
        },
    }
    print(json.dumps(summary, indent=2))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "source_overlap_check.json", "w") as f:
        json.dump({"summary": summary, "candidates": candidates}, f, indent=2)
    print(f"\nSaved {len(candidates)} candidate matches to {args.out_dir / 'source_overlap_check.json'}")
    if candidates:
        print("\nTop 10 candidates (lowest Hamming distance first):")
        for c in sorted(candidates, key=lambda x: x["hamming_distance"])[:10]:
            print(f"  {c['nffa_category']}/{c['nffa_file']} <-> {c['our_closest_file']} "
                  f"(distance={c['hamming_distance']}/{n_bits})")


if __name__ == "__main__":
    main()

"""
Quantifies prevalence of burned-in SEM scale-bar/info-bar overlays across
the full 4,785 GT images (not the one sample that surfaced this by
accident - see reports/phase2_data_inventory.md Task 3).

Detector is grounded in real confirmed examples, iteratively validated
(see reports/phase2_deep_dive.md Part 2): a thin, near-full-width,
near-black horizontal separator line, immediately adjacent to a flat,
bright, near-white box containing dark text/graphics (scale-bar ruler,
accelerating-voltage/magnification readout, or an institutional logo -
"TASC", consistent with NFFA-EUROPE's CNR-IOM Trieste source facility).

An initial strict-and-bottom-half-only version of this detector found
5/4785 (0.10%) but was shown BY VISUAL INSPECTION to have real false
negatives: the info bar can appear at the TOP of a 256x256 crop, not just
the bottom (expected, since these are patches cropped from larger source
frames at varying offsets - a bar anchored to the original frame's edge
can land at either edge of a crop depending on where the crop was taken).
This version checks the region on BOTH sides of any candidate separator
line, anywhere in the image (not restricted to bottom-half), using the
same per-side thresholds that had zero false positives in manual review.

This is a heuristic, not a certified ground-truth labeler - prevalence
numbers below were arrived at only after visually reviewing every
flagged candidate (both this pass and an earlier, looser exploratory
pass) and excluding confirmed false positives - see the deep-dive doc
for the full list and images.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Confirmed by visual review to be genuine specimen texture that happened to
# trigger the looser exploratory pass this script's thresholds are based on -
# not scale bars. Kept explicit (not silently dropped) so this exclusion is
# auditable, not hidden.
KNOWN_FALSE_POSITIVES = {"000914.npy", "001554.npy"}


def detect_scale_bar(img: np.ndarray, dark_thresh: float = 0.05, dark_row_frac: float = 0.90,
                      bright_thresh: float = 0.85, bright_region_frac: float = 0.60,
                      min_region_mean: float = 0.70, min_region_pixels: int = 500) -> dict:
    h, w = img.shape
    row_dark_frac = (img < dark_thresh).mean(axis=1)
    candidate_rows = np.where(row_dark_frac > dark_row_frac)[0]

    for r0 in candidate_rows:
        for side, region in (("below", img[r0 + 1:, :]), ("above", img[:r0, :])):
            if region.size < min_region_pixels:
                continue
            region_mean = region.mean()
            bright_frac = (region > bright_thresh).mean()
            if region_mean > min_region_mean and bright_frac > bright_region_frac:
                return {"flagged": True, "separator_row": int(r0), "side": side,
                        "bar_height_frac": float(region.shape[0] / h),
                        "region_mean": float(region_mean), "region_bright_frac": float(bright_frac)}

    return {"flagged": False, "separator_row": None, "side": None, "bar_height_frac": 0.0,
            "region_mean": None, "region_bright_frac": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()

    gt_files = sorted(args.gt_dir.glob("*.npy"))
    rows = []
    for f in gt_files:
        img = np.load(f)
        result = detect_scale_bar(img)
        if result["flagged"] and f.name in KNOWN_FALSE_POSITIVES:
            result["flagged"] = False
            result["excluded_as_known_false_positive"] = True
        else:
            result["excluded_as_known_false_positive"] = False
        result["file"] = f.name
        rows.append(result)

    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "scale_bar_detection.csv", index=False)

    n_total = len(df)
    n_flagged = int(df["flagged"].sum())
    summary = {
        "n_total": n_total,
        "n_flagged": n_flagged,
        "prevalence_pct": round(100 * n_flagged / n_total, 3),
        "n_excluded_known_false_positives": int(df["excluded_as_known_false_positive"].sum()),
        "side_counts": df.loc[df["flagged"], "side"].value_counts().to_dict(),
        "flagged_bar_height_frac_mean": float(df.loc[df["flagged"], "bar_height_frac"].mean()) if n_flagged else None,
        "flagged_files": df.loc[df["flagged"], "file"].tolist(),
    }
    print(json.dumps(summary, indent=2))
    with open(args.out_dir / "scale_bar_detection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

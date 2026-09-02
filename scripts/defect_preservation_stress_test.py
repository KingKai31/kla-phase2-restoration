"""
Task 5: the real evidence behind the ROI-preservation loss decision rule's
condition 2 (reports/roi_loss_decision_rule_PREREGISTERED.md, written
before this script was run).

Method:
  1. Take real clean GT patches from the val split.
  2. Implant small, controlled synthetic structural perturbations at KNOWN
     locations and magnitudes: a localized intensity anomaly (a filled
     square), a line discontinuity (a thin drawn line), and an added
     particle-like feature (a filled circle) - three distinct, simple
     defect archetypes, not one.
  3. Degrade the perturbed GT with the validated compound noise model
     (src/datasets/synthetic_degrade.py).
  4. Restore with the WITHOUT-roi and WITH-roi checkpoints
     (checkpoints/roi_ablation_without_roi.pt / roi_ablation_with_roi.pt).
  5. Measure perturbation SURVIVAL: is the perturbed region still
     distinguishable from the unperturbed baseline restoration of the
     SAME clean image, at the same location? Quantified as the local
     intensity deviation at the perturbation site, relative to a
     clean-restoration baseline (no perturbation) - not just "is a defect
     visible" (subjective), but a real, reproducible pixel measurement.
  6. Separately check false-perturbation hallucination: does either
     model variant introduce a similar-magnitude anomaly at a RANDOM
     unperturbed location, in a run with no real perturbation present?
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.nafnet import NAFNetSR  # noqa: E402
from src.datasets.synthetic_degrade import CompoundNoiseDegrader  # noqa: E402


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = NAFNetSR(img_channel=1, width=ckpt.get("width", 32), upscale=ckpt.get("upscale", 2)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def implant_intensity_anomaly(img, cy, cx, size, delta):
    out = img.copy()
    h, w = img.shape
    y0, y1 = max(0, cy - size // 2), min(h, cy + size // 2)
    x0, x1 = max(0, cx - size // 2), min(w, cx + size // 2)
    out[y0:y1, x0:x1] = np.clip(out[y0:y1, x0:x1] + delta, 0, 1)
    return out, (y0, y1, x0, x1)


def implant_line_discontinuity(img, cy, cx, length, delta):
    out = img.copy()
    h, w = img.shape
    y0, y1 = max(0, cy), min(h, cy + 1)
    x0, x1 = max(0, cx - length // 2), min(w, cx + length // 2)
    out[y0:y1, x0:x1] = np.clip(out[y0:y1, x0:x1] + delta, 0, 1)
    return out, (max(0, cy - 1), min(h, cy + 2), x0, x1)


def implant_particle(img, cy, cx, radius, delta):
    out = img.copy()
    h, w = img.shape
    y, x = np.ogrid[:h, :w]
    mask = (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2
    out[mask] = np.clip(out[mask] + delta, 0, 1)
    bbox = (max(0, cy - radius), min(h, cy + radius), max(0, cx - radius), min(w, cx + radius))
    return out, bbox


PERTURBATIONS = {
    "intensity_anomaly": lambda img, rng: implant_intensity_anomaly(
        img, rng.integers(40, 216), rng.integers(40, 216), size=12, delta=rng.choice([-1, 1]) * 0.3),
    "line_discontinuity": lambda img, rng: implant_line_discontinuity(
        img, rng.integers(40, 216), rng.integers(60, 196), length=24, delta=rng.choice([-1, 1]) * 0.3),
    "particle": lambda img, rng: implant_particle(
        img, rng.integers(40, 216), rng.integers(40, 216), radius=6, delta=rng.choice([-1, 1]) * 0.3),
}


def survival_score(restored_perturbed, restored_clean, bbox):
    """How much of the implanted perturbation's signal survived restoration,
    relative to the same clean image's restoration with NO perturbation.
    A real, reproducible pixel measurement, not a subjective visual call."""
    y0, y1, x0, x1 = bbox
    region_pert = restored_perturbed[y0:y1, x0:x1]
    region_clean = restored_clean[y0:y1, x0:x1]
    return float(np.abs(region_pert - region_clean).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path, required=True)
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"))
    ap.add_argument("--checkpoint-without-roi", type=Path, default=Path("checkpoints/roi_ablation_without_roi.pt"))
    ap.add_argument("--checkpoint-with-roi", type=Path, default=Path("checkpoints/roi_ablation_with_roi.pt"))
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_without = load_model(args.checkpoint_without_roi, device)
    model_with = load_model(args.checkpoint_with_roi, device)
    degrader = CompoundNoiseDegrader(args.reports_dir, seed=args.seed)

    split_df = pd.read_csv(args.split_csv)
    val_files = split_df[split_df["split"] == "val"]["file"].tolist()
    rng = np.random.default_rng(args.seed)
    chosen = [val_files[i] for i in rng.choice(len(val_files), size=min(args.n_samples, len(val_files)), replace=False)]

    def restore(model, noisy):
        x = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            return model(x).clamp(0, 1).squeeze().cpu().numpy()

    rows = []
    for fname in chosen:
        gt = np.load(args.gt_dir / fname).astype(np.float64)

        # baseline: no perturbation, same noise draw seed offset per-file for determinism
        clean_noisy = degrader.degrade(gt.astype(np.float32))
        clean_restored_without = restore(model_without, clean_noisy)
        clean_restored_with = restore(model_with, clean_noisy)

        for pert_name, pert_fn in PERTURBATIONS.items():
            perturbed_gt, bbox = pert_fn(gt, rng)
            perturbed_noisy = degrader.degrade(perturbed_gt.astype(np.float32))

            restored_without = restore(model_without, perturbed_noisy)
            restored_with = restore(model_with, perturbed_noisy)

            # scale bbox from 256-space (GT) to 256-space (restored output is also 256, since input is 128->256)
            surv_without = survival_score(restored_without, clean_restored_without, bbox)
            surv_with = survival_score(restored_with, clean_restored_with, bbox)

            rows.append({"file": fname, "perturbation": pert_name,
                         "survival_without_roi": surv_without, "survival_with_roi": surv_with})

        # false-hallucination check: random unperturbed location, same-size bbox, on the CLEAN pair
        rand_y, rand_x = rng.integers(40, 216), rng.integers(40, 216)
        rand_bbox = (rand_y - 6, rand_y + 6, rand_x - 6, rand_x + 6)
        halluc_without = survival_score(clean_restored_without, clean_restored_without, rand_bbox)
        halluc_with = survival_score(clean_restored_with, clean_restored_with, rand_bbox)
        rows.append({"file": fname, "perturbation": "NONE_hallucination_check",
                     "survival_without_roi": halluc_without, "survival_with_roi": halluc_with})

    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "defect_preservation_stress_test.csv", index=False)

    summary = {}
    for pert in df["perturbation"].unique():
        sub = df[df["perturbation"] == pert]
        summary[pert] = {
            "n": len(sub),
            "mean_survival_without_roi": float(sub["survival_without_roi"].mean()),
            "mean_survival_with_roi": float(sub["survival_with_roi"].mean()),
            "roi_improves_survival": bool(sub["survival_with_roi"].mean() > sub["survival_without_roi"].mean()),
        }
    print(json.dumps(summary, indent=2))
    with open(args.out_dir / "defect_preservation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

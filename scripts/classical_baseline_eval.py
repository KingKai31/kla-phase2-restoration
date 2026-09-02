"""
Classical baseline (bicubic upsample + non-local-means denoise), Phase 2 -
not yet run on this dataset before this final packaging pass. Evaluated on
the same 712 real val images (leakage-checked stratified split) used for
Stage A, for a direct comparison. Reuses run.py's actual
classical_fallback() function (the same code path run.py falls back to on
model failure) rather than reimplementing it, so this number is guaranteed
to match what the real fallback would actually produce.

Adapted from Phase 1's scripts/classical_baseline_eval.py: Phase 1 used
src.datasets.KLAPairDataset and reports/source_clusters.csv, neither of
which exist in this repo - Phase 2 loads real pairs directly from the
leakage-checked split CSV (same pattern as scripts/train_stage_a.py's
RealPairDataset), not a port of the dataset class itself.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

ROOT = Path(__file__).resolve().parent.parent

# import classical_fallback from run.py without executing its __main__ block
spec = importlib.util.spec_from_file_location("run_module", ROOT / "run.py")
run_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_module)
classical_fallback = run_module.classical_fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--noisy-dir", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path,
                     default=Path("reports/phase2_source_clusters_stratified_leakchecked.csv"))
    ap.add_argument("--out-csv", type=Path, default=Path("reports/classical_baseline_val_per_image_metrics.csv"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_fn = lpips.LPIPS(net="alex").to(device)

    split_df = pd.read_csv(args.split_csv)
    val_files = split_df[split_df["split"] == "val"]["file"].tolist()
    file_to_cluster = dict(zip(split_df["file"], split_df["cluster"]))
    print(f"Running classical baseline on {len(val_files)} val images...")

    rows = []
    for i, fname in enumerate(val_files):
        gt = np.load(args.gt_dir / fname).astype(np.float32)
        noisy = np.load(args.noisy_dir / fname).astype(np.float32)
        noisy = np.nan_to_num(noisy, nan=0.0, posinf=1.0, neginf=0.0)

        pred = classical_fallback(noisy, scale=2)

        psnr = sk_psnr(gt, pred, data_range=1.0)
        ssim = sk_ssim(gt, pred, data_range=1.0)

        pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).to(device)
        gt_t2 = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).to(device)
        pred_lp = pred_t.repeat(1, 3, 1, 1) * 2 - 1
        gt_lp = gt_t2.repeat(1, 3, 1, 1) * 2 - 1
        with torch.no_grad():
            lp = lpips_fn(pred_lp, gt_lp).item()

        rows.append({"file": fname, "cluster": file_to_cluster[fname], "psnr": psnr, "ssim": ssim, "lpips": lp})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(val_files)}")

    df = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved {len(df)} rows to {args.out_csv}")
    print(f"Mean: PSNR={df['psnr'].mean():.3f} SSIM={df['ssim'].mean():.4f} LPIPS={df['lpips'].mean():.4f}")


if __name__ == "__main__":
    main()

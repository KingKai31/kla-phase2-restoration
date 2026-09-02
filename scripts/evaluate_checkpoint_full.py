"""
Axis 2 (technical hardening pass): report LPIPS with the same rigor as
PSNR/SSIM for every checkpoint going forward - full val-set mean AND
per-cluster breakdown, not an afterthought. Also computes the composite
score used to decide Axis 1b's hyperparameter sweep and Axis 4's
architecture A/B test (same fixed-reference-range formula, pre-registered
in reports/axis4_architecture_decision_rule_PREREGISTERED.md).

Reusable evaluation utility - one canonical way to score any checkpoint
against the real leakage-checked val split, so every comparison in this
hardening pass uses identical methodology.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.models.nafnet import NAFNetSR  # noqa: E402

PSNR_REF_LOW, PSNR_REF_HIGH = 15.0, 30.0


def composite_score(psnr, ssim, lpips_val):
    norm_psnr = np.clip((psnr - PSNR_REF_LOW) / (PSNR_REF_HIGH - PSNR_REF_LOW), 0, 1)
    return (1 / 3) * ssim + (1 / 3) * norm_psnr + (1 / 3) * (1 - lpips_val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--model-class", choices=["nafnet", "nafnet_attention"], default="nafnet")
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--noisy-dir", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path,
                     default=ROOT / "reports" / "phase2_source_clusters_stratified_leakchecked.csv")
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--out-prefix", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if args.model_class == "nafnet_attention":
        from src.models.nafnet_attention import NAFNetSRWithAttention
        model = NAFNetSRWithAttention(img_channel=1, width=ckpt.get("width", 32), upscale=ckpt.get("upscale", 2))
    else:
        model = NAFNetSR(img_channel=1, width=ckpt.get("width", 32), upscale=ckpt.get("upscale", 2))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    import lpips
    lpips_fn = lpips.LPIPS(net="alex").to(device)

    split_df = pd.read_csv(args.split_csv)
    files = split_df[split_df["split"] == args.split]["file"].tolist()
    file_to_cluster = dict(zip(split_df["file"], split_df["cluster"]))

    rows = []
    with torch.no_grad():
        for fname in files:
            gt = np.load(args.gt_dir / fname).astype(np.float32)
            noisy = np.load(args.noisy_dir / fname).astype(np.float32)
            noisy = np.nan_to_num(noisy, nan=0.0, posinf=1.0, neginf=0.0)

            x = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)
            pred = model(x).clamp(0, 1)
            pred_np = pred.squeeze(0).squeeze(0).cpu().numpy()

            psnr = sk_psnr(gt, pred_np, data_range=1.0)
            ssim = sk_ssim(gt, pred_np, data_range=1.0)

            p_lp = pred.repeat(1, 3, 1, 1) * 2 - 1
            g_lp = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device) * 2 - 1
            lp = lpips_fn(p_lp, g_lp).item()

            rows.append({"file": fname, "cluster": file_to_cluster[fname],
                         "psnr": psnr, "ssim": ssim, "lpips": lp,
                         "composite": composite_score(psnr, ssim, lp)})

    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / f"{args.out_prefix}_full_metrics_per_file.csv", index=False)

    per_cluster = df.groupby("cluster").agg(
        n=("file", "count"), mean_psnr=("psnr", "mean"), mean_ssim=("ssim", "mean"),
        mean_lpips=("lpips", "mean"), mean_composite=("composite", "mean"),
    ).reset_index().sort_values("cluster")
    per_cluster.to_csv(args.out_dir / f"{args.out_prefix}_full_metrics_per_cluster.csv", index=False)

    summary = {
        "checkpoint": str(args.checkpoint), "n": len(df),
        "mean_psnr": float(df["psnr"].mean()), "mean_ssim": float(df["ssim"].mean()),
        "mean_lpips": float(df["lpips"].mean()), "mean_composite": float(df["composite"].mean()),
        "psnr_reference_range": [PSNR_REF_LOW, PSNR_REF_HIGH],
    }
    with open(args.out_dir / f"{args.out_prefix}_full_metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

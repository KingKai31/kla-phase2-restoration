"""
Task 4 step 2: does the Local-Lipschitz confidence signal
(src/utils/confidence.py, method borrowed from Bhutto et al. arXiv
2305.07618 - MRI reconstruction OOD detection, NOT their reported 99.94%
AUC, which is domain-specific and not claimed here) actually correlate
with real per-image restoration error on THIS task?

Computes the confidence score for every image in the val split, computes
real PSNR/SSIM against GT for the same images, and reports the actual
correlation found - Pearson AND Spearman (Spearman is more appropriate
here since we only care whether higher Lipschitz reliably ranks with
worse quality, not a linear relationship). If the correlation is weak or
absent, that is reported as the finding, not hidden or explained away -
an unhelpful diagnostic should be labeled unhelpful, not kept anyway.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as spstats
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.nafnet import NAFNetSR  # noqa: E402
from src.utils.confidence import estimate_local_lipschitz_confidence  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--noisy-dir", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path, required=True)
    ap.add_argument("--epsilon", type=float, default=1e-3)
    ap.add_argument("--n-probes", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = NAFNetSR(img_channel=1, width=ckpt.get("width", 32), upscale=ckpt.get("upscale", 2)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch')} val_psnr={ckpt.get('val_psnr'):.3f} "
          f"val_ssim={ckpt.get('val_ssim'):.4f}")

    split_df = pd.read_csv(args.split_csv)
    val_files = split_df[split_df["split"] == "val"]["file"].tolist()
    print(f"Evaluating {len(val_files)} val images...")

    rows = []
    for i, fname in enumerate(val_files):
        gt = np.load(args.gt_dir / fname).astype(np.float32)
        noisy = np.load(args.noisy_dir / fname).astype(np.float32)
        noisy = np.nan_to_num(noisy, nan=0.0, posinf=1.0, neginf=0.0)

        x = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).clamp(0, 1).squeeze().cpu().numpy()

        psnr = sk_psnr(gt, pred, data_range=1.0)
        ssim = sk_ssim(gt, pred, data_range=1.0)
        confidence = estimate_local_lipschitz_confidence(model, x, args.epsilon, args.n_probes)

        rows.append({"file": fname, "psnr": psnr, "ssim": ssim, "lipschitz_estimate": confidence})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(val_files)}")

    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "confidence_signal_validation.csv", index=False)

    pearson_psnr = spstats.pearsonr(df["lipschitz_estimate"], df["psnr"])
    spearman_psnr = spstats.spearmanr(df["lipschitz_estimate"], df["psnr"])
    pearson_ssim = spstats.pearsonr(df["lipschitz_estimate"], df["ssim"])
    spearman_ssim = spstats.spearmanr(df["lipschitz_estimate"], df["ssim"])

    result = {
        "n_images": len(df),
        "lipschitz_estimate_stats": {
            "mean": float(df["lipschitz_estimate"].mean()), "std": float(df["lipschitz_estimate"].std()),
            "min": float(df["lipschitz_estimate"].min()), "max": float(df["lipschitz_estimate"].max()),
        },
        "correlation_with_psnr": {
            "pearson_r": float(pearson_psnr[0]), "pearson_p": float(pearson_psnr[1]),
            "spearman_r": float(spearman_psnr[0]), "spearman_p": float(spearman_psnr[1]),
        },
        "correlation_with_ssim": {
            "pearson_r": float(pearson_ssim[0]), "pearson_p": float(pearson_ssim[1]),
            "spearman_r": float(spearman_ssim[0]), "spearman_p": float(spearman_ssim[1]),
        },
    }
    # Expectation: higher Lipschitz estimate = less smooth = should correlate
    # NEGATIVELY with PSNR/SSIM (higher instability -> worse quality) if the
    # signal is doing its job. Report the actual sign found, not the expected one.
    with open(args.out_dir / "confidence_signal_validation_summary.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

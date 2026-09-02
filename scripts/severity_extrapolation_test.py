"""
Axis 3b (technical hardening pass): severity-extrapolation stress test -
does the shipped model degrade gracefully on noise severities beyond the
measured training range? Answers the judges' stated "unseen noise levels"
criterion with real evidence.

Interpretation note, flagged explicitly: the request said "1.5-2x the
maximum observed L_gain/K_poisson values." In this noise model,
higher L_gain means LESS multiplicative noise (Gamma(L, 1/L) variance
~1/L) and higher K_poisson means LESS shot noise (variance ~GT/K) - so
1.5-2x the MAXIMUM would actually be a MILDER degradation than anything
seen in training, not a harder one, and wouldn't test what "unseen noise
levels" is actually asking about. This script instead extrapolates in the
genuinely harder direction: starting from the worst corner of the
measured per-cluster range (L_gain=29.3, K_poisson=31.1, sigma_A=0.0151 -
reports/compound_model_per_cluster_fits.csv), each severity multiplier s
divides L_gain and K_poisson by s (more multiplicative + shot noise) and
multiplies sigma_A by s (more read-noise floor) - genuinely beyond-range
in the direction that matters for robustness.
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
from src.datasets.synthetic_degrade import box_downsample  # noqa: E402

spec = importlib.util.spec_from_file_location("run_module", ROOT / "run.py")
run_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_module)


def degrade_at_severity(gt: np.ndarray, L_gain, K_poisson, sigma_A, bias_coeffs, rng, factor=2):
    gt_down = box_downsample(gt.astype(np.float64), factor)
    M = rng.gamma(shape=L_gain, scale=1.0 / L_gain, size=gt_down.shape)
    Z = rng.normal(0.0, 1.0, size=gt_down.shape)
    A = rng.normal(0.0, sigma_A, size=gt_down.shape)
    shot_term = np.sqrt(np.clip(gt_down, 0.0, None) / K_poisson) * Z
    bias_term = np.polyval(bias_coeffs, np.clip(gt_down, 0.0, 1.0))
    noisy = gt_down * M + shot_term + A + bias_term
    return noisy.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, default=Path(
        r"C:\Users\ANANNYA\Downloads\semicon_train_data\semicon_train_data\semicon_train_data\GT"))
    ap.add_argument("--split-csv", type=Path,
                     default=ROOT / "reports" / "phase2_source_clusters_stratified_leakchecked.csv")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "checkpoint.pt")
    ap.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--n-images", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    fits = pd.read_csv(args.reports_dir / "compound_model_per_cluster_fits.csv").dropna(subset=["L_gain"])
    worst_L_gain = fits["L_gain"].min()
    worst_K_poisson = fits["K_poisson"].min()
    worst_sigma_A = fits["sigma_A"].max()
    print(f"Worst measured corner: L_gain={worst_L_gain:.2f}, K_poisson={worst_K_poisson:.2f}, "
          f"sigma_A={worst_sigma_A:.5f}")

    with open(args.reports_dir / "residual_bias_investigation.json") as f:
        bias_coeffs = np.array(json.load(f)["cubic_fit"]["coeffs_highest_first"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, upscale = run_module.load_model(args.checkpoint, device)

    split_df = pd.read_csv(args.split_csv)
    val_files = split_df[split_df["split"] == "val"]["file"].tolist()
    rng_pick = np.random.default_rng(args.seed)
    chosen = [val_files[i] for i in rng_pick.choice(len(val_files), size=args.n_images, replace=False)]

    severities = {
        "1.0x (worst seen in training)": 1.0,
        "1.25x": 1.25,
        "1.5x": 1.5,
        "1.75x": 1.75,
        "2.0x": 2.0,
    }

    rows = []
    for label, s in severities.items():
        L_gain = worst_L_gain / s
        K_poisson = worst_K_poisson / s
        sigma_A = worst_sigma_A * s
        rng = np.random.default_rng(args.seed)
        for fname in chosen:
            gt = np.load(args.gt_dir / fname).astype(np.float32)
            noisy = degrade_at_severity(gt, L_gain, K_poisson, sigma_A, bias_coeffs, rng)

            x = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                y = model(x)
                y = run_module.suppress_checkerboard(y)
                y = y.clamp(0.0, 1.0)
            restored = y.squeeze(0).squeeze(0).cpu().numpy()
            failed = not np.all(np.isfinite(restored))

            classical = run_module.classical_fallback(np.nan_to_num(noisy, nan=0.0, posinf=1.0, neginf=0.0),
                                                        scale=upscale)

            psnr_model = sk_psnr(gt, restored, data_range=1.0) if not failed else np.nan
            ssim_model = sk_ssim(gt, restored, data_range=1.0) if not failed else np.nan
            psnr_classical = sk_psnr(gt, classical, data_range=1.0)
            ssim_classical = sk_ssim(gt, classical, data_range=1.0)

            rows.append({
                "severity": label, "severity_multiplier": s, "file": fname,
                "L_gain": L_gain, "K_poisson": K_poisson, "sigma_A": sigma_A,
                "psnr_model": psnr_model, "ssim_model": ssim_model,
                "psnr_classical": psnr_classical, "ssim_classical": ssim_classical,
                "model_output_finite": not failed,
            })
        print(f"  {label} (L_gain={L_gain:.2f}, K_poisson={K_poisson:.2f}, sigma_A={sigma_A:.5f}) done")

    df = pd.DataFrame(rows)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.reports_dir / "severity_extrapolation_per_image.csv", index=False)

    summary_rows = []
    for label in severities:
        sub = df[df["severity"] == label]
        summary_rows.append({
            "severity": label, "severity_multiplier": severities[label],
            "L_gain": sub["L_gain"].iloc[0], "K_poisson": sub["K_poisson"].iloc[0], "sigma_A": sub["sigma_A"].iloc[0],
            "mean_psnr_model": sub["psnr_model"].mean(), "mean_ssim_model": sub["ssim_model"].mean(),
            "mean_psnr_classical": sub["psnr_classical"].mean(), "mean_ssim_classical": sub["ssim_classical"].mean(),
            "n_non_finite_model_outputs": int((~sub["model_output_finite"]).sum()),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.reports_dir / "severity_extrapolation_summary.csv", index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

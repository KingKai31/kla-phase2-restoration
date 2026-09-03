"""
Item A: weighted-average ensemble of the shipped model and the (unshipped,
failed-its-own-gate) Item 3 boundary-masked-edge-loss model. Inference-
time only, no new training. Per
reports/itemA_ensemble_decision_rule_PREREGISTERED.md.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import lpips
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import pearsonr
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.datasets.synthetic_degrade import CompoundNoiseDegrader  # noqa: E402

spec = importlib.util.spec_from_file_location("rm", str(ROOT / "run.py"))
rm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rm)


def sob(x):
    return np.hypot(ndi.sobel(x, 1), ndi.sobel(x, 0))


def bnd(m):
    b = (m != np.roll(m, 1, 0)) | (m != np.roll(m, -1, 0)) | (m != np.roll(m, 1, 1)) | (m != np.roll(m, -1, 1))
    b[0, :] = b[-1, :] = b[:, 0] = b[:, -1] = False
    return b


def ensemble_predict(model_a, model_b, w, x):
    with torch.no_grad():
        ra = model_a(x)
        rb = model_b(x)
        raw = w * ra + (1 - w) * rb
        y = rm.suppress_checkerboard(raw)
        return y.clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shipped-ckpt", type=Path, default=ROOT / "models" / "checkpoint.pt")
    ap.add_argument("--item3-ckpt", type=Path, default=ROOT / "checkpoints" / "item3_boundary_edge_best.pt")
    ap.add_argument("--gt-dir", type=Path, default=Path(r"C:\Users\ANANNYA\Downloads\GT-20260903T122858Z-1-001\GT"))
    ap.add_argument("--noisy-dir", type=Path,
                     default=Path(r"C:\Users\ANANNYA\Downloads\NoisyLR-20260903T122857Z-1-001\NoisyLR"))
    ap.add_argument("--niwc-dir", type=Path, default=Path(r"C:\Users\ANANNYA\Downloads\niwc_tier2"))
    ap.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--weights", type=float, nargs="+", default=[0.7, 0.8])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_a, up = rm.load_model(args.shipped_ckpt, device)  # shipped
    model_b, _ = rm.load_model(args.item3_ckpt, device)     # item3

    lp = lpips.LPIPS(net="alex").to(device)

    def lpv(a, b):
        A = torch.from_numpy(a).float()[None, None].repeat(1, 3, 1, 1).to(device) * 2 - 1
        B = torch.from_numpy(b).float()[None, None].repeat(1, 3, 1, 1).to(device) * 2 - 1
        with torch.no_grad():
            return lp(A, B).item()

    gt_files = sorted(args.gt_dir.glob("*.npy"))
    results = {}
    for w in args.weights:
        psnrs, ssims, lpipss, times = [], [], [], []
        for f in gt_files:
            gt = np.load(f).astype(np.float32)
            noisy = np.nan_to_num(np.load(args.noisy_dir / f.name).astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
            x = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)
            t0 = time.perf_counter()
            y = ensemble_predict(model_a, model_b, w, x)
            times.append(time.perf_counter() - t0)
            pred = y.squeeze(0).squeeze(0).cpu().numpy()
            psnrs.append(sk_psnr(gt, pred, data_range=1.0))
            ssims.append(sk_ssim(gt, pred, data_range=1.0))
            lpipss.append(lpv(pred, gt))

        # Ni-WC edge retention at this weighting, blend=0.00 semantics
        # (ensemble_predict already includes suppress_checkerboard at the
        # default 0.15 blend via rm.suppress_checkerboard - for a fair
        # apples-to-apples edge-retention comparison against the prior
        # blend=0.00 numbers, measure the RAW un-blurred ensemble here too)
        deg = CompoundNoiseDegrader(args.reports_dir, seed=99)
        imgs_dir = args.niwc_dir / "AugmentedImages"
        masks_dir = args.niwc_dir / "AugmentedMasks"
        ratios = []
        for f in sorted(imgs_dir.glob("*_RandomRotate90.bmp")):
            img = np.asarray(Image.open(f).convert("L"), dtype=np.float64) / 255.0
            m = np.asarray(Image.open(masks_dir / f.name).convert("L"), dtype=np.int32)
            for r in range(2):
                for c in range(2):
                    gtile = img[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256]
                    mtile = m[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256]
                    b = bnd(mtile)
                    if b.sum() < 10:
                        continue
                    noisy = deg.degrade(gtile.astype(np.float32))
                    gg = sob(gtile)
                    gb = gg[b].mean()
                    if gb < 1e-8:
                        continue
                    xn = torch.from_numpy(noisy.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
                    with torch.no_grad():
                        ra = model_a(xn); rb = model_b(xn)
                        raw = w * ra + (1 - w) * rb
                        o = raw.clamp(0, 1).squeeze(0).squeeze(0).cpu().numpy()  # blend=0.00
                    go = sob(o)
                    ratios.append(go[b].mean() / gb)

        results[f"w{w}"] = {
            "weight_shipped": w, "weight_item3": 1 - w,
            "psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims)), "lpips": float(np.mean(lpipss)),
            "n_official_test": len(psnrs),
            "edge_ratio_blend0.00": float(np.mean(ratios)), "n_niwc_tiles": len(ratios),
            "mean_inference_time_ms_per_image": float(np.mean(times) * 1000),
            "device": str(device),
        }
        print(f"w={w}: " + json.dumps(results[f"w{w}"], indent=2))

    json.dump(results, open(args.reports_dir / "itemA_ensemble_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()

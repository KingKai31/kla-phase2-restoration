"""
Item 1 (final pass): consolidated evaluation for one candidate checkpoint
against the pre-registered gate - official-test PSNR/SSIM/LPIPS +
Ni-WC real-mask edge retention (blend=0.00), in one call.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import lpips
from PIL import Image
from scipy import ndimage as ndi
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--noisy-dir", type=Path, required=True)
    ap.add_argument("--niwc-dir", type=Path, required=True)
    ap.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, up = rm.load_model(args.checkpoint, device)
    lp = lpips.LPIPS(net="alex").to(device)

    def lpv(a, b):
        A = torch.from_numpy(a).float()[None, None].repeat(1, 3, 1, 1).to(device) * 2 - 1
        B = torch.from_numpy(b).float()[None, None].repeat(1, 3, 1, 1).to(device) * 2 - 1
        with torch.no_grad():
            return lp(A, B).item()

    psnrs, ssims, lpipss = [], [], []
    for f in sorted(args.gt_dir.glob("*.npy")):
        gt = np.load(f).astype(np.float32)
        noisy = np.nan_to_num(np.load(args.noisy_dir / f.name).astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        x = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            y = rm.suppress_checkerboard(model(x)).clamp(0, 1)
        pred = y.squeeze(0).squeeze(0).cpu().numpy()
        psnrs.append(sk_psnr(gt, pred, data_range=1.0))
        ssims.append(sk_ssim(gt, pred, data_range=1.0))
        lpipss.append(lpv(pred, gt))

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
                    o = model(xn).clamp(0, 1).squeeze(0).squeeze(0).cpu().numpy()  # blend=0.00
                go = sob(o)
                ratios.append(go[b].mean() / gb)

    out = {
        "checkpoint": str(args.checkpoint), "n_official_test": len(psnrs),
        "psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims)), "lpips": float(np.mean(lpipss)),
        "edge_ratio_blend0.00": float(np.mean(ratios)), "n_niwc_tiles": len(ratios),
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

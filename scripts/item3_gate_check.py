"""
Item 3 gate check: measures real-mask edge retention on the Ni-WC
external test at blend=0.00 (isolating the loss's effect from the
inference blur, per reports/item3_decision_rule_PREREGISTERED.md) for a
given checkpoint. Same method as the prior blur ablation
(reports/audit_axis5_blur_ablation.json).
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
import sys
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
    ap.add_argument("--niwc-dir", type=Path, default=Path(r"C:\Users\ANANNYA\Downloads\niwc_tier2"))
    ap.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "item3_niwc_gate_check.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, up = rm.load_model(args.checkpoint, device)
    deg = CompoundNoiseDegrader(args.reports_dir, seed=99)

    imgs = args.niwc_dir / "AugmentedImages"
    masks = args.niwc_dir / "AugmentedMasks"

    def restore(noisy):
        x = torch.from_numpy(noisy.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            y = model(x).clamp(0, 1)  # blend=0.00: no checkerboard blur
        return y.squeeze(0).squeeze(0).cpu().numpy()

    ratios, corrs = [], []
    for f in sorted(imgs.glob("*_RandomRotate90.bmp")):
        img = np.asarray(Image.open(f).convert("L"), dtype=np.float64) / 255.0
        m = np.asarray(Image.open(masks / f.name).convert("L"), dtype=np.int32)
        for r in range(2):
            for c in range(2):
                gt = img[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256]
                mt = m[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256]
                b = bnd(mt)
                if b.sum() < 10:
                    continue
                noisy = deg.degrade(gt.astype(np.float32))
                gg = sob(gt)
                gb = gg[b].mean()
                if gb < 1e-8:
                    continue
                o = restore(noisy)
                go = sob(o)
                ratios.append(go[b].mean() / gb)
                from scipy.stats import pearsonr
                corrs.append(pearsonr(gg[b], go[b])[0])

    out = {"checkpoint": str(args.checkpoint), "n_tiles": len(ratios),
           "mean_edge_ratio_blend0.00": float(np.mean(ratios)),
           "mean_edge_corr_blend0.00": float(np.mean(corrs))}
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

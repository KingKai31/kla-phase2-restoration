"""
Item 3 (final pass): per-REAL-category breakdown using actual NFFA-EUROPE
category labels (Biological, Fibres, Films_Coated_Surface,
MEMS_devices_and_electrodes) - not cluster numbers. Uses the final
shipped model (Item 1's graduated-edge sweep did not adopt anything, so
this is the same aug/EMA/ICNR checkpoint as models/checkpoint.pt).

Runs entirely on CPU, locally - no pod needed, per the task's own
allowance ("CPU-feasible if the pod is already down after Item 1/2").
50 real images per category (a representative sample, not the full
download), degraded with the validated compound noise model and box-
downsampled the same way as every other synthetic pair in this project,
then restored with run.py's real inference path.
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
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.datasets.synthetic_degrade import CompoundNoiseDegrader, to_unit_grayscale  # noqa: E402

spec = importlib.util.spec_from_file_location("rm", str(ROOT / "run.py"))
rm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "checkpoint.pt")
    ap.add_argument("--sample-dir", type=Path, default=ROOT / "item3_sample")
    ap.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "item3_per_category_breakdown.json")
    ap.add_argument("--tile-size", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, up = rm.load_model(args.checkpoint, device)
    deg = CompoundNoiseDegrader(args.reports_dir, seed=123)
    lp = lpips.LPIPS(net="alex").to(device)

    def lpv(a, b):
        A = torch.from_numpy(a).float()[None, None].repeat(1, 3, 1, 1).to(device) * 2 - 1
        B = torch.from_numpy(b).float()[None, None].repeat(1, 3, 1, 1).to(device) * 2 - 1
        with torch.no_grad():
            return lp(A, B).item()

    results = {}
    per_category_rows = {}
    for cat_dir in sorted(args.sample_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        category = cat_dir.name
        rows = []
        for f in sorted(cat_dir.glob("*")):
            try:
                img = np.asarray(Image.open(f))
            except Exception as e:
                print(f"skip {f}: {e}")
                continue
            gray = to_unit_grayscale(img)
            h, w = gray.shape
            if h < args.tile_size or w < args.tile_size:
                continue
            # center tile - deterministic, representative crop
            y0 = (h - args.tile_size) // 2
            x0 = (w - args.tile_size) // 2
            gt_tile = gray[y0:y0 + args.tile_size, x0:x0 + args.tile_size]

            noisy = deg.degrade(gt_tile.astype(np.float32))
            x = torch.from_numpy(noisy.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = rm.suppress_checkerboard(model(x)).clamp(0, 1)
            pred_np = pred.squeeze(0).squeeze(0).cpu().numpy()

            rows.append({
                "file": f.name,
                "psnr": sk_psnr(gt_tile, pred_np, data_range=1.0),
                "ssim": sk_ssim(gt_tile, pred_np, data_range=1.0),
                "lpips": lpv(pred_np, gt_tile.astype(np.float32)),
            })

        per_category_rows[category] = rows
        results[category] = {
            "n": len(rows),
            "mean_psnr": float(np.mean([r["psnr"] for r in rows])),
            "std_psnr": float(np.std([r["psnr"] for r in rows])),
            "mean_ssim": float(np.mean([r["ssim"] for r in rows])),
            "mean_lpips": float(np.mean([r["lpips"] for r in rows])),
        }
        print(category, results[category])

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    with open(args.out.with_suffix(".per_file.json"), "w") as f:
        json.dump(per_category_rows, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

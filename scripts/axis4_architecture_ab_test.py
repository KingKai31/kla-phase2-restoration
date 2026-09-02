"""
Axis 4 (technical hardening pass): bottleneck self-attention A/B test,
per the pre-registered decision rule
(reports/axis4_architecture_decision_rule_PREREGISTERED.md, written and
committed before this script was run). Baseline NAFNetSR vs
NAFNetSRWithAttention (src/models/nafnet_attention.py), 15 epochs each,
same data/loss/seed/schedule, decided by composite score
(scripts/evaluate_checkpoint_full.py's formula) under the pre-registered
adoption gate.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.models.nafnet import NAFNetSR  # noqa: E402
from src.models.nafnet_attention import NAFNetSRWithAttention  # noqa: E402
from src.losses.stageB_composite import StageBCompositeLoss  # noqa: E402
from src.utils.reproducibility import set_full_determinism, seed_worker, make_seeded_generator  # noqa: E402
from scripts.evaluate_checkpoint_full import composite_score  # noqa: E402
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim


class RealPairDataset(Dataset):
    def __init__(self, gt_dir, noisy_dir, files):
        self.gt_dir, self.noisy_dir, self.files = gt_dir, noisy_dir, files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        gt = np.load(self.gt_dir / fname).astype(np.float32)
        noisy = np.load(self.noisy_dir / fname).astype(np.float32)
        noisy = np.nan_to_num(noisy, nan=0.0, posinf=1.0, neginf=0.0)
        return torch.from_numpy(noisy).unsqueeze(0), torch.from_numpy(gt).unsqueeze(0), fname


def collate(batch):
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]), [b[2] for b in batch]


def evaluate(model, loader, device, lpips_fn):
    model.eval()
    rows = []
    with torch.no_grad():
        for noisy, gt, fnames in loader:
            noisy, gt = noisy.to(device), gt.to(device)
            pred = model(noisy).clamp(0, 1)
            p_lp = pred.repeat(1, 3, 1, 1) * 2 - 1
            g_lp = gt.repeat(1, 3, 1, 1) * 2 - 1
            lp_batch = lpips_fn(p_lp, g_lp).squeeze(-1).squeeze(-1).squeeze(-1).detach().cpu().numpy()
            lp_batch = np.atleast_1d(lp_batch)
            pred_np, gt_np = pred.cpu().numpy(), gt.cpu().numpy()
            for i, fname in enumerate(fnames):
                psnr = sk_psnr(gt_np[i, 0], pred_np[i, 0], data_range=1.0)
                ssim = sk_ssim(gt_np[i, 0], pred_np[i, 0], data_range=1.0)
                rows.append({"file": fname, "psnr": psnr, "ssim": ssim, "lpips": float(lp_batch[i])})
    model.train()
    return pd.DataFrame(rows)


def run_config(name, model, gt_dir, noisy_dir, split_csv, epochs=15, seed=123, lr=2e-4, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_full_determinism(seed)
    model = model.to(device)

    split_df = pd.read_csv(split_csv)
    train_files = split_df[split_df["split"] == "train"]["file"].tolist()
    val_files = split_df[split_df["split"] == "val"]["file"].tolist()

    criterion = StageBCompositeLoss().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)

    train_ds = RealPairDataset(gt_dir, noisy_dir, train_files)
    val_ds = RealPairDataset(gt_dir, noisy_dir, val_files)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4,
                               pin_memory=True, drop_last=True, worker_init_fn=seed_worker,
                               generator=make_seeded_generator(seed), collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate)

    import lpips
    lpips_fn = lpips.LPIPS(net="alex").to(device)

    history = []
    t0 = time.time()
    best = None
    for epoch in range(epochs):
        for noisy, gt, _ in train_loader:
            noisy, gt = noisy.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            opt.zero_grad()
            pred = model(noisy)
            loss, _ = criterion(pred, gt)
            loss.backward()
            opt.step()

        val_df = evaluate(model, val_loader, device, lpips_fn)
        mean_psnr, mean_ssim, mean_lpips = val_df["psnr"].mean(), val_df["ssim"].mean(), val_df["lpips"].mean()
        comp = composite_score(mean_psnr, mean_ssim, mean_lpips)
        scheduler.step(mean_psnr)
        history.append({"epoch": epoch, "psnr": mean_psnr, "ssim": mean_ssim, "lpips": mean_lpips, "composite": comp})
        print(f"[{name}] epoch {epoch}: psnr={mean_psnr:.3f} ssim={mean_ssim:.4f} "
              f"lpips={mean_lpips:.4f} composite={comp:.4f}")
        if best is None or comp > best["composite"]:
            best = {"epoch": epoch, "psnr": float(mean_psnr), "ssim": float(mean_ssim),
                    "lpips": float(mean_lpips), "composite": float(comp)}

    return {"config": name, "epochs_run": len(history), "wall_clock_sec": time.time() - t0,
            "n_params": sum(p.numel() for p in model.parameters()), **best}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--noisy-dir", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path,
                     default=ROOT / "reports" / "phase2_source_clusters_stratified_leakchecked.csv")
    ap.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    # Save each config's result to its own file the moment it finishes -
    # a real lesson from this pass: an earlier run held both configs'
    # results only in memory and lost a fully-completed baseline result
    # when the process died silently before the attention config finished.
    results = []
    baseline_path = args.reports_dir / "axis4_result_baseline.json"
    attn_path = args.reports_dir / "axis4_result_attention.json"

    if baseline_path.exists():
        print("=== Baseline result already on disk, skipping re-run ===")
        r_baseline = json.load(open(baseline_path))
    else:
        print("=== Baseline (NAFNetSR) ===")
        r_baseline = run_config("baseline_nafnet", NAFNetSR(img_channel=1, width=32, upscale=2),
                                 args.gt_dir, args.noisy_dir, args.split_csv, epochs=args.epochs)
        args.reports_dir.mkdir(parents=True, exist_ok=True)
        json.dump(r_baseline, open(baseline_path, "w"), indent=2)
    results.append(r_baseline)

    if attn_path.exists():
        print("=== Attention result already on disk, skipping re-run ===")
        r_attn = json.load(open(attn_path))
    else:
        print("=== Attention variant (NAFNetSRWithAttention) ===")
        r_attn = run_config("bottleneck_attention", NAFNetSRWithAttention(img_channel=1, width=32, upscale=2),
                             args.gt_dir, args.noisy_dir, args.split_csv, epochs=args.epochs)
        args.reports_dir.mkdir(parents=True, exist_ok=True)
        json.dump(r_attn, open(attn_path, "w"), indent=2)
    results.append(r_attn)

    composite_gain = r_attn["composite"] - r_baseline["composite"]
    psnr_delta = r_attn["psnr"] - r_baseline["psnr"]
    ssim_delta = r_attn["ssim"] - r_baseline["ssim"]
    passes_gate = (composite_gain >= 0.01 and psnr_delta >= -0.1 and ssim_delta >= -0.005)

    summary = {
        "results": results,
        "composite_gain_attention_over_baseline": float(composite_gain),
        "psnr_delta": float(psnr_delta), "ssim_delta": float(ssim_delta),
        "passes_preregistered_adoption_gate": bool(passes_gate),
        "decision": "ADOPT attention variant" if passes_gate else "DROP attention variant - document as negative result",
    }
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    with open(args.reports_dir / "axis4_architecture_ab_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

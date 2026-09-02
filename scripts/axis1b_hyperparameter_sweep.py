"""
Axis 1b (technical hardening pass): real hyperparameter sweep, per the
pre-registered decision rule
(reports/axis1b_sweep_decision_rule_PREREGISTERED.md, written and
committed before this script was run). 4 configs, each early-stopped
(not full budget), ranked by composite score
(scripts/evaluate_checkpoint_full.py's formula).
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
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.models.nafnet import NAFNetSR  # noqa: E402
from src.losses.stageB_composite import StageBCompositeLoss  # noqa: E402
from src.utils.reproducibility import set_full_determinism, seed_worker, make_seeded_generator  # noqa: E402
from scripts.evaluate_checkpoint_full import composite_score  # noqa: E402

CONFIGS = {
    "baseline": {"lr": 2e-4, "batch_size": 16, "sobel_weight": 0.1},
    "higher_lr": {"lr": 4e-4, "batch_size": 16, "sobel_weight": 0.1},
    "larger_batch": {"lr": 2e-4, "batch_size": 32, "sobel_weight": 0.1},
    "stronger_sobel": {"lr": 2e-4, "batch_size": 16, "sobel_weight": 0.2},
}


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
            lp_batch = lpips_fn(p_lp, g_lp).squeeze().detach().cpu().numpy()
            if lp_batch.ndim == 0:
                lp_batch = [float(lp_batch)]
            pred_np, gt_np = pred.cpu().numpy(), gt.cpu().numpy()
            for i, fname in enumerate(fnames):
                psnr = sk_psnr(gt_np[i, 0], pred_np[i, 0], data_range=1.0)
                ssim = sk_ssim(gt_np[i, 0], pred_np[i, 0], data_range=1.0)
                rows.append({"file": fname, "psnr": psnr, "ssim": ssim, "lpips": float(lp_batch[i])})
    model.train()
    return pd.DataFrame(rows)


def run_config(name, cfg, gt_dir, noisy_dir, split_csv, reports_dir, device, epochs=60, patience=15, seed=123):
    set_full_determinism(seed)
    split_df = pd.read_csv(split_csv)
    train_files = split_df[split_df["split"] == "train"]["file"].tolist()
    val_files = split_df[split_df["split"] == "val"]["file"].tolist()

    model = NAFNetSR(img_channel=1, width=32, upscale=2).to(device)
    criterion = StageBCompositeLoss(sobel_weight=cfg["sobel_weight"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=6)

    train_ds = RealPairDataset(gt_dir, noisy_dir, train_files)
    val_ds = RealPairDataset(gt_dir, noisy_dir, val_files)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4,
                               pin_memory=True, drop_last=True, worker_init_fn=seed_worker,
                               generator=make_seeded_generator(seed), collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate)

    import lpips
    lpips_fn = lpips.LPIPS(net="alex").to(device)

    best_composite, best_epoch, epochs_since_best = -np.inf, -1, 0
    best_metrics = None
    ckpt_path = reports_dir.parent / "checkpoints" / f"axis1b_{name}.pt"
    history = []
    t0 = time.time()

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

        if comp > best_composite:
            best_composite, best_epoch, epochs_since_best = comp, epoch, 0
            best_metrics = {"psnr": float(mean_psnr), "ssim": float(mean_ssim), "lpips": float(mean_lpips),
                             "composite": float(comp)}
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "width": 32, "upscale": 2}, ckpt_path)
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                print(f"[{name}] early stop at epoch {epoch}")
                break

    return {
        "config": name, "params": cfg, "best_epoch": best_epoch, "epochs_run": len(history),
        "wall_clock_sec": time.time() - t0, **best_metrics,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--noisy-dir", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path,
                     default=ROOT / "reports" / "phase2_source_clusters_stratified_leakchecked.csv")
    ap.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for name, cfg in CONFIGS.items():
        print(f"=== Running config: {name} {cfg} ===")
        r = run_config(name, cfg, args.gt_dir, args.noisy_dir, args.split_csv, args.reports_dir, device)
        results.append(r)

    df = pd.DataFrame(results).sort_values("composite", ascending=False)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.reports_dir / "axis1b_sweep_results.csv", index=False)

    baseline = next(r for r in results if r["config"] == "baseline")
    winner = df.iloc[0].to_dict()
    composite_gain = winner["composite"] - baseline["composite"]
    psnr_delta = winner["psnr"] - baseline["psnr"]
    ssim_delta = winner["ssim"] - baseline["ssim"]
    passes_gate = (winner["config"] != "baseline" and composite_gain >= 0.01
                   and psnr_delta >= -0.1 and ssim_delta >= -0.005)

    summary = {
        "results": results,
        "winner": winner["config"], "winner_composite": float(winner["composite"]),
        "baseline_composite": float(baseline["composite"]),
        "composite_gain_over_baseline": float(composite_gain),
        "passes_preregistered_adoption_gate": bool(passes_gate),
    }
    with open(args.reports_dir / "axis1b_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

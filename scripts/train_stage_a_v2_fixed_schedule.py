"""
Axis 1c (technical hardening pass): re-run Stage A with the LR schedule
fix already validated for Stage B (reports/STAGE_B_RESULTS.md) - Stage
A's original cosine schedule was sized for 150 epochs but early-stopped
at 39 (best at epoch 13), so LR barely decayed (reports/STAGE_A_RESULTS.md).
This isolates ONE variable only (the schedule) - same real data, same
split, same loss, same seed, same everything else as the original Stage
A run - to honestly check whether Stage A stopped short of real
convergence or was already done.

Longer early-stop patience (40 vs the original 25) specifically to give
ReduceLROnPlateau room to decay multiple times and see if that unlocks
further improvement, not just to run longer for its own sake.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.nafnet import NAFNetSR  # noqa: E402
from src.losses.stageB_composite import StageBCompositeLoss  # noqa: E402
from src.utils.reproducibility import set_full_determinism, seed_worker, make_seeded_generator  # noqa: E402


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
    noisy = torch.stack([b[0] for b in batch])
    gt = torch.stack([b[1] for b in batch])
    fnames = [b[2] for b in batch]
    return noisy, gt, fnames


def evaluate_per_file(model, loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for noisy, gt, fnames in loader:
            noisy, gt = noisy.to(device), gt.to(device)
            pred = model(noisy).clamp(0, 1)
            pred_np, gt_np = pred.cpu().numpy(), gt.cpu().numpy()
            for i, fname in enumerate(fnames):
                psnr = sk_psnr(gt_np[i, 0], pred_np[i, 0], data_range=1.0)
                ssim = sk_ssim(gt_np[i, 0], pred_np[i, 0], data_range=1.0)
                rows.append({"file": fname, "psnr": psnr, "ssim": ssim})
    model.train()
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--noisy-dir", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path,
                     default=Path("reports/phase2_source_clusters_stratified_leakchecked.csv"))
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--patience", type=int, default=40, help="early-stop patience on val PSNR")
    ap.add_argument("--plateau-patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--checkpoint-out", type=Path, default=Path("checkpoints/stage_a_v2_best.pt"))
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_full_determinism(args.seed)

    split_df = pd.read_csv(args.split_csv)
    train_files = split_df[split_df["split"] == "train"]["file"].tolist()
    val_files = split_df[split_df["split"] == "val"]["file"].tolist()
    file_to_cluster = dict(zip(split_df["file"], split_df["cluster"]))
    print(f"train={len(train_files)} val={len(val_files)}")

    model = NAFNetSR(img_channel=1, width=32, upscale=2).to(device)
    criterion = StageBCompositeLoss().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=args.plateau_patience)

    train_ds = RealPairDataset(args.gt_dir, args.noisy_dir, train_files)
    val_ds = RealPairDataset(args.gt_dir, args.noisy_dir, val_files)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                               pin_memory=True, drop_last=True, worker_init_fn=seed_worker,
                               generator=make_seeded_generator(args.seed), collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate)

    best_psnr = -np.inf
    best_epoch = -1
    epochs_since_best = 0
    history = []
    run_start = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        for noisy, gt, _ in train_loader:
            noisy, gt = noisy.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            opt.zero_grad()
            pred = model(noisy)
            loss, parts = criterion(pred, gt)
            loss.backward()
            opt.step()

        val_df = evaluate_per_file(model, val_loader, device)
        val_psnr, val_ssim = val_df["psnr"].mean(), val_df["ssim"].mean()
        scheduler.step(val_psnr)
        epoch_time = time.time() - epoch_start
        history.append({"epoch": epoch, "val_psnr": float(val_psnr), "val_ssim": float(val_ssim),
                         "epoch_time_sec": epoch_time, "lr": opt.param_groups[0]["lr"]})
        print(f"epoch {epoch}: val_psnr={val_psnr:.3f} val_ssim={val_ssim:.4f} "
              f"lr={opt.param_groups[0]['lr']:.2e} time={epoch_time:.1f}s")

        if val_psnr > best_psnr:
            best_psnr, best_epoch, epochs_since_best = val_psnr, epoch, 0
            args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "width": 32, "upscale": 2, "epoch": epoch,
                "val_psnr": val_psnr, "val_ssim": val_ssim, "seed": args.seed,
            }, args.checkpoint_out)
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    total_time = time.time() - run_start

    ckpt = torch.load(args.checkpoint_out, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    final_val_df = evaluate_per_file(model, val_loader, device)
    final_val_df["cluster"] = final_val_df["file"].map(file_to_cluster)

    per_cluster = final_val_df.groupby("cluster").agg(
        n=("file", "count"), mean_psnr=("psnr", "mean"), std_psnr=("psnr", "std"),
        mean_ssim=("ssim", "mean"), std_ssim=("ssim", "std"),
    ).reset_index().sort_values("cluster")

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    final_val_df.to_csv(args.reports_dir / "stage_a_v2_val_per_file_metrics.csv", index=False)
    per_cluster.to_csv(args.reports_dir / "stage_a_v2_per_cluster_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(args.reports_dir / "stage_a_v2_training_history.csv", index=False)

    summary = {
        "n_train": len(train_files), "n_val": len(val_files),
        "epochs_run": len(history), "epochs_budgeted": args.epochs,
        "best_epoch": best_epoch, "best_val_psnr": float(best_psnr),
        "best_val_ssim": float(ckpt["val_ssim"]),
        "comparison_to_original_stage_a": {
            "original_best_val_psnr": 23.483065963141588,
            "original_best_val_ssim": 0.5975757922307565,
            "original_best_epoch": 13,
            "psnr_gain": float(best_psnr - 23.483065963141588),
        },
        "n_clusters_in_val": int(per_cluster.shape[0]),
        "total_wall_clock_sec": total_time,
        "mean_epoch_time_sec": float(np.mean([h["epoch_time_sec"] for h in history])),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint": str(args.checkpoint_out),
    }
    with open(args.reports_dir / "stage_a_v2_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

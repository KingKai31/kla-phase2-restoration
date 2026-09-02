"""
Axis 1a (technical hardening pass): fine-tune from the best real-data-only
checkpoint (stage_a_v2_best.pt, Axis 1c) using the real leakage-checked
data plus an EXPANDED synthetic pool - 4 NFFA-EUROPE categories
(Biological, Fibres, Films_Coated_Surface, MEMS_devices_and_electrodes),
not just the 3 used in the original, failed Stage B attempt. Same
compound noise model applied directly to real external clean images
(src/datasets/synthetic_degrade.py) - not a different mechanism, see
reports/HARDENING_AXIS_3_AND_5.md's framing correction on this point.

Reports full PSNR/SSIM/LPIPS/composite (scripts/evaluate_checkpoint_full.py),
not just PSNR - per Axis 2's rigor requirement for every checkpoint going
forward.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.models.nafnet import NAFNetSR  # noqa: E402
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--noisy-dir", type=Path, required=True)
    ap.add_argument("--synthetic-gt-dir", type=Path, default=Path("/workspace/synthetic_external_v2/GT"))
    ap.add_argument("--synthetic-noisy-dir", type=Path, default=Path("/workspace/synthetic_external_v2/NoisyLR"))
    ap.add_argument("--synthetic-manifest", type=Path,
                     default=Path("/workspace/synthetic_external_v2/synthetic_manifest.csv"))
    ap.add_argument("--split-csv", type=Path,
                     default=ROOT / "reports" / "phase2_source_clusters_stratified_leakchecked.csv")
    ap.add_argument("--init-checkpoint", type=Path, default=ROOT / "checkpoints" / "stage_a_v2_best.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--plateau-patience", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--checkpoint-out", type=Path, default=ROOT / "checkpoints" / "axis1a_best.pt")
    ap.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_full_determinism(args.seed)

    split_df = pd.read_csv(args.split_csv)
    train_files = split_df[split_df["split"] == "train"]["file"].tolist()
    val_files = split_df[split_df["split"] == "val"]["file"].tolist()
    file_to_cluster = dict(zip(split_df["file"], split_df["cluster"]))

    synth_manifest = pd.read_csv(args.synthetic_manifest)
    synth_files = synth_manifest["file"].tolist()
    print(f"real train={len(train_files)} synthetic train={len(synth_files)} val={len(val_files)}")

    model = NAFNetSR(img_channel=1, width=32, upscale=2).to(device)
    ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Initialized from {args.init_checkpoint} (val_psnr={ckpt['val_psnr']:.3f})")

    criterion = StageBCompositeLoss().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=args.plateau_patience)

    real_train_ds = RealPairDataset(args.gt_dir, args.noisy_dir, train_files)
    synth_train_ds = RealPairDataset(args.synthetic_gt_dir, args.synthetic_noisy_dir, synth_files)
    train_ds = ConcatDataset([real_train_ds, synth_train_ds])
    val_ds = RealPairDataset(args.gt_dir, args.noisy_dir, val_files)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                               pin_memory=True, drop_last=True, worker_init_fn=seed_worker,
                               generator=make_seeded_generator(args.seed), collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate)

    import lpips
    lpips_fn = lpips.LPIPS(net="alex").to(device)

    best_composite, best_epoch, epochs_since_best = -np.inf, -1, 0
    history = []
    run_start = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
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
        epoch_time = time.time() - epoch_start
        history.append({"epoch": epoch, "psnr": mean_psnr, "ssim": mean_ssim, "lpips": mean_lpips,
                         "composite": comp, "epoch_time_sec": epoch_time, "lr": opt.param_groups[0]["lr"]})
        print(f"epoch {epoch}: psnr={mean_psnr:.3f} ssim={mean_ssim:.4f} lpips={mean_lpips:.4f} "
              f"composite={comp:.4f} time={epoch_time:.1f}s")

        if comp > best_composite:
            best_composite, best_epoch, epochs_since_best = comp, epoch, 0
            args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "width": 32, "upscale": 2, "epoch": epoch,
                        "val_psnr": float(mean_psnr), "val_ssim": float(mean_ssim),
                        "val_lpips": float(mean_lpips), "composite": float(comp)}, args.checkpoint_out)
            # Save an incremental summary too - lesson from Axis 4's crash
            json.dump({"best_epoch": epoch, "psnr": float(mean_psnr), "ssim": float(mean_ssim),
                        "lpips": float(mean_lpips), "composite": float(comp)},
                       open(args.reports_dir / "axis1a_best_so_far.json", "w"), indent=2)
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    total_time = time.time() - run_start

    ckpt_best = torch.load(args.checkpoint_out, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_best["model_state_dict"])
    final_val_df = evaluate(model, val_loader, device, lpips_fn)
    final_val_df["cluster"] = final_val_df["file"].map(file_to_cluster)

    per_cluster = final_val_df.groupby("cluster").agg(
        n=("file", "count"), mean_psnr=("psnr", "mean"), mean_ssim=("ssim", "mean"),
        mean_lpips=("lpips", "mean"),
    ).reset_index().sort_values("cluster")

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    final_val_df.to_csv(args.reports_dir / "axis1a_val_per_file_metrics.csv", index=False)
    per_cluster.to_csv(args.reports_dir / "axis1a_per_cluster_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(args.reports_dir / "axis1a_training_history.csv", index=False)

    summary = {
        "n_train_real": len(train_files), "n_train_synthetic": len(synth_files),
        "n_val": len(val_files), "init_checkpoint": str(args.init_checkpoint),
        "init_val_psnr": float(ckpt["val_psnr"]),
        "epochs_run": len(history), "best_epoch": best_epoch,
        "best_val_psnr": float(ckpt_best["val_psnr"]), "best_val_ssim": float(ckpt_best["val_ssim"]),
        "best_val_lpips": float(ckpt_best["val_lpips"]), "best_composite": float(ckpt_best["composite"]),
        "psnr_gain_over_init": float(ckpt_best["val_psnr"] - ckpt["val_psnr"]),
        "total_wall_clock_sec": total_time,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint": str(args.checkpoint_out),
    }
    with open(args.reports_dir / "axis1a_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

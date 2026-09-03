"""
Item 2 (new technique classes): severity-ordered curriculum fine-tune.
Never tried in this project - a training-STRATEGY change, not a loss or
architecture change. Real training pairs are ranked into severity thirds
by their assigned cluster's fitted K_poisson (the single measured
variable with the strongest documented correlation to restoration
difficulty in this project, r=0.688, p=0.0016,
reports/phase2_deep_dive.md Part 8) - lower K_poisson = more shot-noise
variance = a harder real example. Clusters 9 and 16 (too small for a
stable noise-model fit) have no severity score and are treated as
"middle" (included in all phases) as a safe fallback.

Schedule over the fine-tune's epoch budget:
  - first 30% of epochs: sample ONLY from the mildest third (highest
    K_poisson) - literally "sample only," per the pre-registered spec.
  - middle 40%: sample uniformly across ALL training files (matches
    current/every prior training run's behavior).
  - final 30%: OVERSAMPLE the harshest third (lowest K_poisson) - a
    reweighted mix (harshest third gets 3x the per-file sampling weight
    of the rest), not exclusive-only, per the pre-registered spec's use
    of "oversample" rather than "sample only" for this phase.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.models.nafnet import NAFNetSR  # noqa: E402
from src.losses.stageB_composite import StageBCompositeLoss  # noqa: E402
from src.utils.reproducibility import set_full_determinism, seed_worker, make_seeded_generator  # noqa: E402


def build_severity_tiers(split_df: pd.DataFrame, fits_csv: Path):
    fits = pd.read_csv(fits_csv).dropna(subset=["K_poisson"])
    fits_sorted = fits.sort_values("K_poisson")
    n = len(fits_sorted)
    harsh_clusters = set(fits_sorted.iloc[: n // 3]["cluster"])
    mild_clusters = set(fits_sorted.iloc[-(n // 3):]["cluster"])
    # everything else (including clusters 9/16 with no fit) is "middle"
    cluster_to_tier = {}
    for c in split_df["cluster"].unique():
        if c in harsh_clusters:
            cluster_to_tier[c] = "harsh"
        elif c in mild_clusters:
            cluster_to_tier[c] = "mild"
        else:
            cluster_to_tier[c] = "middle"
    return cluster_to_tier


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


def evaluate(model, loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for noisy, gt, fnames in loader:
            noisy, gt = noisy.to(device), gt.to(device)
            pred = model(noisy).clamp(0, 1)
            p, g = pred.cpu().numpy(), gt.cpu().numpy()
            for i, f in enumerate(fnames):
                rows.append({"file": f, "psnr": sk_psnr(g[i, 0], p[i, 0], data_range=1.0),
                             "ssim": sk_ssim(g[i, 0], p[i, 0], data_range=1.0)})
    model.train()
    df = pd.DataFrame(rows)
    return {"psnr": float(df["psnr"].mean()), "ssim": float(df["ssim"].mean())}


def phase_for_epoch(epoch, total_epochs):
    frac = epoch / total_epochs
    if frac < 0.30:
        return "mild_only"
    elif frac < 0.70:
        return "uniform"
    else:
        return "harsh_oversample"


def make_loader(train_ds, tiers, files, phase, batch_size, seed):
    if phase == "mild_only":
        weights = [1.0 if tiers[f] == "mild" else 0.0 for f in files]
    elif phase == "harsh_oversample":
        weights = [3.0 if tiers[f] == "harsh" else 1.0 for f in files]
    else:
        weights = [1.0 for f in files]
    sampler = WeightedRandomSampler(weights, num_samples=len(files), replacement=True,
                                     generator=make_seeded_generator(seed))
    return DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True,
                       drop_last=True, worker_init_fn=seed_worker, collate_fn=collate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--noisy-dir", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path,
                     default=ROOT / "reports" / "phase2_source_clusters_stratified_leakchecked.csv")
    ap.add_argument("--fits-csv", type=Path, default=ROOT / "reports" / "compound_model_per_cluster_fits.csv")
    ap.add_argument("--init-checkpoint", type=Path, default=ROOT / "models" / "checkpoint.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--checkpoint-out", type=Path, default=ROOT / "checkpoints" / "item2_curriculum_best.pt")
    ap.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_full_determinism(args.seed)

    split_df = pd.read_csv(args.split_csv)
    train_files = split_df[split_df["split"] == "train"]["file"].tolist()
    val_files = split_df[split_df["split"] == "val"]["file"].tolist()
    file_to_cluster = dict(zip(split_df["file"], split_df["cluster"]))

    cluster_to_tier = build_severity_tiers(split_df, args.fits_csv)
    file_tier = {f: cluster_to_tier[file_to_cluster[f]] for f in train_files}
    tier_counts = pd.Series(file_tier.values()).value_counts().to_dict()
    print(f"train={len(train_files)} val={len(val_files)} tier_counts={tier_counts}", flush=True)

    model = NAFNetSR(img_channel=1, width=32, upscale=2).to(device)
    ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Initialized from {args.init_checkpoint} (val_psnr={ckpt['val_psnr']:.3f})", flush=True)

    criterion = StageBCompositeLoss().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=6)

    train_ds = RealPairDataset(args.gt_dir, args.noisy_dir, train_files)
    val_ds = RealPairDataset(args.gt_dir, args.noisy_dir, val_files)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate)

    best_psnr, best_epoch, epochs_since_best, history = -np.inf, -1, 0, []
    t0 = time.time()
    for epoch in range(args.epochs):
        te = time.time()
        phase = phase_for_epoch(epoch, args.epochs)
        train_loader = make_loader(train_ds, file_tier, train_files, phase, 16, args.seed + epoch)
        for noisy, gt, _ in train_loader:
            noisy, gt = noisy.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            opt.zero_grad()
            loss, _ = criterion(model(noisy), gt)
            loss.backward()
            opt.step()
        m = evaluate(model, val_loader, device)
        scheduler.step(m["psnr"])
        history.append({"epoch": epoch, "phase": phase, **m, "epoch_time_sec": time.time() - te,
                         "lr": opt.param_groups[0]["lr"]})
        print(f"epoch {epoch} [{phase}]: psnr={m['psnr']:.3f} ssim={m['ssim']:.4f} "
              f"lr={opt.param_groups[0]['lr']:.1e} {time.time()-te:.1f}s", flush=True)
        if m["psnr"] > best_psnr:
            best_psnr, best_epoch, epochs_since_best = m["psnr"], epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "width": 32, "upscale": 2,
                        "epoch": epoch, "val_psnr": m["psnr"], "val_ssim": m["ssim"]}, args.checkpoint_out)
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(f"Early stop at epoch {epoch}", flush=True)
                break

    pd.DataFrame(history).to_csv(args.reports_dir / "item2_curriculum_training_history.csv", index=False)
    result = {"init_checkpoint": str(args.init_checkpoint), "init_val_psnr": float(ckpt["val_psnr"]),
              "tier_counts": tier_counts, "epochs_run": len(history), "best_epoch": best_epoch,
              "best_val_psnr": float(best_psnr), "best_val_ssim": float(history[best_epoch]["ssim"]),
              "wall_clock_sec": time.time() - t0, "checkpoint": str(args.checkpoint_out)}
    json.dump(result, open(args.reports_dir / "item2_curriculum_results.json", "w"), indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

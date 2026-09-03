"""
Item 3 (time-boxed): fine-tune from the currently-shipped
stage_a_aug_raw_best.pt, replacing the diluted SobelEdgeLoss with
BoundaryMaskedEdgeLoss (src/losses/boundary_masked_edge.py) in the active
loss stack. Per reports/item3_decision_rule_PREREGISTERED.md.

Short fine-tune, not from scratch - a refinement expected to converge
fast. Saves best-by-val-PSNR AND tracks the mask-count-weighted edge loss
for a sanity read during training.
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
from src.losses.stageB_composite import (  # noqa: E402
    CharbonnierLoss, RangeConsistencyPenalty,
)
from src.losses.boundary_masked_edge import BoundaryMaskedEdgeLoss  # noqa: E402
from src.utils.reproducibility import set_full_determinism, seed_worker, make_seeded_generator  # noqa: E402

import lpips
from pytorch_msssim import MS_SSIM


class Item3Loss(torch.nn.Module):
    """Same stack as StageBCompositeLoss, but SobelEdgeLoss ->
    BoundaryMaskedEdgeLoss (direct substitution, not a 6th term)."""

    def __init__(self, charbonnier_weight=1.0, msssim_weight=0.2, lpips_weight=0.075,
                 edge_weight=0.1, range_weight=0.05, data_range=1.0):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.msssim = MS_SSIM(data_range=data_range, size_average=True, channel=1)
        self.lpips_fn = lpips.LPIPS(net="alex")
        for p in self.lpips_fn.parameters():
            p.requires_grad_(False)
        self.edge = BoundaryMaskedEdgeLoss(percentile=90.0)
        self.range_penalty = RangeConsistencyPenalty(0.0, data_range)
        self.w_char, self.w_msssim, self.w_lpips = charbonnier_weight, msssim_weight, lpips_weight
        self.w_edge, self.w_range = edge_weight, range_weight

    def forward(self, raw_pred, target):
        pred_c = torch.clamp(raw_pred, 0.0, 1.0)
        target_c = torch.clamp(target, 0.0, 1.0)
        char_loss = self.charbonnier(raw_pred, target)
        msssim_loss = 1.0 - self.msssim(pred_c, target_c)
        pred_lp = pred_c.repeat(1, 3, 1, 1) * 2 - 1
        target_lp = target_c.repeat(1, 3, 1, 1) * 2 - 1
        lpips_loss = self.lpips_fn(pred_lp, target_lp).mean()
        edge_loss = self.edge(pred_c, target_c)
        range_loss = self.range_penalty(raw_pred)
        total = (self.w_char * char_loss + self.w_msssim * msssim_loss + self.w_lpips * lpips_loss
                 + self.w_edge * edge_loss + self.w_range * range_loss)
        return total, {"charbonnier": char_loss.item(), "ms_ssim_loss": msssim_loss.item(),
                       "lpips": lpips_loss.item(), "boundary_edge": edge_loss.item(),
                       "range_penalty": range_loss.item()}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--noisy-dir", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path,
                     default=ROOT / "reports" / "phase2_source_clusters_stratified_leakchecked.csv")
    ap.add_argument("--init-checkpoint", type=Path, default=ROOT / "checkpoints" / "stage_a_aug_raw_best.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--checkpoint-out", type=Path, default=ROOT / "checkpoints" / "item3_boundary_edge_best.pt")
    ap.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_full_determinism(args.seed)

    split_df = pd.read_csv(args.split_csv)
    train_files = split_df[split_df["split"] == "train"]["file"].tolist()
    val_files = split_df[split_df["split"] == "val"]["file"].tolist()
    print(f"train={len(train_files)} val={len(val_files)}", flush=True)

    model = NAFNetSR(img_channel=1, width=32, upscale=2).to(device)
    ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Initialized from {args.init_checkpoint} (val_psnr={ckpt['val_psnr']:.3f})", flush=True)

    criterion = Item3Loss().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)

    train_ds = RealPairDataset(args.gt_dir, args.noisy_dir, train_files)
    val_ds = RealPairDataset(args.gt_dir, args.noisy_dir, val_files)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True,
                               drop_last=True, worker_init_fn=seed_worker,
                               generator=make_seeded_generator(args.seed), collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate)

    best_psnr, best_epoch, epochs_since_best, history = -np.inf, -1, 0, []
    t0 = time.time()
    for epoch in range(args.epochs):
        te = time.time()
        for noisy, gt, _ in train_loader:
            noisy, gt = noisy.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            opt.zero_grad()
            loss, parts = criterion(model(noisy), gt)
            loss.backward()
            opt.step()
        m = evaluate(model, val_loader, device)
        scheduler.step(m["psnr"])
        history.append({"epoch": epoch, **m, "epoch_time_sec": time.time() - te,
                         "lr": opt.param_groups[0]["lr"]})
        print(f"epoch {epoch}: psnr={m['psnr']:.3f} ssim={m['ssim']:.4f} "
              f"lr={opt.param_groups[0]['lr']:.1e} {time.time()-te:.1f}s", flush=True)
        if m["psnr"] > best_psnr:
            best_psnr, best_epoch, epochs_since_best = m["psnr"], epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "width": 32, "upscale": 2,
                        "epoch": epoch, "val_psnr": m["psnr"], "val_ssim": m["ssim"]}, args.checkpoint_out)
            json.dump({"best_epoch": epoch, **m}, open(args.reports_dir / "item3_best_so_far.json", "w"), indent=2)
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(f"Early stop at epoch {epoch}", flush=True)
                break

    pd.DataFrame(history).to_csv(args.reports_dir / "item3_training_history.csv", index=False)
    result = {"init_checkpoint": str(args.init_checkpoint), "init_val_psnr": float(ckpt["val_psnr"]),
              "epochs_run": len(history), "best_epoch": best_epoch, "best_val_psnr": float(best_psnr),
              "best_val_ssim": float(history[best_epoch]["ssim"]),
              "wall_clock_sec": time.time() - t0, "checkpoint": str(args.checkpoint_out)}
    json.dump(result, open(args.reports_dir / "item3_results.json", "w"), indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

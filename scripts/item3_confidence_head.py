"""
Item 3 (new technique classes): auxiliary confidence/residual-magnitude
head. Strictly additive - the shipped model's weights are FROZEN and its
main restoration output (run.py's actual output) is architecturally
untouched. A small trainable head reads the frozen decoder's pre-
up_head feature map and predicts a per-pixel |prediction - GT| map -
"where is the model's own restoration likely to be wrong."

Per reports/new_techniques_decision_rules_PREREGISTERED.md: adopt as a
real, demonstrable finding only if Spearman r >= 0.3 between the
predicted confidence map and the model's actual per-pixel error on
held-out data. Cannot regress the main model by construction (frozen
weights, separate head, never invoked by run.py).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.models.nafnet import NAFNetSR  # noqa: E402
from src.utils.reproducibility import set_full_determinism, seed_worker, make_seeded_generator  # noqa: E402


def frozen_forward_with_feature(model: NAFNetSR, inp: torch.Tensor):
    """Replicates NAFNetSR.forward exactly, but also returns the
    pre-up_head decoder feature map (width channels, padded LR
    resolution) - the same tap point Item 1's proposed decoder residual
    block would sit at. No gradient needed through the main model since
    it is frozen for this item."""
    _, _, h, w = inp.shape
    x = model._pad_to_multiple(inp)
    x = model.intro(x)
    skips = []
    for encoder, down in zip(model.encoders, model.downs):
        x = encoder(x)
        skips.append(x)
        x = down(x)
    x = model.middle(x)
    for decoder, up, skip in zip(model.decoders, model.ups, reversed(skips)):
        x = up(x)
        x = x + skip
        x = decoder(x)
    feature = x  # (B, width, Hpad, Wpad) - pre-up_head
    out = model.up_head(x)
    base = F.interpolate(inp, scale_factor=model.upscale, mode="bilinear", align_corners=False)
    out = out[:, :, : h * model.upscale, : w * model.upscale] + base
    return out, feature


class ConfidenceHead(nn.Module):
    """Small, separate trainable head: width-channel LR feature ->
    1-channel HR confidence map, mirroring up_head's own PixelShuffle
    structure but independently trained. Softplus keeps the output
    non-negative, matching a residual-magnitude target."""

    def __init__(self, width: int = 32, upscale: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width, upscale ** 2, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(upscale)
        self.upscale = upscale

    def forward(self, feature, out_h, out_w):
        y = self.act(self.conv1(feature))
        y = self.conv2(y)
        y = self.shuffle(y)
        y = y[:, :, :out_h, :out_w]
        return F.softplus(y)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--noisy-dir", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path,
                     default=ROOT / "reports" / "phase2_source_clusters_stratified_leakchecked.csv")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "checkpoint.pt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--head-out", type=Path, default=ROOT / "checkpoints" / "item3_confidence_head.pt")
    ap.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_full_determinism(args.seed)

    split_df = pd.read_csv(args.split_csv)
    train_files = split_df[split_df["split"] == "train"]["file"].tolist()
    val_files = split_df[split_df["split"] == "val"]["file"].tolist()
    print(f"train={len(train_files)} val={len(val_files)}", flush=True)

    main_model = NAFNetSR(img_channel=1, width=32, upscale=2).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    main_model.load_state_dict(ckpt["model_state_dict"])
    main_model.eval()
    for p in main_model.parameters():
        p.requires_grad_(False)

    head = ConfidenceHead(width=32, upscale=2).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

    train_ds = RealPairDataset(args.gt_dir, args.noisy_dir, train_files)
    val_ds = RealPairDataset(args.gt_dir, args.noisy_dir, val_files)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True,
                               drop_last=True, worker_init_fn=seed_worker,
                               generator=make_seeded_generator(args.seed), collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate)

    best_val_loss, best_epoch, epochs_since_best, history = np.inf, -1, 0, []
    t0 = time.time()
    for epoch in range(args.epochs):
        te = time.time()
        head.train()
        for noisy, gt, _ in train_loader:
            noisy, gt = noisy.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            with torch.no_grad():
                pred, feature = frozen_forward_with_feature(main_model, noisy)
                pred = pred.clamp(0, 1)
                target_residual = (pred - gt).abs()
            opt.zero_grad()
            conf = head(feature, gt.shape[2], gt.shape[3])
            loss = F.l1_loss(conf, target_residual)
            loss.backward()
            opt.step()

        head.eval()
        val_losses = []
        with torch.no_grad():
            for noisy, gt, _ in val_loader:
                noisy, gt = noisy.to(device), gt.to(device)
                pred, feature = frozen_forward_with_feature(main_model, noisy)
                pred = pred.clamp(0, 1)
                target_residual = (pred - gt).abs()
                conf = head(feature, gt.shape[2], gt.shape[3])
                val_losses.append(F.l1_loss(conf, target_residual).item())
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "val_l1_loss": val_loss, "epoch_time_sec": time.time() - te,
                         "lr": opt.param_groups[0]["lr"]})
        print(f"epoch {epoch}: val_l1_loss={val_loss:.5f} lr={opt.param_groups[0]['lr']:.1e} "
              f"{time.time()-te:.1f}s", flush=True)

        if val_loss < best_val_loss:
            best_val_loss, best_epoch, epochs_since_best = val_loss, epoch, 0
            torch.save({"head_state_dict": head.state_dict(), "epoch": epoch, "val_l1_loss": val_loss},
                       args.head_out)
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(f"Early stop at epoch {epoch}", flush=True)
                break

    pd.DataFrame(history).to_csv(args.reports_dir / "item3_confidence_head_training_history.csv", index=False)

    # ---- final evaluation: real Spearman correlation on held-out val ----
    best_ckpt = torch.load(args.head_out, map_location=device, weights_only=False)
    head.load_state_dict(best_ckpt["head_state_dict"])
    head.eval()

    per_image_corrs = []
    all_conf_sample, all_err_sample = [], []
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for noisy, gt, fnames in val_loader:
            noisy, gt = noisy.to(device), gt.to(device)
            pred, feature = frozen_forward_with_feature(main_model, noisy)
            pred = pred.clamp(0, 1)
            target_residual = (pred - gt).abs()
            conf = head(feature, gt.shape[2], gt.shape[3])
            for i in range(noisy.shape[0]):
                c = conf[i, 0].cpu().numpy().ravel()
                e = target_residual[i, 0].cpu().numpy().ravel()
                r, p = spearmanr(c, e)
                per_image_corrs.append({"file": fnames[i], "spearman_r": float(r), "spearman_p": float(p)})
                idx = rng.choice(len(c), size=min(500, len(c)), replace=False)
                all_conf_sample.append(c[idx])
                all_err_sample.append(e[idx])

    per_image_df = pd.DataFrame(per_image_corrs)
    per_image_df.to_csv(args.reports_dir / "item3_confidence_per_image_correlation.csv", index=False)

    pooled_conf = np.concatenate(all_conf_sample)
    pooled_err = np.concatenate(all_err_sample)
    pooled_r, pooled_p = spearmanr(pooled_conf, pooled_err)

    summary = {
        "n_val_images": len(per_image_df),
        "per_image_mean_spearman_r": float(per_image_df["spearman_r"].mean()),
        "per_image_median_spearman_r": float(per_image_df["spearman_r"].median()),
        "per_image_std_spearman_r": float(per_image_df["spearman_r"].std()),
        "pooled_spearman_r": float(pooled_r),
        "pooled_spearman_p": float(pooled_p),
        "pooled_n_pixels_sampled": len(pooled_conf),
        "best_epoch": best_epoch, "best_val_l1_loss": float(best_val_loss),
        "wall_clock_sec": time.time() - t0,
        "passes_preregistered_gate_r_ge_0.3": bool(per_image_df["spearman_r"].mean() >= 0.3),
        "head_checkpoint": str(args.head_out),
    }
    with open(args.reports_dir / "item3_confidence_head_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""
Improvement pass, Items 1a + 1b + 2 (one retrain, three folded changes -
explicitly permitted), per the pre-registered rules in
reports/item1_item2_decision_rule_PREREGISTERED.md, written and committed
before this script ran.

  1a. Dihedral augmentation (8 flips/rotations) applied IDENTICALLY to
      each GT/NoisyLR pair. Physically valid: SEM images have no canonical
      orientation, and box-downsampling commutes with these transforms, so
      rot90/flip of the pair is still a valid pair.
  1b. EMA of weights (decay 0.999) tracked alongside raw weights; both
      evaluated every epoch, both best-by-val-PSNR checkpoints saved.
  2.  ICNR initialization of every conv feeding a PixelShuffle (`ups` and
      `up_head`): each output channel's r^2 sub-pixel filters start
      identical, so the initial upsample is nearest-neighbor and the
      checkerboard is suppressed at the source rather than by run.py's
      post-hoc 15% box blur. The architecture is unchanged - the
      checkpoint loads into the same NAFNetSR class run.py already ships.

Everything else (split, loss, seed, LR, batch, ReduceLROnPlateau) is
identical to Stage A so the comparison against the shipped 23.483 dB is
clean. Final evaluation reports raw_best and ema_best, each at blend 0.15
(the shipped run.py behaviour) and blend 0.00, with a period-2
checkerboard energy ratio, and applies Gates 1 and 2 in code.

Partial results are persisted after every improving epoch (lesson from
the disk-quota crash in the hardening pass).
"""
import argparse
import copy
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
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.models.nafnet import NAFNetSR  # noqa: E402
from src.losses.stageB_composite import StageBCompositeLoss  # noqa: E402
from src.utils.reproducibility import set_full_determinism, seed_worker, make_seeded_generator  # noqa: E402

BASELINE_PSNR = 23.483065963141588   # shipped models/checkpoint.pt, internal val
BASELINE_SSIM = 0.5975757922307565
GATE1_PSNR_MIN = BASELINE_PSNR + 0.100
GATE1_SSIM_MIN = BASELINE_SSIM - 0.005
GATE1_LPIPS_SLACK = 0.005
GATE2_PSNR_SLACK = 0.010
GATE2_CHECKER_MAX_RATIO = 1.10


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def dihedral(arr: np.ndarray, k: int) -> np.ndarray:
    """k in 0..7: rot90 by (k % 4), then horizontal flip if k >= 4."""
    out = np.rot90(arr, k % 4)
    if k >= 4:
        out = out[:, ::-1]
    return np.ascontiguousarray(out)


class RealPairDataset(Dataset):
    def __init__(self, gt_dir, noisy_dir, files, augment: bool):
        self.gt_dir, self.noisy_dir, self.files, self.augment = gt_dir, noisy_dir, files, augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        gt = np.load(self.gt_dir / fname).astype(np.float32)
        noisy = np.load(self.noisy_dir / fname).astype(np.float32)
        noisy = np.nan_to_num(noisy, nan=0.0, posinf=1.0, neginf=0.0)
        if self.augment:
            # np.random is per-worker seeded by seed_worker (src/utils/reproducibility.py)
            k = int(np.random.randint(0, 8))
            gt, noisy = dihedral(gt, k), dihedral(noisy, k)
        return torch.from_numpy(noisy).unsqueeze(0), torch.from_numpy(gt).unsqueeze(0), fname


def collate(batch):
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]), [b[2] for b in batch]


# ----------------------------------------------------------------------------
# ICNR + EMA + inference-time helpers (blur identical to run.py's)
# ----------------------------------------------------------------------------
def icnr_(conv: nn.Conv2d, scale: int = 2):
    """PixelShuffle takes input channel c*r^2 + i*r + j -> output channel c,
    sub-position (i, j). Making each consecutive block of r^2 filters identical
    means every sub-position starts with the same value: a nearest-neighbour
    upsample, no checkerboard."""
    w = conv.weight.data
    out_ch, in_ch, kh, kw = w.shape
    assert out_ch % (scale ** 2) == 0
    sub = torch.zeros(out_ch // (scale ** 2), in_ch, kh, kw, device=w.device, dtype=w.dtype)
    nn.init.kaiming_normal_(sub, a=0, mode="fan_in", nonlinearity="relu")
    w.copy_(sub.repeat_interleave(scale ** 2, dim=0))
    if conv.bias is not None:
        conv.bias.data.zero_()


def apply_icnr(model: NAFNetSR):
    n = 0
    for up in model.ups:
        icnr_(up[0], scale=2); n += 1
    icnr_(model.up_head[0], scale=model.upscale); n += 1
    return n


@torch.no_grad()
def ema_update(ema_model, model, decay):
    for pe, pm in zip(ema_model.parameters(), model.parameters()):
        pe.lerp_(pm, 1.0 - decay)
    for be, bm in zip(ema_model.buffers(), model.buffers()):
        be.copy_(bm)


def suppress_checkerboard(y, blend):
    if blend <= 0:
        return y
    kernel = torch.ones(1, 1, 3, 3, device=y.device, dtype=y.dtype) / 9.0
    return (1 - blend) * y + blend * F.conv2d(y, kernel, padding=1)


def period2_energy(x):
    """Energy of the period-2 (checkerboard-frequency) residual."""
    up = F.interpolate(F.avg_pool2d(x, 2), scale_factor=2, mode="nearest")
    return ((x - up) ** 2).mean().item()


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------
def evaluate(model, loader, device, blend=0.15, lpips_fn=None, checker=False):
    model.eval()
    rows, e_pred, e_gt = [], 0.0, 0.0
    with torch.no_grad():
        for noisy, gt, fnames in loader:
            noisy, gt = noisy.to(device), gt.to(device)
            pred = suppress_checkerboard(model(noisy), blend).clamp(0, 1)
            if checker:
                e_pred += period2_energy(pred) * pred.shape[0]
                e_gt += period2_energy(gt) * gt.shape[0]
            lp = None
            if lpips_fn is not None:
                lp = lpips_fn(pred.repeat(1, 3, 1, 1) * 2 - 1, gt.repeat(1, 3, 1, 1) * 2 - 1)
                lp = np.atleast_1d(lp.squeeze(-1).squeeze(-1).squeeze(-1).cpu().numpy())
            p, g = pred.cpu().numpy(), gt.cpu().numpy()
            for i, f in enumerate(fnames):
                row = {"file": f, "psnr": sk_psnr(g[i, 0], p[i, 0], data_range=1.0),
                       "ssim": sk_ssim(g[i, 0], p[i, 0], data_range=1.0)}
                if lp is not None:
                    row["lpips"] = float(lp[i])
                rows.append(row)
    model.train()
    df = pd.DataFrame(rows)
    out = {"psnr": float(df["psnr"].mean()), "ssim": float(df["ssim"].mean())}
    if lpips_fn is not None:
        out["lpips"] = float(df["lpips"].mean())
    if checker:
        n = len(df)
        out["checker_ratio_pred_over_gt"] = float((e_pred / n) / max(e_gt / n, 1e-12))
    return out, df


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--noisy-dir", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path,
                     default=ROOT / "reports" / "phase2_source_clusters_stratified_leakchecked.csv")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--plateau-patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--no-icnr", action="store_true")
    ap.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    ap.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--tag", type=str, default="stage_a_aug")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_full_determinism(args.seed)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    split_df = pd.read_csv(args.split_csv)
    train_files = split_df[split_df["split"] == "train"]["file"].tolist()
    val_files = split_df[split_df["split"] == "val"]["file"].tolist()
    print(f"train={len(train_files)} val={len(val_files)} augment={not args.no_augment} icnr={not args.no_icnr}")

    model = NAFNetSR(img_channel=1, width=32, upscale=2).to(device)
    n_icnr = 0 if args.no_icnr else apply_icnr(model)
    print(f"ICNR applied to {n_icnr} pixel-shuffle convs")
    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    criterion = StageBCompositeLoss().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=args.plateau_patience)

    train_ds = RealPairDataset(args.gt_dir, args.noisy_dir, train_files, augment=not args.no_augment)
    val_ds = RealPairDataset(args.gt_dir, args.noisy_dir, val_files, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                               pin_memory=True, drop_last=True, worker_init_fn=seed_worker,
                               generator=make_seeded_generator(args.seed), collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate)

    raw_ckpt = args.checkpoint_dir / f"{args.tag}_raw_best.pt"
    ema_ckpt = args.checkpoint_dir / f"{args.tag}_ema_best.pt"
    best = {"raw": {"psnr": -np.inf, "epoch": -1}, "ema": {"psnr": -np.inf, "epoch": -1}}
    epochs_since_best, history, t0 = 0, [], time.time()

    def save(m, path, epoch, metrics):
        torch.save({"model_state_dict": m.state_dict(), "width": 32, "upscale": 2, "epoch": epoch,
                    "val_psnr": metrics["psnr"], "val_ssim": metrics["ssim"], "seed": args.seed,
                    "augment": not args.no_augment, "icnr": not args.no_icnr}, path)

    for epoch in range(args.epochs):
        te = time.time()
        for noisy, gt, _ in train_loader:
            noisy, gt = noisy.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            opt.zero_grad()
            loss, _ = criterion(model(noisy), gt)
            loss.backward()
            opt.step()
            ema_update(ema_model, model, args.ema_decay)

        m_raw, _ = evaluate(model, val_loader, device, blend=0.15)
        m_ema, _ = evaluate(ema_model, val_loader, device, blend=0.15)
        scheduler.step(m_raw["psnr"])
        history.append({"epoch": epoch, "raw_psnr": m_raw["psnr"], "raw_ssim": m_raw["ssim"],
                         "ema_psnr": m_ema["psnr"], "ema_ssim": m_ema["ssim"],
                         "lr": opt.param_groups[0]["lr"], "epoch_time_sec": time.time() - te})
        print(f"epoch {epoch}: raw psnr={m_raw['psnr']:.3f} ssim={m_raw['ssim']:.4f} | "
              f"ema psnr={m_ema['psnr']:.3f} ssim={m_ema['ssim']:.4f} | lr={opt.param_groups[0]['lr']:.1e} "
              f"| {time.time() - te:.1f}s", flush=True)

        improved = False
        if m_raw["psnr"] > best["raw"]["psnr"]:
            best["raw"] = {**m_raw, "epoch": epoch}; save(model, raw_ckpt, epoch, m_raw); improved = True
        if m_ema["psnr"] > best["ema"]["psnr"]:
            best["ema"] = {**m_ema, "epoch": epoch}; save(ema_model, ema_ckpt, epoch, m_ema); improved = True
        if improved:
            epochs_since_best = 0
            json.dump({"best": best, "epochs_run": epoch + 1},
                      open(args.reports_dir / f"item1_{args.tag}_best_so_far.json", "w"), indent=2)
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    train_time = time.time() - t0
    pd.DataFrame(history).to_csv(args.reports_dir / f"item1_{args.tag}_training_history.csv", index=False)

    # ---------------- final evaluation: raw/ema x blend 0.15/0.00, with LPIPS + checkerboard --------
    import lpips
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    base_lpips_path = args.reports_dir / "item1_baseline_shipped_full_metrics_summary.json"
    base_lpips = json.load(open(base_lpips_path))["mean_lpips"] if base_lpips_path.exists() else None

    final = {}
    for name, path in [("raw_best", raw_ckpt), ("ema_best", ema_ckpt)]:
        ck = torch.load(path, map_location=device, weights_only=False)
        m = NAFNetSR(img_channel=1, width=32, upscale=2).to(device)
        m.load_state_dict(ck["model_state_dict"])
        for blend in (0.15, 0.0):
            met, df = evaluate(m, val_loader, device, blend=blend, lpips_fn=lpips_fn, checker=True)
            met["epoch"] = ck["epoch"]
            final[f"{name}_blend{blend:.2f}"] = met
            df.to_csv(args.reports_dir / f"item1_{args.tag}_{name}_blend{blend:.2f}_per_file.csv", index=False)
            print(name, blend, met, flush=True)

    # Gate 1: better of raw/ema at the SHIPPED inference setting (blend 0.15)
    cands = {k: v for k, v in final.items() if k.endswith("blend0.15")}
    g1_name = max(cands, key=lambda k: cands[k]["psnr"])
    g1 = cands[g1_name]
    gate1 = {
        "candidate": g1_name, "psnr": g1["psnr"], "ssim": g1["ssim"], "lpips": g1["lpips"],
        "psnr_gain_vs_shipped": g1["psnr"] - BASELINE_PSNR,
        "pass_psnr": g1["psnr"] >= GATE1_PSNR_MIN,
        "pass_ssim": g1["ssim"] >= GATE1_SSIM_MIN,
        "pass_lpips": (g1["lpips"] <= base_lpips + GATE1_LPIPS_SLACK) if base_lpips is not None else None,
        "baseline_lpips_used": base_lpips,
    }
    gate1["PASS"] = bool(gate1["pass_psnr"] and gate1["pass_ssim"] and (gate1["pass_lpips"] in (True, None)))

    # Gate 2: blur removal, evaluated on the Gate-1 candidate weights
    k0 = g1_name.replace("blend0.15", "blend0.00")
    b15, b0 = final[g1_name], final[k0]
    gate2 = {
        "psnr_blend0.15": b15["psnr"], "psnr_blend0.00": b0["psnr"],
        "ssim_blend0.15": b15["ssim"], "ssim_blend0.00": b0["ssim"],
        "checker_ratio_blend0.00": b0["checker_ratio_pred_over_gt"],
        "checker_ratio_blend0.15": b15["checker_ratio_pred_over_gt"],
        "pass_psnr": b0["psnr"] >= b15["psnr"] - GATE2_PSNR_SLACK,
        "pass_ssim": b0["ssim"] >= b15["ssim"],
        "pass_checker": b0["checker_ratio_pred_over_gt"] <= GATE2_CHECKER_MAX_RATIO,
    }
    gate2["PASS"] = bool(gate2["pass_psnr"] and gate2["pass_ssim"] and gate2["pass_checker"])

    result = {
        "config": {"augment": not args.no_augment, "icnr": not args.no_icnr, "ema_decay": args.ema_decay,
                   "lr": args.lr, "batch_size": args.batch_size, "seed": args.seed},
        "baseline_shipped": {"psnr": BASELINE_PSNR, "ssim": BASELINE_SSIM, "lpips": base_lpips},
        "epochs_run": len(history), "best": best, "train_wall_clock_sec": train_time,
        "mean_epoch_time_sec": float(np.mean([h["epoch_time_sec"] for h in history])),
        "final_eval": final, "gate1_new_checkpoint": gate1, "gate2_drop_blur": gate2,
        "checkpoints": {"raw_best": str(raw_ckpt), "ema_best": str(ema_ckpt)},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    with open(args.reports_dir / f"item1_{args.tag}_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"gate1": gate1, "gate2": gate2}, indent=2), flush=True)


if __name__ == "__main__":
    main()

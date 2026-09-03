"""
Item 1 (final pass): graduated boundary-masked auxiliary loss. Reuses
BoundaryMaskedEdgeLoss (src/losses/boundary_masked_edge.py) UNMODIFIED,
added as a 6th term ALONGSIDE the existing unchanged 5-term stack (not a
replacement, unlike Item 3), at a small weight. Fine-tunes from the
CURRENTLY SHIPPED checkpoint. Per
reports/item1_graduated_edge_decision_rule_PREREGISTERED.md.
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
from src.losses.boundary_masked_edge import BoundaryMaskedEdgeLoss  # noqa: E402
from src.utils.reproducibility import set_full_determinism, seed_worker, make_seeded_generator  # noqa: E402


class SixTermLoss(torch.nn.Module):
    """The exact, unmodified StageBCompositeLoss (5 terms) plus
    BoundaryMaskedEdgeLoss as an ADDITIONAL 6th term - the base stack is
    not edited, matching this project's established pattern of adding
    variants alongside a validated module instead of modifying it."""

    def __init__(self, boundary_edge_weight: float):
        super().__init__()
        self.base = StageBCompositeLoss()  # unchanged: char=1.0, msssim=0.2, lpips=0.075, sobel=0.1, range=0.05
        self.boundary_edge = BoundaryMaskedEdgeLoss(percentile=90.0)
        self.w_boundary = boundary_edge_weight

    def forward(self, raw_pred, target):
        total, parts = self.base(raw_pred, target)
        pred_c = torch.clamp(raw_pred, 0.0, 1.0)
        target_c = torch.clamp(target, 0.0, 1.0)
        boundary_loss = self.boundary_edge(pred_c, target_c)
        total = total + self.w_boundary * boundary_loss
        parts["boundary_edge"] = boundary_loss.item()
        return total, parts


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


def run_one_weight(weight, gt_dir, noisy_dir, split_csv, init_checkpoint, checkpoint_out,
                    reports_dir, epochs=25, patience=10, lr=5e-5, seed=123):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_full_determinism(seed)

    split_df = pd.read_csv(split_csv)
    train_files = split_df[split_df["split"] == "train"]["file"].tolist()
    val_files = split_df[split_df["split"] == "val"]["file"].tolist()

    model = NAFNetSR(img_channel=1, width=32, upscale=2).to(device)
    ckpt = torch.load(init_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    criterion = SixTermLoss(boundary_edge_weight=weight).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)

    train_ds = RealPairDataset(gt_dir, noisy_dir, train_files)
    val_ds = RealPairDataset(gt_dir, noisy_dir, val_files)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True,
                               drop_last=True, worker_init_fn=seed_worker,
                               generator=make_seeded_generator(seed), collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate)

    best_psnr, best_epoch, epochs_since_best, history = -np.inf, -1, 0, []
    t0 = time.time()
    for epoch in range(epochs):
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
        print(f"[w={weight}] epoch {epoch}: psnr={m['psnr']:.3f} ssim={m['ssim']:.4f} "
              f"lr={opt.param_groups[0]['lr']:.1e} {time.time()-te:.1f}s", flush=True)
        if m["psnr"] > best_psnr:
            best_psnr, best_epoch, epochs_since_best = m["psnr"], epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "width": 32, "upscale": 2,
                        "epoch": epoch, "val_psnr": m["psnr"], "val_ssim": m["ssim"]}, checkpoint_out)
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                print(f"[w={weight}] early stop at epoch {epoch}", flush=True)
                break

    pd.DataFrame(history).to_csv(reports_dir / f"item1_final_w{weight}_training_history.csv", index=False)
    return {"weight": weight, "init_val_psnr": float(ckpt["val_psnr"]), "epochs_run": len(history),
            "best_epoch": best_epoch, "best_val_psnr": float(best_psnr),
            "best_val_ssim": float(history[best_epoch]["ssim"]),
            "wall_clock_sec": time.time() - t0, "checkpoint": str(checkpoint_out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--noisy-dir", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path,
                     default=ROOT / "reports" / "phase2_source_clusters_stratified_leakchecked.csv")
    ap.add_argument("--init-checkpoint", type=Path, default=ROOT / "models" / "checkpoint.pt")
    ap.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    ap.add_argument("--weights", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    args = ap.parse_args()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for w in args.weights:
        print(f"=== Running weight {w} ===", flush=True)
        ckpt_out = args.checkpoint_dir / f"item1_final_w{w}.pt"
        r = run_one_weight(w, args.gt_dir, args.noisy_dir, args.split_csv, args.init_checkpoint,
                            ckpt_out, args.reports_dir)
        results.append(r)
        json.dump(results, open(args.reports_dir / "item1_final_sweep_results.json", "w"), indent=2)
        print(json.dumps(r, indent=2), flush=True)

    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()

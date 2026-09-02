"""
Short, controlled comparison: WITH vs WITHOUT the ROI-preservation term
(Task 3 / reports/roi_loss_decision_rule_PREREGISTERED.md, written and
committed BEFORE this script was run). Same data, same stratified split,
same random seed/init, same hyperparameters - only the loss stack differs.

This checks decision-rule condition 1 (PSNR/SSIM regression guardrail)
only. Condition 2 (defect-preservation benefit) requires
scripts/defect_preservation_stress_test.py, run separately.
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
from src.losses.roi_preservation import StageBCompositeLossWithROI  # noqa: E402
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
        return torch.from_numpy(noisy).unsqueeze(0), torch.from_numpy(gt).unsqueeze(0)


def evaluate(model, loader, device):
    model.eval()
    psnrs, ssims = [], []
    with torch.no_grad():
        for noisy, gt in loader:
            noisy, gt = noisy.to(device), gt.to(device)
            pred = model(noisy).clamp(0, 1)
            pred_np, gt_np = pred.cpu().numpy(), gt.cpu().numpy()
            for i in range(pred_np.shape[0]):
                psnrs.append(sk_psnr(gt_np[i, 0], pred_np[i, 0], data_range=1.0))
                ssims.append(sk_ssim(gt_np[i, 0], pred_np[i, 0], data_range=1.0))
    model.train()
    return float(np.mean(psnrs)), float(np.mean(ssims))


def train_one_config(name, use_roi, gt_dir, noisy_dir, train_files, val_files,
                      device, seed, epochs, batch_size, lr):
    print(f"\n=== Training config: {name} (use_roi={use_roi}) ===")
    set_full_determinism(seed)

    model = NAFNetSR(img_channel=1, width=32, upscale=2).to(device)
    base_loss = StageBCompositeLoss().to(device)
    criterion = StageBCompositeLossWithROI(base_loss, roi_weight=0.1).to(device) if use_roi else base_loss
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = RealPairDataset(gt_dir, noisy_dir, train_files)
    val_ds = RealPairDataset(gt_dir, noisy_dir, val_files)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4,
                               pin_memory=True, drop_last=True, worker_init_fn=seed_worker,
                               generator=make_seeded_generator(seed))
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    history = []
    t0 = time.time()
    for epoch in range(epochs):
        for noisy, gt in train_loader:
            noisy, gt = noisy.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            opt.zero_grad()
            pred = model(noisy)
            loss, parts = criterion(pred, gt)
            loss.backward()
            # gradient-norm monitor: watch for the ROI term (or any single term)
            # silently dominating - same discipline as the existing 5-term stack
            total_norm = torch.norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]))
            opt.step()
        val_psnr, val_ssim = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "val_psnr": val_psnr, "val_ssim": val_ssim,
                         "grad_norm_last_batch": float(total_norm), **parts})
        print(f"  epoch {epoch}: val_psnr={val_psnr:.3f} val_ssim={val_ssim:.4f} "
              f"loss_parts={ {k: round(v, 4) for k, v in parts.items()} }")
    elapsed = time.time() - t0
    print(f"  {name} total training time: {elapsed:.1f}s")
    return history, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--noisy-dir", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    ap.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"),
                     help="Both configs' final checkpoints are saved here, needed for "
                          "scripts/defect_preservation_stress_test.py's paired comparison")
    args = ap.parse_args()

    split_df = pd.read_csv(args.split_csv)
    train_files = split_df[split_df["split"] == "train"]["file"].tolist()
    val_files = split_df[split_df["split"] == "val"]["file"].tolist()
    print(f"train={len(train_files)} val={len(val_files)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hist_without, model_without = train_one_config("without_roi", False, args.gt_dir, args.noisy_dir,
                                                     train_files, val_files, device, args.seed,
                                                     args.epochs, args.batch_size, args.lr)
    hist_with, model_with = train_one_config("with_roi", True, args.gt_dir, args.noisy_dir,
                                              train_files, val_files, device, args.seed,
                                              args.epochs, args.batch_size, args.lr)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for tag, model, hist in [("without_roi", model_without, hist_without), ("with_roi", model_with, hist_with)]:
        torch.save({
            "model_state_dict": model.state_dict(), "width": 32, "upscale": 2,
            "epoch": args.epochs - 1, "val_psnr": hist[-1]["val_psnr"], "val_ssim": hist[-1]["val_ssim"],
            "seed": args.seed, "use_roi": tag == "with_roi",
        }, args.checkpoint_dir / f"roi_ablation_{tag}.pt")
        print(f"Saved checkpoint: {args.checkpoint_dir / f'roi_ablation_{tag}.pt'}")

    final_without = hist_without[-1]
    final_with = hist_with[-1]
    psnr_delta = final_with["val_psnr"] - final_without["val_psnr"]
    ssim_delta = final_with["val_ssim"] - final_without["val_ssim"]

    result = {
        "epochs": args.epochs, "seed": args.seed,
        "final_without_roi": {"val_psnr": final_without["val_psnr"], "val_ssim": final_without["val_ssim"]},
        "final_with_roi": {"val_psnr": final_with["val_psnr"], "val_ssim": final_with["val_ssim"]},
        "psnr_delta_with_minus_without": psnr_delta,
        "ssim_delta_with_minus_without": ssim_delta,
        "guardrail_psnr_ok (delta >= -0.1dB)": psnr_delta >= -0.1,
        "guardrail_ssim_ok (delta >= -0.005)": ssim_delta >= -0.005,
        "history_without_roi": hist_without,
        "history_with_roi": hist_with,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "roi_loss_ablation_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== FINAL COMPARISON ===")
    print(json.dumps({k: v for k, v in result.items() if not k.startswith("history")}, indent=2))


if __name__ == "__main__":
    main()

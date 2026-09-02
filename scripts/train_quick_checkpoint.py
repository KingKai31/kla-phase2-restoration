"""
Quick training run that SAVES a checkpoint (unlike scripts/roi_loss_ablation.py,
which only tracks in-memory metrics) - needed as a real trained model to
validate the Local-Lipschitz confidence signal against
(scripts/validate_confidence_signal.py). Not a final Stage A/B model -
short, for diagnostic-validation purposes only, using the base 5-term
Stage B loss (no ROI term, keeps this independent of the ROI decision).
"""
import argparse
import sys
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--noisy-dir", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--checkpoint-out", type=Path, default=Path("checkpoints/quick_diag_checkpoint.pt"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_full_determinism(args.seed)

    split_df = pd.read_csv(args.split_csv)
    train_files = split_df[split_df["split"] == "train"]["file"].tolist()
    val_files = split_df[split_df["split"] == "val"]["file"].tolist()

    model = NAFNetSR(img_channel=1, width=32, upscale=2).to(device)
    criterion = StageBCompositeLoss().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_ds = RealPairDataset(args.gt_dir, args.noisy_dir, train_files)
    val_ds = RealPairDataset(args.gt_dir, args.noisy_dir, val_files)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                               pin_memory=True, drop_last=True, worker_init_fn=seed_worker,
                               generator=make_seeded_generator(args.seed))
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    for epoch in range(args.epochs):
        for noisy, gt in train_loader:
            noisy, gt = noisy.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            opt.zero_grad()
            pred = model(noisy)
            loss, parts = criterion(pred, gt)
            loss.backward()
            opt.step()
        val_psnr, val_ssim = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: val_psnr={val_psnr:.3f} val_ssim={val_ssim:.4f}")

    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "width": 32, "upscale": 2, "epoch": args.epochs - 1,
        "val_psnr": val_psnr, "val_ssim": val_ssim, "seed": args.seed,
    }, args.checkpoint_out)
    print(f"Saved checkpoint to {args.checkpoint_out}")


if __name__ == "__main__":
    main()

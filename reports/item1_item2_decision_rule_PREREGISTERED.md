# Improvement pass — Items 1a/1b/1c/2: pre-registered decision rules

Written and committed BEFORE any training run or measurement. Same
discipline that made the ROI-loss, correlated-noise, Axis 1b, and Axis 4
negative results credible.

## Baseline (fixed, not re-measured to move the target)

The **shipped** model, `models/checkpoint.pt` (= `stage_a_best.pt`):
internal val **PSNR 23.483 dB, SSIM 0.5976** (n=712, leakage-checked
split). Val LPIPS was never computed for it; it is measured once, before
any comparison, with `scripts/evaluate_checkpoint_full.py` and recorded
in `reports/item1_baseline_shipped_full_metrics_summary.json`. That number
becomes the LPIPS baseline. **Measured seed-variance floor: 0.026 dB**
(Axis 1c, two identical-config runs).

## One retrain, three folded changes (explicitly permitted)

`scripts/train_stage_a_aug_ema.py`, from scratch, same split/loss/seed/
LR/batch as Stage A, with:
- **1a. Augmentation:** random dihedral transform (8 flips/rotations)
  applied identically to each GT/NoisyLR pair. Physically valid - SEM
  images have no canonical orientation.
- **1b. EMA:** exponential moving average of weights (decay 0.999),
  tracked alongside the raw weights. Both evaluated every epoch.
- **2. ICNR initialization** of every conv that feeds a PixelShuffle
  (`ups` and `up_head`), so the checkerboard is suppressed at the source
  and the post-hoc 15% blur in `run.py` can be dropped.

Attribution between the three is deliberately NOT claimed - the gate is
on the combined result against the shipped baseline, as instructed.

## Gate 1 (Items 1a/1b) - adopt a new checkpoint only if

For the better of {raw_best, ema_best} (each selected by val PSNR):
- val PSNR >= **23.583 dB** (baseline + 0.100, ~4x the seed floor), AND
- val SSIM >= baseline SSIM - 0.005, AND
- val LPIPS <= baseline LPIPS + 0.005.
If neither weight set clears all three: **no new checkpoint ships;
Stage A remains**, and the result is documented as negative.

## Gate 2 (Item 2, blur removal) - drop the `run.py` blur only if

Evaluated on the checkpoint that passes Gate 1 (or on the shipped model
if none does):
- val PSNR at blend=0.00 >= val PSNR at blend=0.15 - 0.010, AND
- val SSIM at blend=0.00 >= val SSIM at blend=0.15, AND
- period-2 (checkerboard) energy ratio at blend=0.00 <= 1.10x that of GT
  (defined in the script: energy of `x - nearest_upsample(avgpool2x2(x))`,
  prediction over GT).
If it fails, the blur stays and ICNR is reported as "did not remove the
need for the blur."

## Gate 3 (Item 1c, TTA) - adopt 8-fold dihedral TTA in `run.py` only if

- val PSNR gain >= **0.100 dB** over the same checkpoint without TTA, AND
- SSIM and LPIPS do not regress, AND
- **per-image inference time < 1.0 second on the A100** (stated here,
  before measuring; the shipped path is ~6 ms warm, so 8x is ~50 ms).
Both numbers are reported regardless of outcome.

## Fallback rule (non-negotiable)

`models/checkpoint.pt` and `run.py` at git HEAD are always the shipped,
verified artifact. Any candidate is staged in the working tree, the FULL
compliance chain is run on it (fresh venv + no-internet + wrong-cwd +
25-test suite + shape/range/NaN checks), and only then committed. A red
chain means `git checkout -- run.py models/checkpoint.pt`.

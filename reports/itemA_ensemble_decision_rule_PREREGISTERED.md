# Item A — checkpoint ensemble: pre-registered decision rule

Written and committed BEFORE any ensemble measurement. **20-minute
time-box including evaluation.**

## The two checkpoints

- **Shipped** (`models/checkpoint.pt` = `stage_a_aug_raw_best.pt`):
  official-test PSNR 24.004, SSIM 0.6257, LPIPS 0.1616; edge retention
  0.705 (blend=0.00).
- **Item 3** (`checkpoints/item3_boundary_edge_best.pt`, not shipped -
  failed its own gate): official-test PSNR 23.746, SSIM 0.6314, LPIPS
  0.1559; edge retention 0.819 (blend=0.00).

## The test

Weighted average of the two models' raw (pre-checkerboard-suppression)
output tensors, `w * shipped + (1-w) * item3`, then run.py's normal
post-processing (suppress_checkerboard, clamp, sanitize). Sweep
`w in {0.7, 0.8}` (70/30 and 80/20 toward the shipped model, as
requested) - no new training, inference-time only.

## Gate - adopt only if ALL three hold

1. **Official-test PSNR improves by >= 0.026** (the measured seed-
   variance floor) over the shipped model's 24.004 - i.e. **>= 24.030**.
2. **Edge retention (Ni-WC, blend=0.00) improves by >= 0.020 absolute**
   over the shipped model's 0.705 - i.e. **>= 0.725**.
3. **Inference time stays within ~2x single-model cost** (stated before
   measuring, per instruction) - shipped model's warm per-image cost is
   ~6ms (`reports/run_py_compliance_checklist.md`); ensemble runs both
   models per image, so ~2x (~12ms) is the expected, acceptable cost;
   flagged as a problem only if it comes in meaningfully above 2x
   (e.g. >20ms/image), which would suggest something is wrong, not just
   "two forward passes cost twice as much."

**All three must hold, both weightings tested, best reported.** If
neither weighting clears all three, or the 20-minute window closes
first: **not adopted**, documented as a real negative/neutral result.
No change to `models/checkpoint.pt` unless this gate passes.

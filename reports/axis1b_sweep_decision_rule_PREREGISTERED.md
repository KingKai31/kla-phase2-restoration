# Axis 1b — hyperparameter sweep: pre-registered decision rule

Written and committed BEFORE any sweep run, same standard as every other
comparison in this project.

## Configurations (4, each run to early-stopping, not full budget)

| Config | LR | Batch | Loss weights | Rationale |
|---|---|---|---|---|
| `baseline` | 2e-4 | 16 | default (Charbonnier=1.0, MS-SSIM=0.2, LPIPS=0.075, Sobel=0.1, range=0.05) | matches Stage A's original hyperparameters exactly, for a like-for-like reference point under this sweep's own eval protocol |
| `higher_lr` | 4e-4 | 16 | default | tests whether Stage A's convergence (Axis 1c) was LR-limited, not just schedule-limited |
| `larger_batch` | 2e-4 | 32 | default | tests whether batch-size-driven gradient noise was a real factor |
| `stronger_sobel` | 2e-4 | 16 | Sobel doubled to 0.2 | directly motivated by Axis 5's real finding (the model preserves only 68.7% of true edge magnitude at real annotated boundaries vs. classical's 88.1%) - the cheapest, already-existing lever to test rather than a new loss term |

Same real leakage-checked split, same seed, `ReduceLROnPlateau` (the
already-validated schedule fix), for all 4 - only the swept parameter
differs per config.

**Epoch budget: capped at 60, early-stop patience 15** - explicitly a
short, bounded, directional sweep (matching the "not full budget"
instruction), not a from-scratch full Stage A run repeated 4 times.

## Decision metric, pre-registered

Same composite score as Axis 4
(`scripts/evaluate_checkpoint_full.py`, fixed 15-30dB PSNR reference
range, equal-weighted SSIM/norm-PSNR/(1-LPIPS)) - not raw PSNR alone,
since a hyperparameter change could trade PSNR against LPIPS/SSIM.

## Decision rule

1. Rank all 4 configs (baseline included) by mean composite score on the
   val split.
2. **Adopt the winning config as the new default hyperparameters for
   subsequent axis work (Axis 1a's retrain, etc.) only if it beats
   `baseline` by at least 0.01 composite** (same margin used in Axis 4's
   rule) **and does not regress PSNR by more than 0.1dB or SSIM by more
   than 0.005 relative to baseline.**
3. If no config clears that bar, **`baseline`'s hyperparameters (Stage
   A's original recipe) remain in use** and the sweep is reported as
   informational - real evidence that this hyperparameter space is
   already reasonably chosen, not a failure to find something.

## Time-box

One sweep, 4 configs, run once, 60-epoch cap each. Not iterated with
additional configs if the result is ambiguous - report what the
pre-registered rule says.

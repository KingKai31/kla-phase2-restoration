# Item 3 — boundary-masked edge loss: pre-registered decision rule

Written and committed BEFORE any training run or measurement. **Hard
time-box: 45-60 minutes wall-clock from this commit.** If unresolved by
then, STOP and ship `b309040` (the currently-pushed, verified commit)
as final, documenting a stopped-for-time result.

## The fix, following directly from the Axis 5 diagnosis

`reports/TECHNICAL_AUDIT.md` §9: the existing `SobelEdgeLoss`
(`src/losses/stageB_composite.py`) averages gradient-magnitude error over
**all** pixels; real structural boundaries are <5% of pixels, so the term
is diluted by flat-region gradient noise, and doubling its weight (Axis
1b) changed nothing because a diluted term scaled by 2 is still diluted.

**New term, `BoundaryMaskedEdgeLoss`:** compute the same Charbonnier-on-
gradient-magnitude loss, but only over pixels where the **GT** gradient
magnitude is in the top decile (>=90th percentile, per-image) - i.e. only
at real boundaries, not diluted by the 90% flat/background pixels.
Replaces the existing diluted Sobel term in the active loss stack (not
added as a 6th term) - direct substitution isolates the comparison and
avoids re-tuning a 6-way weight balance under time pressure.

## Base for this attempt

Fine-tuned from `checkpoints/stage_a_aug_raw_best.pt` (the currently
shipped, winning checkpoint - internal PSNR 23.798, official-test PSNR
24.004), not from scratch - a refinement, expected to converge fast.
Same real leakage-checked split, same LR/schedule pattern as prior
fine-tunes in this project.

## Gate - adopt only if ALL three hold

1. **Real-mask edge retention (Ni-WC external test,
   `scripts/niwc_external_validation.py`'s boundary-ratio method,
   blend=0.00 to isolate the loss's effect from the inference blur)
   improves by >= 0.030 absolute** over the current shipped model's
   0.705 (measured in the prior pass's blur ablation) - i.e. reaches
   **>= 0.735**.
2. **Official test-set PSNR/SSIM/LPIPS do not regress beyond the measured
   seed-variance floor:** PSNR >= 24.004 - 0.026 = **23.978**, SSIM >=
   0.6257 - 0.005 = **0.6207**, LPIPS <= 0.1616 + 0.005 = **0.1666**
   (the 0.026dB PSNR floor is Axis 1c's measured two-identical-run gap;
   0.005 SSIM/LPIPS margins match every other gate's convention in this
   project).
3. **The full compliance chain passes with zero changes to `run.py`'s
   core contract** - only `models/checkpoint.pt` changes, same as the
   Item 1 swap. 25-test suite + fresh-venv/no-internet/wrong-cwd combined
   check must all pass on the new checkpoint before it is considered
   shippable.

**All three must hold.** If any fails, or the 45-60 minute window closes
first: **do not adopt.** `b309040` remains final, unmodified, unpushed-
over. The attempt is documented as a real negative (or stopped-for-time)
result with real numbers, same standard as ROI-preservation and the
correlated-noise fix.

## Non-negotiable safety rule

`models/checkpoint.pt` at `b309040` is never overwritten in the working
tree until Gate 1-3 all pass. No commit is made unless the gate passes.
If time runs out mid-training, the training process is simply killed and
nothing is committed - `git status` stays clean relative to `b309040`.

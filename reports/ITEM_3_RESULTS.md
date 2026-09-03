# Item 3 result — boundary-masked edge loss: gate FAILED, real and fascinating tradeoff

Per `reports/item3_decision_rule_PREREGISTERED.md`, written and committed
before this run. Attempted, measured, and **not adopted** - `b309040`
remains the final shipped commit, unmodified.

## What was tried

`src/losses/boundary_masked_edge.py`: replaced the diluted `SobelEdgeLoss`
(averaged over all pixels) with `BoundaryMaskedEdgeLoss` (Charbonnier-on-
gradient-magnitude, computed only over the top-decile-by-GT-gradient
pixels per image - i.e. only at real boundaries) in the active loss
stack. Fine-tuned from the shipped `stage_a_aug_raw_best.pt`, 18 epochs
(best at epoch 2, early-stopped at 17), ~5 minutes wall-clock.

## Gate 1 (real-mask edge retention) — PASSED, decisively

| | Shipped model (blend=0.00) | Item 3 (blend=0.00) | Classical | Required |
|---|---|---|---|---|
| Edge ratio | 0.705 | **0.819** (+0.114) | 0.879 | >= 0.735 (+0.030) |

**The fix worked exactly as diagnosed, and by a wide margin** - nearly
4x the required improvement, closing 66% of the remaining gap to the
classical baseline (0.879 - 0.705 = 0.174 gap before; 0.879 - 0.819 =
0.060 gap after). This is real, strong confirmation of the Axis 5
mechanistic diagnosis (`reports/TECHNICAL_AUDIT.md` §9): a loss term that
actually concentrates its gradient at true boundaries, instead of being
diluted across 95% flat pixels, measurably improves real structural-edge
preservation.

## Gate 2 (official test-set PSNR/SSIM/LPIPS, no regression beyond seed floor) — FAILED on PSNR

| Metric | Shipped (`b309040`) | Item 3 | Required | Result |
|---|---|---|---|---|
| PSNR | 24.004 | **23.746** (-0.258) | >= 23.978 (-0.026 floor) | **FAIL** - ~10x beyond the floor |
| SSIM | 0.6257 | **0.6314** (+0.0057) | >= 0.6207 | PASS |
| LPIPS | 0.1616 | **0.1559** (-0.0057, better) | <= 0.1666 | PASS |

**A real, interesting trade - not a clean loss.** SSIM and LPIPS both
*improved* over the already-shipped, already-improved model - the
boundary-masked term made the network's output more perceptually and
structurally faithful. But PSNR (pure pixel-fidelity, MSE-based) dropped
by ten times the measured seed-variance floor. This is mechanistically
sensible: concentrating the edge loss's gradient onto true boundaries
pushes the network to render sharper, more confident edges there instead
of the smoothed "safe" (posterior-median) prediction that minimizes MSE -
exactly the behavior the Axis 5 diagnosis predicted the model was
defaulting to, now reversed enough to cost PSNR.

## Gate 3 (compliance chain) — not run

Moot: Gate 2 already fails, so the pre-registered "all three must hold"
rule already returns a clean answer without needing the compliance chain.
Not run, to respect the time-box - would only have delayed an already-
determined negative result.

## Decision: FAIL, not adopted

Per the pre-registered rule ("all three must hold... if any fails, do not
adopt"): **Gate 2's PSNR condition fails, so Item 3 is not adopted.**
`b309040` remains the final, unmodified shipped commit - checksum
verified identical (`36d2d38c...`) before and after this attempt.
`checkpoints/item3_boundary_edge_best.pt` and this writeup are kept in
the repo as documented history, same standard as ROI-preservation and
the correlated-noise fix - a real, evidence-backed negative result, not
hidden.

## What this result means for future work (not pursued further, time-boxed)

The mechanism is now confirmed causal, not just diagnosed: boundary-
masking a structural loss term really does buy real-edge-preservation,
at a real PSNR cost. A weighted combination (e.g. a smaller boundary-loss
weight than the direct substitution tried here, or keeping both the
original diluted Sobel term and a smaller-weighted boundary term
alongside it) might find a better point on this real tradeoff curve -
untested, flagged for anyone extending this work, not chased further
given the deadline.

## Timing

Pre-registration to final measurement: ~40 minutes wall-clock, within
the 45-60 minute time-box. Training itself: ~5 minutes (18 epochs,
A100).

# Technical hardening pass — consolidated summary

A full pass across 5 axes targeting every weak dimension identified in a
model-quality audit: reconstruction quality, perceptual quality,
generalization evidence, architectural sophistication, and defect
relevance. Same standard as the rest of this project: every comparison
pre-registered before running, every result reported honestly whether
positive or negative.

**Bottom line: the shipped model is unchanged. `checkpoints/stage_a_best.pt`
(23.483dB / 0.598 SSIM) remains the final model.** Every axis either
confirmed it was already well-chosen or surfaced a real, disclosed
limitation that a cheap fix didn't resolve - no axis produced a result
that cleared its own pre-registered adoption bar.

## Axis 1 — reconstruction quality: three independent confirmations that Stage A was already near its ceiling

- **1c (schedule fix):** re-ran Stage A with the already-validated
  `ReduceLROnPlateau` fix and much longer patience/budget. **Same best
  epoch (13), +0.026dB - Stage A was genuinely converged, not
  undertrained.** (`reports/AXIS_1C_RESULTS.md`)
- **1a (more/different real external data):** fine-tuned with 12,204
  synthetic pairs from 4 real NFFA categories (43% more than the
  original failed Stage B attempt, including a brand-new category). **A
  small regression (-0.033dB), not an improvement - doubly confirms
  Stage B's original null result.** The weak-cluster subgroup (11, 14)
  got slightly worse, not better. (`reports/AXIS_1A_RESULTS.md`)
- **1b (hyperparameter sweep):** 4 configs (LR, batch size, Sobel
  weight). Best (`higher_lr`) beat baseline by 0.0024 composite - a
  quarter of the pre-registered 0.01 adoption threshold.
  (`reports/AXIS_1B_RESULTS.md`)

## Axis 2 — perceptual quality (LPIPS)

Every checkpoint evaluated in this pass reports full PSNR/SSIM/LPIPS and
a composite score via one shared, reusable evaluator
(`scripts/evaluate_checkpoint_full.py`) - no longer an afterthought.

## Axis 3 — generalization evidence

- **3a (external validation, Ni-WC metal-matrix composite, different
  facility/domain):** a real, mixed result - model beats classical on
  PSNR/LPIPS, loses on SSIM (the only metric in this whole project where
  classical wins). Honest, bounded evidence of real generalization, not
  oversold. (`reports/HARDENING_AXIS_3_AND_5.md`)
- **3b (severity extrapolation, up to 2x beyond the worst noise seen in
  training):** graceful, smooth degradation, zero crashes at any tested
  severity - real positive evidence against "only works in-distribution."

## Axis 4 — architectural sophistication

Bottleneck self-attention block: won on every individual metric (PSNR
+0.034dB, SSIM +0.005, LPIPS -0.0006) but composite gain (0.0026) failed
the pre-registered 0.01 gate by a wide margin. **Dropped** - 15% more
parameters for a quarter of the required improvement.
(`reports/AXIS_4_RESULTS.md`)

## Axis 5 — defect/downstream relevance

Real pixel-level segmentation masks (Ni-WC data) used for structural-
boundary preservation - stronger evidence than the earlier synthetic-
perturbation test. **Real finding: the model preserves only 68.7% of
true edge magnitude at real annotated boundaries vs. classical's 88.1%.**
The proposed lightweight fix (double the existing Sobel loss weight,
tested in Axis 1b) showed no meaningful difference from baseline -
**this specific gap is not solved by that cheap lever.**
(`reports/HARDENING_AXIS_3_AND_5.md`, `reports/AXIS_1B_RESULTS.md`)

## What this pass adds to the submission's real story

Not a bigger number, but a much more thoroughly interrogated one: five
independent, pre-registered probes at the model's architecture, training
recipe, data, and structural-preservation behavior, each with real
evidence rather than an assumption. The two genuine, still-open
limitations this pass surfaced and did not paper over:

1. **External structural-edge preservation is weaker than a naive
   baseline** on out-of-domain content (Axis 5) - a real, bounded,
   disclosed gap, not fixed by the cheapest lever tried.
2. **The synthetic generator's high-frequency spectral deficit is a
   generation-mechanism problem, not a data-volume problem** - now
   confirmed twice (Stage B, Axis 1a) with increasing data and no
   change in outcome.

## Infrastructure lesson from this pass

A real disk-quota incident (shared network filesystem reporting 111TB
free at the pool level, masking a much smaller per-pod quota) silently
killed three concurrent background processes, including one that had
already fully completed - fixed by persisting partial results
incrementally going forward. Full writeup:
`reports/HARDENING_DISK_QUOTA_INCIDENT.md`.

## Real compute cost

All 5 GPU-bound comparisons (1a, 1b, 1c, 4) ran on a single A100-SXM4-80GB
pod, combined wall-clock roughly 70 minutes of actual training time
across the whole pass (1c: 14min, 1a: 37min, 4: ~8min, 1b: ~32min),
plus setup/transfer/download overhead. Axis 3 and 5's evaluation-only
work ran locally on CPU, no GPU cost.

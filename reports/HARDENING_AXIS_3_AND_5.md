# Technical hardening pass — Axis 3 (generalization) and Axis 5 (defect relevance)

Both run locally on CPU against `checkpoints/stage_a_best.pt` (the shipped
model) - no GPU/pod needed for either, since they're evaluation-only.

---

## Axis 3a — genuine external validation (Ni-WC metal-matrix composite, Zenodo CC-BY-4.0)

**Data honesty check, done before using it:** the Zenodo record
(10.5281/zenodo.17315241) ships only an "Augmented" set - 405 files that
are 9 augmented variants (5 geometric, 4 photometric) of just **45 truly
independent 512x512 source crops**. Using all 405 would overstate the
effective sample size with correlated near-duplicates. This test uses
only the `RandomRotate90` variant of each of the 45 base crops - a pure
rigid transform (no warping, no intensity change), confirmed present for
all 45/45 - then tiles each into 4 non-overlapping 256x256 regions
(180 tiles total, from 45 independent images, reported as both numbers).

**Method:** each real 256x256 tile is a genuine SEM image from a
different facility and a completely different specimen domain (metal
matrix composite, not semiconductor). Degraded with our own validated
compound noise model, restored with the shipped model and, separately,
with run.py's real classical fallback.

| | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Model (Stage A) | 17.730 | 0.3261 | 0.4150 |
| Classical (bicubic+NLM) | 16.591 | 0.3792 | 0.6422 |

**A real, mixed result - not a clean win, reported honestly.** The model
beats classical on PSNR (+1.14dB) and LPIPS (much better, 0.415 vs
0.642), but **loses on SSIM** (0.326 vs 0.379) - the only place in this
entire project where the classical baseline beats the model on any
metric. On genuinely out-of-domain content, the model's advantage is real
but partial, not uniform - a legitimate generalization signal (it isn't
just memorizing this dataset's specific texture), stated at the strength
the evidence actually supports, not oversold.

Full per-tile data: `reports/niwc_external_validation_per_tile.csv`.
Summary: `reports/niwc_external_validation_summary.json`.

---

## Axis 5a/5b/5c — real-mask structural-boundary preservation

Same 45-crop/180-tile Ni-WC set, using its **real pixel-level segmentation
masks** (not synthetic perturbations, unlike the earlier ROI-loss stress
test). Boundary pixels = any pixel whose class differs from a
4-connected neighbor in the real annotation. At those exact locations,
compared each restoration's Sobel gradient magnitude to the clean
reference's:

| | Mean edge ratio (1.0 = fully preserved) | Mean edge correlation with true pattern |
|---|---|---|
| Model (Stage A) | 0.687 | 0.631 |
| Classical (bicubic+NLM) | 0.881 | 0.581 |

**Axis 5c applies: the model does NOT preserve real annotated structure
as well as the naive classical baseline, on this genuinely different
domain.** The model retains only 68.7% of the true edge magnitude at real
structural boundaries, versus classical's 88.1% - the model measurably
over-smooths real fine structure it hasn't been trained on, even though
its spatial *pattern* of edges correlates slightly better with ground
truth (0.631 vs 0.581) and its overall PSNR/LPIPS are better. This is
consistent across all 4 magnification levels in the dataset (600x-1000x),
not a fluke of one subset.

**Precise, bounded claim (same standard as every other finding in this
project):** this is real evidence from a genuinely different domain
(metal composite defects, not semiconductor inspection defects) with real
annotations, not synthetic ones - stronger evidence than the earlier
synthetic-perturbation test, but still not a claim about semiconductor
defect preservation specifically, since no real semiconductor defect
annotations exist for this project's own data.

**What this means for Axis 5c's fork:** the model is genuinely weaker
than a naive baseline at real fine-structure retention on unfamiliar
content. A targeted, lighter fix (higher weight on the existing Sobel
edge-loss term, tested with a pre-registered rule, NOT a new 6th loss
term) is worth trying - **but this requires a real training run and is
therefore GPU-bound, blocked on the fresh RunPod pod's environment
finishing setup.** Flagged as the next concrete step once the pod is
ready, not attempted on CPU.

---

## Axis 3b — severity-extrapolation stress test

**Interpretation note, flagged before running:** the original ask was
"1.5-2x the maximum observed L_gain/K_poisson." In this noise model,
*higher* L_gain means *less* multiplicative noise (Gamma(L,1/L) variance
~1/L) and *higher* K_poisson means *less* shot noise - so 1.5-2x the
*maximum* would test a *milder* degradation than anything already seen in
training, not a harder one, and wouldn't answer what "unseen noise
levels" is actually asking. This test instead extrapolates from the
worst already-seen corner of the measured range (L_gain=29.3,
K_poisson=31.1, sigma_A=0.0151 - the real per-cluster fit minimums/
maximum) in the genuinely harder direction: each severity multiplier
divides L_gain/K_poisson and multiplies sigma_A.

| Severity | L_gain | K_poisson | sigma_A | Model PSNR | Model SSIM | Classical PSNR | Non-finite outputs |
|---|---|---|---|---|---|---|---|
| 1.0x (worst seen) | 29.32 | 31.07 | 0.0151 | 19.89 | 0.402 | 17.24 | 0/60 |
| 1.25x | 23.46 | 24.85 | 0.0189 | 18.89 | 0.359 | 16.44 | 0/60 |
| 1.5x | 19.55 | 20.71 | 0.0227 | 18.11 | 0.325 | 15.78 | 0/60 |
| 1.75x | 16.75 | 17.75 | 0.0265 | 17.49 | 0.299 | 15.23 | 0/60 |
| 2.0x | 14.66 | 15.53 | 0.0303 | 17.01 | 0.278 | 14.75 | 0/60 |

**Real, positive evidence of graceful degradation.** PSNR declines
smoothly and predictably (19.89dB -> 17.01dB, roughly -1.4dB per 0.25x
step, no cliff), **zero non-finite/crashed outputs at any severity
tested, including 2x beyond the worst noise level ever seen in
training.** The model's margin over the classical baseline stays roughly
stable through the extrapolated range (+2.65dB at 1.0x, +2.26dB at
2.0x) - it doesn't collapse faster than the naive baseline under stress,
real evidence against "the model only works in-distribution and falls
apart outside it."

Full per-image data: `reports/severity_extrapolation_per_image.csv`.
Summary: `reports/severity_extrapolation_summary.csv`.

---

## Status: Axis 3 and 5's evaluation-only work is done

Axis 3a, 3b, and 5a/b/c are complete - all three ran on CPU, no GPU/pod
dependency. The one GPU-bound follow-up this axis pair identified (a
Sobel-weight ablation, per 5c) is queued behind the fresh pod's setup,
not attempted here.

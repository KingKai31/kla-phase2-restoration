# Official test-set results — the true numbers, clearly labeled

**These are the only numbers in this project measured against real,
externally-released ground truth.** Every other metric in this repo
(Stage A's 23.483dB, every hardening-pass number, every per-cluster
table) was measured on an *internal* stratified validation proxy split
carved out of our own training delivery. This is different: KLA's Phase 2
test set, released 2026-09-03, paired NoisyLR inputs with real clean GT.

## The data

| | Count | Shape | dtype | Range | Finite |
|---|---|---|---|---|---|
| NoisyLR (input) | 297 | (128, 128) | float32 | [-0.180, 2.038] | all |
| GT (reference) | 297 | (256, 256) | float32 | [0.000, 1.000] | all |

Exact filename pairing (`000000.npy` … `000296.npy`) - identical
convention to the training delivery, including NoisyLR overshooting
[0,1] in both directions exactly as the training data does.

**Leakage check, done before trusting any number:** perceptual-hash
(aHash, 256-bit, same method as `scripts/check_source_overlap.py`)
comparison of all 297 test GT images against all 4,785 training GT
images. Median minimum Hamming distance **81/256 bits** - the test set is
overwhelmingly disjoint from training. Exactly one candidate sat at the
conservative match threshold (test `000285` vs. train `003616`, distance
10): visually and numerically verified **not** a duplicate (pixel
correlation 0.33, mean |diff| 0.139 - a distinct particle-structure image
vs. a flat granular texture, a coarse-brightness coincidence). Figure:
`reports/figures/official_test_borderline_pair_check.png`. **No leakage.**

## Results — the shipped `run.py`, exactly as submitted, on all 297 images

Restored outputs written to `test_predictions/` (in the repo, per the
submission checklist). All 297: model path succeeded, **zero classical
fallbacks, zero load failures.** Every output verified (256, 256)
float32, in [0, 1], finite, exact 2x resolution, exact filename match.

| Method | PSNR (dB) | SSIM | LPIPS ↓ |
|---|---|---|---|
| **Shipped model (Stage A, run.py)** | **23.761** | **0.6090** | **0.1943** |
| Classical baseline (bicubic + NLM, run.py's real fallback) | 20.332 | 0.5133 | 0.4993 |

### Statistical significance (paired, n=297, same images scored by both)

Paired Wilcoxon signed-rank test + bootstrap 95% CI (2,000 resamples)
on the per-image difference, model minus classical:

| Metric | Mean diff | Bootstrap 95% CI | Wilcoxon p |
|---|---|---|---|
| PSNR | **+3.43 dB** | [+3.21, +3.66] | 1.9e-50 |
| SSIM | **+0.096** | [+0.083, +0.109] | 1.3e-41 |
| LPIPS | **-0.305** (better) | [-0.321, -0.290] | 1.9e-50 |

**All three improvements are decisive and tightly bounded** - not one of
them is close to a noise-level result.

## How this compares to the internal proxy split

| | Internal val proxy (n=712) | Official test (n=297) |
|---|---|---|
| Model PSNR | 23.483 | 23.761 |
| Model SSIM | 0.5976 | 0.6090 |
| Model LPIPS | not computed | 0.1943 |
| Classical PSNR | 20.273 | 20.332 |
| Model gain over classical | +3.21 dB | +3.43 dB |

**The internal proxy was honest, not optimistic.** The official test
numbers are slightly *better* than the internal split's, and the
model-vs-classical margin is slightly *larger*. Every internal number in
this project was, if anything, a mild under-estimate of real held-out
performance - the stratified, leakage-checked split did its job.

## What this does and does not say

- **Does say:** the shipped model generalizes to KLA's real held-out data
  at least as well as it did internally; the +3.4dB margin over a naive
  method is real and decisive; nothing in the pipeline broke on
  never-seen inputs (0 fallbacks).
- **Does not say:** anything about KLA's final scoring weights (unknown),
  or about the Axis 5 structural-edge-preservation gap - that finding
  (`reports/TECHNICAL_HARDENING_PASS_SUMMARY.md`) was measured on
  different-domain data with real masks; this test set has no
  segmentation masks, so it cannot confirm or refute it. That remains the
  most important open limitation, unchanged by these numbers.

Per-image data: `reports/official_test_set_per_image_metrics.csv`.
Summary: `reports/official_test_set_summary.json`. Significance:
`reports/official_test_set_significance.json`.

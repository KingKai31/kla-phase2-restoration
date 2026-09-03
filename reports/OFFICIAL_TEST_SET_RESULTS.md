# Official test-set results — the true numbers, clearly labeled

**These are the only numbers in this project measured against real,
externally-released ground truth.** Every other metric in this repo
was measured on an *internal* stratified validation proxy split carved
out of our own training delivery. This is different: KLA's Phase 2 test
set, released 2026-09-03, paired NoisyLR inputs with real clean GT.

**Updated after the improvement pass (same day):** the shipped model
changed (Item 1 - dihedral augmentation + EMA + ICNR, see
`reports/ITEM_1_2_RESULTS.md`). This document reports both the original
Stage A model's official numbers and the improved model's, so the real
gain is stated against the true benchmark, not just the internal split.

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
submission checklist) using the **final shipped model** (Stage A +
augmentation/EMA/ICNR + decoder-capacity block, `models/checkpoint.pt`
sha256 `7b77678d1742...c9fb0412`), regenerated from scratch after that
model was adopted. All
297: model path succeeded, **zero classical fallbacks, zero load
failures.** Every output verified (256, 256) float32, in [0, 1], finite,
exact 2x resolution, exact filename match.

| Method | PSNR (dB) | SSIM | LPIPS ↓ |
|---|---|---|---|
| **Shipped model, FINAL (+ decoder-capacity block)** | **24.001** | **0.6251** | **0.1605** |
| Stage A + augmentation/EMA/ICNR (the prior shipped model) | 24.004 | 0.6257 | 0.1616 |
| Shipped model, original (Stage A) | 23.761 | 0.6090 | 0.1943 |
| Classical baseline (bicubic + NLM, run.py's real fallback) | 20.332 | 0.5133 | 0.4993 |

### Item 1 vs. original Stage A, on the official test set (paired, n=297)

Paired Wilcoxon + bootstrap 95% CI (2,000 resamples) - the real,
official-benchmark version of the internal Gate 1 comparison:

| Metric | Mean diff (new − old) | Bootstrap 95% CI | Wilcoxon p |
|---|---|---|---|
| PSNR | **+0.243 dB** | [+0.219, +0.269] | 1.4e-49 |
| SSIM | **+0.0167** | [+0.0152, +0.0183] | 1.6e-49 |
| LPIPS | **-0.0327** (better) | [-0.0370, -0.0285] | 1.8e-38 |

**Decisive on the real benchmark, not just the internal split** - every
CI is tight and entirely on the improving side, at p<1e-37 for all three
metrics. Slightly smaller PSNR gain than the internal split showed
(+0.243 vs. +0.315dB) but the same direction, same order of magnitude,
same statistical certainty.

### Final model vs. classical baseline (paired, n=297)

Classical predictions are unaffected by Item 1 (bicubic+NLM doesn't
depend on the trained model), so this reuses the same classical
per-image numbers, paired against the final model's real outputs -
computed fresh, not estimated:

| Metric | Mean diff | Bootstrap 95% CI | Wilcoxon p |
|---|---|---|---|
| PSNR | **+3.67 dB** | [+3.44, +3.92] | 1.9e-50 |
| SSIM | **+0.112** | [+0.100, +0.125] | 1.3e-49 |
| LPIPS | **-0.338** (better) | [-0.353, -0.323] | 1.9e-50 |

(Original-model-vs-classical significance, unchanged, retained below for
provenance.)

| Metric | Mean diff | Bootstrap 95% CI | Wilcoxon p |
|---|---|---|---|
| PSNR | +3.43 dB | [+3.21, +3.66] | 1.9e-50 |
| SSIM | +0.096 | [+0.083, +0.109] | 1.3e-41 |
| LPIPS | -0.305 (better) | [-0.321, -0.290] | 1.9e-50 |

Full detail: `reports/official_test_set_significance_item1_vs_classical.json`.

### Final shipped model vs. the prior shipped model (paired, n=297)

The decoder-capacity block that produced the final model was adopted on a
**real-mask edge-retention** gate (0.705 -> 0.735), not a PSNR gate. Its
effect on the official-test pixel metrics is reported exactly here,
because it is a clean example of statistical significance without
practical significance:

| Metric | Mean diff (final − prior) | Wilcoxon p | vs. the 0.026 dB reproducibility floor |
|---|---|---|---|
| PSNR | −0.0033 dB | 0.003 | ~8x **smaller** than the floor |
| SSIM | −0.00063 | 2.2e-14 | negligible |
| LPIPS | −0.0011 (better) | 0.0035 | negligible |

At n=297 even micro-differences reach significance. The pre-registered
gate deliberately used the measured same-seed reproducibility floor
rather than a p-value for exactly this reason. **Read plainly: the pixel
metrics are unchanged in any way that matters, and the +0.030
edge-retention gain is an order of magnitude larger than any of these
deltas.** The two "vs. classical" tables above were computed on the prior
model's outputs; since the final model differs from it by less than the
reproducibility floor on every pixel metric, those margins carry over
unchanged (24.001 − 20.332 = +3.67 dB).

## How this compares to the internal proxy split

| | Internal val proxy (n=712) | Official test (n=297) |
|---|---|---|
| Final model PSNR | 23.731 | 24.001 |
| Final model SSIM | 0.6138 | 0.6251 |
| Final model LPIPS | 0.1666* | 0.1605 |
| Classical PSNR | 20.273 | 20.332 |

\* internal-val LPIPS was not recomputed for the decoder-capacity
checkpoint (its gate was defined on official-test metrics plus real-mask
edge retention); the value shown is the prior model's, for scale only.

**The internal proxy remains honest, not optimistic** - the official test
numbers are again slightly *better* than the internal split's, for the
final model exactly as they were for the original one.

## What this does and does not say

- **Does say:** the improved model generalizes to KLA's real held-out
  data at least as well as it did internally, and the Item 1 gain
  (augmentation + EMA) is real and decisive on the actual benchmark, not
  an internal-split artifact; nothing in the pipeline broke on never-seen
  inputs (0 fallbacks) before or after the change.
- **Does not say:** anything about KLA's final scoring weights (unknown),
  or about the Axis 5 structural-edge-preservation gap - that finding
  (`reports/TECHNICAL_HARDENING_PASS_SUMMARY.md`) was measured on
  different-domain data with real masks; this test set has no
  segmentation masks, so it cannot confirm or refute it, and Item 1 did
  not target that mechanism. **That remains the most important open
  limitation, unchanged by these numbers or this pass** (Item 3, the fix
  that follows from the diagnosed mechanism, was not reached/not gated
  in - see `reports/ITEM_1_2_RESULTS.md`).

Per-image data: `reports/official_test_set_per_image_metrics_item1.csv`
(final model), `reports/official_test_set_per_image_metrics.csv`
(original model, retained). Summaries:
`reports/official_test_set_summary_item1.json`,
`reports/official_test_set_summary.json`. Significance:
`reports/official_test_set_significance_item1_vs_stageA.json`,
`reports/official_test_set_significance.json`.

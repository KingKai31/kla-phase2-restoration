# Axis 4 — bottleneck self-attention A/B test: pre-registered decision rule

Written and committed BEFORE any comparison is run, per this project's
established practice (ROI-preservation loss, correlated-noise spectral
fix). This is a short, bounded, directional test - not a
full-convergence architecture search.

## What's being compared

- **Baseline**: `NAFNetSR` exactly as shipped (`src/models/nafnet.py`),
  unchanged.
- **Variant**: the identical backbone with ONE lightweight multi-head
  self-attention block added at the bottleneck (after the existing
  `middle` NAFBlocks, at the lowest spatial resolution - 8x8 for a
  128x128 input, 4 heads, standard residual + LayerNorm wrapping, no
  positional encoding needed at this resolution). Cheap by construction:
  at 8x8=64 tokens, the attention matrix is 64x64, trivial next to the
  convolutional cost of the rest of the network. This targets the
  long-range/periodic-structure limitation flagged as a real
  architectural gap in earlier strategy notes - NAFNet's convolutions are
  local; a single global attention pass at the bottleneck is the cheapest
  way to test whether that gap is actually costing real quality.

## Training protocol (identical for both configs)

- Same real leakage-checked stratified split, same 5-term
  `StageBCompositeLoss` (no ROI term - dropped, not reintroduced), same
  seed, same optimizer/LR (`ReduceLROnPlateau`, matching the fix already
  validated for Stage B).
- **15 epochs each, not full convergence** - explicitly a short, bounded,
  directional comparison per the instruction that started this pass.
  Stage A's own best epoch was 13/39, so 15 epochs is enough to see real
  early-training signal without paying for a full run twice.

## Decision metric: composite score, pre-registered before running

Raw PSNR/SSIM/LPIPS don't answer "which is better" on their own when they
trade off against each other. Composite score, same "fixed reference
range, not min-max across just these two values" principle used in
Phase 1's own composite scoring:

```
norm_psnr = clip((psnr - 15) / (30 - 15), 0, 1)   # fixed 15-30dB reference range
composite = (1/3) * ssim + (1/3) * norm_psnr + (1/3) * (1 - lpips)
```

Computed on the same val split, same images, for both configs.

## Decision rule

**Adopt the attention variant for the final model only if BOTH hold:**

1. Composite score improves by **at least 0.01** over baseline (a real,
   meaningful margin - comparable in scale to the composite-score margins
   Phase 1 itself treated as decisive, 1.6-7.9%).
2. **Neither PSNR regresses by more than 0.1dB nor SSIM by more than
   0.005** individually - a guardrail against a composite-score win that
   is actually one metric cratering to inflate another (the same
   regression-guardrail structure used for the ROI-preservation loss's
   own decision rule).

**If either condition fails: the attention block is dropped and the
result is reported as a negative/null finding - not tuned post-hoc, not
retried with a different head count or insertion point.** Same standard
as the ROI-preservation loss and the correlated-noise spectral fix: a
pre-registered rule is only meaningful if it's actually followed when the
result doesn't cooperate.

## Time-box

One comparison, 15 epochs per config, run once. Not iterated on
insertion point, head count, or epoch budget if the first result is
ambiguous - report what the pre-registered rule says and move on, per
the explicit "don't chase a result" instruction for this axis.

# Item 3 result — auxiliary confidence/residual head: real signal, below the pre-registered bar

Per `reports/new_techniques_decision_rules_PREREGISTERED.md`. Strictly
additive: the shipped model's weights were frozen throughout, and its
main restoration output is architecturally untouched - verified before
training (see "Safety verification" below).

## What was built

A small head (two convs + PixelShuffle, ~8k params) reads the frozen
decoder's feature map immediately before the pixel-shuffle head (the
same tap point Item 1 inserts its capacity increase at) and predicts a
per-pixel `|prediction - GT|` map, trained on real per-pixel residuals
from the shipped model's actual restoration - "where is the model's own
output likely to be wrong." Trained in 70 seconds (14 epochs, frozen
main network means only the small head needs gradients).

## Result — real, universal, but below the pre-registered "meaningful" bar

| Metric | Value |
|---|---|
| **Per-image mean Spearman r** (the pre-registered gate metric) | **0.226** |
| Per-image median | 0.209 |
| Per-image std | 0.126 |
| Per-image min / max | 0.012 / 0.737 |
| Images with r > 0 | **712 / 712 (100%)** |
| Images individually significant (p<0.05) | **712 / 712 (100%)** |
| Pooled correlation across 356,000 sampled pixels | 0.434 (p≈0) |

**Fails the pre-registered gate** (mean 0.226 < the required 0.3). Per
the rule agreed in advance, **not adopted as a "genuine, demonstrable"
finding for the deck** - reported as a real, honest, partial result
instead.

## Why the pooled number (0.434) is not the number that matters

The pooled correlation, computed across all sampled pixels from every
image at once, is noticeably higher than the per-image mean. This
inflation is expected and was anticipated in the pre-registration: pooling
partly captures "some images are just harder overall" (higher average
confidence AND higher average error together), which is a different,
weaker claim than "within this one image, the map correctly points to
where the errors are." **The per-image metric is the one that matches the
actual proposed use case** (an inspector looking at a single restored
image, asking "where in this image should I double-check") - that is why
it, not the pooled number, was the pre-registered gate.

## The real, non-trivial part of this result

This is not a null result. **Every single one of the 712 held-out
validation images** shows a positive correlation, and every one is
individually statistically significant. That is a genuinely consistent,
non-spurious signal - the confidence head learned something real about
where the model tends to err, not noise that happened to average out
positive. It simply isn't strong enough, on a typical image, to clear the
bar set in advance for calling it a demonstrable innovation. About a
quarter of images (75th percentile = 0.301) already individually clear
the 0.3 bar - the signal is real and occasionally strong, just not
reliably strong enough on a typical image to adopt as a headline claim.

## Safety verification (no risk to the shipped model)

- The main model's parameters were loaded once, set to `.eval()`, and
  every parameter had `requires_grad_(False)` set before any training
  step - verified locally before this ever ran on the pod
  (`main model has no grad (frozen): True`).
- `run.py`'s inference path does not import or reference this head at
  all - it is a separate script, separate checkpoint
  (`checkpoints/item3_confidence_head.pt`), never loaded by the shipped
  inference code.
- `models/checkpoint.pt` (sha256 `36d2d38c...`) is unchanged by this item
  by construction, not just by choice.

## Not pursued further this pass

Given the gate result, no further tuning of the head architecture or
loss was attempted, per the same-standard discipline of this project -
report the honest result and move on rather than iterate until a
pre-registered threshold is cleared. Full per-image data:
`reports/item3_confidence_per_image_correlation.csv`.

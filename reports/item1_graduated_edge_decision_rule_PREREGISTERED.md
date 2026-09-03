# Item 1 (final pass) — graduated boundary-masked auxiliary loss: pre-registered decision rule

Written and committed BEFORE any training run or measurement.

## What's different from Item 3 (not a repeat)

Item 3 (`reports/ITEM_3_RESULTS.md`) **replaced** the diluted `SobelEdgeLoss`
with `BoundaryMaskedEdgeLoss` at full weight (0.1, the same weight the
term it replaced had) - a hard substitution. It worked mechanistically
(edge retention 0.705 -> 0.819) but cost 0.258dB PSNR, ~10x the
reproducibility floor, and was not adopted.

This attempt: **add** the exact same `BoundaryMaskedEdgeLoss` class
(`src/losses/boundary_masked_edge.py`, unmodified - reused, not
rewritten) as a **6th term alongside** the existing, unchanged 5-term
stack (Charbonnier + MS-SSIM + LPIPS + `SobelEdgeLoss` + range-
consistency), at a small weight, fine-tuning from the **currently
shipped checkpoint** (`models/checkpoint.pt`, sha256 `36d2d38c...`), not
from Item 3's checkpoint and not from scratch. The existing diluted Sobel
term is NOT removed - this tests a small nudge on top of the validated
recipe, not a replacement.

## Configurations - 3 weights, one training pass each

| Config | Boundary-edge weight | Base stack |
|---|---|---|
| `w0.05` | 0.05 | unchanged (char=1.0, msssim=0.2, lpips=0.075, sobel=0.1, range=0.05) |
| `w0.10` | 0.10 | unchanged |
| `w0.20` | 0.20 | unchanged |

Same real leakage-checked split, same LR (5e-5, fine-tune scale), same
`ReduceLROnPlateau`, same short-patience early stopping as prior
fine-tunes in this project (Item 3's own recipe).

## Gate - adopt the best-clearing weight only if it passes ALL THREE

For each weight, evaluated on the **official 297-image test set**:

1. **PSNR does not regress beyond 0.026dB** below the shipped model's
   24.004 - i.e. **>= 23.978**. Reported and referred to throughout as a
   **same-seed, different-schedule reproducibility bound** (Axis 1c), NOT
   a formal cross-seed variance estimate - this precise framing is used
   in every place this number is cited, per the correction already on
   record in `reports/ITEM_B_RESULTS.md`.
2. **Real-mask edge retention (Ni-WC, blend=0.00) improves by >= 0.020
   absolute** over the shipped model's 0.705 - i.e. **>= 0.725**. (A
   more modest bar than Item 3's 0.030, since this is a smaller,
   more conservative intervention by design.)
3. **SSIM and LPIPS do not regress** below the shipped model's 0.6257 /
   above 0.1616 respectively.

**Among weights that clear all three, adopt the one with the best edge-
retention improvement.** If none clear the gate, this is reported as a
real negative result: the PSNR/edge-retention tradeoff is structural to
this loss family at this data scale, not resolvable by weight tuning
alone - precise, valuable information about the model's loss landscape,
not a failure to try hard enough.

## Time-box

~45 minutes soft limit for this item (3 short fine-tunes + evaluation).
If exceeded without a clear signal, stop, document as time-boxed, move
to Item 2.

## Non-negotiable safety rule

`models/checkpoint.pt` is never overwritten in the working tree until a
weight passes the gate. No commit swaps the shipped checkpoint unless the
gate passes for that specific weight.

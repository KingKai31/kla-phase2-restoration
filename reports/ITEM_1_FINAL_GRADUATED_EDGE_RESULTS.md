# Item 1 (final pass) result — graduated boundary-edge auxiliary loss: gate FAILED at all 3 weights

Per `reports/item1_graduated_edge_decision_rule_PREREGISTERED.md`. Not a
repeat of Item 3: this **added** `BoundaryMaskedEdgeLoss` as a 6th term
alongside the unchanged 5-term stack (the diluted `SobelEdgeLoss` stayed
in place), at three small weights, fine-tuning from the currently
shipped checkpoint - the literature-motivated "small-weight auxiliary
term" configuration, not a hard replacement.

## Official-test-set results, all three weights

| Weight | PSNR (need ≥23.978) | SSIM (need ≥0.6257) | LPIPS (need ≤0.1616) | Gate |
|---|---|---|---|---|
| 0.05 | 23.906 (-0.098dB) | 0.6298 (better) | 0.1574 (better) | **FAIL** - PSNR only |
| 0.10 | 23.781 (-0.223dB) | 0.6310 (better) | 0.1571 (better) | **FAIL** - PSNR only |
| 0.20 | 23.542 (-0.462dB) | 0.6316 (better) | 0.1623 (worse) | **FAIL** - PSNR and LPIPS |

Internal val showed the identical monotonic pattern before the official-
test numbers were even measured (23.798 init -> 23.596 / 23.445 / 23.172
for w=0.05/0.10/0.20 respectively) - a clean, real dose-response, not
noise.

## Decision: NONE adopted - a real, structural negative result

Per the pre-registered rule ("if none of the three weights clear the
gate: this is a real, important negative result... it would mean the
trade-off is structural, not a matter of weight tuning"): **that is
exactly what was found.** Even the smallest tested weight (0.05 - a
twentieth the scale of the base Charbonnier term) cost 0.098dB, roughly
4x the reproducibility floor. SSIM and LPIPS improved at the two smaller
weights, consistent with Item 3's finding that this mechanism genuinely
trades pixel fidelity for perceptual/structural quality - but the
PSNR cost is not eliminable by using a smaller weight; it appears
immediately and scales smoothly with weight.

**Not adopted at any weight.** `models/checkpoint.pt` unchanged - still
the aug/EMA/ICNR checkpoint from `b309040`/Item 1-2, checksum
`36d2d38c...`. The Ni-WC real-mask edge-retention check (Gate condition
2) was not run for any weight - moot, since Gate condition 1 (PSNR)
already fails for all three, and the pre-registered "all three must
hold" rule means no amount of edge-retention improvement can rescue a
PSNR failure.

## Real, precise, publication-grade information this produces

The PSNR/edge-retention tradeoff identified in Item 3 is **not a
threshold effect that only appears at high loss weight** - it is present,
in roughly the same direction and shape, from the very first non-zero
weight tested. This is a stronger and more precise characterization of
the model's loss landscape than Item 3 alone provided: the boundary-
masked term's PSNR cost is a smooth, monotonic function of its weight,
starting near zero weight - not a sharp cliff that a small-enough weight
could avoid. Combined with Item 3's result, this closes the "maybe a
smaller weight would have worked" question definitively: it does not,
at least not within the tested range (0.05-0.20), and the mechanism is
structural to this loss family, not a tuning artifact.

## A real infrastructure catch during setup

The pod's on-disk `models/checkpoint.pt` was found stale (checksum
`19bf6df1...`, the pre-Item-1 Stage A checkpoint) before this run started -
never updated after the earlier Item 1/2 local swap. Caught and fixed
(re-transferred, re-verified `36d2d38c...`) before initializing this
fine-tune, which would otherwise have silently started from the wrong
base model.

## Timing

Pre-registration through final measurement: ~35 minutes (3 short
fine-tunes, ~3.5-4 min each on the A100, plus setup/eval), within the
45-minute soft time-box.

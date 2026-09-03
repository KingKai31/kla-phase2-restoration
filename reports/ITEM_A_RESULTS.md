# Item A result — checkpoint ensemble: gate FAILED, not adopted

Per `reports/itemA_ensemble_decision_rule_PREREGISTERED.md`.

| Weighting | PSNR (need >= 24.030) | Edge ratio (need >= 0.725) | Verdict |
|---|---|---|---|
| 70% shipped / 30% Item 3 | 23.961 | 0.761 | **FAIL** (PSNR) |
| 80% shipped / 20% Item 3 | 23.979 | 0.753 | **FAIL** (PSNR) |

**Both weightings score PSNR *below* the shipped model alone (24.004) -
not just below the required +0.026dB gate.** Edge retention improved
comfortably past its own gate in both cases, but per the pre-registered
"all three must hold" rule, PSNR failing alone is decisive.

## Why, mechanistically

Ensembling reduces error when the two models' errors are largely
uncorrelated noise - averaging cancels noise while preserving signal.
Here, Item 3's PSNR deficit relative to the shipped model (23.746 vs
24.004, -0.258dB, from `reports/ITEM_3_RESULTS.md`) is not noise - it is
a **consistent bias** from a real loss-function change (the model
learned to render sharper, more confident edges at the cost of
pixel-level fidelity everywhere). Blending two models with a shared bias
in the same direction doesn't cancel it - it just interpolates between
the two endpoints. The measured PSNR values fall almost exactly on the
line between the two source checkpoints (24.004 and 23.746), which is
the expected result for a bias, not the noise-cancellation an ensemble
is actually useful for.

## Decision: not adopted

`models/checkpoint.pt` unchanged. Inference-timing gate not conclusively
tested as designed (measured on CPU locally, ~82-103ms/image, not the
A100 the pre-registered ~12ms expectation was calibrated against) - moot,
since Gate 1 already fails on its own. Real evidence, real negative
result, same standard as every other gate in this project.

**Timing:** pre-registration through final measurement, ~12 minutes,
within the 20-minute time-box.

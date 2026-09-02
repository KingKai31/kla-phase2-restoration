# ROI-preservation loss (6th term) — pre-registered decision rule

**Written and committed BEFORE running the comparison below. Not adjusted
after seeing results.**

## Decision rule

Keep the ROI-preservation term (`src/losses/roi_preservation.py`) in the
final Stage B loss stack **only if both of the following hold**:

1. **Regression guardrail:** on a short controlled comparison (same data,
   same split, same hyperparameters, only the loss stack differs),
   val PSNR does not drop by more than **0.1 dB**, and val SSIM does not
   drop by more than **0.005**, relative to the same setup without the
   ROI term.
2. **Structure-preservation benefit:** on the defect-preservation stress
   test (`scripts/defect_preservation_stress_test.py` - synthetic
   perturbations implanted at known locations, degraded with the
   validated compound noise model, restored with and without the ROI
   term), the WITH-ROI model shows a **measurably higher perturbation
   survival rate** than the WITHOUT-ROI model, without a corresponding
   increase in false-perturbation hallucination.

**If either condition fails, the term is dropped or reported as a
negative/null result - not tuned post-hoc until it looks better.** If the
short comparison shows the ROI term amplifying noise instead of
protecting structure (a real, explicitly named risk in the term's own
docstring - high local variance can indicate noise OR real structure, and
the naive percentile-based mask can't fully distinguish them), that will
be reported as a finding, not hidden or explained away.

## What's measured now vs. later

This document is written before Task D's short PSNR/SSIM comparison AND
before Task 5's defect-preservation stress test. Condition 1 is checked
first (cheap, fast); condition 2 requires Task 5's stress test to exist.
Both must pass for the term to be kept - a good PSNR/SSIM result alone is
not sufficient justification, since PSNR/SSIM are not what this term was
added to improve.

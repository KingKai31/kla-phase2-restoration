# ROI-preservation loss — final decision (applying the pre-registered rule)

Per `reports/roi_loss_decision_rule_PREREGISTERED.md`, written and
committed before any comparison was run.

## Condition 1 — regression guardrail: PASSED

`without_roi` final: PSNR=23.325 / SSIM=0.5843. `with_roi` final:
PSNR=23.361 / SSIM=0.5860. Both deltas positive (+0.036dB, +0.0017),
comfortably inside the pre-registered thresholds. Full detail:
`reports/roi_loss_ablation_result.json`.

## Condition 2 — defect-preservation benefit: FAILED

`scripts/defect_preservation_stress_test.py`, n=100 val images, 3 real
perturbation archetypes (intensity anomaly, line discontinuity, particle),
paired Wilcoxon signed-rank test (not just "is the mean higher" - a
delta this small needs a real significance check):

| Perturbation | Δ survival (with − without) | Relative Δ | Wilcoxon p | Significant? |
|---|---|---|---|---|
| Intensity anomaly | +0.00042 | +0.18% | 0.062 | No |
| Line discontinuity | +0.00039 | +0.44% | 0.184 | No |
| Particle | +0.00023 | +0.11% | 0.102 | No |

**None of the three perturbation types show a statistically significant
survival improvement.** The positive-looking deltas from an earlier,
buggy version of this script (which additionally had a broken
hallucination check - comparing an array to itself, always exactly 0,
fixed before this final run) were not real signal.

**Hallucination check (fixed): statistically significant, wrong
direction.** Comparing two independently-noised restorations of the same
clean image (no perturbation, either time) at a random location:
`with_roi` shows a **higher** discrepancy (0.0538 vs 0.0534, p=0.039) -
small in absolute terms, but real, and in exactly the direction the
term's own docstring flagged as a risk (amplifying sensitivity to noise
rather than protecting real structure).

## Decision: DROP the ROI-preservation term

Per the pre-registered rule ("If either condition fails, the term is
dropped or reported as a negative/null result - not tuned post-hoc until
it looks better"): **condition 2 failed, so the term is not adopted for
the Stage B loss stack.** This is reported as a genuine negative result,
not hidden or reframed - `src/losses/roi_preservation.py` stays in the
repo as working, tested code with a clear real finding attached, not
deleted, in case a differently-tuned version (different boost factor,
different ROI-selection criterion) is worth revisiting later with a fresh
pre-registered comparison.

**Practical implication:** Stage B's loss stack remains the existing
5-term `StageBCompositeLoss` (Charbonnier + MS-SSIM + LPIPS + Sobel +
range-consistency) - unchanged by this investigation.

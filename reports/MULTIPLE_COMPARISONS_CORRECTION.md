# Multiple-comparisons correction (Task 6, final packaging)

Benjamini-Hochberg FDR correction (`statsmodels.stats.multitest.multipletests`,
`method="fdr_bh"`, alpha=0.05) applied within each decision's own test
family - not pooled across unrelated decisions, since each family was
pre-registered and used to inform one specific, separate call. Stage B's
own comparison isn't included here: it never ran a formal significance
test (the per-cluster deltas were reported descriptively as "noise-level,"
reports/STAGE_B_RESULTS.md), and Stage B isn't shipping regardless.

## Family 1: ROI-loss defect-preservation decision (4 tests)

The tests behind `reports/roi_loss_FINAL_DECISION.md`'s condition 2.

| Test | Raw p | BH-adjusted p | Significant (BH)? |
|---|---|---|---|
| Intensity anomaly | 0.0624 | 0.1248 | No |
| Line discontinuity | 0.1844 | 0.1844 | No |
| Particle | 0.1017 | 0.1356 | No |
| Hallucination check | 0.0388 | 0.1248 | **No (was borderline-significant uncorrected)** |

**One real change: the hallucination check's uncorrected p=0.039 does NOT
survive BH correction (adjusted p=0.125).** `reports/roi_loss_FINAL_DECISION.md`
described this as "statistically significant, wrong direction" - that
claim is corrected below. This does not change the actual decision: the
ROI term was dropped because it failed to show a significant
*defect-preservation benefit* (condition 2's real requirement, and none of
the three benefit tests were ever close to significant, corrected or not).
The corrected reading is, if anything, cleaner: no test in this family
shows a significant effect in either direction after correction - a
uniformly null result, not "no benefit but real, if small, harm."

## Family 2: confidence-signal correlation validation (4 tests)

The tests behind the confidence signal's kept-with-scoped-claim decision.

| Test | Raw p | BH-adjusted p | Significant (BH)? |
|---|---|---|---|
| PSNR, Pearson | 1.94e-75 | 3.87e-75 | Yes |
| PSNR, Spearman | 5.50e-78 | 2.20e-77 | Yes |
| SSIM, Pearson | 2.33e-07 | 3.11e-07 | Yes |
| SSIM, Spearman | 1.88e-06 | 1.88e-06 | Yes |

**No change.** All four remain significant after correction by many orders
of magnitude - the PSNR-vs-SSIM split (real signal for one, not the other)
that scopes the confidence signal's claim is unaffected.

## Updated files

- `reports/roi_loss_FINAL_DECISION.md`: hallucination-check line corrected
  to state the BH-adjusted result, not just the raw p-value.
- `reports/defect_preservation_summary.json`: BH-adjusted p-values added
  alongside the raw ones for each of the 4 tests.
- `src/losses/roi_preservation.py`: STATUS docstring updated to say
  "nominally significant before correction, not significant after BH
  correction" rather than just "significant."

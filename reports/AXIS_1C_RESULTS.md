# Axis 1c results — was Stage A undertrained? No.

**Question:** Stage A's original cosine LR schedule was sized for 150
epochs but early-stopped at 39 (best at epoch 13), so LR barely decayed
(reports/STAGE_A_RESULTS.md). Did that mean Stage A stopped before
genuinely converging?

**Test:** re-ran Stage A with everything identical (same real
leakage-checked data, split, loss, seed) except the schedule -
`ReduceLROnPlateau` (the already-validated Stage B fix) instead of
cosine, patience raised from 25 to 40, epoch budget raised from 150 to
250 to remove any remaining ceiling.

| | Original Stage A | Axis 1c (fixed schedule) |
|---|---|---|
| Best epoch | 13 | **13** |
| Best val PSNR | 23.483 | 23.509 |
| Best val SSIM | 0.5976 | 0.5977 |
| Epochs actually run | 39 | 54 |
| Epoch budget | 150 | 250 |

**Answer: no, it was not undertrained.** With 65% more patience, a
schedule proven to actually decay (validated in Stage B, which used the
same fix and did show multiple real LR reductions), and a much larger
epoch ceiling, the model still found its best result at the exact same
epoch (13) with a gain of **+0.026dB - pure noise, not a real
improvement.** The original cosine schedule's failure to decay meaningfully
turned out not to matter in practice: the model had genuinely reached its
real optimum for this architecture/data/loss combination well before
either schedule got a chance to matter.

**Conclusion for the hardening pass:** Stage A's 23.483dB/0.598 SSIM is a
real, converged result, not an artifact of an undertrained run. Any
further improvement has to come from somewhere other than training
longer or fixing the schedule - consistent with what Axis 1a (more/
different data) and Axis 1b (different hyperparameters) are actually
testing.

Full data: `reports/stage_a_v2_results.json`,
`reports/stage_a_v2_per_cluster_metrics.csv`,
`reports/stage_a_v2_training_history.csv`.

# Axis 1a results — more real external data: a doubly-confirmed null result

**Question:** the original Stage B fine-tune (3 NFFA categories, 8,526
synthetic pairs) showed no measurable improvement. Would a 4th category
(MEMS_devices_and_electrodes, the largest single category, 4,681 real
images) and a larger pool (12,204 pairs, +43% over Stage B) change that?

**Method:** fine-tuned from `stage_a_v2_best.pt` (23.509dB, the Axis 1c
checkpoint - marginally the current best real-only result) on real
4,073-pair train data + 12,204 synthetic pairs from 4 real NFFA
categories, same compound noise model applied directly to real external
clean images (the same mechanism Task C used, not a different or "fixed"
one - see the framing correction in `reports/HARDENING_AXIS_3_AND_5.md`).

| | Init (Stage A v2) | Axis 1a (4-category fine-tune) |
|---|---|---|
| Val PSNR | 23.509 | 23.476 |
| Val SSIM | 0.5977 | 0.5975 |
| Val LPIPS | not computed | 0.1829 |
| Best epoch | 13 | 15 (of 41 run) |

**Result: a small, real regression, not an improvement.** PSNR dropped
-0.033dB. Composite score (which also weighs LPIPS/SSIM) ticked up
marginally (+0.0007) - not a meaningful win, well within noise.

## Per-cluster: uniformly flat-to-negative, including the weak subgroup

17 of 20 clusters showed a negative PSNR delta (all small, mostly
<0.1dB). **Clusters 11 and 14 - the persistent weak subgroup flagged in
Part 8 of `reports/phase2_deep_dive.md` - both got slightly *worse*
(-0.058dB, -0.061dB), not better**, despite MEMS's dense fine-grained
texture being exactly the kind of content hypothesized to help them.

## This doubly-confirms Stage B's finding, not just repeats it

Two independent fine-tune attempts, different synthetic pool sizes (3
categories/8,526 pairs vs. 4 categories/12,204 pairs), different init
checkpoints (`stage_a_best.pt` vs. `stage_a_v2_best.pt`), same real
outcome: **no measurable benefit from this synthetic augmentation
approach, regardless of how much of it there is.** This strengthens
rather than merely repeats the Part 6 insurance-check explanation (the
~22% high-frequency spectral deficit) - if the problem were simply "not
enough synthetic data yet," a 43% larger, more diverse pool should have
shown at least a partial improvement. It didn't. The deficit is a
property of the noise model's generation mechanism itself, not a data-
volume problem, exactly as the spectral-fix investigation's mechanistic
analysis predicted (`reports/SPECTRAL_FIX_ATTEMPT.md`).

**Conclusion for the hardening pass:** Axis 1a is closed as a real,
doubly-confirmed negative result. `checkpoints/axis1a_best.pt` is not
adopted - `stage_a_v2_best.pt` (or the original `stage_a_best.pt`, which
are statistically indistinguishable per Axis 1c) remains the best
real-data-only candidate. Further attempts at external-data augmentation
via this generator are not worth pursuing without first fixing the
generator's actual spectral gap (the correlated-noise fix already tried
and failed - see `reports/SPECTRAL_FIX_ATTEMPT.md` for what direction
would actually be needed).

Full data: `reports/axis1a_results.json`,
`reports/axis1a_per_cluster_metrics.csv`,
`reports/axis1a_training_history.csv`.

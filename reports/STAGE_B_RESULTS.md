# Stage B results (Task 6) — real fine-tune, real negative result

**Run:** `scripts/train_stage_b.py`, fine-tuned from `checkpoints/stage_a_best.pt`
(val PSNR=23.483dB) on real 4,073 train pairs + Task C's 8,526 synthetic
pairs (12,599 total), same 5-term `StageBCompositeLoss` (ROI term
excluded per the pre-registered decision), `ReduceLROnPlateau` in place
of Stage A's fixed cosine schedule.

## Headline: essentially no improvement — a real null result

| | Stage A | Stage B |
|---|---|---|
| Val PSNR | 23.483 dB | 23.499 dB |
| Val SSIM | 0.5976 | 0.5950 |
| Gain | — | **+0.016 dB** (noise-level, not a real improvement) |

Best epoch 9 of 35 run (early-stopped). The LR schedule fix worked as
intended this time - `ReduceLROnPlateau` engaged and decayed the LR five
times (5e-5 down to 3.1e-6) as val PSNR plateaued, unlike Stage A's cosine
schedule which barely moved before early stopping cut it off. That
confirms the schedule itself wasn't the limiting factor here: **the model
converged and found no further exploitable signal in the added synthetic
data**, not that it needed more time to get there.

## Per-cluster: uniformly flat, not a localized effect

`reports/stage_a_per_cluster_metrics.csv` vs. `reports/stage_b_per_cluster_metrics.csv`,
all 20 clusters: **16 of 20 clusters changed by less than ±0.05dB.** The
two requested watch clusters specifically:

| Cluster | Stage A PSNR | Stage B PSNR | Delta |
|---|---|---|---|
| 11 | 19.669 | 19.665 | -0.004 dB |
| 14 | 19.203 | 19.195 | -0.008 dB |

**Clusters 11 and 14 did not improve - they stayed flat, within
measurement noise.** Per the explicit branch flagged in
`reports/phase2_deep_dive.md` Part 8 ("if they stay flat despite more
data, that would point toward [something other than a data-coverage
gap]"): the fact that the flatness is **uniform across all 20 clusters,
not specific to 11/14**, points away from "this architecture can't
recover high-frequency texture" and toward a more mundane explanation -
the synthetic data simply didn't carry a strong enough or well-matched
enough training signal to move the model past where Stage A already
landed it.

## A real, plausible reason why, not just a shrug

`reports/phase2_deep_dive.md` Part 6 (insurance check) already disclosed
that the synthetic generator's output has a real, measured gap: **~22%
high-frequency spectral deficit relative to real NoisyLR data at the
highest frequencies.** That is exactly the content that would matter most
for teaching a model to recover the fine, dense texture that clusters
11/14 (and difficult-cluster restoration in general) depend on. This is a
plausible, disclosed-in-advance explanation for a real, observed null
result - not being invoked after the fact to explain away a
disappointment; it was already on record as a known limitation before
this run.

## What this does and doesn't mean

- **Does not mean the synthetic pipeline is broken** - the insurance
  check (Part 6) showed strong bulk statistical agreement; it just isn't
  a strong enough training signal on the specific high-frequency
  reconstruction task to move Stage B's numbers here.
- **Does not mean Stage B should be abandoned** - `checkpoints/stage_b_best.pt`
  performs statistically the same as Stage A (23.499 vs 23.483dB), so
  there's no regression risk to using it, but also no demonstrated benefit
  from the added synthetic data as generated.
- **A real, disclosed negative result**, consistent with this project's
  standing rule to report honest findings rather than reframe a flat
  result as a win. Reported per the same standard as the ROI-loss decision
  (`reports/CASE_STUDY_rigor_in_practice.md`) - pre-registered signal to
  watch, measured, reported as it came out.

## Open, not chased further right now

If revisiting this is worth the time later: check whether upsampling the
synthetic pairs' weight in the batch composition, a longer fine-tune with
a higher initial LR, or specifically improving the generator's
high-frequency fidelity (the disclosed spectral gap) changes this result.
Not pursued now per the standing "don't rabbit-hole, report and move on"
discipline - flagging as a real open question for anyone extending this
work, not a chased-to-completion investigation.

## Files

- `checkpoints/stage_b_best.pt` (on the pod)
- `reports/stage_b_results.json`, `stage_b_per_cluster_metrics.csv`,
  `stage_b_val_per_file_metrics.csv`, `stage_b_training_history.csv`

## Background download status (checked opportunistically, not blocking)

A 4th Tier-1 category (MEMS_devices_and_electrodes, 4,681 images)
finished downloading before this Stage B run started but was **not**
folded in - the pod had only 8.7GB free and the remaining 6 categories
need roughly 12GB more to finish, so generating more synthetic pairs
would have competed with the download for the same tight budget. As of
this report: still only the same 4 categories present, 8.2GB free, 57%
disk used, download still running. **This is a real, disclosed risk: the
pod's 20GB disk is very likely to fill before all 6 remaining categories
finish**, independent of anything Stage B did - worth a decision (prioritize
specific categories, increase disk, or accept partial Tier-1 coverage)
rather than assuming it will resolve itself.

# Stage A results (Task 6) — real training run, real 4,785 pairs

**Run:** `scripts/train_stage_a.py` on the leakage-checked stratified split
(`reports/phase2_source_clusters_stratified_leakchecked.csv` - 4 cross-split
near-duplicate pairs found and fixed first, see
`reports/split_leakage_check.json`). NAFNetSR (6.82M params), base 5-term
`StageBCompositeLoss`, A100-SXM4-80GB.

## Headline numbers

| | |
|---|---|
| Train / val | 4,073 / 712 |
| Best epoch | 13 (of 39 run, early-stopped at patience=25) |
| **Best val PSNR** | **23.483 dB** |
| **Best val SSIM** | **0.5976** |
| Wall clock | 581.6s (9.7 min) total, 14.9s/epoch mean |

Matches the pre-registered epoch-time estimate (~13.2-15.5s/epoch,
confirmed on real DataLoader I/O + full loss stack) almost exactly - no
surprise there. Convergence is fast and clean (val PSNR: 22.34 -> 23.48
over the first ~13 epochs, then flat) - the cosine LR schedule was set for
a 150-epoch span but early stopping fired at epoch 39, so LR only decayed
~2% from its initial value. Disclosed, not fixed: a shorter or
step-decay schedule matched to the real ~15-epoch convergence point might
reach the same or slightly better result faster, but this hasn't been
tested and the current result is not blocked on it.

## Per-cluster breakdown (`reports/stage_a_per_cluster_metrics.csv`)

Full 20-cluster range: PSNR from 13.3 to 28.4 dB. Most clusters land in a
reasonably tight 22-26 dB band; three are worth flagging honestly rather
than averaging away:

- **Cluster 9 (n=2): PSNR=13.28 dB.** This is the same cluster already
  flagged in `reports/phase2_deep_dive.md` as too small to trust for noise
  fitting (excluded from the synthetic generator's parameter pool). With
  only 2 val images, this number is not statistically meaningful on its
  own - it is reported for completeness, not as evidence the model fails
  on a real subgroup. Needs more data before it means anything.
- **Clusters 11 (n=41, PSNR=19.67) and 14 (n=46, PSNR=19.20)** are
  reasonably sized and genuinely below the rest of the distribution (next
  lowest full-size cluster is ~22 dB). This is a real, if moderate,
  weak-subgroup signal - exactly the kind of thing Part 1's clustering was
  built to catch. Not yet root-caused; worth a visual check before Stage B
  if time allows, but not blocking Task C/6's continuation per the
  standing "don't rabbit-hole" guidance already applied to the cluster-18
  noise-model anomaly.
- **Best cluster (15, n=19): PSNR=28.42 dB** - healthy, no concern.

## Files

- `checkpoints/stage_a_best.pt` (on the pod)
- `reports/stage_a_results.json` - full summary
- `reports/stage_a_per_cluster_metrics.csv` - all 20 clusters
- `reports/stage_a_val_per_file_metrics.csv` - per-image PSNR/SSIM
- `reports/stage_a_training_history.csv` - full epoch-by-epoch curve

## Next: Stage B

Fine-tune from `checkpoints/stage_a_best.pt` adding Task C's 8,526
synthetic external pairs (train-only) plus whatever additional Tier 1
categories finish downloading in the background by the time Stage B
starts.

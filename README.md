# KLA Hackathon — Phase 2: SEM/NFFA-EUROPE Image Restoration

New repo per judges' preference for a fresh repo per phase (not extending
Phase 1's [kla-ps01-restoration](https://github.com/KingKai31/kla-ps01-restoration)).
Same task type (restore degraded images: multiplicative + additive noise +
spatial downsampling), completely new dataset: real SEM (scanning electron
microscope) images derived from **NFFA-EUROPE** (CC-BY 4.0), 10 documented
categories, 4,785 GT/NoisyLR pairs.

## Status

**Data understanding, noise-model characterization, and a validated
synthetic generator are complete. No restoration model has been trained
and no architecture decisions have been made yet** - see
[reports/phase2_data_inventory.md](reports/phase2_data_inventory.md) (raw
inventory) and [reports/phase2_deep_dive.md](reports/phase2_deep_dive.md)
(clustering proxy, scale-bar investigation, noise-model comparison,
parameter characterization, synthetic generator, insurance check) for the
full findings - real evidence, real figures, real numbers on the full
4,785-pair dataset, not assumptions carried over from Phase 1.

**Noise model finalized, synthetic generator built and validated**
(`src/datasets/synthetic_degrade.py`, `CompoundNoiseDegrader`). Insurance
check against 200 real held-out pairs: bulk statistics (mean, std) match
strongly (KS tests find no significant difference), visual samples across
6 diverse specimen types are convincing. Two real, disclosed gaps: a
heavier real-data tail in extreme max values than the Gaussian
approximation produces (KS p=0.001), and a ~22% high-frequency spectral
deficit at the highest spatial frequencies checked - both flagged
explicitly, of similar kind/magnitude to gaps Phase 1 itself judged
low-impact and proceeded past. Cluster 18's noise-fit anomaly was checked
(bounded, time-boxed investigation) and plausibly explained by its
unusually dark/low-brightness image content destabilizing its own
per-cluster fit - not fully certain, documented as a known limitation.

Headline findings:
- Structurally very similar to Phase 1's data convention (GT/NoisyLR
  folders, `.npy` format, 256↔128 only, GT strictly `[0,1]`, NoisyLR
  overshoots both directions, negative pixels present).
- **No category labels ship with this data delivery.** Decision: don't
  block on a corrected download - built an unsupervised-clustering proxy
  instead (20 clusters, real imbalance found: 12 to 526 images per
  cluster, 43.8x ratio). These are proxy groups, not verified NFFA
  category names. **Train/val split built STRATIFIED per cluster** (not
  Phase 1's whole-cluster assignment) so every cluster - even the n=12
  one - has guaranteed representation in both splits: 4,069 train / 716
  val, verified programmatically.
- **Scale-bar/info-panel overlays quantified: 10/4,785 (0.21%).** Excluded
  from all training/fitting going forward (permanent, re-derivable list:
  `reports/scale_bar_excluded_files.txt`). Checked whether they
  contaminate the noise-model fit: no - the apparent 2.5x higher noise in
  bar regions is fully explained by their higher average brightness once
  normalized the multiplicative way (ratio drops to 1.003x); masking
  wasn't structurally necessary for the fit, but the files are excluded
  anyway since they contain non-specimen content.
- **Noise model: the compound model is ADOPTED as the real Phase 2 noise
  model, replacing Phase 1's pure-Gamma.** Tested rigorously on all 4,775
  non-bar pairs: compound (quadratic multiplicative + linear Poisson-like
  term) fits dramatically better (R²=0.9997) than pure Gamma (R²=0.9926,
  Phase 1's model) or pure Poisson (R²=0.9719) alone - physically
  consistent with real SEM detector physics (Poisson electron counting
  plus multiplicative detector gain). **Fully characterized via 200
  bootstrap fits**: L_gain (multiplicative) mean 39.3, range 29.7-50.6;
  K_poisson (shot-noise) mean 104.4, range 76.9-214.2; σ_A (additive
  floor) small, often near-zero. Real per-cluster variation found beyond
  bootstrap noise (L_gain 29-80, K_poisson 46-364 across clusters) - the
  Phase 2 analogue of Phase 1's "randomize L across its full range"
  finding. A secondary S-shaped residual-bias finding was investigated,
  not left loose: not explained by a simple linear effect, but a cubic
  empirical correction captures 99.4% of it, documented as a known minor
  approximation (same pattern as Phase 1's own unexplained
  brightness-overprediction).

**Phase 1's fitted noise-model numbers (Gamma L range, additive σ, etc.) do
NOT apply to this dataset** and are not carried over anywhere in this repo.

## What's ported from Phase 1 vs. what's new

Ported (domain-independent infrastructure - evaluation, statistics,
compliance patterns, none of it depends on Phase 1's specific fitted
numbers or trained weights):

| File | What it is |
|---|---|
| `run.py.template` | Phase 1's `run.py` verbatim - the guardrail patterns (`sanitize_output()`, `classical_fallback()`, per-image exception handling, cwd-independent checkpoint resolution) are reusable; the model class/checkpoint loading inside it is Phase-1-specific and needs rewriting once a Phase 2 model exists. Renamed `.template` so it's never mistaken for a runnable script. |
| `tests/test_run_py_robustness.py.template` | Phase 1's 25-test robustness suite - the test *cases* (corrupt files, wrong shape, NaN/Inf, tiny images, etc.) are domain-independent; it imports Phase 1's `run.py` and needs repointing once Phase 2's `run.py` exists. |
| `scripts/generate_ppt_report.py.template` | The statistical-significance (paired Wilcoxon + bootstrap CI), composite-score-sensitivity, and classical-baseline-comparison *functions* are generic PSNR/SSIM/LPIPS math - the narrative text in each is written for Phase 1's specific findings and needs a full rewrite, not a numbers swap. |
| `scripts/compute_per_image_metrics.py`, `classical_baseline_eval.py`, `performance_profile.py`, `gt_noise_ceiling_check.py`, `ensemble_check.py`, `quick_test_visualize.py` | Reusable as-is in structure; each imports Phase 1's `src.models.nafnet`/`src.datasets.kla_dataset`, which don't exist yet here - won't run until Phase 2's model/dataset code is built. |
| `src/utils/reproducibility.py` | Fully domain-independent (`set_full_determinism`, `seed_worker`, `make_seeded_generator`) - usable as-is once training starts. |
| `.gitignore` | Same patterns (data/checkpoints/outputs excluded, figures NOT blanket-ignored per the documented Phase 1 incident). |

**Not ported, by explicit instruction:** Phase 1's fitted noise-model
numbers, trained checkpoints, or `src/models/nafnet.py` (architecture
decisions are out of scope for this data-understanding pass).

## Data

See [data/README.md](data/README.md) for the real local path and layout.
Not committed (gitignored, same pattern as Phase 1).

## Repo layout

```
reports/                    phase2_data_inventory.md (start here), figures, JSON summaries
scripts/phase2_data_inventory.py   the script behind every number in the inventory doc
scripts/*.template, run.py.template, tests/*.template   ported Phase 1 infrastructure, needs adaptation
src/utils/reproducibility.py       ported as-is, domain-independent
data/                        not committed - see data/README.md
```

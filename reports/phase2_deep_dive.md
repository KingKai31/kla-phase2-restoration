# Phase 2 deep dive — clustering proxy, scale-bar investigation, noise-model comparison

Follow-up to [reports/phase2_data_inventory.md](phase2_data_inventory.md),
covering the three explicitly requested investigations: (1) an
unsupervised-clustering category proxy, (2) full quantification of the
scale-bar/info-panel finding plus a noise-contamination check, (3) a
rigorous Gamma-multiplicative vs. Poisson-shot-noise comparison on the
full dataset. Every number below is produced by a committed, re-runnable
script — no manual/ad-hoc analysis.

---

## Part 1 — Unsupervised category-proxy clustering

**Script:** `scripts/cluster_categories_proxy.py`. Same technique as Phase
1's `scripts/cluster_sources.py` (KMeans on cheap per-image statistics),
enriched with frequency-domain and gradient-orientation features since SEM
category content is far more texturally/directionally distinct than Phase
1's data (fibres vs. particles vs. porous membranes differ strongly in
directionality and frequency content, not just intensity). Full detail on
the feature set and rationale is in the script's own docstring.

**These are unsupervised proxy groups, not verified NFFA category names -
stated with the same honesty standard as Phase 1's OOD-proxy caveat.**
Nothing here recovers Tips/Particles/Fibres/etc. as ground truth; it only
gives *some* grouping structure so validation can catch a hidden weak
subgroup, which was the actual stated goal.

**Result (`reports/phase2_source_clusters.csv`, `reports/phase2_source_split_summary.json`,
visual sanity check: `reports/figures/phase2_cluster_sample_grid.png`):**

- 20 clusters (deliberately more than the real category count of 10, same
  margin-above-true-count logic as Phase 1, to avoid forcing genuinely
  different subgroups into one bucket).
- **Real, significant imbalance**: cluster sizes range from **12 to 526**
  images - a **43.8x ratio** between the largest and smallest cluster.
  This directly confirms the concern flagged in the original Task 3
  ("if categories are imbalanced, that matters for training strategy") -
  it does matter here, concretely.
- Train/val split (whole-cluster assignment, GroupKFold-style, same
  pattern as Phase 1): 4,054 train / 731 val (15.3% actual, target 15%),
  val clusters = {4, 6, 19}.

**Visual sanity check, honest read:** some clusters are strongly visually
coherent (cluster 0: uniform granular texture; cluster 1: round
particle/nanosphere clusters), while others mix related-but-distinct
textures more loosely - expected for a statistics-based proxy on real,
complex image content, not a failure of the method. **Two clusters are
worth calling out specifically:**

- **Cluster 16 (n=23) spontaneously grouped 5 of the 10 confirmed
  scale-bar-contaminated images** (`003717, 003910, 003915, 003923,
  004326.npy` - exactly the 5 "top-of-crop, large-info-panel" style ones,
  see Part 2) **without being told to** - a genuine, unprompted validation
  that the clustering is picking up real structure, not noise. It did
  *not* catch the other 5 scale-bar images (the smaller, bottom-of-crop
  "2 μm" style bars occupying only ~11-23% of the frame vs. 56-86% for
  the ones it caught) - explainable: a small bar in an otherwise-normal
  specimen image doesn't dominate whole-image statistics enough to pull
  it into a distinct cluster, which is a reasonable, non-alarming
  limitation to state plainly rather than paper over.
- **Cluster 9 (n=12, the smallest cluster) looks visually like degenerate/
  pure-noise images** in the sample grid - flagged as worth a closer look
  in a future pass (not one of the three requested investigations here,
  so not deep-dived, but noted rather than silently ignored).

---

## Part 2 — Scale-bar prevalence and noise-model contamination check

**Script:** `scripts/scale_bar_detection.py`. Detector is grounded
directly in the confirmed real example (`GT/000007.npy`): a thin,
near-full-width, near-black separator line immediately adjacent to a
flat, near-white box containing dark text/graphics. Full detector logic
and iteration history (an initial bottom-half-only version undercounted -
see the script's own docstring) is documented in the script.

### Prevalence (full dataset, all 4,785 images)

**10 / 4,785 images (0.21%) confirmed to contain a burned-in scale-bar or
instrument-info-panel overlay**, after visually reviewing every candidate
from both a strict and a relaxed detector pass and excluding 1 confirmed
false positive (`000914.npy` - real dense powder/particle texture with no
actual overlay, visually confirmed, see
`reports/figures/scale_bar_relaxed_new_candidates.png`).

**The bar can appear at either the top or the bottom of a 256×256 crop**
(9 bottom, 1 top in the final confirmed set) - physically consistent with
these being patches cropped from larger source frames at varying offsets,
so a bar anchored to one edge of the original frame can land at either
edge of a crop depending on where it was taken. An initial detector that
only checked the bottom half found just 5/4,785 (0.10%) - **the corrected,
both-directions detector roughly doubled that** after real additional
true positives were found by visual review (`reports/figures/scale_bar_relaxed_new_candidates.png`
shows 5 genuine new finds: tick-mark ruler style, an accelerating-
voltage/magnification readout, and even an institutional **"TASC" logo**
- consistent with NFFA-EUROPE's real source facility, CNR-IOM Trieste).

**Bottom line: real, but low-prevalence (~0.2%).** Full flagged file list:
`reports/scale_bar_detection.csv`, `reports/scale_bar_excluded_files.txt`.

### Noise-model contamination check

Compared residual (`NoisyLR - GT_down`) variance **inside** vs.
**outside** the detected bar region, within the same 10 images
(`reports/scale_bar_contamination_check_normalized.csv`):

| | mean GT brightness | raw residual variance | Gamma-normalized (resid²/GT²) | Poisson-normalized (resid²/GT) |
|---|---|---|---|---|
| Bar region | 0.902 | 0.0358 | 0.0641 | 0.0407 |
| Specimen region | 0.551 | 0.0143 | 0.0639 | 0.0263 |
| **Ratio (bar/specimen)** | 1.64x | **2.50x** | **1.003x** | 1.55x |

**Raw residual variance in bar regions looked 2.5x higher than specimen
regions at first glance - but this was almost entirely explained by
brightness, not by the bar overlay being treated specially.** Bar regions
are brighter on average (0.90 vs. 0.55, since they're mostly white
background), and once that's normalized out the way a multiplicative-
noise model would (dividing by GT²), **the gap disappears almost
completely (ratio 1.003x - a 0.3% difference)**. Normalizing the Poisson
(linear) way instead still leaves a 55% gap - itself a second, independent
piece of evidence favoring the quadratic/multiplicative scaling used in
Part 3, found as an unplanned side effect of this check.

**Direct answer to the hypothesis in the task:** scale-bar regions do
**not** show anomalously different (e.g. near-zero) noise statistics once
brightness is properly accounted for - the same per-pixel noise model
appears to apply uniformly whether the underlying content is real
specimen or a synthetic overlay. **This means Task 5's noise-model fit
(Part 3) does not require masking bar regions to avoid contaminating the
brightness-vs-variance relationship** - excluding the 10 known files was
still done anyway (free, removes any doubt), but the reassuring finding is
it wasn't actually necessary for the noise-scaling conclusion.

**This does NOT mean the bar content is harmless for training a
restoration model**, a separate concern from the noise-fitting question:
the bar/text pixels are still a synthetic graphic, not real specimen
structure, and a model would be "restoring" a rendered artifact if trained
on a patch containing one. At 0.2% prevalence this is a small, easily
excludable population (the exact 10 filenames are known) rather than a
pipeline-level concern.

---

## Part 3 — Gamma-multiplicative vs. Poisson shot-noise, full dataset

**Script:** `scripts/noise_model_comparison.py`. Bins every pixel (from
all 4,775 pairs, the 10 known scale-bar images excluded) by GT brightness,
computes the empirical variance of `NoisyLR - GT_down` per bin, and fits
both candidate functional forms directly - not assumed from a first-pass
pooled check, which cannot distinguish linear from quadratic scaling.

**Full result** (`reports/noise_model_comparison_full.json`, figure:
`reports/figures/noise_model_comparison_full.png`):

| Model | Form | R² | AIC |
|---|---|---|---|
| Poisson shot-noise | `Var = c·x + d` (linear) | 0.9719 | -313.1 |
| Gamma-multiplicative (Phase 1's model) | `Var = a·x² + b` (quadratic) | 0.9926 | -346.3 |

**The quadratic (Gamma-multiplicative) model fits clearly better than the
linear (Poisson) model** - visually confirmed too: the Poisson curve
systematically sits above the empirical points across the low-to-mid
brightness range (0.1-0.5) while the quadratic curve tracks them closely
across nearly the entire range. The Poisson fit's intercept `d` also comes
out **negative** (-0.0045), which is physically implausible for a variance
term - a sign the linear form is straining to fit a curve it doesn't
structurally match.

**Going further than a binary "which one wins" - a compound model fits
dramatically better than either pure model alone:**

| Model | Form | R² | AIC |
|---|---|---|---|
| Compound (both terms) | `Var = a·x² + c·x + e` | **0.9997** | **-422.8** |

Both coefficients come out substantial and positive (`a=0.0236`,
`c=0.0123`) - the AIC improvement (-422.8 vs. -346.3) is far larger than
the 1-extra-parameter penalty would predict by chance, meaning this isn't
overfitting: **there is a real, non-trivial linear (Poisson-like)
component in the noise, in addition to a larger quadratic (multiplicative-
like) component.** Visual confirmation: `reports/figures/noise_model_compound_comparison.png` -
the compound curve visually overlays the empirical points almost exactly
across the full brightness range.

**Honest, physically-grounded verdict:** neither Phase 1's pure Gamma-
multiplicative model nor a naive pure-Poisson model is the complete
picture here. **A compound model - a dominant multiplicative/gain-like
quadratic term plus a genuine, smaller Poisson/shot-noise linear term -
fits far better than either alone.** This is physically sensible for real
SEM secondary-electron detectors: photomultiplier/microchannel-plate
detectors apply a random multiplicative *gain* to each Poisson-distributed
detected electron before digitization, producing exactly this kind of
compound (Poisson-times-Gamma-gain) noise signature - a well-documented
phenomenon in electron detection, not a novel claim invented to explain
this result away.

**Secondary finding, flagged not smoothed over:** the mean residual per
brightness bin is **not flat at zero** (right panel of
`noise_model_comparison_full.png`) - it dips to about -0.005 in the
low-mid brightness range (~0.15-0.3) and rises to about +0.007 in the
mid-high range (~0.75-0.85), a mild but clear systematic S-shaped bias.
Small relative to the noise's own scale (residual std is ~0.12-0.19
implied by the fitted variances), but a real deviation from the "additive
component has zero mean everywhere" assumption baked into both candidate
models - not yet explained, flagged as an open question for whoever
builds the final synthetic-degradation generator, not resolved here.

### Implication for the next phase

**Recommendation, not a unilateral architecture decision:** any synthetic
degradation model built for Phase 2 training augmentation should use the
compound form (`Var(x) = a·x² + c·x + e`, fit per-image or pooled the same
way Phase 1's per-source Gamma parameters varied) rather than assuming a
pure multiplicative model carries over unchanged from Phase 1. The
brightness-dependent mean-bias finding above should also be checked before
finalizing that generator, not assumed away.

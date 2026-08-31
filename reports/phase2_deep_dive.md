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
- **Split strategy, updated by explicit decision: stratified per cluster,
  not Phase 1's whole-cluster assignment.** `scripts/build_stratified_split.py`.
  Phase 1 assigned *entire* clusters to train or val (GroupKFold-style) to
  approximate true OOD generalization. Here, given the real 43.8x
  imbalance, whole-cluster assignment risks a rare cluster (as small as
  n=12) landing entirely on one side - zero validation coverage for that
  visual sub-population, which defeats the actual goal (catching a hidden
  weak subgroup at validation time, not simulating OOD generalization).
  Instead, every cluster is split ~85/15 proportionally, with a floor of
  at least 1 image on each side even for the smallest cluster. Result
  (`reports/phase2_source_clusters_stratified.csv`): **4,069 train / 716
  val (15.0% overall), every one of the 20 clusters represented in both
  splits** (individual cluster val-fractions range 13.0%-16.7%, tight
  around the 15% target even for n=12/n=23 clusters) - verified
  programmatically (the script asserts this and fails loudly if it isn't
  true, not just visually checked). Accepted tradeoff: near-duplicate-
  style patches from the same cluster can now appear on both sides of the
  split, which Phase 1's design specifically avoided - a deliberate choice
  for this different goal, not an oversight.

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

**Bottom line: real, but low-prevalence (~0.2%). Decision: exclude these
10 files from all training/fitting going forward** (already applied to
Part 3's noise-model fit and the parameter characterization below).
**Permanent, reproducible exclusion list:** `reports/scale_bar_excluded_files.txt`
(the exact 10 filenames, one per line - generated by, and re-derivable
from, `scripts/scale_bar_detection.py`, not a manual list). Full detection
detail: `reports/scale_bar_detection.csv`.

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

### Residual-bias investigation (follow-up, not left loose)

The secondary finding above - mean residual per brightness bin not flat at
zero, dipping to about -0.005 at low-mid brightness (~0.15-0.3) and rising
to about +0.007 at mid-high brightness (~0.75-0.85), an S-shape with both
tails curving back toward zero - was investigated rather than left as an
open question. **Script:** the fit is in `reports/residual_bias_investigation.json`,
figure: `reports/figures/residual_bias_cubic_correction.png`.

- **Magnitude, precisely bounded:** max |bias| = 0.0072, which is up to
  **~12% of the local noise std at low brightness** and ~4.5% at high
  brightness (`reports/residual_bias_investigation.json`) - real, not
  negligible, but small relative to the noise itself.
- **Ruled out the simple explanation:** a pure linear bias (i.e. `E[M] !=
  1` exactly, the simplest possible explanation) only explains 81% of the
  variance in the bias curve (R²=0.81) - **not sufficient**, so this isn't
  just a miscalibrated multiplicative mean.
- **A cubic empirical correction captures it almost perfectly: R²=0.994.**
  Residual-after-correction drops to **<0.00067 absolute everywhere**
  (<0.6% of the local noise std at any brightness) - effectively
  eliminated as a practical concern once corrected for.
- **Root physical mechanism not identified** - candidates considered
  (a subtle tone-curve/gamma mismatch between GT and NoisyLR generation, an
  interaction between the multiplicative and Poisson noise components) but
  not conclusively distinguished with the data available. **Documented as
  a known minor approximation with an empirical correction, exactly the
  same pattern as Phase 1's own unexplained ~15% high-brightness
  overprediction** (which Phase 1 also corrected empirically via a fitted
  `L_eff(x)` curve rather than deriving it from first principles) - a
  precedented, not novel, way to close this out.

---

## Part 4 — Full compound-model parameter characterization

**Script:** `scripts/compound_model_characterization.py`. A single 128×128
image doesn't carry enough brightness range/pixel count to stably fit a
3-parameter curve on its own (unlike Phase 1's per-image *ratio-only*
Gamma fit, which needed just 1 parameter per image) - so parameters are
characterized via **200 bootstrap fits, each on a random 300-image subset**
(with replacement) of the 4,775 non-bar-contaminated pairs, giving a real
distribution rather than a single point estimate. All three variance
components are physically non-negative, so fits are **bounds-constrained**
(`a, c, e >= 0`) - an earlier unconstrained version let `e` wander slightly
negative on most subsets, a fitting artifact with no physical meaning, not
a real "negative noise floor."

**Physical interpretation** (`NoisyLR = GT·M + sqrt(GT/K)·Z + A`): `M ~
Gamma(L_gain, 1/L_gain)` is the multiplicative detector-gain term, the
`sqrt(GT/K)·Z` term is the Gaussian-approximated Poisson shot-noise
component, `A ~ N(0, σ_A)` is a constant read-noise floor.

| Parameter | Mean | Std | Range (min-max) | p5-p95 |
|---|---|---|---|---|
| **L_gain** (multiplicative, 1/a) | 39.3 | 3.9 | 29.7 - 50.6 | 33.0 - 45.6 |
| **K_poisson** (shot-noise, 1/c) | 104.4 | 20.3 | 76.9 - 214.2 | 82.5 - 141.9 |
| **σ_A** (additive floor, √e) | 0.0019 | 0.0026 | ~0 - 0.0188 | 0.00035 - 0.0076 |

Full distributions: `reports/compound_model_bootstrap_fits.csv`, figure:
`reports/figures/compound_model_parameter_distributions.png`.
**σ_A is small and often near-zero** - the noise here is dominated by the
two brightness-dependent terms, with little evidence of a meaningful
brightness-independent read-noise floor (median σ_A = 0.00136).

**Per-cluster fits** (`reports/compound_model_per_cluster_fits.csv`) -
does noise vary by visual/texture sub-population, or is it dominated by
detector physics regardless of specimen content? **Real, substantial
variation found**, well beyond bootstrap sampling noise (which gave only
~10% relative spread on L_gain): across clusters with stable fits,
**L_gain ranges roughly 29-80 and K_poisson ranges roughly 46-364** - a
~2.7x and ~8x spread respectively. This is the direct Phase 2 analogue of
Phase 1's "L varies a lot across sources (3.8-50.9) - randomize across
this full range" finding, and should be treated the same way: **the
synthetic generator should randomize (L_gain, K_poisson, σ_A) across
their full measured ranges, not use one fixed triple.**

**Cluster 18 anomaly - bounded follow-up check, done:** cluster 18 (n=93,
1.94% of data) fit to `L_gain=239, K_poisson=31` - qualitatively different
from every other cluster. A short, focused check (5 sample images
visually inspected: `reports/figures/cluster18_samples.png`, plus pooled
pixel-level brightness stats) found a **plausible, evidence-backed partial
explanation, not a fully certain one**: this cluster's images are
consistently **dark, low-brightness specimen textures** (visually: fine
cellular-network and speckled patterns on a dark background) - pooled
pixel median brightness **0.217**, versus ~0.38 for two comparably-sized
normal clusters checked (8 and 0), and only **0.43% of its pixels exceed
brightness 0.8** versus 2.4-4.0% for the comparison clusters. With so few
high-brightness samples to constrain the quadratic term, the compound
model's `a` (quadratic) vs. `c` (linear) coefficients become poorly
separable for this cluster specifically - a known regression-stability
issue with polynomial fits over a compressed input range, not necessarily
evidence of a genuinely different physical noise mechanism.

**Documented as a known limitation, not fully resolved:** cluster 18
(n=93, ~1.9% of data) shows a distinct noise-regime *fit* not captured by
the population-level parameter ranges above - most likely (not certain)
driven by its unusually narrow, low-brightness pixel distribution making
its own compound-model fit unstable, rather than a confirmed different
physical detector regime. Synthetic data generation (Part 5, below) draws
from the full population-level distribution, which may underrepresent
this specific dark-image regime if it is in fact physically distinct -
flagged for anyone extending this work, not chased further per the
explicit time-box on this check.

---

## Part 5 — Synthetic data generator

**Module:** `src/datasets/synthetic_degrade.py`, class `CompoundNoiseDegrader`
(mirrors Phase 1's `SpeckleAdditiveDegrader` structure). Implements:

```
NoisyLR = box_downsample(GT, 2) * M + sqrt(box_downsample(GT,2)/K) * Z + A + bias(x)
  M ~ Gamma(L_gain, 1/L_gain), mean 1        (multiplicative detector gain)
  Z ~ N(0, 1)                                 (Poisson shot noise, Gaussian-approximated)
  A ~ N(0, sigma_A)                           (constant read-noise floor)
  bias(x)                                     cubic empirical correction (Part 3)
```

`(L_gain, K_poisson, sigma_A)` are drawn **per generated image from the 17
real per-cluster fitted triples** (`reports/compound_model_per_cluster_fits.csv`,
excluding cluster 18 and the two too-small clusters) - real measured
population diversity, the same "sample from real per-source fits, don't
use one fixed value" principle as Phase 1's L pool. No blur kernel is
modeled (same as Phase 1, for the same reason - this pass focused on the
noise model family, not spatial blur; revisit if the FFT check below shows
a spectral gap wide enough to matter).

---

## Part 6 — Insurance check: does synthetic data actually look real?

**Script:** `scripts/insurance_check.py`, same methodology as Phase 1's
insurance check (statistical, spectral, visual), run against **200 real
held-out (val-split) pairs** the generator was calibrated on in aggregate
but not fit to individually. Full numbers: `reports/phase2_insurance_check_summary.json`.

**Bulk statistics match strongly:**

| | real mean | synth mean | KS stat | KS p-value |
|---|---|---|---|---|
| per-image mean | 0.441 | 0.441 | 0.020 | 0.9999 (no difference) |
| per-image std | 0.185 | 0.183 | 0.045 | 0.988 (no difference) |
| per-image min | -0.004 | -0.003 | 0.135 | 0.052 (borderline) |
| per-image max | 1.430 | 1.366 | 0.195 | **0.001 (real difference)** |

**Visual grid** (`reports/figures/phase2_insurance_check_visual_grid.png`,
6 diverse samples - metal edge, granular texture, sharp geometric edge,
cellular network, wire/fibre structures, flat granular surface): synthetic
NoisyLR is visually convincing against real NoisyLR across all 6, no
obvious distinguishing artifact at a glance.

**Two real, disclosed gaps, not hidden:**

1. **Extreme-value (max) mismatch, confirmed by KS test.** Real data has
   a somewhat heavier right tail than this model produces - likely because
   the Gaussian approximations used for the shot-noise and read-noise
   terms are lighter-tailed than whatever the true noise process is.
   Consistent with Phase 1's own earlier finding that real additive
   residuals have "heavier-than-Gaussian tails" (excess kurtosis ~2.7) -
   the same class of approximation, now re-observed on different data.
2. **High-frequency spectral deficit**
   (`reports/figures/phase2_insurance_check_radial_spectrum.png`):
   low-to-mid frequencies match almost perfectly (<2% mismatch), but
   synthetic power falls increasingly short of real power at high spatial
   frequencies, reaching **~22% less power at the highest frequencies
   checked**. Plausible cause: this generator applies i.i.d. per-pixel
   noise, and real noise may carry some spatial correlation this doesn't
   capture (structurally the same class of gap Phase 1 found in its own
   insurance check, ~10-22% in a mid-frequency band, attributed there to
   noise being injected before downsampling).

**Verdict: convincing enough to proceed, gaps disclosed rather than
hidden.** The statistics that matter most for training realism (mean,
std, overall visual appearance) match strongly; the two gaps found are
specific, bounded, and of a similar kind and magnitude to gaps Phase 1
itself judged low-impact and proceeded past. Not claiming a perfect match
- if training results ever show artifacts traceable to extreme highlights
or fine high-frequency texture specifically, these two flagged gaps are
the first place to look.

---

## Implication for the next phase

**Recommendation, not a unilateral architecture decision:** the compound
model (`Var(x) = a·x² + c·x + e`) is adopted as the Phase 2 noise model,
replacing Phase 1's pure-multiplicative form - a stronger, more physically
grounded fit (R²=0.9997 vs. 0.9926) that matches real SEM detector physics
(Poisson electron-counting noise plus multiplicative detector gain) rather
than an ad-hoc curve fit. The synthetic generator (Part 5) samples
`(L_gain, K_poisson, σ_A)` from their full measured per-cluster ranges (not
a single fixed triple) and applies the cubic brightness-dependent
mean-bias correction (Part 3), and has been validated against real
held-out data (Part 6) with two specific, disclosed gaps rather than an
unverified assumption that it works.

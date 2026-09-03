# Technical audit — exhaustive, diagnostic only

A Michelin-level critique of every ingredient: data, noise model,
architecture, loss, training, evaluation, inference, engineering, and the
Axis 5 finding. Every claim cites the actual file/line. Ratings are 1-10
against best available practice, not against "good enough." **Nothing was
fixed in this pass** - one cheap diagnostic ablation was run (dimension 9)
because it converts a hypothesis into a measurement; no shipped artifact
changed.

**Context the audit must not lose:** on KLA's real held-out test set the
shipped model scores 23.761dB / 0.609 SSIM / 0.194 LPIPS, +3.43dB over
classical (`reports/OFFICIAL_TEST_SET_RESULTS.md`). This is a working,
honest, decisively-better-than-baseline system. The gaps below are about
what separates it from the best possible version, not about whether it
works.

---

## Priority ranking — what would actually move the needle

| Rank | Dimension | Rating | Expected real-world lift | Cost |
|---|---|---|---|---|
| 1 | Training strategy (no augmentation, no EMA, no TTA) | 5/10 | **Highest** - the only untried levers with reliable, documented PSNR gains | Low |
| 2 | Loss function (Charbonnier-dominant, diluted Sobel) | 6/10 | High - the actual mechanism behind Axis 5 | Medium |
| 3 | Axis 5 specifically | (diagnosed) | High - now mechanistically located, not guessed | Medium |
| 4 | Engineering (zero src tests, 3x model duplication) | 5/10 | Medium - risk reduction, not score | Low |
| 5 | Evaluation (single seed everywhere) | 7/10 | Medium - several "wins" may be inside seed noise | Low |
| 6 | Noise model (i.i.d., in-sample R², Gaussian-Poisson) | 6/10 | Medium - only matters if synthetic data is ever used | High |
| 7 | Data (coarse clustering, coarse leakage hash) | 6/10 | Low-medium - official test confirms no inflation | Medium |
| 8 | Architecture (NAFNet-lite) | 7/10 | Low - Axis 4 measured the ceiling | High |
| 9 | Inference pipeline | 7/10 | ~Zero for score, real for deployment | Low |

---

## 1. DATA — 6/10

**What's done.** 4,785 real GT/NoisyLR pairs; 10 scale-bar images
excluded via a two-direction detector with a hardcoded false-positive
set (`scripts/scale_bar_detection.py`); 20 KMeans proxy clusters; a
per-cluster stratified 85/15 split with a programmatic coverage assertion
(`scripts/build_stratified_split.py:66-69`); two perceptual-hash leakage
checks (internal 4 pairs fixed; external 0 matches; official test set 0
matches).

**Where it's suboptimal, with evidence.**

- **The split's own docstring admits the flaw** (`build_stratified_split.py:17-21`):
  near-duplicate patches from one cluster can land on both sides. The
  hash check that mitigates this is a 16x16 average hash
  (`scripts/check_split_leakage.py`) - it catches translated/re-exposed
  copies but is blind to flips, 90-degree rotations, and crops, all
  plausible in SEM acquisition. **Mitigating evidence:** the official
  test set scored *higher* than the internal val split (23.76 vs 23.48),
  which would not happen if val were meaningfully inflated by leakage.
  Rated 6 not 4 because of that external confirmation.
- **The clustering feature vector is dominated by a raw thumbnail.**
  `cluster_categories_proxy.py:78-94` builds 6 + 16 (histogram) + 8 (FFT)
  + 8 (orientation) + **64 (8x8 thumbnail)** = 102 dims, then
  `StandardScaler` (line 120) equalizes every dim. The 64 thumbnail dims
  therefore carry ~63% of the distance metric - clusters are largely
  coarse spatial-layout/brightness groupings, not texture groupings.
  This is the mechanistic reason the real-label alignment was only
  "moderate" (top-3 hit 51-64%, `reports/cluster_alignment_summary.json`).
  A texture-descriptor-weighted or PCA-whitened feature set would give
  clusters that mean more.
- **Two conflicting split columns exist.** `cluster_categories_proxy.py:131-143`
  still writes a whole-cluster `split` column into
  `phase2_source_clusters.csv`; `build_stratified_split.py` writes a
  different `split` into a second CSV. A future reader loading the wrong
  one gets a different experiment. Real confusion risk.
- **Per-cluster reporting rests on n as small as 2** (cluster 9). Every
  per-cluster table in the repo reports means with no CI; cluster 9's
  13.3dB is quoted repeatedly despite being statistically meaningless.
- **Synthetic-degradation representativeness is the weakest link: 3/10
  on its own.** i.i.d. noise produces a white spectrum; real noise is
  ~22% richer at high frequency (`reports/phase2_insurance_check_summary.json`).
  Confirmed non-useful for training twice (Stage B, Axis 1a). Any future
  external-data plan is dead until this is fixed (see dimension 2).
- **Data volume:** 4,785 pairs is small for 6.8M params, but Axis 1c
  (same best epoch with 3x the budget) and the official test both say
  the model is not overfitting. Adequate for this backbone.

**Unaddressed contamination risk:** the scale-bar detector was validated
on this delivery only; KLA's test set was not re-scanned for overlays.
Low risk (0.21% base rate), but unchecked.

---

## 2. NOISE MODEL — 6/10

**What's done.** `Var = a·x² + c·x + e` (Gamma-multiplicative + Gaussian-
approximated Poisson + read floor), fit to brightness-binned residual
variance; R²=0.9997 vs 0.9926/0.9719 for the pure forms; 200-fit
bootstrap; per-cluster fits; a cubic empirical bias correction.

**Is it the best physical model? No - it is a good variance-curve model.**

- **The fit is in-sample on 25 aggregate bins.** `noise_model_comparison.py:84-117`
  bins all pixels into 25 brightness bins, fits 3 parameters to those 25
  points, and reports R² on the same 25 points. R²=0.9997 with 3
  parameters on 25 nearly-smooth points is not the strong evidence it
  reads as. There is no held-out validation of the variance curve
  anywhere in the repo. The compound form is almost certainly right
  (physics agrees), but the *number* overstates certainty.
- **"Variance" is E[resid²] around zero, not around the bin mean.**
  Acknowledged at `noise_model_comparison.py:108-109`. With a measured
  bias up to 0.0072, the bias² term (up to 5e-5) is folded into a
  variance that is ~1e-3 at mid-brightness - a ~5% contamination of the
  fitted curve at the bias peaks. Small, real, uncorrected.
- **Poisson is Gaussian-approximated** (`synthetic_degrade.py:97-98`).
  This fails exactly where cluster 18 lives - dark pixels with low
  effective counts (K_poisson=31, GT~0.2 -> ~6 effective counts), where
  a Gaussian is visibly wrong (negative-going tails, no skew). The
  cluster-18 "anomaly" may partly be this approximation, not physics.
- **The model is spatially i.i.d.** No pixel-to-pixel correlation term.
  Real SEM noise has raster-scan structure (line-correlated), and the
  detector gain has a finite bandwidth. This is the root of the spectral
  tilt; the correlated-noise fix tried in `reports/SPECTRAL_FIX_ATTEMPT.md`
  blurred noise (wrong direction). The *right* fix is a
  frequency-domain-shaped noise (fit the residual power spectrum directly
  and sample noise with that spectrum), or a learned degradation model
  (e.g., a small GAN/flow on the residual). Neither was attempted.
- **No dark current, no scan-line artifact, no gamma/tone-curve term.**
  The cubic bias correction (`reports/residual_bias_investigation.json`)
  is almost certainly a tone-curve mismatch between how GT and NoisyLR
  were generated - fitting a gamma/LUT would give a *mechanism* instead
  of a polynomial.
- **Bootstrap rigor: fine. Per-cluster rigor: absent.**
  `compound_model_characterization.py:143-151` fits each cluster once
  with no CI; cluster 18's L_gain=239 is published as a point estimate
  with no uncertainty, then excluded from the generator pool on that
  basis. Bootstrapping *within* each cluster would have cost minutes.
- **Bounds fix was correct** (`compound_model_characterization.py:72-79`)
  and honestly documented.

---

## 3. ARCHITECTURE — 7/10

**What's done.** A NAFNet-lite (`src/models/nafnet.py`): width 32, encoder
blocks (1,1,1,2), 2 middle, decoder (1,1,1,1), SimpleGate, simplified
channel attention, PixelShuffle 2x head, global bilinear residual, 6.82M
params.

**Is NAFNet the best choice? For this task, yes-adjacent.** Grayscale,
128->256, 4.8k images, PSNR/SSIM-scored: a convolutional restorer is the
correct family. Diffusion is the wrong tool (samples, not conditional
means - loses PSNR by construction). Restormer/SwinIR/HAT are stronger on
public SR benchmarks by 0.2-0.5dB but at 5-30x the params on 4.8k images
the gain would likely not survive; Axis 4 (`reports/AXIS_4_RESULTS.md`)
measured the value of adding global mixing at +0.03dB - a direct
measurement that long-range context is not the bottleneck here.

**Specific implementation critique.**

- **This is shallow.** Original NAFNet uses (2,2,4,8)/12/(2,2,2,2); this
  uses (1,1,1,2)/2/(1,1,1,1) - roughly 1/4 the depth. Not measured
  whether depth helps. A width-48 or (2,2,2,4) variant is the untried
  architecture question, more promising than attention.
- **The global residual base is bilinear** (`nafnet.py:155-156`).
  Bilinear is a low-pass filter; the residual branch must synthesize
  *all* high-frequency content on top of a blurred base. A bicubic or
  learned-upsample base gives the network a sharper starting point.
  Directly relevant to edge preservation (dimension 9).
- **PixelShuffle after a 3x3 conv** (`nafnet.py:121-124`) is the classic
  checkerboard source; the fix chosen is a post-hoc 15% box-blur at
  inference (`run.py:203-206`), which is a global high-frequency
  attenuator applied to every output. ICNR initialization or a
  Conv->PixelShuffle->Conv head removes the artifact at the source with
  no output blur. Measured cost of the blur: dimension 9.
- **Fixed 2x upscale** (`nafnet.py:12-18`): correct for this task; not a
  real constraint. Removing it buys nothing here.
- **LayerNorm2d / SimpleGate / SCA:** standard, correct, well-implemented.
- **Reflect padding to a multiple of 16** (`nafnet.py:128-133`): the <=8px
  failure is documented and safely handled. Fine.

---

## 4. LOSS FUNCTION — 6/10

`src/losses/stageB_composite.py:71-115`: Charbonnier 1.0 + MS-SSIM 0.2 +
LPIPS 0.075 + Sobel 0.1 + range 0.05. Weights were never ablated; only
Sobel 0.1->0.2 was tested (Axis 1b: no effect).

**Per-term audit.**

- **Charbonnier (w=1.0) - carrying the model, and causing the Axis 5
  problem.** An L1-type pixel loss under multiplicative+Poisson noise
  drives the network toward the conditional *median*, which at a
  boundary between two textures is a smoothed transition, not a sharp
  one. This is textbook regression-to-the-mean; it is *why* pixel-loss
  SR models over-smooth and why NLM (a non-parametric patch average that
  never regresses toward a global mean) keeps edges. Pulling its weight:
  yes. Optimal: no - it dominates every other term by construction
  (typical magnitudes: Charbonnier ~0.03, Sobel ~0.02x0.1, LPIPS
  ~0.2x0.075).
- **MS-SSIM (w=0.2).** Correct use (clamped inputs, line 91-95), 5 default
  scales on 256x256 -> smallest 16x16, fine. Pulling weight: moderately.
- **LPIPS (w=0.075) - domain-mismatched.** AlexNet ImageNet features on
  grayscale-replicated SEM (`stageB_composite.py:97-99`). It rewards
  "natural-image-like" statistics, not SEM-like ones. It measurably
  helps LPIPS-the-metric; whether it helps *inspection* is unknown. The
  frozen-parameter handling (line 79-80) is correct.
- **Sobel (w=0.1) - the term that should fix Axis 5, and structurally
  cannot.** `stageB_composite.py:46-54` computes Charbonnier on gradient
  *magnitude* (orientation discarded) *averaged over every pixel*. Real
  structural boundaries are <5% of pixels; the loss is dominated by
  flat-region gradient noise. Doubling the weight (Axis 1b) changed
  nothing because doubling a diluted signal is still diluted - this is
  the exact mechanistic explanation for that null result. Also computed
  on the *clamped* prediction (line 101), so out-of-range pixels
  contribute zero gradient. **A boundary-weighted edge loss (mask by
  target gradient percentile, or top-k) is the correct design.**
- **Range penalty (w=0.05).** Correct, on raw output (line 102), small.
  Fine.

**Better loss design given everything learned:** (a) an FFT/spectral
loss (L1 on log-magnitude spectra) - directly targets both the
high-frequency deficit and edge retention; (b) a gradient-masked
Charbonnier that up-weights the top-10% target-gradient pixels; (c) drop
or halve LPIPS. Zero of these were tried. This is the highest-leverage
*medium-cost* change in the project.

---

## 5. TRAINING STRATEGY — 5/10

**What's done.** Stage A from scratch: Adam 2e-4, batch 16, cosine (later
plateau) schedule, early stop on val PSNR, single seed 123, full RNG
determinism (`src/utils/reproducibility.py`). Stage B / Axis 1a external
augmentation: tested, null, dropped correctly.

**The lowest-rated dimension because the biggest wins are the cheapest
untried ones.**

- **Zero data augmentation.** `scripts/train_stage_a.py:38-50`
  `RealPairDataset.__getitem__` loads and returns - no flips, no
  rotations, nothing. SEM images have no canonical orientation; the
  dihedral group gives 8x effective data for free. On 4,785 images this
  is the single most reliable untried PSNR gain in the project
  (literature and Phase-1-adjacent experience: +0.1-0.3dB). It is
  genuinely surprising this was never enabled.
- **No EMA of weights.** Standard in every modern SR recipe (+0.05-0.15dB
  typical, free at inference).
- **No test-time augmentation in `run.py`.** Averaging the 8 dihedral
  passes costs 8x inference (~50ms/image, still trivial) and typically
  +0.1-0.3dB PSNR with reduced artifacts. Zero training cost. Not present.
- **No weight decay, no gradient clipping, no mixed precision.** First
  two are minor; fp16 would halve training time (matters at pod cost).
- **Staged approach:** correct given B/1a failed. Curriculum (easy->hard
  noise) and progressive resolution: low expected value at 128->256.
  Self-supervised pretraining (Noise2Noise on the NoisyLR pool alone) is
  plausible but 4.8k images is thin. Distillation: no teacher exists.
  **The real gap is not an exotic strategy - it is the absent basics.**
- Axis 1c proved convergence is real; Axis 1b proved LR/batch are near
  optimal. Those are settled.

---

## 6. EVALUATION METHODOLOGY — 7/10

**What's done.** Paired Wilcoxon + bootstrap CI on the official test set
and on model-vs-classical; Benjamini-Hochberg across the two decision
families with one claim corrected as a result; pre-registered rules for
every comparison; per-cluster reporting; leakage checks; an honest
external-domain test.

**This is above typical hackathon standard and near publication grade on
the tests it runs. The gaps are in what is not run.**

- **Single seed for every training comparison.** Axis 1b/4's "gains"
  (0.0024-0.0026 composite) were declared below a 0.01 gate that was
  chosen, not derived. The only seed-variance datapoint in the project
  is Axis 1c: two identical-config runs differed by 0.026dB. Nobody knows
  whether 0.01 composite is above or below run-to-run noise. Three seeds
  per config would have cost ~30 minutes and made every gate defensible.
- **No CI on any per-cluster number** (n from 2 to 78).
- **Composite score reference range (15-30dB, `evaluate_checkpoint_full.py`)
  is arbitrary** and the equal-weighting of SSIM/PSNR/(1-LPIPS) is a
  guess at KLA's unknown weighting. Sensitivity to that choice was
  analyzed in Phase 1 but not repeated here.
- **The official test set has no per-cluster breakdown** (clusters are
  defined only on training data) - the weak-subgroup question cannot be
  checked on the numbers that matter most.
- **The Ni-WC external test's effective n is 45, not 180**, correctly
  disclosed, but the Wilcoxon on 180 correlated tiles is not reported
  there anyway - good, but the per-tile CSV invites misuse.
- **LPIPS was absent from Stage A/B's original eval** and back-filled;
  the official test set is the first place all three metrics coexist on
  the shipped model.

---

## 7. INFERENCE PIPELINE (run.py) — 7/10

**What's done.** Self-contained, cwd-independent, socket-verified
offline, universal `sanitize_output` gate, per-image try/except with a
real classical fallback, 25 adversarial tests passing, 72ms/image cold /
6ms warm.

- **Correctness and robustness: 9/10.** Nothing found that would fail a
  grader.
- **Performance left on the table (irrelevant to score, real for
  deployment):** images processed one at a time (`run.py:264-296` loop),
  no batching, no `torch.autocast`/fp16, no `channels_last`. A batched
  fp16 path is ~3-5x faster on GPU. `test_predictions/` for 297 images
  took ~3 minutes on CPU; fine.
- **The checkerboard blur is a global quality tax** (`run.py:203-206`,
  blend 0.15, applied unconditionally). Measured cost on the shipped
  model: dimension 9 - small but real, and avoidable at the source.
- **No TTA** - the cheapest quality lever available in this file.
- **Silent CPU fallback** (`run.py:256`): if CUDA is absent the run is
  ~10x slower with no warning. Acceptable, worth a log line.
- **`torch.load(..., weights_only=False)`** (`run.py:195`) - fine for a
  self-bundled checkpoint, would be flagged in a security review.
- **Placeholder on load failure** writes a 256x256 constant for an input
  of unknown true size - spec-compliant, scores zero. Correct choice.

---

## 8. REPRODUCIBILITY & ENGINEERING — 5/10

**What's done.** Full determinism utilities; exact-pinned requirements
(two real gaps found and fixed this project: PyWavelets, pytorch_msssim);
exceptional documentation; incremental result persistence added after a
real crash; clean commit history with real messages.

**What a senior reviewer would flag.**

- **`src/` has zero tests.** `tests/test_run_py_robustness.py` imports
  *no* `src` module (verified by grep). The model (`nafnet.py`), all
  losses, the degrader, and the confidence signal - 732 lines - are
  untested except indirectly through run.py's *separate inlined copy*.
  The shipped model class and the trained model class are not the same
  object and nothing asserts they agree.
- **The model is defined three times.** `src/models/nafnet.py`,
  `run.py:60-191` (inline copy), and `src/models/nafnet_attention.py:46-112`
  (a full copy-paste rather than a subclass). A one-line divergence in
  any copy is silent. `CharbonnierLoss` is defined twice
  (`charbonnier_msssim.py:8`, `stageB_composite.py:27`).
- **Dead code:** `src/losses/charbonnier_msssim.py` - `CharbonnierMSSSIMLoss`
  is imported nowhere; Stage A actually trains on `StageBCompositeLoss`
  (`train_stage_a.py:34`), contradicting that file's own docstring
  ("Stage A loss"). The project's own cleanliness audit missed it.
- **`RealPairDataset` / `collate` / `evaluate` are copy-pasted across six
  scripts** (`train_stage_a.py`, `_v2`, `train_stage_b.py`, `axis1a`,
  `axis1b`, `axis4`). There is no `src/datasets/real_pairs.py`.
- **Hardcoded Windows absolute paths as defaults** in
  `niwc_external_validation.py:110`, `severity_extrapolation_test.py:59`,
  and the inline audit scripts - non-portable, will break on the pod or
  any other machine.
- **No CI, no linter config, no type checking, no `pyproject.toml`.**
- **Documentation: 9/10** - genuinely best-in-class, and the reason this
  audit could be done from the repo alone. One stale spot:
  `phase2_deep_dive.md` Part 1 still describes the whole-cluster split
  logic that `build_stratified_split.py` superseded.

---

## 9. THE AXIS 5 FINDING — mechanistic diagnosis, now measured

**The number:** model 68.5% vs classical 87.9% edge-magnitude retention
at real annotated boundaries (n=167 tiles, 45 independent crops,
`reports/audit_axis5_blur_ablation.json`).

**Hypothesis 1 - the inference blur is an asymmetric handicap. TESTED.**
`run.py:203-206` applies a 15% 3x3 box blur to every model output;
`run.py:209-218` applies none to the classical path. Ablation with
`blend=0.0`:

| | Edge ratio | Edge corr | PSNR | SSIM |
|---|---|---|---|---|
| Model, shipped (blend 0.15) | 0.685 | 0.634 | 17.75 | 0.340 |
| Model, no blur (blend 0.00) | **0.705** | 0.634 | 17.75 | 0.345 |
| Classical | 0.879 | 0.583 | 16.68 | 0.394 |

**Verdict: real but minor.** The blur accounts for 0.020 of the 0.194
gap - about 10%. It is a free +2 points (and +0.005 SSIM) with no PSNR
cost, and should be removed at the source (ICNR / conv-after-shuffle),
but it is *not* the explanation. Reported as measured, not as the answer
one might have hoped for.

**Hypothesis 2 - regression-to-the-mean of the pixel loss. The dominant
mechanism, by elimination and by code.** With Charbonnier at w=1.0
(`stageB_composite.py:94,104`) the network is an estimator of the
conditional median of the clean pixel given a noisy neighborhood. At a
texture boundary under multiplicative + Poisson noise, the median of the
posterior is a *blend* of the two sides - a soft edge. NLM
(`run.py:215`, `denoise_nl_means`) does not estimate a posterior; it
averages patches that already match, so a boundary pixel is averaged
only with other boundary pixels and stays sharp. This is why a naive
method wins a fidelity-at-edges metric while losing PSNR everywhere
else: PSNR rewards the median; edge retention punishes it. The
unchanged edge *correlation* (0.634 either way) supports this - the
model knows *where* the edges are as well as ever; it under-states their
*amplitude*, which is precisely what a median estimator does.

**Hypothesis 3 - the Sobel term is structurally unable to counteract
H2. Supported by Axis 1b.** `stageB_composite.py:46-54` averages
gradient-magnitude error over all 65,536 pixels; boundary pixels are a
few percent. Doubling w_sobel (Axis 1b) gave composite 0.6601 vs 0.6600.
A diluted term scaled by 2 is still diluted. A boundary-masked or top-k
edge loss is the direct fix; it was never built.

**Hypothesis 4 - the bilinear residual base** (`nafnet.py:155`) starts
every output blurred; the residual branch must recover all edge
amplitude from a low-pass prior. Contributory, unmeasured, cheap to test
(swap for bicubic).

**Hypothesis 5 - domain shift.** Ni-WC is a different specimen class;
some smoothing on unfamiliar texture is expected. Contributory. Cannot
be separated from H2 without semiconductor masks, which do not exist.

**Ruled out - receptive field / architecture.** The 8x8 bottleneck at
128 input sees the whole image; adding global attention (Axis 4) moved
nothing. This is not a "the network can't see the edge" problem.

**Best real diagnosis:** ~10% inference blur (measured), remainder
dominated by a pixel-loss median estimator that a diluted edge term
cannot correct, on top of a low-pass residual prior. The fix is a loss
redesign (masked edge term and/or spectral term) plus removing the blur
at the source - not a bigger model, not more data.

---

## What this audit does not claim

- That any of the above would change KLA's score by a known amount. The
  official test result is decisive as it stands; the ranked list is
  expected value, not a guarantee.
- That the Axis 5 metric (Sobel magnitude at mask boundaries) is the
  metric KLA uses. It is the closest proxy available, on a different
  domain, and it is the honest thing to lead with.

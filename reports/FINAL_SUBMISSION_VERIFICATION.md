# Final submission verification — Phase 2 (SEM/NFFA-EUROPE data)

**Verdict: submission-ready.** Every hard-gate requirement passes,
re-verified against the exact shipped checkpoint and the exact
`requirements.txt`/`submission_requirements.txt` files as they currently
exist. Full detail: `reports/run_py_compliance_checklist.md`.

---

## A. Hard-gate requirements

All 15 checks in `reports/run_py_compliance_checklist.md` pass, including
the combined fresh-venv + wrong-cwd + no-internet verification run
together in one process, and the 25-test adversarial pytest suite
(`tests/test_run_py_robustness.py`). One real bug was found and fixed
during this pass (PyWavelets missing on both the pod and the local dev
machine - the exact Phase 1 mistake recurring on new infrastructure,
caught by the ported regression test before shipping, not after).

---

## B. Model quality — the full honest story, all three stages

**Shipped model: `checkpoints/stage_a_best.pt`, trained on real data
only.** Val PSNR=23.483dB, val SSIM=0.5976 (712 real leakage-checked val
images, 20 clusters). This is not the only thing that was tried - the
full arc below is the actual submission narrative, not a cleaned-up
success story.

| Method | PSNR | SSIM | LPIPS | n |
|---|---|---|---|---|
| Classical baseline (bicubic + NLM, run.py's real fallback) | 20.273 | 0.5066 | 0.5128 | 712 |
| **Stage A (shipped)** | **23.483** | **0.5976** | not computed† | 712 |
| Stage B (real + 8,526 synthetic pairs) | 23.499 | 0.5950 | not computed† | 712 |

† LPIPS wasn't part of Stage A/B's own per-image eval loop (only PSNR/SSIM
were tracked during training); not recomputed for this table since Stage
B isn't shipping and the PSNR/SSIM comparison already answers the
question each step needed.

**The AI model gains +3.21dB PSNR over the classical baseline** - a real,
large, unambiguous margin (n=712).

### Stage A -> Stage B: a genuine, disclosed null result

Fine-tuning from Stage A on real + 8,526 synthetic pairs (Task C, 3
verified NFFA-EUROPE categories, compound noise model) gave **+0.016dB
PSNR - noise-level, not a real improvement**, and was **uniformly flat
across all 20 clusters** (16/20 changed by <0.05dB), including the two
specific weak clusters (11, 14) it was hoped might benefit. Full detail:
`reports/STAGE_B_RESULTS.md`. Plausibly explained by an already-disclosed
~22% high-frequency spectral deficit in the synthetic generator
(`reports/phase2_deep_dive.md` Part 6) - not shipped, since it showed no
demonstrated benefit over Stage A.

### One targeted fix attempt, correctly abandoned

A pre-registered, time-boxed attempt to close the spectral deficit by
adding spatial correlation to the synthetic noise **failed its own gate**:
it made the mismatch *worse* (17.6% deficit baseline -> 44-47% with
correlation), for a real, understood mechanistic reason (correlation is a
blur-like operation that moves spectral power the wrong direction for a
generator that's already low-frequency-heavy). No retraining was run past
this point, per the pre-registered rule. Full detail:
`reports/SPECTRAL_FIX_ATTEMPT.md`.

**This is the actual "failures and successes" story:** a real architecture
choice (the compound noise model) that measurably beat the alternatives,
a real augmentation hypothesis that was tested honestly and didn't pay
off, and a real, mechanistically-understood fix attempt that was
correctly killed by its own pre-registered gate rather than pushed
through. Also see `reports/CASE_STUDY_rigor_in_practice.md` for the
ROI-preservation loss's own parallel arc (built -> pre-registered ->
bug found and fixed -> real negative result -> dropped).

### Other findings that inform this model's real, scoped capabilities

- **Local-Lipschitz confidence signal**: kept, validated as a real signal
  for PSNR-type error (Pearson r=-0.614, p<1e-75, survives BH correction)
  but NOT for SSIM-type error (r=+0.192, weak/wrong-signed) - never state
  as a general confidence measure without this split.
  (`src/utils/confidence.py`, `reports/confidence_signal_validation_summary.json`)
- **Cluster-alignment against real NFFA labels**: a real but moderate
  signal (top-3 cluster hit rate 51-64% vs. a 31% uninformative baseline
  across 3 categories), useful for stratified validation coverage, not a
  category proxy. (`reports/phase2_deep_dive.md` Part 7)
- **Clusters 11/14 (weak subgroup, ~19-20dB vs. the 22-26dB band)**: a
  real, partially-explained cause - a population-level K_poisson-vs-PSNR
  correlation (r=0.688, p=0.0016) plus visibly denser high-frequency
  texture content, not a model generalization failure. Stayed flat after
  Stage B's synthetic augmentation - inconclusive on whether that's a
  data-coverage gap or something structural, since the flatness was
  uniform across all clusters, not localized. (`reports/phase2_deep_dive.md` Part 8)
- **GT noise-ceiling check**: 27.6dB implied ceiling, well above Stage A's
  23.483dB - not currently the binding constraint.
- **Multiple-comparisons correction** (`reports/MULTIPLE_COMPARISONS_CORRECTION.md`):
  applied to the two decision-relevant test families. Confidence-signal
  correlations unaffected. The ROI-loss hallucination check's raw p=0.039
  does not survive BH correction (adjusted p=0.125) - corrected
  everywhere it's cited; doesn't change the drop decision.

### Inference timing

**72.08 ms/image, full cold-start end-to-end** (50 synthetic images,
A100-SXM4-80GB) - directly comparable in kind to Phase 1's own H100
headline figure (76.4ms/image), close in magnitude despite different
hardware. Warm/local component breakdown: 6.04ms/image full run.py path,
13.09ms/image amortized total. H100 pricing checked (~$2.99/hr, RunPod
on-demand) but a dedicated pod wasn't rented for this - flagged as
available, not assumed necessary.

---

## C. Engineering rigor

- **Reproducibility**: full RNG determinism utilities carried over from
  Phase 1 (`src/utils/reproducibility.py`) - `set_full_determinism`,
  `seed_worker`, `make_seeded_generator` - used in both Stage A and B
  training.
- **Formal test suite**: `tests/test_run_py_robustness.py`, 25 tests,
  25 passed against the real shipped checkpoint.
- **Real bug found and fixed**: PyWavelets missing (Section A) - the
  identical Phase 1 mistake on new infrastructure, caught before shipping.
- **Data leakage checks, two kinds**: (1) source-image overlap between our
  4,785 real pairs and the external NFFA download - 0 candidate matches
  at a conservative perceptual-hash threshold (`reports/source_overlap_check.json`);
  (2) internal train/val leakage within our own real split - 4 real
  cross-split near-duplicate pairs found and fixed
  (`reports/split_leakage_check.json`), producing the leak-checked split
  used for every Stage A/B number in this document.
- **License compliance**: NFFA-EUROPE data is CC-BY-4.0 (verified against
  the B2SHARE record directly, not assumed from a HuggingFace mirror,
  which was found to genuinely differ in image count and stated license -
  not used). Attribution requirement noted for the README/submission
  materials.
- **Repo cleanliness**: see Section E and the separate cleanliness-audit
  pass before this document was finalized.

---

## D. Known limitations (disclosed, not hidden)

- **Only 4 of 10 real NFFA-EUROPE categories were downloaded and used**
  (Biological, Fibres, Films_Coated_Surface for Task C's synthetic data;
  MEMS_devices_and_electrodes finished too late to fold in before Stage B
  ran). The remaining 6 categories were left downloading at low priority
  in the background per explicit instruction not to block on them - final
  status depends on whether the pod's disk holds out (see the disk-risk
  flag in `reports/STAGE_B_RESULTS.md`).
- **Clusters 11 and 14 remain a real, ~19-20dB weak subgroup** relative to
  the ~22-26dB band the rest of the data sits in - partially explained
  (noise-parameter regime + texture density), not fully resolved, and
  synthetic augmentation didn't move it.
- **Cluster 18's own noise-model-fit anomaly** remains a documented,
  plausible-but-not-certain explanation (unusually dark, low-brightness
  content destabilizing its own curve fit), not definitively resolved.
- **Cluster-alignment against real labels is a moderate signal, not a
  category proxy** - do not use these clusters as if they were verified
  ground-truth category labels.
- **The confidence signal only works for PSNR-type error**, not SSIM-type
  - never state it as a general uncertainty measure.
- **Inputs <=8px per side** fall back to the classical restoration path (a
  real architectural constraint from NAFNetSR's reflect-padding, inherited
  unchanged from Phase 1, safely handled but not expected to matter at
  this dataset's real 128x128 resolution).
- **The synthetic generator has two disclosed statistical gaps** versus
  real data: a heavier real max-value tail (KS p=0.001) and the
  ~22% high-frequency spectral deficit discussed in Section B - neither
  fully closed.

---

## E. Dropped, not pending

- **Stage B (real + synthetic fine-tune): dropped by explicit decision,
  not outstanding.** A genuine null result on its own merits (Section B) -
  Stage A ships instead. Kept in the repo as documented history (checkpoint,
  scripts, and full writeup), not deleted.
- **The spatial-noise-correlation spectral fix: dropped by its own
  pre-registered gate**, not abandoned mid-investigation - tested exactly
  as planned, failed cleanly, stopped per the rule agreed in advance.
- **The ROI-preservation loss (6th loss term): dropped by its own
  pre-registered rule** - Stage B (and thus this submission) uses the
  plain validated 5-term `StageBCompositeLoss`.
- **Tier 2 external datasets (Zenodo: Ni-WC metal composite, steel/hydrogen-
  embrittlement, fiber-composite): not pursued.** B2SHARE's Tier 1 data was
  already progressing and sufficient; a real, separate TLS-connectivity
  issue to Zenodo was also found and not chased further, since it wasn't
  needed once Tier 1 proved usable.
- **Remaining 6/10 Tier 1 NFFA categories: still downloading in the
  background at low priority, not blocking this submission** - whatever
  finishes is a bonus for any future extension, not a requirement this
  submission depends on.

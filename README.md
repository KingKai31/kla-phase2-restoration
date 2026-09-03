# KLA Hackathon — Phase 2: SEM/NFFA-EUROPE Image Restoration

New repo per judges' preference for a fresh repo per phase (not extending
Phase 1's [kla-ps01-restoration](https://github.com/KingKai31/kla-ps01-restoration)).
Same task type (restore degraded images: multiplicative + additive noise +
spatial downsampling), completely new dataset: real SEM (scanning electron
microscope) images derived from **NFFA-EUROPE** (CC-BY 4.0), 4,785
GT/NoisyLR pairs, 20 unsupervised-clustering proxy groups (no category
labels ship with the delivery).

## Status: submission-ready

**Official test-set results (KLA's Phase 2 test set, real GT, n=297,
leakage-checked against training): PSNR=23.761dB, SSIM=0.609,
LPIPS=0.194 - +3.43dB over the classical baseline (paired Wilcoxon
p<1e-49, bootstrap 95% CI [+3.21, +3.66]).** Restored outputs are in
`test_predictions/`. Full writeup:
**[reports/OFFICIAL_TEST_SET_RESULTS.md](reports/OFFICIAL_TEST_SET_RESULTS.md)**
- these are the only numbers here measured against externally-released
ground truth; everything below is the internal proxy split.

**Shipped model: `checkpoints/stage_a_best.pt` (`models/checkpoint.pt` in
the submission), trained on the real 4,785 pairs only.** Internal val
PSNR=23.483dB, val SSIM=0.5976 (712-image leakage-checked val split). Full verification,
the complete honest three-stage story (Stage A ships; a synthetic-data
Stage B fine-tune was tried and gave a genuine null result; one targeted
fix attempt failed its own pre-registered gate), and every disclosed
limitation: **[reports/FINAL_SUBMISSION_VERIFICATION.md](reports/FINAL_SUBMISSION_VERIFICATION.md)**
- start there.

Run inference with:
```
python run.py <input_dir> <output_dir>
```
`run.py` is fully self-contained (model architecture inlined, no `src/`
dependency) - see `reports/run_py_compliance_checklist.md` for the full
verification chain (adversarial pytest suite, fresh-venv + no-internet +
wrong-cwd combined check, real timing).

## Key findings, in the order they were built

1. **Data understanding** (`reports/phase2_data_inventory.md`): raw
   inventory, pairing check, scale-bar/info-panel overlays quantified at
   0.21% and excluded.
2. **Deep dive** (`reports/phase2_deep_dive.md`, Parts 1-8): unsupervised
   clustering as an honestly-scoped validation-coverage proxy (not a
   category proxy - later validated against real NFFA labels as "real but
   moderate," Part 7); a compound noise model
   (`Var = a·x² + c·x + e`, Gamma-multiplicative + Poisson-shot + read-noise
   floor) adopted over pure-Gamma/pure-Poisson (R²=0.9997 vs 0.9926/0.9719),
   fully characterized via bootstrap, physically grounded in real SEM
   detector physics; a synthetic generator built and insurance-checked
   with two disclosed statistical gaps; a bounded investigation of two
   weak clusters (11, 14) with a real, partial explanation (Part 8).
3. **RunPod migration**: A100-SXM4-80GB, environment verified, real timing
   established before any training.
4. **External data (Tier 1)**: 4 of 10 real-labeled NFFA-EUROPE categories
   downloaded and verified (Biological, Fibres, Films_Coated_Surface,
   MEMS_devices_and_electrodes) via B2SHARE (CC-BY-4.0, verified directly -
   a HuggingFace mirror was checked and found to genuinely differ in count
   and license, not used). Zero source-overlap with our own training data
   (perceptual-hash checked). Remaining 6 categories were left downloading
   at low priority, not blocking any deliverable.
5. **A 6th loss term (ROI-preservation), built and honestly dropped**:
   pre-registered a keep/drop rule before any comparison, found and fixed
   a self-comparison bug in the evaluation script, got a real negative
   result, dropped it per the rule - documented as a rigor case study in
   `reports/CASE_STUDY_rigor_in_practice.md`.
6. **A confidence signal, kept with a precisely scoped claim**:
   Local-Lipschitz sensitivity probing validated as real for PSNR-type
   error (r=-0.614, survives Benjamini-Hochberg correction) but not for
   SSIM-type error - never cited without that split.
7. **Stage A** (`reports/STAGE_A_RESULTS.md`): real training run on the
   leakage-checked stratified split, per-cluster reporting, 23.483dB/0.598
   SSIM.
8. **Stage B, a genuine null result** (`reports/STAGE_B_RESULTS.md`):
   fine-tuning with 8,526 synthetic pairs (Task C) added no measurable
   improvement, uniformly flat across all 20 clusters.
9. **One targeted, pre-registered fix attempt, correctly abandoned**
   (`reports/SPECTRAL_FIX_ATTEMPT.md`): spatial noise correlation was
   hypothesized to close the synthetic generator's high-frequency spectral
   deficit; measured, made it worse for a real mechanistic reason, and was
   dropped per its own gate rather than pushed through.
10. **A post-ship technical hardening pass, 5 axes, all pre-registered**
    (`reports/TECHNICAL_HARDENING_PASS_SUMMARY.md`): re-tested the
    schedule fix, more/different real external data, a bottleneck
    self-attention block, a 4-config hyperparameter sweep, real external
    validation (Ni-WC metal-matrix composite, CC-BY-4.0), a beyond-
    training-range noise-severity stress test, and real-mask structural-
    edge preservation. Every comparison either confirmed Stage A was
    already well-chosen or surfaced a real, disclosed limitation that a
    cheap fix didn't resolve - **the shipped model is unchanged.**

**Phase 1's fitted noise-model numbers do NOT apply to this dataset** and
are not carried over anywhere in this repo - every number here is
re-derived from this phase's own data.

## Repo layout

```
run.py                       self-contained inference script (ships models/checkpoint.pt)
tests/                        adversarial pytest robustness suite (25 tests)
submission_requirements.txt   minimal, exact-pinned deps for run.py only
requirements.txt              full training-environment dependencies
reports/                      every finding, in the order above - FINAL_SUBMISSION_VERIFICATION.md is the entry point
scripts/                      every analysis/training script behind the reports
src/models/nafnet.py          NAFNetSR architecture (also inlined in run.py for self-containment)
src/losses/                   Stage A/B loss stack (roi_preservation.py is dropped, kept as documented history)
src/datasets/synthetic_degrade.py   CompoundNoiseDegrader, the validated synthetic generator
src/utils/                     reproducibility helpers, the confidence signal
data/                          not committed - see data/README.md for the real local layout
```

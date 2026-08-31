# KLA Hackathon — Phase 2: SEM/NFFA-EUROPE Image Restoration

New repo per judges' preference for a fresh repo per phase (not extending
Phase 1's [kla-ps01-restoration](https://github.com/KingKai31/kla-ps01-restoration)).
Same task type (restore degraded images: multiplicative + additive noise +
spatial downsampling), completely new dataset: real SEM (scanning electron
microscope) images derived from **NFFA-EUROPE** (CC-BY 4.0), 10 documented
categories, 4,785 GT/NoisyLR pairs.

## Status

**Data-understanding pass complete.** No model has been trained, no
architecture decisions have been made yet - see
[reports/phase2_data_inventory.md](reports/phase2_data_inventory.md) for
the full findings (real evidence, real figures, not assumptions carried
over from Phase 1).

Headline findings:
- Structurally very similar to Phase 1's data convention (GT/NoisyLR
  folders, `.npy` format, 256↔128 only, GT strictly `[0,1]`, NoisyLR
  overshoots both directions, negative pixels present).
- **No category labels ship with this data delivery** - the source
  dataset's 10-category taxonomy is documented, but nothing in the files
  links a specific image to a specific category. Open decision needed on
  how to proceed (see the inventory doc's Task 3 section).
- First-pass noise-model check: Phase 1's Gamma-multiplicative-plus-
  additive shape looks like a reasonable starting hypothesis here too, but
  distinguishing it from a physically-plausible Poisson-shot-noise
  alternative needs the brightness-dependent heteroscedasticity check
  Phase 1 did - not yet done, correctly deferred to the next phase.
- Real, unquantified finding: at least one image has a burned-in SEM
  scale-bar annotation in the pixel data - prevalence across the full
  dataset not yet checked.

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

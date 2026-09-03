# Items 1 & 2 results — the shipped model changed, and it's a real win

Per `reports/item1_item2_decision_rule_PREREGISTERED.md`, written and
committed before this training run or any comparison.

**One retrain folded three changes** (explicitly permitted): dihedral
augmentation (Item 1a), EMA of weights (Item 1b), ICNR init on every
pixel-shuffle conv (Item 2). `scripts/train_stage_a_aug_ema.py`. Same
split, loss, seed, LR, batch as the original Stage A.

## Gate 1 (adopt a new checkpoint) — PASS, decisive

| | Shipped baseline (Stage A) | New (`raw_best`, epoch 98) | Threshold | Pass? |
|---|---|---|---|---|
| PSNR | 23.4831 | **23.7978** (+0.3148) | ≥ 23.5831 | ✅ |
| SSIM | 0.59758 | **0.61323** (+0.01565) | ≥ 0.59258 | ✅ |
| LPIPS | 0.18613 | **0.16662** (−0.01951, better) | ≤ 0.19113 | ✅ |

**+0.315dB is ~12x the measured seed-variance floor** (0.026dB, Axis 1c -
two identical-config runs). This is not inside noise. `raw_best` beat
`ema_best` at the shipped inference setting (blend 0.15) and was adopted;
EMA was real but slightly behind (23.79 vs 23.80 at its own best epoch -
full comparison in `reports/item1_stage_a_aug_results.json`).

**Adopted.** `models/checkpoint.pt` now ships `checkpoints/stage_a_aug_raw_best.pt`
(sha256 `36d2d38c...`). This is the single best result in the entire
project - exactly the lever the audit ranked #1.

## Gate 2 (drop the inference-time blur) — FAIL, honest negative

| | blend 0.15 (shipped) | blend 0.00 (no blur) | Requirement | Pass? |
|---|---|---|---|---|
| PSNR | 23.798 | 23.733 (−0.065) | ≥ 23.788 (−0.010 slack) | ❌ |
| SSIM | 0.6132 | 0.6147 | ≥ 0.6132 | ✅ |
| Checkerboard energy ratio | 0.204 | 0.226 | ≤ 1.10 | ✅ |

**The blur stays in `run.py`, unchanged** (already the 0.15 default - no
code change needed either way since this gate failed).

**A real partial win inside the negative result:** the checkerboard-energy
ratio is well under 1.0 at blend=0.00 (0.226 - the prediction has *less*
period-2 energy than GT itself, not more), meaning **ICNR did suppress
checkerboard at the source, as designed.** The blur's small PSNR/SSIM
benefit is therefore not masking a checkerboard artifact - it's a
separate, generic mild-denoise effect, and removing it costs more than
the pre-registered slack allows. Reported honestly rather than reframed:
Item 2's specific goal (make the blur removable) is not achieved, even
though its mechanism (ICNR) worked as intended.

## Full compliance chain, re-verified on the exact new artifact

`run.py` source is **unchanged** (Gate 2 failed, so no blur-related edit
was needed); only `models/checkpoint.pt` changed. Re-verified anyway,
same standard as every checkpoint swap in this project:

- 25-test adversarial suite (`tests/test_run_py_robustness.py`): **25/25 pass**, against the new checkpoint.
- Fresh venv (rebuilt from `submission_requirements.txt` on the pod's
  clean root filesystem) + wrong-cwd + no-internet, combined, one
  process: **PASS** - 0 network calls, cwd unrelated to the repo,
  checkpoint resolved via absolute script path, 5/5 outputs spec-compliant.
- Smoke test through `run.py`'s real `load_model`/`suppress_checkerboard`/
  `sanitize_output` path: shape/range/finite all correct.

## Official test-set confirmation (not just internal)

Re-ran the shipped `run.py` (new checkpoint) against KLA's real 297-image
test set. Paired against the original model on the exact same images:

| Metric | Mean diff (new − old) | Bootstrap 95% CI | Wilcoxon p |
|---|---|---|---|
| PSNR | +0.243 dB | [+0.219, +0.269] | 1.4e-49 |
| SSIM | +0.0167 | [+0.0152, +0.0183] | 1.6e-49 |
| LPIPS | −0.0327 (better) | [−0.0370, −0.0285] | 1.8e-38 |

**Confirmed on the real benchmark, not just the internal proxy** - same
direction, same order of magnitude, comparably decisive p-values. Full
detail: `reports/OFFICIAL_TEST_SET_RESULTS.md`.

## What Items 1/2 do not address

**Axis 5's diagnosed mechanism is untouched.** Augmentation, EMA, and
ICNR all target reconstruction quality broadly; none of them change the
Charbonnier-dominated loss or the pixel-averaged Sobel term identified as
the actual cause of the real-edge-preservation gap
(`reports/TECHNICAL_AUDIT.md` §9). That gap is expected to persist
unchanged by this pass - Item 3 (the loss redesign that follows from the
diagnosis) is the only planned intervention that targets it, and is
lower priority given the deadline.

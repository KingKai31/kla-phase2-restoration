# Spectral-fix attempt (Task 6, step 1) — gate failed, stopping per pre-registered rule

**Pre-registered rule (this session):** re-run the insurance check's radial
FFT comparison with a spatially-correlated noise variant. If the ~22%
high-frequency spectral deficit doesn't shrink to less than half the
baseline, stop - don't retrain, fall back to shipping Stage A as final.

## What was tried

`scripts/spectral_fix_experiment.py`: generates the additive noise
components (shot term + read-noise floor) on a coarser `k x k` grid, then
nearest-neighbor upsamples to full resolution before adding - correlates
noise within each block while leaving the per-pixel marginal variance
exactly unchanged (no rescale hack needed, since nearest-upsampling
doesn't average). Swept `k` in {1 (baseline, numerically identical to the
current production generator), 2, 3, 4} against the same 200 real
held-out val pairs and radial-power-spectrum method as
`scripts/insurance_check.py`.

## Result: the deficit got worse, not better

| Block size | High-freq deficit (target: <8.8%, half of baseline) | Low/mid-freq excess |
|---|---|---|
| 1 (baseline) | 17.6% | +2.8% |
| 2 | 47.4% | +16.7% |
| 3 | 44.5% | +23.0% |
| 4 | 46.5% | +18.5% |

**Every correlated-noise configuration made both the high-frequency
deficit and the low/mid-frequency excess substantially worse.** The best
of the three (k=3) still nearly *tripled* the deficit rather than halving
it. Full numbers: `reports/spectral_fix_experiment_summary.json`.

## Why — a real mechanism, not a fluke

The insurance check's actual mismatch is a spectral **tilt**, not a flat
high-frequency-only gap: synthetic already has ~2.8% *more* power than
real at low/mid frequencies (radius <30) and ~18% *less* at high
frequencies (radius >60). Spatial correlation is, by construction, a
blur-like operation - it always redistributes power from high frequencies
toward low frequencies for a fixed total variance. That is exactly the
wrong direction for a generator that already has too much low-frequency
power and too little high-frequency power. This was flagged in the
script's own docstring before running the experiment (reasoning first,
then measured anyway, since the whole point of pre-registering was to let
the measurement decide rather than substitute intuition for it) - the
result confirms the mechanism rather than surprising it.

**What this means for the actual generator gap:** closing it for real
would need the opposite kind of change - adding independent, finer-grained
high-frequency content (or increasing the noise magnitude specifically at
fine scales), not correlating/smoothing the existing noise. Not pursued
further in this pass - out of scope for the time-boxed attempt that was
agreed to, and the pre-registered gate says stop here regardless.

## Decision: gate failed, stop per pre-registered rule

Per the explicit rule: **"If the spectral deficit doesn't meaningfully
shrink... STOP HERE - don't retrain, fall back to option (a)
immediately."** No retraining was run. Falling back to shipping
`checkpoints/stage_a_best.pt` as the final Stage A/B model, per the
already-agreed fallback.

## Honest summary of the full arc (for the submission narrative)

1. Stage A trained on real data alone: 23.483dB / 0.598 SSIM, with two
   honestly-flagged weak clusters (11, 14).
2. Diagnosed the weak clusters: ruled out cluster 18's own brightness
   explanation (doesn't transfer), found a real population-level
   K_poisson-vs-PSNR correlation (r=0.688, p=0.0016) plus a visible
   fine-texture content difference - a real, if partial, explanation.
3. Stage B (real + 8,526 synthetic pairs): no measurable improvement
   (+0.016dB, noise-level), uniformly flat across all 20 clusters -
   a genuine negative result, plausibly explained by the generator's
   already-disclosed high-frequency spectral deficit.
4. One targeted, time-boxed fix attempt (spatial noise correlation):
   failed its own pre-registered gate - made the spectral mismatch worse,
   for a real, understood physical reason (correlation moves power the
   wrong direction for this specific tilt).
5. **Final model: `checkpoints/stage_a_best.pt`** (23.483dB / 0.598 SSIM),
   trained on real data only. The synthetic-augmentation path was
   explored honestly, diagnosed, and one fix attempted and correctly
   abandoned when the evidence didn't support it - not hidden, not
   retried past the agreed timebox.

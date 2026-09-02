# Axis 1b results — hyperparameter sweep: another real-but-too-small win

Per the pre-registered decision rule
(`reports/axis1b_sweep_decision_rule_PREREGISTERED.md`): 4 configs, real
leakage-checked data only, early-stopped (not full budget), ranked by
composite score.

| Config | LR | Batch | Sobel weight | PSNR | SSIM | LPIPS | Composite |
|---|---|---|---|---|---|---|---|
| `baseline` | 2e-4 | 16 | 0.1 | 23.496 | 0.5994 | 0.1859 | 0.6600 |
| **`higher_lr`** | 4e-4 | 16 | 0.1 | 23.503 | 0.6013 | 0.1810 | **0.6624** |
| `larger_batch` | 2e-4 | 32 | 0.1 | 23.411 | 0.5955 | 0.1878 | 0.6562 |
| `stronger_sobel` | 2e-4 | 16 | 0.2 | 23.459 | 0.6013 | 0.1849 | 0.6601 |

**Winner by composite: `higher_lr`** (4e-4), beating baseline by 0.0024 -
**below the pre-registered 0.01 adoption threshold.** Per the rule agreed
before running ("adopt only if... at least 0.01... otherwise baseline's
hyperparameters remain in use and the sweep is reported as
informational"):

**Decision: keep baseline hyperparameters. No config adopted.**

## What this sweep actually answers

- **`higher_lr` converged faster** (best at epoch 8 vs. baseline's 14)
  and reached a marginally better result - a real, if small, signal that
  the original LR wasn't badly chosen, just not quite optimal. Not
  enough to change the recipe on its own.
- **`larger_batch` is the clear loser** - worse on every metric. Real
  evidence that batch size 16 (vs. 32) was the right call, not an
  arbitrary choice.
- **`stronger_sobel` (doubled edge-loss weight, directly motivated by
  Axis 5's real finding that the model under-preserves true structural
  edges) shows essentially no difference from baseline** (0.6601 vs.
  0.6600 composite, SSIM actually ties with `higher_lr`'s SSIM). **This
  answers Axis 5c's proposed follow-up directly: simply doubling the
  existing Sobel term's weight is not a real fix for the edge-preservation
  gap found on the Ni-WC external data.** Whatever is causing that gap
  isn't solved by this cheap lever - a real, disclosed negative result
  for Axis 5c's specific suggestion, not chased further with additional
  weight values per the same-axis time-box discipline.

## Pattern across this entire hardening pass

Every GPU-bound comparison in this pass - the fixed LR schedule (Axis
1c), more/different real external data (Axis 1a), a bottleneck
self-attention block (Axis 4), and now 4 hyperparameter variations
(Axis 1b) - returned either a genuine null result or a real-but-too-small
win that correctly failed its own pre-registered gate. This is a coherent
signal, not four unrelated misses: **Stage A's original recipe
(architecture, hyperparameters, schedule) was already close to whatever
this data/loss/model combination can achieve without a more fundamental
change** (e.g., actually fixing the synthetic generator's spectral gap,
which Axis 1a's result shows is the real bottleneck for external-data
augmentation specifically).

Full data: `reports/axis1b_sweep_summary.json`, `reports/axis1b_sweep_results.csv`.

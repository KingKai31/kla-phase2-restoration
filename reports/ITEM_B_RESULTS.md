# Item B result — multi-seed variance: infeasible within the time-box, stopped honestly

Per the standing rule ("if either fails its gate or runs long, drop it
and move to C"). No training was run for this item - the numbers needed
to make that call were already on record.

## Inference-time seed variance: not applicable, not just impractical

Checked first, since it would have been nearly free if available: the
shipped architecture (`src/models/nafnet.py`) has **zero dropout, zero
stochastic layers anywhere** (grep confirms it). `run.py`'s forward pass
is already proven fully deterministic for a fixed checkpoint and input
(`tests/test_src_modules.py::test_deterministic_for_fixed_weights`).
Repeated forward passes on the same input would return bit-identical
output every time - a "variance" measurement here would report exactly
0.0, which is true but answers a different, uninteresting question. This
path is mathematically inapplicable, not merely infeasible.

## Training-time seed variance: real, but doesn't fit the box

The only real question left is: does re-running the shipped recipe
(augmentation + EMA + ICNR) from a different random seed land at a
meaningfully different PSNR? Answering that validly requires a
comparably-converged run, not a truncated one - a truncated run would
measure undertraining noise, not the shipped recipe's real variance, and
reporting it as "the seed-variance floor" would be a real overclaim this
project's standard doesn't allow.

**The actual cost of one such run is already on record:**
`reports/item1_stage_a_aug_results.json` - the run that produced the
shipped checkpoint - took **3,889.8 seconds (~65 minutes)** wall-clock on
an A100. One additional seed alone exceeds Item B's entire 15-minute
time-box by more than 4x, and exceeds this whole pass's 45-minute total
budget on its own. Two seeds, as requested, would take ~130 minutes.

## Decision: stop here, honestly, before spending any time-box on it

Per the standing time-box discipline, this is assessed and stopped
**before** running anything, not discovered mid-run - the cost was
already known from Item 1's own log, so no time was spent attempting a
partial version that couldn't answer the real question. This is not a
failed gate (no gate was run) - it is a real, correctly-scoped decision
not to spend the pass's time budget on an experiment that cannot fit it
without producing a number that shouldn't be trusted.

## What this leaves genuinely open (disclosed, not hidden)

The project's existing "0.026dB seed-variance floor" (used as a gate
threshold throughout Items 1-3 and Axis 1b/4) is more precisely a
**same-seed, different-schedule reproducibility floor** - Axis 1c
compared two runs at the *same* seed (123) with a different LR schedule,
not two different seeds with an identical recipe. It is a real, measured
number and a reasonable proxy (Axis 1c's own conclusion was that the
schedule change didn't matter, supporting treating this gap as noise),
but it is not literally what "seed-variance" means, and a true multi-seed
measurement of the final shipped recipe remains unmeasured. Flagged
honestly here rather than left to be assumed more rigorous than it is.

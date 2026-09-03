# Item 2 result — rigorous, repeated inference-time benchmark

Fixes a real methodological gap: every prior timing claim in this
project was a single measurement. This is N=20 independent, genuinely
**cold-start** runs - a fresh Python subprocess each time (script start
-> import -> load checkpoint -> inference on 50 images -> write), exactly
matching how a grading harness invokes `python run.py <in> <out>`, not a
warm in-process loop.

## Result (final shipped checkpoint, `models/checkpoint.pt`, A100-SXM4-80GB)

| | ms/image |
|---|---|
| **Mean** | **88.35** |
| Std | 2.13 |
| Median | 87.96 |
| Min | 84.96 |
| Max | 93.61 |

**A tight, credible distribution** - std is ~2.4% of the mean, and the
full min-max range spans only 8.65ms. This is a materially stronger
claim than a single-measurement number: it bounds the real run-to-run
variance a grading harness would actually see, not just one lucky (or
unlucky) sample.

## Context vs. earlier single measurements

Earlier reports cited 72.08ms/image (a single cold-start run,
`reports/run_py_compliance_checklist.md`) and 76.4ms/image (Phase 1's
H100 figure). This N=20 mean (88.35ms) sits in the same order of
magnitude but is measured on a different, later checkpoint (aug/EMA/ICNR
vs. plain Stage A) and via `subprocess.run` (full new-process overhead
each time, including Python interpreter startup) rather than however the
single prior measurement was invoked - the two numbers are not directly
comparable methodologically, and this repeated measurement is the one to
cite going forward since it properly bounds variance.

Full data (all 20 per-run values): `reports/item2_timing_benchmark.json`.

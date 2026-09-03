# Three new technique classes — results

Per `reports/new_techniques_decision_rules_PREREGISTERED.md`, written and
committed before any of the three ran. All three are different in kind
from every prior attempt in this project (which only ever changed
loss-term weights on the same architecture and training scheme).

Baseline throughout: the shipped checkpoint (`models/checkpoint.pt`,
sha256 `36d2d38c...`) - official-test PSNR 24.004, SSIM 0.6257, LPIPS
0.1616; Ni-WC real-mask edge retention 0.705 (blend=0.00).

---

## Item 3 — auxiliary confidence head: real signal, below the bar. NOT adopted.

Full detail: `reports/ITEM_3_CONFIDENCE_HEAD_RESULTS.md`.

Per-image mean Spearman r = **0.226** (gate required >= 0.3). Not
adopted as a demonstrable finding. **But not a null result:** all
**712/712** held-out validation images show a positive, individually
significant correlation between the predicted confidence map and the
model's actual per-pixel error. Real, consistent, just modest on a
typical image. Zero risk taken: main model frozen throughout, `run.py`
never references the head.

---

## Item 2 — severity curriculum: FAILS, and the schedule never completed. NOT adopted.

| | Shipped | Item 2 curriculum | Gate |
|---|---|---|---|
| Official-test PSNR | 24.004 | **23.976** (-0.028) | needed >= 24.054 (+0.05) |
| Official-test SSIM | 0.6257 | 0.6246 | no regression |
| Official-test LPIPS | 0.1616 | 0.1629 | no regression |

**Fails decisively** - the gate required a +0.05dB improvement; this
produced a small regression instead.

**A real methodological finding inside the failure, worth stating:** the
curriculum **never reached its own final phase.** Training on the
mildest third only (the prescribed first 30% of epochs) drove validation
PSNR steadily down - 23.798 at init to 23.446 by epoch 11 - because
narrowing the training distribution moves the model away from the full
distribution it is evaluated on. It recovered sharply the moment the
uniform phase began (23.446 -> 23.634 in one epoch, a clean, legible
signal that the mild-only phase was the cause), but by then early
stopping (patience counted from epoch 1's best) terminated the run at
epoch 17 - and the harsh-oversample phase does not begin until epoch 28.
**So the "oversample the hardest examples last" half of the hypothesis
was never actually tested**, and this configuration cannot be said to
have refuted it - only that this schedule, at this epoch budget and
patience, self-terminates before completing. Reported as a partial,
honestly-bounded negative result rather than a clean refutation.

---

## Item 1 — decoder-only capacity increase: **PASSES ALL THREE GATE CONDITIONS**

One additional lightweight NAFBlock in the decoder's final stage,
immediately before the pixel-shuffle head. Near-identity at
initialization by construction (NAFBlock's `beta`/`gamma` residual gates
initialize to exact zero) - verified locally as **numerically exact**
(max output difference vs. the shipped model at init: 0.0).

| Gate condition | Required | Measured | Result |
|---|---|---|---|
| Ni-WC real-mask edge retention | >= 0.725 | **0.735** (+0.030 over 0.705) | **PASS** |
| Official-test PSNR | >= 23.978 | **24.001** (-0.003, well inside the floor) | **PASS** |
| Param increase | < 15% | **+0.12%** | **PASS** |
| Inference-time increase | < 15% | **+8.26%** forward-pass (~+0.8% end-to-end) | **PASS** |

Supporting: official-test SSIM 0.6251 (vs 0.6257, negligible), LPIPS
**0.1605** (vs 0.1616, slightly better).

**This is the first intervention in this entire multi-pass effort to
clear its own pre-registered gate.** It improves the project's single
most important known limitation - real structural-edge preservation -
by +0.030, closing ~17% of the remaining gap to the classical
baseline's 0.879, at essentially zero cost in every other metric. Where
the loss-based attempts (Item 3 of the prior pass, and the graduated
weights) all bought edge retention *by paying PSNR*, this one buys it
for free, which is consistent with the hypothesis it was designed to
test: the decoder genuinely lacked the *capacity* to represent sharp
boundaries, and no amount of loss reweighting could substitute for that.

### ADOPTED — with explicit approval, after the full chain passed

`run.py`'s inlined model could not load this checkpoint (verified:
`RuntimeError: Unexpected key(s) in state_dict: "extra_decoder_block.*"`),
so adoption required editing the single most fail-critical file - more
than the pure checkpoint swap the pre-registration assumed. That was
surfaced as a decision and **explicitly approved** before proceeding.

**What changed, and how the risk was contained:**

- `run.py`'s existing `NAFNetSR` class and all building blocks were left
  **completely untouched** - still byte-for-byte AST-identical to
  `src/models/nafnet.py`. The new architecture was **added** as a
  subclass (`NAFNetSRDecoderCapacity`, inlined from
  `src/models/nafnet_decoder_capacity.py`), and exactly one line changed
  in `load_model()`. Net diff: **50 insertions, 1 deletion.**
- The AST-guard test was reworked to guard **two** class groups against
  their own source modules, and now also asserts `load_model()` actually
  constructs the adopted class (so `run.py` cannot silently revert to the
  plain backbone). Docstrings are stripped before comparison - they
  cannot cause behavioral divergence, and `run.py`'s inlined copy
  legitimately carries extra explanation for a reader of the
  self-contained file.
- **The guard was fault-injected to prove it works:** a deliberate
  `* 1.01` inserted into `run.py`'s inlined `forward` made the test fail
  exactly as intended (`assert not ['NAFNetSRDecoderCapacity']`), then
  the file was restored and re-verified. A green guard that has never
  been shown to fail proves nothing.

**Full compliance chain, re-run against the post-swap combination
(not reused from before the swap):**

| Check | Result |
|---|---|
| 25-test adversarial suite | **25/25 pass** |
| Full local suite (incl. reworked guard) | **51/51 pass** |
| Fresh venv + no-internet + wrong-cwd, combined, one process | **PASS** - 0 network calls, cwd `/tmp`, 5/5 spec-compliant, new `run.py` + sha-verified new checkpoint |
| `models/checkpoint.pt` sha256 | `7b77678d1742...c9fb0412` (Item 1's checkpoint, confirmed) |
| Official test set, regenerated | 297/297, **0 fallbacks**, all (256,256) float32, finite, in [0,1] |

**Final official-test numbers (adopted model, n=297):** PSNR **24.001**,
SSIM **0.6251**, LPIPS **0.1605**. Ni-WC real-mask edge retention
**0.735**. Inference **86.08 +/- 1.52 ms/image** (N=20 cold-start runs).

**Honest nuance on the paired comparison vs. the previous shipped
model:** at n=297, even micro-differences reach statistical
significance. PSNR is -0.0033dB (p=0.003) and SSIM -0.00063 (p=2.2e-14) -
both *statistically* significant, both roughly **8x smaller than the
0.026dB same-seed reproducibility floor**, i.e. statistically detectable
but practically meaningless. LPIPS improved (-0.0011, p=0.0035). The
pre-registered gate deliberately used the reproducibility floor rather
than a significance test for exactly this reason. The real payoff is the
**+0.030 edge-retention gain**, an effect an order of magnitude larger
than any of these deltas.

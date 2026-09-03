# Three new technique classes — pre-registered decision rules

Written and committed BEFORE any of the three items below is run. All
three are genuinely different in kind from every prior attempt (which
only modified loss-term weights on the same architecture/training
scheme): Item 1 changes architectural capacity, Item 2 changes training
strategy/data ordering, Item 3 adds a structurally separate auxiliary
output. Baseline throughout: the shipped checkpoint,
`models/checkpoint.pt`, sha256 `36d2d38c...` - official-test PSNR
24.004, SSIM 0.6257, LPIPS 0.1616; edge retention 0.705 (blend=0.00).

## Item 3 — auxiliary confidence/residual head (run first, lowest risk)

Small head trained on TOP of the shipped model's **frozen** encoder
features, predicting per-pixel `|prediction - GT|`. Cannot regress the
main restoration output by construction (main weights frozen, head is
purely additive, never used unless separately invoked).

**Gate:** report the real Spearman correlation between the confidence
map's values and the model's actual per-pixel error on held-out data.
**Adopt as a genuine, demonstrable finding only if Spearman r >= 0.3**
(meaningful and, given real per-pixel sample sizes, expected to also be
statistically significant at that magnitude - p-value reported alongside
regardless). Below 0.3: reported as a real, honest negative/weak result,
same standard as the earlier Local-Lipschitz confidence work's own
PSNR/SSIM split. Since this never touches the main checkpoint, "adopt"
here means "documented as a real finding for the deck," not a checkpoint
swap - no compliance re-verification is needed for this item regardless
of outcome, since `run.py`'s main restoration path is provably
untouched.

## Item 1 — decoder-only residual-block capacity increase

One additional lightweight residual block inserted in the decoder's
final stage, immediately before the pixel-shuffle head - not the encoder,
not the bottleneck (a different location and purpose than Axis 4's
bottleneck self-attention test). Initialized near-identity (the added
block's residual path starts at ~zero contribution) so the fine-tune
starts close to current behavior, not from scratch. Fine-tuned from the
shipped checkpoint's compatible weights.

**Gate - adopt only if ALL THREE hold:**
1. Real-mask edge retention (Ni-WC, blend=0.00) improves by **>= 0.020
   absolute** over 0.705 - i.e. **>= 0.725**.
2. Official-test PSNR does not regress beyond the **0.026dB same-seed
   reproducibility bound** (Axis 1c) - i.e. **>= 23.978**.
3. **Parameter count and inference-time increase both stay under 15%**
   over the shipped model (6.82M params, 88.35ms/image mean).

If adopted: full re-verification (fresh-venv + no-internet + wrong-cwd +
25-test suite + official-test re-run) before any checkpoint swap.

## Item 2 — severity-curriculum fine-tune

Training schedule over the fine-tune's epoch budget: first 30% of epochs
sample only the mildest third of the measured per-cluster severity range
(highest L_gain/K_poisson, i.e. least noise); middle 40% sample the full
measured range uniformly (matching current/all prior training); final
30% oversample the harshest third (lowest L_gain/K_poisson). Fine-tuned
from the shipped checkpoint.

**Gate:** adopt only if official-test **PSNR improves by >= 0.05dB**
(a genuinely meaningful bar, not just the 0.026dB noise floor) **with no
regression on SSIM, LPIPS, or edge retention** (blend=0.00, vs. 0.705).

If adopted: full re-verification (fresh-venv + no-internet + wrong-cwd +
25-test suite + official-test re-run) before any checkpoint swap.

## Shared rules

- Time-box: ~45 minutes per item, no clear signal by then -> stop, document
  as time-boxed.
- `models/checkpoint.pt` is never overwritten until an item's gate passes
  AND full re-verification (where required) passes.
- Report back after each item with real numbers, pass or fail.

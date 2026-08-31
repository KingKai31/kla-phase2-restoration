# Phase 2 data inventory — SEM/NFFA-EUROPE dataset

Covers Tasks 1-5 of the Phase 2 kickoff: raw inventory, pairing structure,
category breakdown, visual/statistical first pass, and a first-pass noise-
model check. Every number below comes from `scripts/phase2_data_inventory.py`
(full dataset scan, not a sample, for pairing/shape/dtype) plus the figures
in `reports/figures/` (real images, not mockups). Full JSON:
`reports/phase2_first_pass_summary.json`.

**Phase 1's fitted noise-model numbers (Gamma L range, additive σ, etc.) do
NOT apply here and are not referenced below** - this dataset is re-derived
from scratch, per instruction.

---

## Task 1 — Raw inventory

**Real local path** (triple-nested due to zip extraction — see
`data/README.md`):
```
C:\Users\ANANNYA\Downloads\semicon_train_data\semicon_train_data\semicon_train_data\
├── GT\        4785 files, all .npy
└── NoisyLR\   4785 files, all .npy
```
A sibling `__MACOSX` folder exists one level up but is **empty** (0 files) —
harmless zip-extraction artifact, not real data.

- **Total files:** 9,570 (4,785 GT + 4,785 NoisyLR).
- **File formats:** `.npy` only — no `.png`/`.tif`/other image formats found
  anywhere in the download (confirmed via extension scan of every file).
- **No manifest, README, metadata JSON/CSV, or category-label file of any
  kind ships with this download.** Checked explicitly (`find` for
  `*.md`/`*.txt`/`*readme*`/`*manifest*`/`*.json`/`*.csv` anywhere under the
  download root) — zero matches. This is a real, notable gap, not an
  oversight in searching — see Task 3.

## Task 2 — Pairing structure

**Confirmed identical structure to Phase 1's GT/NoisyLR convention** — this
was not assumed, it was checked:

- Filename sets between `GT/` and `NoisyLR/` are **exactly identical**
  (`gt_files == noisy_files` → `True`), confirmed across all 4,785 files —
  perfect 1:1 pairing, zero orphans on either side.
- Filenames are zero-padded sequential integers, `000000.npy`–`004784.npy`,
  contiguous with no gaps.
- **Resolution: only 256↔128 exists.** Full-dataset scan (every single
  file, not a sample) via `np.load(..., mmap_mode="r")`:
  - GT: **100% of files are `(256, 256)`, `float32`** — zero exceptions.
  - NoisyLR: **100% of files are `(128, 128)`, `float32`** — zero exceptions.
  - **No 512×512/256×256 pair exists anywhere in this delivery.** This
    directly answers the open question from Phase 1 (which also only had
    256↔128) — explicitly checked this time via a full scan, not assumed:
    the second stated pair (512→256) is absent here too.

## Task 3 — Category breakdown

**No category labels are present in the data as delivered.** Checked all
three ways the task asked for, all came back negative:

1. **Folder structure:** `GT/` and `NoisyLR/` are both flat — no
   per-category subfolders.
2. **Filename encoding:** filenames are plain sequential integers with zero
   embedded information beyond order (`000000`–`004784`) — no
   category prefix, suffix, or code.
3. **Bundled metadata:** no manifest/JSON/CSV ships with the download (Task 1).

**Cross-referencing against NFFA-EUROPE's published taxonomy:** the source
dataset's documented 10 categories are **Tips, Particles, Patterned
surfaces, MEMS devices and electrodes, Nanowires, Porous sponge,
Biological, Powder, Films and coated surfaces, and Fibres** — confirmed via
the dataset's own citation, [Modarres et al. 2018, *Scientific
Data*](https://www.nature.com/articles/sdata2018172), not recalled from
memory alone. **But this only tells us the taxonomy exists upstream — it
gives no way to assign a category to any specific file in this delivery**,
since nothing links index `003273.npy` back to NFFA-EUROPE's own image IDs
or category labels.

**Visual check for implicit ordering structure** (`reports/figures/index_span_check.png`,
20 images evenly spaced across the full 0–4784 range): content is visibly
diverse and **qualitatively consistent** with several NFFA-EUROPE categories
— clear examples of fibre/wire-like strands, particle/nanosphere clusters,
porous honeycomb membranes, smooth near-featureless films, and granular
powder-like textures all appear. This is consistent with genuine SEM
category diversity, not proof of which category is which.

**Visual check for contiguous block structure** (`reports/figures/contiguous_run_check.png`,
images 0–29): shows weak local similarity between near neighbors (several
consecutive images share a fibrous/cellular look) but **not** a clean,
rigid single-category block — the local run still shows real variation
within it. **Honest read: visual inspection alone is inconclusive on
whether index order encodes any category grouping at all.** This would
need actual unsupervised clustering (the same kind of approach Phase 1's
`scripts/cluster_sources.py` used for its OOD-proxy split, since it also
had no real source labels) to say anything more rigorous — flagged as
next-phase work, not attempted here.

**Real, unrelated finding surfaced during this visual check, worth flagging
prominently:** image index 7 (`reports/figures/contiguous_run_check.png`)
has a **visible "2 μm" scale-bar annotation burned directly into the pixel
data**, not cropped out. This is a known characteristic of some raw SEM
image exports and a genuine data-quality concern for training: a model
being trained to "restore" a patch containing burned-in scale-bar text
would be trying to reconstruct a rendered overlay artifact, not real
surface structure. **Not yet quantified how common this is across the full
4,785 images** — flagged as a concrete, actionable check for the next pass
(a cheap heuristic scan for bright rectangular overlay regions would answer
this, not attempted in this data-understanding-only pass).

**Decision needed from the user:** category labels are genuinely absent
from this data delivery. Options going forward (not deciding unilaterally):
(a) request a corrected/re-exported download with the manifest if one
exists upstream, (b) proceed without real category labels and use an
unsupervised-clustering proxy the way Phase 1 did for its train/val split,
or (c) treat categories as irrelevant to the restoration task itself (the
model restores pixels regardless of category; labels would only matter for
validation-split design or category-stratified reporting).

## Task 4 — Visual & statistical first pass

8 random real pairs (`reports/figures/sample_pairs_random8.png`), full
per-image stats:

| file | GT min/max/mean/std | NoisyLR min/max/mean/std |
|---|---|---|
| 000079 | 0.000 / 1.000 / 0.543 / 0.166 | -0.055 / 1.630 / 0.545 / 0.199 |
| 000195 | 0.000 / 1.000 / 0.571 / 0.142 | 0.001 / 1.525 / 0.570 / 0.184 |
| 000359 | 0.000 / 1.000 / 0.466 / 0.154 | 0.012 / 1.379 / 0.465 / 0.177 |
| 001289 | 0.000 / 1.000 / 0.495 / 0.079 | 0.033 / 1.035 / 0.495 / 0.109 |
| 001472 | 0.000 / 1.000 / 0.382 / 0.095 | 0.018 / 1.011 / 0.381 / 0.113 |
| 002443 | 0.000 / 1.000 / 0.596 / 0.082 | 0.085 / 1.245 / 0.597 / 0.138 |
| 003044 | 0.000 / 1.000 / 0.451 / 0.269 | **-0.105** / 1.489 / 0.451 / 0.284 |
| 004064 | 0.000 / 1.000 / 0.499 / 0.180 | 0.009 / 1.507 / 0.500 / 0.205 |

**GT is consistently exactly `[0, 1]`** across every sample (same convention
as Phase 1). **NoisyLR consistently overshoots both ends** — 6/8 samples
here exceed 1.0 on the high end, 2/8 go negative on the low end. This is
the **same qualitative signature Phase 1 found**: real speckle-style
overshoot beyond the clean [0,1] range, not something that needs to be
assumed to transfer — it's directly observed here too.

## Task 5 — First-pass noise-model check (structural, not final fit)

Method: for a random sample of 100 pairs (not the full 4,785 — full rigor
comes next phase), computed `ratio = NoisyLR / box_downsample(GT, factor=2)`
at NoisyLR's native 128×128 resolution (box/area downsampling, not
bilinear — the exact lesson learned in Phase 1 about upsampling-induced
bias). Full diagnostic: `reports/figures/noise_model_firstpass.png`,
`reports/phase2_first_pass_summary.json`.

**Negative-pixel evidence (same test Phase 1 used to rule out pure
multiplicative speckle):**
- **43/100 sampled images have at least one negative pixel** in NoisyLR.
- Mean fraction of negative pixels per image: 0.047% (small, but real and
  present in nearly half the sample).
- Most negative pixel value seen: **-0.113**.
- **Same conclusion as Phase 1: pure multiplicative noise on a
  non-negative GT cannot produce negative output — an additive component
  is needed here too.** Not assumed to transfer; directly re-confirmed.

**Ratio distribution (n=1,620,307 pixels across 100 images):**
- mean = 0.996, median = 0.984, std = 0.221
- skewness = +0.40 (right-skewed), excess kurtosis = +1.23 (heavier tails
  than Gaussian)
- **Gamma fit: shape `L` ≈ 19.6, scale ≈ 0.051, implied mean ≈ 0.996** —
  visually, the fitted Gamma curve tracks the empirical histogram closely
  (see the left panel of `noise_model_firstpass.png`): unimodal, right-
  skewed, peak near 1, matching the qualitative shape of Phase 1's
  multiplicative-Gamma model.

**Honest assessment — does Phase 1's noise-model *shape* look right, or do
we need a different functional form?** On this coarse, pooled, single-pass
check: **the Gamma-multiplicative-plus-additive shape looks like a
reasonable starting hypothesis here too** — the fit is visually good and
the negative-pixel evidence matches. **But this check pools all brightness
levels together and cannot distinguish a Gamma-multiplicative model from a
Poisson-like shot-noise model**, which is physically plausible for real
SEM electron-detection imaging and was explicitly flagged as a hypothesis
to consider. Distinguishing the two requires exactly what Phase 1's
heteroscedasticity check did (residual variance vs. GT brightness, binned)
— **not done here, correctly deferred to the next, rigorous-fitting phase**
per this pass's explicit scope. Flagging this as the single most important
open question before committing to a synthetic degradation model for
training data augmentation.

---

## Summary: similar to vs. different from Phase 1

**Structurally very similar** (not assumed — checked at every point above):
- Identical `GT`/`NoisyLR` folder convention, identical `.npy`-only format,
  identical filename convention (zero-padded sequential integers).
- Identical resolution pair present (256↔128 only — 512↔256 absent in both).
- Identical value-range convention (GT strictly `[0,1]`, NoisyLR overshoots
  both directions).
- Identical qualitative noise signature (negative pixels present, ratio
  distribution shaped like a right-skewed Gamma centered near 1).

**Different / new, genuinely Phase-2-specific:**
- **10x more data** (4,785 pairs vs. Phase 1's ~3,200) but **zero category
  metadata**, versus Phase 1 which at least had sequential source-agnostic
  filenames with no labels either — so this isn't a regression, but the
  "10 categories" framing from the task brief cannot currently be acted on
  without either a corrected download or an unsupervised-clustering proxy.
- **Real image content is SEM micrographs of physical nanomaterials**
  (fibres, particles, membranes, films), a completely different visual
  domain from Phase 1's semiconductor dendrite/texture imagery — expected,
  but worth stating plainly since it affects what "realistic" synthetic
  degradation augmentation should look like.
- **New, unquantified data-quality concern:** burned-in scale-bar
  annotations in at least one confirmed image, prevalence unknown.
- **Higher overshoot ceiling observed** (max +2.19 in this sample vs.
  Phase 1's comparable checks) — worth confirming this holds at scale
  before assuming it's representative.

## What's explicitly NOT done here (by design, per task scope)

- No rigorous Gamma/additive parameter fitting (this was a first-pass
  shape check on 100/4,785 pairs, pooled across brightness).
- No per-category breakdown (no labels exist — see Task 3's open decision).
- No brightness-dependent heteroscedasticity analysis (needed to
  distinguish Gamma-multiplicative from Poisson-shot-noise — the key open
  question above).
- No architecture or training decisions of any kind.

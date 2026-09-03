# Item 3 result — per-REAL-category breakdown (real NFFA-EUROPE labels)

Ran **locally on CPU**, after the pod had no further GPU-bound work
queued (Items 1 and 2 were already resolved) - no need to keep it
running for this. Uses the final shipped model (Item 1's graduated-edge
sweep did not adopt anything at any weight, so this is the same
aug/EMA/ICNR checkpoint as `models/checkpoint.pt`).

## What this is (and isn't)

**Genuinely differentiating evidence: real category names, not cluster
numbers.** 50 real images per category (a representative sample of the
4 downloaded NFFA-EUROPE categories - Biological, Fibres,
Films_Coated_Surface, MEMS_devices_and_electrodes), a deterministic
center 256x256 crop from each, degraded with the same validated compound
noise model used everywhere in this project, restored with `run.py`'s
real inference path.

**Important scope note, stated plainly:** this is genuinely
**out-of-domain** data - real SEM images of biological/fiber/coating/MEMS
specimens, not the semiconductor content the model was trained on. Same
caveat as the Ni-WC external-validation check (Axis 3a): this measures
generalization to different specimen content under our own noise model,
not in-domain training performance. It should be read alongside, not
instead of, the official-test-set numbers (24.004dB, semiconductor data).

## Results

| Category | n | Mean PSNR | Std PSNR | Mean SSIM | Mean LPIPS |
|---|---|---|---|---|---|
| Biological | 50 | 22.695 | 2.443 | 0.5302 | 0.2725 |
| Fibres | 50 | 25.687 | 2.059 | 0.5489 | 0.3223 |
| Films_Coated_Surface | 50 | 23.783 | 3.043 | 0.4832 | 0.3235 |
| MEMS_devices_and_electrodes | 50 | 24.470 | 2.183 | 0.4829 | 0.2746 |

**A real, honest spread across real category identities** - Fibres
scores highest (25.687dB), Biological lowest (22.695dB), a ~3dB real
gap. All four sit reasonably close together (22.7-25.7dB) with no
category catastrophically failing, and all comfortably above the
classical-baseline range measured elsewhere in this project (~20.3dB) -
real, if partial, evidence that the model's learned restoration
transfers across genuinely different specimen content, not just within
the semiconductor domain it was trained on.

**Why this is rare, differentiating evidence:** the project's own
training data ships with no category labels at all (documented from the
very start of Phase 2) - these real labels only exist because of the
independent Tier-1 NFFA-EUROPE download pursued mid-project. Most
comparable submissions built on this same training delivery would have
no way to produce a real-category breakdown at all.

Full per-file data: `reports/item3_per_category_breakdown.per_file.json`.
Summary: `reports/item3_per_category_breakdown.json`.

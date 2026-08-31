# Data

Not committed to this repo (see `.gitignore` - `/data/**` is excluded except
this file). This documents the real local layout so scripts can be pointed
at it via `--gt-dir`/`--noisy-dir` args, same pattern as the Phase 1
(KLA PS01) repo.

## Current local path

```
C:\Users\ANANNYA\Downloads\semicon_train_data\semicon_train_data\semicon_train_data\
├── GT\        4785 .npy files, float32, (256, 256), values in [0, 1]
└── NoisyLR\   4785 .npy files, float32, (128, 128), values overshoot [0, 1]
                (min seen so far: -0.113, max seen so far: 2.187)
```

Note the triple-nested `semicon_train_data\semicon_train_data\semicon_train_data\`
path - an artifact of how the zip was extracted (also produced an empty
`__MACOSX` folder one level up, safe to ignore/delete). Point scripts at the
innermost folder, not the top-level download.

## Source

Derived from **NFFA-EUROPE**'s public SEM image dataset (CC-BY 4.0),
[Modarres et al. 2018 - Scientific Data](https://www.nature.com/articles/sdata2018172).
The original dataset's documented 10-category taxonomy: Tips, Particles,
Patterned surfaces, MEMS devices and electrodes, Nanowires, Porous sponge,
Biological, Powder, Films and coated surfaces, Fibres. **No manifest,
category labels, or metadata file ships with this specific
GT/NoisyLR download** - see `reports/phase2_data_inventory.md` for what
that means in practice and what was checked.

## Layout convention (same as Phase 1)

- `GT/<id>.npy` and `NoisyLR/<id>.npy` share the same zero-padded numeric
  filename (`000000.npy`-`004784.npy`) - this is the pairing key, confirmed
  1:1 complete (see the inventory doc).
- Only one resolution pair exists in this delivery: 256↔128. No 512↔256
  pairs were found (same gap Phase 1 had).

# run.py compliance checklist — Phase 2 (SEM/NFFA-EUROPE data)

Verified against the actual `run.py` at the repo root, shipping
`checkpoints/stage_a_best.pt` as `models/checkpoint.pt`
(sha256 `19bf6df1804296916bdfb52e7c51a015f3825417f98c3e25f7cd758db56a0591`,
verified byte-identical between the pod's training checkpoint and the
submission copy). Ships Stage A as final per `reports/STAGE_A_RESULTS.md`
and `reports/SPECTRAL_FIX_ATTEMPT.md` - not a Stage A-then-swap-to-B
situation like Phase 1, so there is only one verification pass needed
here, not a re-verification after a later checkpoint swap.

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Entry script named `run.py` from the start, callable as `python run.py <input-dir> <output-dir>` | **PASS** | Built as `run.py` from the first commit of final packaging, not renamed from a dev/eval script - the exact Phase 1 lesson applied proactively |
| 2 | Reads ALL `.npy` files from input dir | **PASS** | `sorted(args.input_dir.glob("*.npy"))` - 5/5 real held-out samples found and processed in the smoke test |
| 3 | Creates output dir if missing | **PASS** | Ran against a directory confirmed nonexistent beforehand; directory + files present after, both in the local smoke test and the combined pod verification (`/root/verify_output_DOES_NOT_EXIST`) |
| 4 | Exactly one output per input, exact filename match | **PASS** | 5 inputs -> 5 outputs, filenames match exactly in both the smoke test and the adversarial pytest suite (`TestBatchSurvival::test_all_ten_inputs_produce_an_output_file`) |
| 5 | Output shape `(H,W)` | **PASS** | Raw model output is `(1,1,256,256)` NCHW, squeezed to plain `(256,256)` before saving - matches this dataset's own GT/NoisyLR `.npy` convention. All 5 real samples: `ndim==2` |
| 6 | Output values strictly in `[0,1]`, no NaN/Inf | **PASS** | `sanitize_output()` is a universal final gate before every `np.save`, applied regardless of code path. All 5 real samples: min/max within `[0.000, 1.000]`; the full 25-test adversarial suite (`tests/test_run_py_robustness.py`) exercises this against corrupt/NaN/Inf/extreme-value/degenerate inputs directly |
| 7 | Correct target resolution (128->256) | **PASS** | All 5 real samples: input `(128,128)` -> output `(256,256)` |
| 8 | Zero internet access at runtime | **PASS** | Import chain has no `lpips`/pretrained-backbone/download call anywhere; proven directly by monkey-patching `socket.socket.connect`/`socket.getaddrinfo` to hard-raise on any call and running the full pipeline end-to-end - 0 network calls, correct output (`scripts/_no_internet_wrong_cwd_check.py`) |
| 9 | Model weights bundled locally under `models/`, loaded from disk only | **PASS** | `models/checkpoint.pt` present; `torch.load` reads a local path only, no URL |
| 10 | Self-contained, no dependency on `src/` | **PASS** | Model architecture (NAFNetSR and building blocks) inlined directly in `run.py`, not imported - verified by inspection, same pattern as Phase 1 |
| 11 | `requirements.txt` = exact pinned versions | **PASS** | `submission_requirements.txt` built from this pod's actual confirmed-working versions (`torch/torchvision/torchaudio==...+cu124`, `numpy==2.4.6`, `scipy==1.17.1`, `pillow==10.2.0`, `scikit-image==0.26.0`, `PyWavelets==1.9.0`); verified end-to-end against a completely fresh venv installing strictly from this file |
| 12 | Fresh-venv install + run | **PASS** | New venv on the pod's separate root filesystem (to avoid disk contention with the in-progress Tier-1 download on `/workspace`), installed strictly from `submission_requirements.txt`, ran the real checkpoint successfully via CUDA |
| 13 | Wrong-cwd/absolute-path invocation | **PASS** | `os.chdir("/tmp")` before invoking `run.py` via an absolute path - checkpoint still resolved correctly via `Path(__file__).resolve().parent`, applied proactively from the start (not a bug found and fixed this time, since it was built in from Phase 1's lesson) |
| **14** | **All three combined** (fresh venv + wrong cwd + no-internet, simultaneously) | **PASS** | The actual real shipping combination, tested together in one process: 0 network calls, cwd unrelated to the repo, absolute script path, real checkpoint via CUDA - 5/5 outputs spec-compliant |
| 15 | Adversarial robustness (10 cases: corrupt file, wrong ndim x2, NaN/Inf, all-zero, all-constant, tiny 8x8, non-square, extreme values, plus the normal control) | **PASS** | `tests/test_run_py_robustness.py`, 25/25 tests pass against the real checkpoint |

## One real bug found and fixed during this pass

`PyWavelets` was **not installed** - not just unpinned, genuinely missing -
on both the training pod and this local dev machine. `skimage.restoration.estimate_sigma()`
(called inside `classical_fallback()` to set the NLM denoising strength)
has an undeclared optional dependency on it; without it, `estimate_sigma()`
raises `ImportError`, silently caught by `classical_fallback()`'s broad
exception handler, degrading every fallback call to bicubic-only with no
visible warning - **the exact Phase 1 mistake, recurring on entirely new
infrastructure.** Caught by `tests/test_run_py_robustness.py::TestClassicalFallback::test_nlm_denoising_actually_executes`
(ported directly from Phase 1's regression test for this exact bug) before
this pass shipped anything, not discovered after the fact. Fixed:
installed on both machines, pinned in both `requirements.txt` and
`submission_requirements.txt`.

## Real inference timing

- **Full end-to-end `run.py` invocation, 50 synthetic images, cold start
  (process launch, imports, CUDA init, model load, all 50 images, exit):
  3.604s total, 72.08 ms/image**, on this pod's A100-SXM4-80GB. Directly
  comparable in kind to Phase 1's own headline H100 figure (76.4 ms/image,
  3.819s for 50 images) - close in magnitude despite different hardware.
- **Component breakdown** (`scripts/performance_profile.py`, warm/local,
  same pod): model load 344.4ms (one-time), pure forward pass 5.35ms/image,
  full run.py per-image path (forward + checkerboard suppress + clamp +
  sanitize) 6.04ms/image, disk I/O 0.16ms/image - warm-amortized total
  13.09ms/image. The gap between this (13.09ms) and the cold end-to-end
  figure (72.08ms) is process/CUDA-context startup overhead, paid once per
  invocation, not per image - same pattern Phase 1 found and explained for
  its own local-vs-H100 gap.
- **VRAM**: 44.0MB peak for a single image via run.py's real (unbatched)
  per-image path; stays flat (44.3MB) for 16 images since only one is ever
  resident at once - run.py does not batch.
- **H100 not separately benchmarked**: this pod is an A100-SXM4-80GB.
  RunPod's on-demand H100 SXM is ~$2.99/hr - a short dedicated timing run
  would cost well under $1, but wasn't launched unprompted since it means
  renting a second pod purely for one timing number. Flagged as an
  available option, not assumed necessary, since the actual grading
  hardware for this phase isn't confirmed to be H100-specific the way
  Phase 1's was.

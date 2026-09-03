"""
Item 1 gate condition 3: fair inference-time comparison, shipped model
vs. the decoder-capacity variant, on the SAME GPU, same method as the
prior pass's N=20 cold-start benchmark (scripts/item2_repeated_timing_benchmark.py)
but N=10 per model (a supplementary check, not the primary official
benchmark, so a smaller N is proportionate) - a fresh subprocess each
run, matching how a grading harness actually invokes run.py.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def prepare_inputs(input_dir: Path, n: int, seed: int = 0):
    input_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        arr = np.clip(rng.normal(0.55, 0.2, size=(128, 128)).astype(np.float32), 0, 1.5)
        np.save(input_dir / f"img_{i:03d}.npy", arr)


def bench(run_py, checkpoint, n_runs, n_images, work_dir):
    input_dir = work_dir / "input"
    prepare_inputs(input_dir, n_images)
    ms_per_image = []
    for i in range(n_runs):
        output_dir = work_dir / f"output_{i}"
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
        t0 = time.perf_counter()
        result = subprocess.run(
            [sys.executable, str(run_py), str(input_dir), str(output_dir), "--checkpoint", str(checkpoint)],
            capture_output=True, text=True,
        )
        elapsed = time.perf_counter() - t0
        if result.returncode != 0:
            print(f"run {i} FAILED:\n{result.stderr}", file=sys.stderr)
            continue
        ms_per_image.append(elapsed / n_images * 1000)
    return ms_per_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-py", type=Path, required=True)
    ap.add_argument("--shipped-checkpoint", type=Path, required=True)
    ap.add_argument("--item1-checkpoint", type=Path, required=True)
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--n-images-per-run", type=int, default=50)
    ap.add_argument("--work-dir", type=Path, default=Path("/root/item1_timing"))
    ap.add_argument("--out", type=Path, default=Path("reports/item1_decoder_capacity_timing_comparison.json"))
    args = ap.parse_args()

    shipped_ms = bench(args.run_py, args.shipped_checkpoint, args.n_runs, args.n_images_per_run,
                        args.work_dir / "shipped")
    item1_ms = bench(args.run_py, args.item1_checkpoint, args.n_runs, args.n_images_per_run,
                      args.work_dir / "item1")

    out = {
        "shipped": {"mean_ms": float(np.mean(shipped_ms)), "std_ms": float(np.std(shipped_ms)), "n": len(shipped_ms)},
        "item1_decoder_capacity": {"mean_ms": float(np.mean(item1_ms)), "std_ms": float(np.std(item1_ms)), "n": len(item1_ms)},
    }
    out["pct_increase"] = (out["item1_decoder_capacity"]["mean_ms"] - out["shipped"]["mean_ms"]) / out["shipped"]["mean_ms"] * 100
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

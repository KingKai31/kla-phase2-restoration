"""
Item 2 (final pass): rigorous, repeated inference-time benchmark. Fixes
the single-measurement gap - N=20 independent COLD-START full-pipeline
runs (fresh process each time: script start -> import -> load checkpoint
-> inference -> write), matching how KLA will actually invoke run.py,
not a warm-loop microbenchmark. Reports mean/std/min/max, not one number.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def prepare_inputs(input_dir: Path, n: int, seed: int = 0):
    import numpy as np
    input_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        arr = np.clip(rng.normal(0.55, 0.2, size=(128, 128)).astype(np.float32), 0, 1.5)
        np.save(input_dir / f"img_{i:03d}.npy", arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-py", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--n-runs", type=int, default=20)
    ap.add_argument("--n-images-per-run", type=int, default=50)
    ap.add_argument("--work-dir", type=Path, default=Path("/root/item2_timing"))
    ap.add_argument("--out", type=Path, default=Path("reports/item2_timing_benchmark.json"))
    args = ap.parse_args()

    input_dir = args.work_dir / "input"
    prepare_inputs(input_dir, args.n_images_per_run)

    per_run_total_sec = []
    per_run_ms_per_image = []
    for i in range(args.n_runs):
        output_dir = args.work_dir / f"output_{i}"
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
        t0 = time.perf_counter()
        # genuinely cold: a brand-new python process each time, exactly
        # how a grading harness invokes `python run.py <in> <out>`
        result = subprocess.run(
            [sys.executable, str(args.run_py), str(input_dir), str(output_dir),
             "--checkpoint", str(args.checkpoint)],
            capture_output=True, text=True,
        )
        elapsed = time.perf_counter() - t0
        if result.returncode != 0:
            print(f"run {i} FAILED:\n{result.stderr}", file=sys.stderr)
            continue
        per_run_total_sec.append(elapsed)
        per_run_ms_per_image.append(elapsed / args.n_images_per_run * 1000)
        print(f"run {i}: {elapsed:.3f}s total, {elapsed/args.n_images_per_run*1000:.2f} ms/image", flush=True)

    arr = np.array(per_run_ms_per_image)
    out = {
        "n_runs": len(per_run_ms_per_image), "n_images_per_run": args.n_images_per_run,
        "checkpoint": str(args.checkpoint),
        "ms_per_image": {
            "mean": float(arr.mean()), "std": float(arr.std()),
            "min": float(arr.min()), "max": float(arr.max()),
            "median": float(np.median(arr)),
        },
        "total_sec_per_run": {
            "mean": float(np.mean(per_run_total_sec)), "std": float(np.std(per_run_total_sec)),
            "min": float(np.min(per_run_total_sec)), "max": float(np.max(per_run_total_sec)),
        },
        "all_runs_ms_per_image": per_run_ms_per_image,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "all_runs_ms_per_image"}, indent=2))


if __name__ == "__main__":
    main()

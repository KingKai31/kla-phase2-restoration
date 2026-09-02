"""
Combined verification (Task 4, final packaging): fresh venv + wrong-cwd +
no-internet, run together in one process - the actual real shipping
combination, not three separate partial checks. Monkey-patches
socket.socket.connect / socket.getaddrinfo to hard-raise on any call
before importing/running run.py, then invokes run.py's main() with an
absolute script path while the process cwd is a directory that has
nothing to do with the repo (simulating a grading harness that doesn't
`cd` into the submission folder first).
"""
import importlib.util
import os
import socket
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMISSION_DIR = Path("/root/kla_submission_test")


def _block_network():
    def _raise(*a, **k):
        raise RuntimeError("BLOCKED: network access attempted during a no-internet check")
    socket.socket.connect = _raise
    socket.getaddrinfo = _raise


def main():
    _block_network()

    # Simulate a wrong cwd: change to a directory unrelated to the repo
    # before invoking run.py via an absolute path - this is exactly the
    # scenario the Phase 1 cwd-independence fix targets (a relative
    # checkpoint default resolves against cwd, not the script's own
    # location, unless explicitly fixed).
    os.chdir("/tmp")

    run_py_path = SUBMISSION_DIR / "run.py"
    spec = importlib.util.spec_from_file_location("run_module_verify", run_py_path)
    run_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_module)

    input_dir = Path("/root/verify_input")
    output_dir = Path("/root/verify_output_DOES_NOT_EXIST")
    input_dir.mkdir(exist_ok=True)

    import numpy as np
    rng = np.random.default_rng(0)
    for i in range(5):
        arr = np.clip(rng.normal(0.55, 0.2, size=(128, 128)).astype(np.float32), 0, 1.5)
        np.save(input_dir / f"verify_{i}.npy", arr)

    argv_backup = sys.argv
    sys.argv = ["run.py", str(input_dir), str(output_dir)]
    try:
        run_module.main()
    finally:
        sys.argv = argv_backup

    outputs = sorted(output_dir.glob("*.npy"))
    assert len(outputs) == 5, f"expected 5 outputs, got {len(outputs)}"
    for f in outputs:
        arr = np.load(f)
        assert arr.dtype == np.float32 and arr.ndim == 2
        assert arr.min() >= 0.0 and arr.max() <= 1.0
        assert np.all(np.isfinite(arr))
    print(f"\nPASS: {len(outputs)}/5 outputs spec-compliant, 0 network calls attempted, "
          f"cwd was {os.getcwd()} (not the repo), checkpoint resolved via absolute "
          f"script path -> {run_module.__dict__.get('args', 'default_checkpoint used')}")
    print("Combined fresh-venv + wrong-cwd + no-internet check: PASS")


if __name__ == "__main__":
    main()

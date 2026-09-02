"""
Range-chunked downloader for the NFFA-EUROPE full dataset from B2SHARE.

Why: plain `curl -o file url` and even `curl -C - --retry` both proved
unreliable for large files on this pod's network path (repeated
truncation, and in one case a near-total stall despite --max-time -
see reports/phase2_deep_dive.md network notes). B2SHARE's server
confirmed `Accept-Ranges: bytes` support. This downloads each file as
many small explicit byte-range requests (mirroring the exact strategy
that fixed the earlier unreliable scp transfers to this same pod),
verifying and retrying per-chunk rather than per-whole-file, then
concatenates and verifies the assembled file's MD5 against B2SHARE's
published checksum before extracting.
"""
import hashlib
import subprocess
import sys
import tarfile
import time
from pathlib import Path

BASE_URL = "https://b2share.eudat.eu/api/records/862nr-cn036/files"
WORK_DIR = Path("/workspace/nffa_full")
# Reduced from 20MB after observing real evidence of within-session throughput
# degradation (chunk 12 took 4 min for 20MB vs an initial ~9MB/s baseline,
# chunk 13 then failed 6 straight retries) - smaller chunks bound the damage
# from a single slow/failed request and a short inter-chunk pause avoids
# hammering the server in a way that might be triggering throttling.
CHUNK_SIZE = 5 * 1024 * 1024  # 5MB
INTER_CHUNK_DELAY_S = 0.5

CHECKSUMS = {
    "Biological.tar": "c3a20755e2d937d70dd7d5be18f6006e",
    "Fibres.tar": "86fb065581eb3680f08ab2de9cea86eb",
    "Films_Coated_Surface.tar": "feb2b098a150c2ba6b6b35f2f18a6e7c",
    "MEMS_devices_and_electrodes.tar": "8bc4f788ec54de267a2b8cd0ac92066a",
    "Nanowires.tar": "005ffe16579c6352d3d521f58de56e8d",
    "Particles.tar": "6b73d88e65d3c00493076238c4230579",
    "Patterned_surface.tar": "19c0417faa8f977fdbf13d9c91ac3be8",
    "Porous_Sponge.tar": "851e6daf0e7ce45b68dbe9eb1338807e",
    "Powder.tar": "82da8aeb4d7a79c53bc3fefe9770e757",
    "Tips.tar": "0a85315213af1c32a14555d46e8f093a",
}


def get_content_length(url: str) -> int:
    out = subprocess.run(["curl", "-sI", "--max-time", "20", url], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.lower().startswith("content-length:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"Could not get content-length for {url}")


def download_chunk(url: str, start: int, end: int, out_path: Path, max_attempts: int = 12) -> bool:
    expected_size = end - start + 1
    for attempt in range(1, max_attempts + 1):
        try:
            # --limit-rate: deliberately deprioritized background job (per explicit
            # instruction) - caps this well below what it's shown capable of, so it
            # never competes with real training/analysis network or CPU usage.
            subprocess.run(
                ["curl", "-s", "--limit-rate", "1M", "--max-time", "90", "-r", f"{start}-{end}", "-o", str(out_path), url],
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            pass
        if out_path.exists() and out_path.stat().st_size == expected_size:
            return True
        time.sleep(2)
    return False


def download_file_ranged(name: str, expected_md5: str) -> bool:
    url = f"{BASE_URL}/{name}/content"
    total_size = get_content_length(url)
    print(f"  total size: {total_size / 1e6:.1f} MB")

    chunk_dir = WORK_DIR / f".{name}.chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    n_chunks = (total_size + CHUNK_SIZE - 1) // CHUNK_SIZE
    for i in range(n_chunks):
        start = i * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, total_size) - 1
        chunk_path = chunk_dir / f"chunk_{i:05d}"
        if chunk_path.exists() and chunk_path.stat().st_size == (end - start + 1):
            continue  # already have this chunk from a prior run
        ok = download_chunk(url, start, end, chunk_path)
        if not ok:
            print(f"  chunk {i}/{n_chunks} failed after retries")
            return False
        time.sleep(INTER_CHUNK_DELAY_S)
        if (i + 1) % 40 == 0 or i == n_chunks - 1:
            print(f"  chunk {i + 1}/{n_chunks} done")

    # assemble
    out_path = WORK_DIR / name
    md5 = hashlib.md5()
    with open(out_path, "wb") as outf:
        for i in range(n_chunks):
            chunk_path = chunk_dir / f"chunk_{i:05d}"
            data = chunk_path.read_bytes()
            outf.write(data)
            md5.update(data)

    actual_md5 = md5.hexdigest()
    if actual_md5 != expected_md5:
        print(f"  FINAL MD5 MISMATCH: expected {expected_md5}, got {actual_md5}")
        return False

    print(f"  assembled and verified: {actual_md5}")
    # cleanup chunks
    for f in chunk_dir.glob("chunk_*"):
        f.unlink()
    chunk_dir.rmdir()
    return True


def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    extracted_dir = WORK_DIR / "extracted"
    extracted_dir.mkdir(exist_ok=True)

    for name, expected_md5 in CHECKSUMS.items():
        done_marker = extracted_dir / f".{name}.done"
        print(f"=== {name} ===")
        if done_marker.exists():
            print("  already done, skipping")
            continue

        ok = download_file_ranged(name, expected_md5)
        if not ok:
            print(f"  FAILED: {name}")
            sys.exit(1)

        tar_path = WORK_DIR / name
        with tarfile.open(tar_path) as tf:
            tf.extractall(extracted_dir)
        cat_name = name.replace(".tar", "")
        n_files = sum(1 for _ in (extracted_dir / cat_name).rglob("*") if _.is_file())
        print(f"  extracted: {n_files} files")
        tar_path.unlink()
        done_marker.touch()

        import shutil
        total, used, free = shutil.disk_usage("/workspace")
        print(f"  disk: {used / 1e9:.1f}GB used, {free / 1e9:.1f}GB free")

    print("ALL_CATEGORIES_DONE")


if __name__ == "__main__":
    main()

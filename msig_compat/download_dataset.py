"""
Download remaining Mixed-Signals-V2X tars from HuggingFace (sberrio/Mixed-Signals-V2X)
and extract them into data/mixed-signals/. Resumable (huggingface_hub handles partial
downloads / retries internally via hf_hub_download's local_dir caching).

Usage:
    .venv/bin/python msig_compat/download_dataset.py [--files train/mini_2.tar val/mini_9.tar ...]

With no --files, downloads every file in FILES_IN_ORDER not yet extracted (checked via
a marker file data/mixed-signals/.extracted/<safe_name>).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "sberrio/Mixed-Signals-V2X"
REPO_TYPE = "dataset"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_DIR = RAW_DIR / ".cache" / "huggingface"
EXTRACT_ROOT = PROJECT_ROOT / "data" / "mixed-signals"
TEST_EXTRACT_ROOT = PROJECT_ROOT / "data" / "mixed-signals-test-competition"
MARKER_DIR = EXTRACT_ROOT / ".extracted"

# val/ and test/ ship no labels and aren't needed for training/eval (see
# msig_compat/dataset_patch.py), so the train/ tars - the actual
# training+eval pool - are prioritized first; val/ and test/ are downloaded
# for completeness but come last.
FILES_IN_ORDER = (
    ["val/mini_9.tar", "test/test-0000.tar"]
    + [
        f"train/mini_{n}.tar"
        for n in (2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 19, 20, 22, 23, 24, 25, 26,
                  28, 29, 30, 31, 32, 33, 34, 35, 37)
    ]
    + [f"val/mini_{n}.tar" for n in (18, 27, 36)]
    + [f"test/test-000{n}.tar" for n in (1, 2, 3, 4, 5)]
)


def marker_for(rel_path: str) -> Path:
    return MARKER_DIR / rel_path.replace("/", "__")


def download_one(rel_path: str, max_retries: int = 8) -> Path:
    split = rel_path.split("/")[0]
    for attempt in range(1, max_retries + 1):
        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=rel_path,
                local_dir=str(RAW_DIR),
                cache_dir=str(CACHE_DIR),
            )
            return Path(local_path)
        except Exception as e:
            wait = min(60, 5 * attempt)
            print(f"[download] {rel_path} attempt {attempt}/{max_retries} failed: {e!r}; "
                  f"retrying in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"Failed to download {rel_path} after {max_retries} attempts")


def extract_one(tar_path: Path, rel_path: str) -> None:
    split = rel_path.split("/")[0]
    dest = TEST_EXTRACT_ROOT if split == "test" else EXTRACT_ROOT
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {tar_path} -> {dest}", flush=True)
    subprocess.run(["tar", "-xf", str(tar_path), "-C", str(dest)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None)
    args = ap.parse_args()

    files = args.files if args.files else FILES_IN_ORDER
    MARKER_DIR.mkdir(parents=True, exist_ok=True)

    for rel_path in files:
        marker = marker_for(rel_path)
        if marker.exists():
            print(f"[skip] {rel_path} already extracted", flush=True)
            continue
        t0 = time.time()
        print(f"[start] {rel_path}", flush=True)
        tar_path = download_one(rel_path)
        dt = time.time() - t0
        size_gb = tar_path.stat().st_size / 1e9
        print(f"[done-download] {rel_path} ({size_gb:.2f} GB in {dt:.0f}s, "
              f"{size_gb/max(dt,1)*1000:.1f} MB/s)", flush=True)
        extract_one(tar_path, rel_path)
        marker.write_text("ok\n")
        print(f"[done-extract] {rel_path} total {time.time()-t0:.0f}s", flush=True)

    print("[all done]", flush=True)


if __name__ == "__main__":
    main()

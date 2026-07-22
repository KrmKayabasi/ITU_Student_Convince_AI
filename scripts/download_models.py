#!/usr/bin/env python3
"""
Download required ONNX model files for speaker verification.

Downloads (if not already present):
  1. TitaNet-Small  — speaker embedding  (38 MB, 192-dim)
  2. Silero VAD     — voice activity detection  (628 KB)

Usage:
  python3 scripts/download_models.py              # download to default paths
  python3 scripts/download_models.py --force      # re-download even if exists

Sources:
  https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

# ── Model catalogue ──────────────────────────────────────────────────────────

MODELS = [
    {
        "name": "nemo_en_titanet_small",
        "description": "TitaNet-Small speaker embedding (192-dim, 38 MB)",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
               "speaker-recongition-models/nemo_en_titanet_small.onnx",
        "dest": "models/nemo_en_titanet_small.onnx",
        # SHA-256 of the known-good 38 MB file.
        # If the hash doesn't match, the download is treated as corrupt and
        # re-fetched.  Set to "" to skip verification.
        "sha256": "",
    },
    {
        "name": "silero_vad",
        "description": "Silero VAD (628 KB)",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
               "asr-models/silero_vad.onnx",
        "dest": "models/silero_vad.onnx",
        "sha256": "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
    },
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def _sha256_hex(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, description: str, expected_sha256: str = "") -> bool:
    """Download *url* to *dest*.  Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"  → {description}")
    print(f"    {url}", flush=True)

    headers = {"User-Agent": "itu-convince-ai/1.0"}
    retries = 3
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=120) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                start = time.monotonic()
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(256 * 1024)  # 256 KiB
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = downloaded / total * 100
                            elapsed = time.monotonic() - start
                            rate = downloaded / (elapsed + 0.001) / 1024 / 1024
                            print(
                                f"\r    {pct:5.1f}%  {downloaded / 1024**2:.1f} / "
                                f"{total / 1024**2:.1f} MiB  ({rate:.1f} MiB/s)  ",
                                end="", flush=True,
                            )
                print()
                break  # success
        except Exception as exc:
            print(f"\n    ⚠ attempt {attempt}/{retries} failed: {exc}", flush=True)
            if attempt == retries:
                if tmp.exists():
                    tmp.unlink()
                return False
            time.sleep(2 ** attempt)

    # --- verify hash if one was provided ---
    if expected_sha256:
        actual = _sha256_hex(tmp)
        if actual != expected_sha256:
            print(f"    ✗ SHA-256 mismatch!", flush=True)
            print(f"      expected {expected_sha256[:16]}…", flush=True)
            print(f"      got      {actual[:16]}…", flush=True)
            tmp.unlink()
            return False

    tmp.replace(dest)
    size_mb = dest.stat().st_size / 1024**2
    print(f"    ✓ saved {dest} ({size_mb:.1f} MiB)", flush=True)
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Download ONNX models for speaker verification")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if the file already exists",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project root directory (default: detected from script location)",
    )
    args = parser.parse_args()

    # Resolve project root
    if args.project_root:
        root = Path(args.project_root).resolve()
    else:
        # Script is at <root>/scripts/download_models.py
        root = Path(__file__).resolve().parent.parent

    os.chdir(root)
    print(f"Project root: {root}\n", flush=True)

    ok = 0
    fail = 0

    for model in MODELS:
        dest = root / model["dest"]
        print(f"[{model['name']}]", flush=True)

        if dest.exists() and not args.force:
            # Optional hash check on existing file
            if model["sha256"]:
                actual = _sha256_hex(dest)
                if actual == model["sha256"]:
                    print(f"  ✓ already present ({dest.stat().st_size / 1024**2:.1f} MiB)", flush=True)
                    ok += 1
                    continue
                else:
                    print(f"  ⚠ hash mismatch on existing file; re-downloading…", flush=True)
            else:
                print(f"  ✓ already present ({dest.stat().st_size / 1024**2:.1f} MiB)", flush=True)
                ok += 1
                continue

        if _download(model["url"], dest, model["description"], model["sha256"]):
            ok += 1
        else:
            fail += 1
        print()

    # --- summary ---
    print("=" * 60)
    if fail == 0:
        print(f"✓ All {ok} model(s) ready.", flush=True)
        return 0
    else:
        print(f"✗ {fail} model(s) failed to download.", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

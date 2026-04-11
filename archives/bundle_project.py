#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import base64
import hashlib
import datetime as dt
import gzip
from pathlib import Path

EXCLUDE_DIRS = {
    ".git", "__pycache__", ".ipynb_checkpoints", ".venv", "venv", "node_modules", "dist", "build"
}
EXCLUDE_FILES = {
    ".DS_Store",
}
EXCLUDE_SUFFIXES = {
    ".pyc", ".pyo", ".synctex.gz",
}

# Option: include binaries (pdf/png/etc.) in base64. If False, binaries are skipped.
INCLUDE_BINARIES = True

# If including binaries, you may still want to exclude very large files.
MAX_BINARY_BYTES = 50 * 1024 * 1024  # 50 MB


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def is_probably_binary(data: bytes) -> bool:
    # Heuristic: NUL byte usually indicates binary.
    if b"\x00" in data:
        return True
    return False


def read_file(path: Path):
    raw = path.read_bytes()
    file_sha = sha256_bytes(raw)

    # First try to decode as UTF-8 text.
    try:
        text = raw.decode("utf-8")
        # If it decodes, still check for obvious binary signal.
        if is_probably_binary(raw):
            raise UnicodeDecodeError("utf-8", raw, 0, 1, "NUL byte indicates binary")
        return {
            "kind": "text",
            "encoding": "utf-8",
            "sha256": file_sha,
            "content": text,
            "bytes": len(raw),
        }
    except UnicodeDecodeError:
        if not INCLUDE_BINARIES:
            return {
                "kind": "binary_skipped",
                "encoding": None,
                "sha256": file_sha,
                "content": None,
                "bytes": len(raw),
            }

        if len(raw) > MAX_BINARY_BYTES:
            return {
                "kind": "binary_skipped_too_large",
                "encoding": None,
                "sha256": file_sha,
                "content": None,
                "bytes": len(raw),
            }

        b64 = base64.b64encode(raw).decode("ascii")
        return {
            "kind": "binary",
            "encoding": "base64",
            "sha256": file_sha,
            "content": b64,
            "bytes": len(raw),
        }


def should_exclude_dir(dirname: str) -> bool:
    return dirname in EXCLUDE_DIRS


def should_exclude_file(path: Path) -> bool:
    if path.name in EXCLUDE_FILES:
        return True
    for suf in EXCLUDE_SUFFIXES:
        if path.name.endswith(suf):
            return True
    return False


def bundle_project(root_dir: Path, output_json: Path):
    root_dir = root_dir.resolve()

    files_out = []
    total_bytes = 0
    count_text = 0
    count_bin = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # prune excluded dirs in-place
        dirnames[:] = [d for d in dirnames if not should_exclude_dir(d)]

        for fname in filenames:
            fpath = Path(dirpath) / fname
            rel = fpath.relative_to(root_dir).as_posix()

            if should_exclude_file(fpath):
                continue

            info = read_file(fpath)
            total_bytes += info["bytes"]

            if info["kind"] == "text":
                count_text += 1
            elif info["kind"] == "binary":
                count_bin += 1

            files_out.append({
                "path": rel,
                "kind": info["kind"],
                "encoding": info["encoding"],
                "sha256": info["sha256"],
                "bytes": info["bytes"],
                "content": info["content"],
            })

    # Try to infer project version from VERSION file if present
    version_path = root_dir / "VERSION"
    project_version = None
    if version_path.exists():
        try:
            project_version = version_path.read_text(encoding="utf-8").strip()
        except Exception:
            project_version = None

    bundle = {
        "schema_version": "1.0",
        "project_name": "Mem4ristor",
        "project_version": project_version or "unknown",
        "bundle_timestamp_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "root": root_dir.name,
        "stats": {
            "files_total": len(files_out),
            "files_text": count_text,
            "files_binary": count_bin,
            "bytes_total": total_bytes,
            "include_binaries": INCLUDE_BINARIES,
            "max_binary_bytes": MAX_BINARY_BYTES,
            "excluded_dirs": sorted(list(EXCLUDE_DIRS)),
            "excluded_suffixes": sorted(list(EXCLUDE_SUFFIXES)),
        },
        "files": files_out
    }

    output_json.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")

    # Also write gzip version
    gz_path = output_json.with_suffix(output_json.suffix + ".gz")
    with gzip.open(gz_path, "wt", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False)

    print(f"Bundle written: {output_json}")
    print(f"Bundle gz written: {gz_path}")
    print(f"Files: {len(files_out)} (text={count_text}, binary={count_bin}), bytes={total_bytes}")


if __name__ == "__main__":
    # Adjust root_dir if needed
    root_dir = Path(".").resolve()
    output = Path("mem4ristor_v2_bundle.json")

    if not root_dir.exists():
        raise SystemExit(f"Root directory not found: {root_dir.resolve()}")

    bundle_project(root_dir, output)

# SPDX-License-Identifier: LGPL-3.0-or-later
"""Identity manifests for resumable CUDA benchmark results."""

from __future__ import (
    annotations,
)

import hashlib
import json
import os
import subprocess
from typing import (
    Any,
)


def sha256(path: str) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: str, *, content_hash: bool) -> dict[str, Any]:
    """Return an absolute-path, size, timestamp, and optional hash identity."""
    absolute = os.path.abspath(path)
    stat = os.stat(absolute)
    identity: dict[str, Any] = {
        "path": absolute,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if content_hash:
        identity["sha256"] = sha256(absolute)
    return identity


def gpu_identity(index: str) -> dict[str, str]:
    """Return the physical GPU UUID and model name."""
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            index,
            "--query-gpu=uuid,name",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    uuid, name = (field.strip() for field in output.split(",", maxsplit=1))
    return {"index": index, "uuid": uuid, "name": name}


def publish(
    path: str,
    payload: dict[str, Any],
    *,
    fresh: bool,
    has_results: bool,
) -> None:
    """Write a fresh manifest or validate one before resuming results."""
    if has_results and not fresh:
        if not os.path.exists(path):
            raise RuntimeError(f"Cannot resume without benchmark manifest: {path}")
        with open(path, encoding="utf-8") as stream:
            previous = json.load(stream)
        if previous != payload:
            raise RuntimeError(
                f"Benchmark identity changed for {path}; run with BENCH_FRESH=1"
            )
        return

    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)

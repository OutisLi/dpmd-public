# SPDX-License-Identifier: LGPL-3.0-or-later
"""Bind geometry measurements to verified binaries, models and datasets."""

from __future__ import (
    annotations,
)

import hashlib
import json
from pathlib import (
    Path,
)


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def capture_snapshot(root: Path) -> dict:
    """Verify the prepared manifests and fingerprint their complete contents."""
    files = {}
    runtime_manifest = root / "staged_runtime.json"
    systems_manifest = root / "systems/manifest.json"
    expected = {
        Path(record["target"]): record["sha256"]
        for record in json.loads(runtime_manifest.read_text())
    }
    for metadata in sorted((root / "models").glob("*.json")):
        record = json.loads(metadata.read_text())
        expected[metadata.with_suffix(".pt2")] = record["sha256"]
        files[str(metadata.relative_to(root))] = file_digest(metadata)
    for record in json.loads(systems_manifest.read_text()):
        expected[root / "systems" / Path(record["path"]).name] = record["sha256"]
    for path, reference in expected.items():
        actual = file_digest(path)
        assert actual == reference, (path, reference, actual)
        files[str(path.relative_to(root))] = actual
    for manifest in (runtime_manifest, systems_manifest):
        files[str(manifest.relative_to(root))] = file_digest(manifest)
    encoded = json.dumps(files, sort_keys=True).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "files": files}


def verify_snapshot(root: Path, output: Path, snapshot: dict) -> None:
    """Confirm that the files stayed unchanged throughout a completed run."""
    assert capture_snapshot(root) == snapshot, root
    (output / "snapshot_verified.json").write_text(
        json.dumps({"sha256": snapshot["sha256"]}) + "\n"
    )

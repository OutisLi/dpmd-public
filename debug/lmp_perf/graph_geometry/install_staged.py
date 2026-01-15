# SPDX-License-Identifier: LGPL-3.0-or-later
"""Install staged files by atomic replacement and retain the original files."""

from __future__ import (
    annotations,
)

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import (
    Path,
)


def fingerprint(path: Path) -> str | None:
    if path.is_symlink():
        return "symlink:" + os.readlink(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()
    stage = args.stage.resolve()
    prefix = args.prefix.resolve()
    backup = args.backup.resolve()
    backup.mkdir(parents=True, exist_ok=False)
    records = []
    for source in sorted(stage.rglob("*")):
        if not source.is_file() and not source.is_symlink():
            continue
        relative = source.relative_to(stage)
        target = prefix / relative
        before, after = fingerprint(target), fingerprint(source)
        record = {"path": str(target), "before": before, "after": after}
        if before != after:
            target.parent.mkdir(parents=True, exist_ok=True)
            if before is not None:
                saved = backup / relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, saved, follow_symlinks=False)
            descriptor, temporary = tempfile.mkstemp(
                prefix=".install-", dir=target.parent
            )
            os.close(descriptor)
            if source.is_symlink():
                os.unlink(temporary)
                os.symlink(os.readlink(source), temporary)
            else:
                shutil.copy2(source, temporary)
            os.replace(temporary, target)
        assert fingerprint(target) == after, target
        records.append(record)
    (backup / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print("Verified installed files:", len(records))  # noqa: T201


if __name__ == "__main__":
    main()

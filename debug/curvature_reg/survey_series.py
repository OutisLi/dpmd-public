# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201
"""Evaluate a fixed checkpoint manifest, partitioned across independent GPUs."""

from __future__ import (
    annotations,
)

import argparse
import json
import subprocess
import sys
from pathlib import (
    Path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    tasks = json.loads(args.manifest.read_text())["checkpoints"][
        args.shard :: args.shards
    ]
    script = Path(__file__).resolve().parent / "dimer_survey.py"
    failed = []
    print(
        f"STARTUP {len(tasks)} checkpoints in shard {args.shard}/{args.shards}",
        flush=True,
    )
    for item in tasks:
        checkpoint = Path(item["checkpoint"])
        output = (
            Path(item["output"])
            if "output" in item
            else checkpoint.parent / f"survey_full_{item['step']}.npz"
        )
        if output.exists() and output.with_suffix(".json").exists():
            if len(json.loads(output.with_suffix(".json").read_text())["pairs"]) != 24:
                raise ValueError(f"Expected 24 pairs in existing survey {output}")
            print(f"PRESENT {item['run']} {item['step']}", flush=True)
            continue
        with output.with_suffix(".log").open("w") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    str(script),
                    str(checkpoint),
                    f"{item['run']} {item['step']}",
                    "--out",
                    str(output),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            failed.append(item)
            print(
                f"FAILED {item['run']} {item['step']}: {result.returncode}", flush=True
            )
            continue
        record = json.loads(output.with_suffix(".json").read_text())
        if len(record["pairs"]) != 24:
            raise ValueError(f"Expected 24 pairs in {output}")
        print(
            f"DONE {item['run']} {item['step']}: "
            f"{sum(not p['tail_pass'] for p in record['pairs'])} tails over 0.05",
            flush=True,
        )
    if failed:
        raise RuntimeError(f"Failed checkpoint surveys: {failed}")
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()

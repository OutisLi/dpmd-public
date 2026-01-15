# SPDX-License-Identifier: LGPL-3.0-or-later
"""Combine complete cases from a preserved partial batch and a completed batch."""

from __future__ import (
    annotations,
)

import argparse
import json
import statistics
from pathlib import (
    Path,
)
from typing import (
    Any,
)

from artifact_snapshot import (
    capture_snapshot,
    file_digest,
)
from summarize_matrix import (
    read_cases,
    summarize,
)


def read_run(path: Path, completed: bool) -> list[dict[str, Any]]:
    """Require four complete rotated repetitions for every retained case."""
    if completed:
        for subdirectory in (Path(), Path("baseline")):
            snapshot = json.loads((path / subdirectory / "snapshot.json").read_text())
            marker = json.loads(
                (path / subdirectory / "snapshot_verified.json").read_text()
            )
            assert marker["sha256"] == snapshot["sha256"], path
    else:
        assert (path / "interrupted.json").is_file(), path
        assert not (path / "snapshot_verified.json").exists(), path
    cases = json.loads((path / "summary.json").read_text())
    runs = [json.loads(line) for line in (path / "runs.jsonl").read_text().splitlines()]
    sides = json.loads((path / "configuration.json").read_text())["sides"].split(",")
    assert len(sides) == 4 and set(sides) == {
        "before_contract",
        "vector",
        "indexed",
        "auto",
    }
    order = [
        (repeat, side)
        for repeat in range(4)
        for side in sides[repeat:] + sides[:repeat]
    ]
    for case in cases:
        actual = [
            record
            for record in runs
            if record["case"] == case["case"] and not record["pilot"]
        ]
        assert actual == case["records"] and len(actual) == 16, case["case"]
        assert [(record["repeat"], record["side"]) for record in actual] == order, case[
            "case"
        ]
        inputs = {
            file_digest(
                path
                / case["case"]
                / f"{record['repeat']}_{record['side']}"
                / "input.lmp"
            )
            for record in actual
        }
        assert len(inputs) == 1, case["case"]
        assert set(case["samples_ms"]) == {
            "before_contract",
            "vector",
            "indexed",
            "auto",
        }
        for side, values in case["samples_ms"].items():
            records = [record for record in actual if record["side"] == side]
            assert [record["repeat"] for record in records] == list(range(4)), case[
                "case"
            ]
            assert [record["step_ms"] for record in records] == values, case["case"]
            assert statistics.median(values) == case["medians_ms"][side], case["case"]
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--completed", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifacts.resolve()
    paths = [args.partial.resolve(), args.completed.resolve()]
    snapshots = [json.loads((path / "snapshot.json").read_text()) for path in paths]
    baselines = [
        json.loads((path / "baseline/snapshot.json").read_text()) for path in paths
    ]
    origin = json.loads((root / "strict_contract/origin_snapshot.json").read_text())
    restored = json.loads((root / "strict_restoration.json").read_text())
    assert snapshots[0] == snapshots[1] == origin == capture_snapshot(root)
    assert restored["restored_snapshot_sha256"] == origin["sha256"]
    assert baselines[0] == baselines[1] == capture_snapshot(root / "contract_baseline")
    repo = Path(__file__).resolve().parents[3]
    for name, expected in restored["restored_source_hashes"].items():
        assert file_digest(repo / name) == expected, name
    batches = [read_run(paths[0], False), read_run(paths[1], True)]
    assert [len(batch) for batch in batches] == [24, 96]
    assert {case["grade"] for case in batches[0]} == {"nano"}
    assert {case["grade"] for case in batches[1]} == {"mini", "neo", "air", "plus"}
    cases = batches[0] + batches[1]
    verification = read_cases(args.verification.resolve())
    assert len(cases) == len({case["case"] for case in cases}) == 120
    assert {case["case"] for case in cases} == verification.keys()
    for case in cases:
        assert case["baseline_snapshot_sha256"] == baselines[0]["sha256"]
        summarize(case, verification[case["case"]])
    source_runs = []
    for path, batch, completed in zip(paths, batches, (False, True), strict=True):
        names = [
            "summary.json",
            "runs.jsonl",
            "configuration.json",
            "snapshot.json",
            "baseline/snapshot.json",
        ]
        names += (
            ["snapshot_verified.json", "baseline/snapshot_verified.json"]
            if completed
            else ["interrupted.json"]
        )
        source_runs.append(
            {
                "path": str(path),
                "completed_batch": completed,
                "retained_cases": [case["case"] for case in batch],
                "files": {name: file_digest(path / name) for name in names},
            }
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "baseline").mkdir()
    records = {
        "summary.json": cases,
        "snapshot.json": snapshots[0],
        "snapshot_verified.json": {"sha256": snapshots[0]["sha256"]},
        "baseline/snapshot.json": baselines[0],
        "baseline/snapshot_verified.json": {"sha256": baselines[0]["sha256"]},
        "source_runs.json": {
            "sources": source_runs,
            "partial_batch_evidence": {
                "origin_snapshot": str(root / "strict_contract/origin_snapshot.json"),
                "restoration": str(root / "strict_restoration.json"),
                "restoration_sha256": file_digest(root / "strict_restoration.json"),
                "end_snapshot_matches_restored_origin": True,
            },
        },
    }
    for name, record in records.items():
        (output / name).write_text(json.dumps(record, indent=2) + "\n")
    print(f"Verified {len(cases)} complete cases from two preserved batches")  # noqa: T201


if __name__ == "__main__":
    main()

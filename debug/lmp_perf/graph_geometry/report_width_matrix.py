# SPDX-License-Identifier: LGPL-3.0-or-later
"""Verify and summarize automatic-width, fixed-32 and original-vector MD timings."""

from __future__ import (
    annotations,
)

import argparse
import csv
import json
import re
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


def read_completed(path: Path, snapshots: dict[str, dict]) -> list[dict[str, Any]]:
    """Verify timing records, launch order and all three immutable runtimes."""
    for name, expected in snapshots.items():
        directory = path / name
        assert json.loads((directory / "snapshot.json").read_text()) == expected, (
            directory
        )
        marker = json.loads((directory / "snapshot_verified.json").read_text())
        assert marker["sha256"] == expected["sha256"], directory
    configuration = json.loads((path / "configuration.json").read_text())
    assert configuration["baseline_layout"] == "auto"
    assert configuration["original_layout"] == "vector"
    sides = configuration["sides"].split(",")
    assert sides == ["before_contract", "auto", "original_vector"]
    repeats = configuration["repeats"]
    assert repeats >= 3 and repeats % len(sides) == 0
    order = [
        (repeat, side)
        for repeat in range(repeats)
        for side in sides[repeat % len(sides) :] + sides[: repeat % len(sides)]
    ]
    cases = json.loads((path / "summary.json").read_text())
    runs = [json.loads(line) for line in (path / "runs.jsonl").read_text().splitlines()]
    for case in cases:
        actual = [
            record
            for record in runs
            if record["case"] == case["case"] and not record["pilot"]
        ]
        assert actual == case["records"], case["case"]
        assert [(record["repeat"], record["side"]) for record in actual] == order, case[
            "case"
        ]
        inputs = {
            re.sub(
                r'"[^"\n]+\.pt2"',
                '"MODEL"',
                (
                    path
                    / case["case"]
                    / f"{record['repeat']}_{record['side']}"
                    / "input.lmp"
                ).read_text(),
            )
            for record in actual
        }
        assert len(inputs) == 1, case["case"]
        assert all(record["danger"] == 0 for record in actual), case["case"]
        for record in actual:
            directory = path / case["case"] / f"{record['repeat']}_{record['side']}"
            screen = (directory / "screen.txt").read_text()
            loops = list(
                re.finditer(
                    r"Loop time of ([\d.eE+-]+) on \d+ procs for (\d+) steps with (\d+) atoms",
                    screen,
                )
            )
            last = loops[-1]
            assert int(last[2]) == record["steps"] == case["steps"]
            assert int(last[3]) == case["atoms"]
            assert 1000 * float(last[1]) / record["steps"] == record["step_ms"]
            tail = screen[last.end() :]
            assert (
                int(re.search(r"Neighbor list builds = (\d+)", tail)[1])
                == record["builds"]
            )
            assert (
                int(re.search(r"Dangerous builds = (\d+)", tail)[1]) == record["danger"]
            )
            root_key, layout = {
                "before_contract": ("baseline_artifacts", "auto"),
                "auto": ("artifacts", "auto"),
                "original_vector": ("original_artifacts", "vector"),
            }[record["side"]]
            root = Path(configuration[root_key])
            runtime = "fork" if case["flavor"] == "spin" else "public"
            libraries = json.loads((directory / "libraries.json").read_text())
            assert libraries["LAMMPS_PLUGIN_PATH"] == str(
                root / f"{runtime}_{layout}_plugin"
            )
            assert libraries["DP_BACKEND_PLUGIN_PATH"] == str(root / "integrated")
            command = json.loads((directory / "command.json").read_text())
            assert command[0] == str(root / f"{runtime}_lammps/lmp")
        for repeat in range(repeats):
            assert (
                len(
                    {
                        record["builds"]
                        for record in actual
                        if record["repeat"] == repeat
                    }
                )
                == 1
            )
        for side in sides:
            values = [record["step_ms"] for record in actual if record["side"] == side]
            assert values == case["samples_ms"][side]
            assert statistics.median(values) == case["medians_ms"][side]
    return cases


def summarize(case: dict[str, Any]) -> dict[str, Any]:
    medians, samples = case["medians_ms"], case["samples_ms"]
    cost = 100 * (medians["auto"] / medians["before_contract"] - 1)
    choices = [
        item["selected"]
        for record in case["records"]
        if record["side"] == "auto"
        for item in record["calibration"]
    ]
    startup = [
        item["startup_s"]
        for record in case["records"]
        if record["side"] == "auto"
        for item in record["calibration"]
    ]
    return {
        key: case[key]
        for key in ("case", "grade", "flavor", "zbl", "elements", "atoms", "steps")
    } | {
        "fixed32_ms": medians["before_contract"],
        "automatic_ms": medians["auto"],
        "original_vector_ms": medians["original_vector"],
        "automatic_overhead_percent": cost,
        "net_time_reduction_percent": 100
        * (1 - medians["auto"] / medians["original_vector"]),
        "overhead_exceeds_3_percent": cost > 3,
        "all_automatic_samples_exceed_all_fixed32_by_3_percent": min(samples["auto"])
        > 1.03 * max(samples["before_contract"]),
        "fixed32_min_ms": min(samples["before_contract"]),
        "fixed32_max_ms": max(samples["before_contract"]),
        "automatic_min_ms": min(samples["auto"]),
        "automatic_max_ms": max(samples["auto"]),
        "selected_indexed": choices.count("indexed"),
        "selected_vector": choices.count("vector"),
        "calibration_min_s": min(startup),
        "calibration_max_s": max(startup),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--timings", type=Path, nargs="+", required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    work = args.artifacts.resolve()
    verification = args.verification.resolve()
    snapshots = {
        "": capture_snapshot(work),
        "baseline": capture_snapshot(work.parent / "natural32"),
        "original": capture_snapshot(work.parent / "contract_baseline"),
    }
    for name in ("", "baseline"):
        assert (
            json.loads((verification / name / "snapshot.json").read_text())
            == snapshots[name]
        )
        assert (
            json.loads((verification / name / "snapshot_verified.json").read_text())[
                "sha256"
            ]
            == snapshots[name]["sha256"]
        )
    verified = json.loads((verification / "summary.json").read_text())
    assert len(verified) == 120
    assert all(
        value["passed"]
        for case in verified
        for comparison in case["comparisons"].values()
        for value in comparison.values()
    )
    batches = [read_completed(path.resolve(), snapshots) for path in args.timings]
    cases = [case for batch in batches for case in batch]
    assert len(cases) == len({case["case"] for case in cases}) == 120
    assert {case["case"] for case in cases} == {case["case"] for case in verified}
    records = [summarize(case) for case in cases]
    grades = {}
    for grade in ("nano", "mini", "neo", "air", "plus"):
        selected = [record for record in records if record["grade"] == grade]
        assert len(selected) == 24
        grades[grade] = {
            metric: {
                "minimum": min(values),
                "median": statistics.median(values),
                "maximum": max(values),
            }
            for metric in ("automatic_overhead_percent", "net_time_reduction_percent")
            for values in ([record[metric] for record in selected],)
        }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(cases, indent=2) + "\n")
    (output / "comparison.json").write_text(json.dumps(records, indent=2) + "\n")
    (output / "grades.json").write_text(json.dumps(grades, indent=2) + "\n")
    (output / "source_runs.json").write_text(
        json.dumps(
            [
                {
                    "path": str(path.resolve()),
                    "cases": [case["case"] for case in batch],
                    "files": {
                        name: file_digest(path / name)
                        for name in (
                            "summary.json",
                            "runs.jsonl",
                            "configuration.json",
                            "snapshot.json",
                        )
                    },
                }
                for path, batch in zip(args.timings, batches, strict=True)
            ],
            indent=2,
        )
        + "\n"
    )
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output / "snapshots.json").write_text(json.dumps(snapshots, indent=2) + "\n")
    print(json.dumps(grades, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()

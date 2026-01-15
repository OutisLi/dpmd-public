# SPDX-License-Identifier: LGPL-3.0-or-later
"""Summarize accepted geometry timings and their exact-output checks."""

from __future__ import (
    annotations,
)

import argparse
import csv
import json
import statistics
from pathlib import (
    Path,
)
from typing import (
    Any,
)


def read_cases(path: Path) -> dict[str, dict[str, Any]]:
    records = json.loads(path.read_text())
    verified = json.loads((path.parent / "snapshot_verified.json").read_text())
    assert {record["snapshot_sha256"] for record in records} == {verified["sha256"]}, (
        path
    )
    if "baseline_snapshot_sha256" in records[0]:
        baseline = json.loads(
            (path.parent / "baseline/snapshot_verified.json").read_text()
        )
        assert {record["baseline_snapshot_sha256"] for record in records} == {
            baseline["sha256"]
        }, path
    indexed = {record["case"]: record for record in records}
    assert len(indexed) == len(records) == 120, path
    return indexed


def summarize(timing: dict[str, Any], verification: dict[str, Any]) -> dict[str, Any]:
    assert verification["bit_exact"], verification["case"]
    assert timing["snapshot_sha256"] == verification["snapshot_sha256"]
    checked = verification["records"]
    assert len(checked) == 3 and {record["side"] for record in checked} == {
        "vector",
        "indexed",
        "auto",
    }
    for field in ("before_sha256", "after_sha256", "thermo"):
        assert len({json.dumps(record[field]) for record in checked}) == 1, (
            verification["case"]
        )
    medians = timing["medians_ms"]
    samples = timing["samples_ms"]
    sides = {"vector", "indexed", "auto"}
    if "baseline_snapshot_sha256" in timing:
        sides.add("before_contract")
    assert set(medians) == sides
    assert all(len(values) >= 3 for values in samples.values())
    records = timing["records"]
    assert all(not record["pilot"] and record["danger"] == 0 for record in records)
    for repeat in {record["repeat"] for record in records}:
        group = [record for record in records if record["repeat"] == repeat]
        assert (
            len(group) == len(sides) and {record["side"] for record in group} == sides
        )
        assert len({record["builds"] for record in group}) == 1
    calibrations = [
        calibration
        for record in records
        if record["side"] == "auto"
        for calibration in record["calibration"]
    ]
    assert calibrations
    best = min(("vector", "indexed"), key=medians.__getitem__)
    result = {
        key: timing[key]
        for key in ("case", "grade", "flavor", "zbl", "elements", "atoms", "steps")
    }
    for side in medians:
        values = samples[side]
        result[f"{side}_ms"] = medians[side]
        result[f"{side}_min_ms"] = min(values)
        result[f"{side}_max_ms"] = max(values)
        result[f"{side}_mad_ms"] = statistics.median(
            abs(value - medians[side]) for value in values
        )
    result.update(
        indexed_reduction_percent=100 * (1 - medians["indexed"] / medians["vector"]),
        auto_reduction_percent=100 * (1 - medians["auto"] / medians["vector"]),
        best_fixed=best,
        auto_extra_vs_best_percent=100 * (medians["auto"] / medians[best] - 1),
        auto_choices=",".join(item["selected"] for item in calibrations),
        calibration_median_s=statistics.median(
            item["startup_s"] for item in calibrations
        ),
        calibration_max_s=max(item["startup_s"] for item in calibrations),
        bit_exact=True,
        snapshot_sha256=timing["snapshot_sha256"],
        repeats=len(samples["vector"]),
    )
    if "before_contract" in medians:
        result.update(
            net_auto_reduction_percent=100
            * (1 - medians["auto"] / medians["before_contract"]),
            net_indexed_reduction_percent=100
            * (1 - medians["indexed"] / medians["before_contract"]),
            vector_contract_change_percent=100
            * (medians["vector"] / medians["before_contract"] - 1),
            baseline_snapshot_sha256=timing["baseline_snapshot_sha256"],
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    timings = read_cases(args.timing)
    verification = read_cases(args.verification)
    assert timings.keys() == verification.keys()
    rows = [summarize(timing, verification[name]) for name, timing in timings.items()]
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "matrix.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["device", *rows[0]])
        writer.writeheader()
        writer.writerows({"device": args.device, **row} for row in rows)
    report = {
        "device": args.device,
        "timing": str(args.timing.resolve()),
        "verification": str(args.verification.resolve()),
        "cases": rows,
        "auto_over_3_percent_slower_than_best_fixed": [
            row["case"] for row in rows if row["auto_extra_vs_best_percent"] > 3
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(  # noqa: T201
        json.dumps(
            {key: value for key, value in report.items() if key != "cases"}, indent=2
        )
    )


if __name__ == "__main__":
    main()

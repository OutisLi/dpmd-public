# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Pooled step times of several ``bench_sizes.py`` campaigns, and their speedup.

Every run of a point in the given campaigns joins one pool, whose median is the
step time of the point; the interquartile range uses the exclusive quartiles of
``statistics.quantiles``. Two sets of campaigns, a baseline and a candidate,
give the speedup of every point as the ratio of their medians.

Usage
-----
    python pool_sizes.py --base base_trained,stock_head,stock6_head \
        --candidate stock_cur,stock6_cur
"""

from __future__ import (
    annotations,
)

import argparse
import csv
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
GRADES = ("nano", "mini", "neo", "air", "plus")
SIZES = (512, 10_296, 100_800)


def pooled(labels: list[str]) -> dict[tuple[str, int], list[float]]:
    """Return the step times in ms of every run of every point of ``labels``."""
    runs: dict[tuple[str, int], list[float]] = {}
    for label in labels:
        path = os.path.join(HERE, "results", f"sizes_{label}.csv")
        with open(path, encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                n_atoms = int(row["n_atoms"])
                runs.setdefault((row["grade"], n_atoms), []).extend(
                    n_atoms / float(value) for value in row["runs"].split()
                )
    return runs


def spread(values: list[float]) -> float:
    """Return the interquartile range relative to the median."""
    lower, _, upper = statistics.quantiles(values, n=4)
    return (upper - lower) / statistics.median(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, help="comma-separated labels")
    parser.add_argument("--candidate", required=True, help="comma-separated labels")
    args = parser.parse_args()
    base = pooled(args.base.split(","))
    candidate = pooled(args.candidate.split(","))
    for grade in GRADES:
        cells = []
        for n_atoms in SIZES:
            before = statistics.median(base[(grade, n_atoms)])
            after = statistics.median(candidate[(grade, n_atoms)])
            cells.append(f"{before:.3f} → {after:.3f} ({before / after:.2f}x)")
        print(f"| {grade.capitalize():5s} | " + " | ".join(cells) + " |")  # noqa: T201
    for name, runs in (("base", base), ("candidate", candidate)):
        count = min(len(values) for values in runs.values())
        worst = max(runs, key=lambda point: spread(runs[point]))
        print(  # noqa: T201
            f"{name}: {count} runs or more per point, largest interquartile range "
            f"{spread(runs[worst]):.1%} at {worst[0]} {worst[1]}"
        )


if __name__ == "__main__":
    main()

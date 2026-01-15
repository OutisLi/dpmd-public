# SPDX-License-Identifier: LGPL-3.0-or-later
"""Summarize an A/B timing written by run_ab_bench.sh.

For every grade and size the script prints the per-round step times of both
builds, their medians, the relative change of B against A, and the median of
the per-round paired differences, which cancels drift between rounds.

Usage
-----
    python summarize_ab.py LABEL_A LABEL_B ROUNDS
"""

from __future__ import (
    annotations,
)

import argparse
import csv
import os
import statistics

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def step_times(label: str, rounds: int) -> dict[tuple[str, int], list[float]]:
    """Return the step time in ms of every (grade, size) for each round."""
    times: dict[tuple[str, int], list[float]] = {}
    for index in range(rounds):
        path = os.path.join(RESULTS, f"newton_{label}_r{index}.csv")
        with open(path, newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = (row["grade"], int(row["n_atoms"]))
                times.setdefault(key, []).append(float(row["ms_per_step"]))
    return times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label_a")
    parser.add_argument("label_b")
    parser.add_argument("rounds", type=int)
    args = parser.parse_args()
    times_a = step_times(args.label_a, args.rounds)
    times_b = step_times(args.label_b, args.rounds)
    print(  # noqa: T201
        f"| Grade | Atoms | {args.label_a} ms | {args.label_b} ms | change | paired diff ms |\n"
        "|---|---:|---:|---:|---:|---:|"
    )
    for key in sorted(times_a, key=lambda k: (k[0], k[1])):
        a, b = times_a[key], times_b[key]
        median_a, median_b = statistics.median(a), statistics.median(b)
        paired = statistics.median(y - x for x, y in zip(a, b, strict=True))
        print(  # noqa: T201
            f"| {key[0]} | {key[1]} | {median_a:.4f} ({' '.join(f'{v:.3f}' for v in a)}) | "
            f"{median_b:.4f} ({' '.join(f'{v:.3f}' for v in b)}) | "
            f"{100 * (median_b - median_a) / median_a:+.1f}% | {paired:+.4f} |"
        )


if __name__ == "__main__":
    main()

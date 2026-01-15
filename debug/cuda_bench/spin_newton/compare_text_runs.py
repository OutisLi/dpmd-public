# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compare two LAMMPS runs written with %.17g: dumps column by column, thermo row by row.

Every value is printed with 17 significant digits, which identifies a double
uniquely, so identical text means bitwise identical values. For differing
values the script reports the largest absolute difference per column.

Usage
-----
    python compare_text_runs.py --dumps a.dump b.dump --logs a.log b.log
"""

from __future__ import (
    annotations,
)

import argparse
import re

import numpy as np


def dump_frames(path: str) -> list[tuple[int, list[str], np.ndarray, list[list[str]]]]:
    """Return ``(step, column names, numeric rows, text rows)`` per frame, sorted by ID."""
    with open(path, encoding="utf-8") as stream:
        lines = stream.read().splitlines()
    frames, i = [], 0
    while i < len(lines):
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue
        step = int(lines[i + 1])
        count = int(lines[i + 3])
        names = lines[i + 8].split()[2:]
        text = [line.split() for line in lines[i + 9 : i + 9 + count]]
        text.sort(key=lambda row: int(row[0]))
        frames.append((step, names, np.array(text, dtype=float), text))
        i += 9 + count
    return frames


def thermo_rows(path: str) -> tuple[list[str], list[list[str]]]:
    """Return the thermo header and every thermo row of a log as text tokens."""
    with open(path, encoding="utf-8") as stream:
        text = stream.read()
    header, rows = [], []
    for match in re.finditer(r"^\s*Step\s.*$", text, re.MULTILINE):
        header = match.group(0).split()
        block = text[match.end() :].split("Loop time")[0]
        rows += [line.split() for line in block.strip().splitlines() if line.strip()]
    return header, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps", nargs=2, required=True)
    parser.add_argument("--logs", nargs=2, required=True)
    args = parser.parse_args()

    identical = True
    frames_a, frames_b = (dump_frames(path) for path in args.dumps)
    if [f[0] for f in frames_a] != [f[0] for f in frames_b]:
        raise ValueError("the dumps hold different frame sequences")
    for (step, names, values_a, text_a), (_, _, values_b, text_b) in zip(
        frames_a, frames_b, strict=True
    ):
        if text_a == text_b:
            continue
        identical = False
        delta = np.abs(values_a - values_b).max(axis=0)
        print(  # noqa: T201
            f"step {step}: "
            + " ".join(f"{n}={d:.3e}" for n, d in zip(names, delta, strict=True) if d)
        )
    header_a, rows_a = thermo_rows(args.logs[0])
    header_b, rows_b = thermo_rows(args.logs[1])
    if header_a != header_b or len(rows_a) != len(rows_b):
        raise ValueError("the logs hold different thermo layouts")
    for row_a, row_b in zip(rows_a, rows_b, strict=True):
        if row_a != row_b:
            identical = False
            delta = [
                abs(float(x) - float(y)) for x, y in zip(row_a, row_b, strict=True)
            ]
            print(  # noqa: T201
                f"thermo step {row_a[0]}: "
                + " ".join(
                    f"{n}={d:.3e}" for n, d in zip(header_a, delta, strict=True) if d
                )
            )
    print(  # noqa: T201
        f"{len(frames_a)} dump frames, {len(rows_a)} thermo rows: "
        + ("bitwise identical" if identical else "NOT bitwise identical")
    )


if __name__ == "__main__":
    main()

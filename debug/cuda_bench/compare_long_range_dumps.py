# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201
r"""Compare two directories of long-range reproducibility dumps.

The dumps are those of ``bench_long_range.py --dump``: the positions and
forces after a short trajectory, one file per case and size. A file pair is
reported as identical when the files agree byte for byte; otherwise the
largest deviation of a position in Å and of a force relative to the largest
force of the reference are given. Dumps of two builds whose short-range
model is the same agree bit for bit in the ``*_sr`` cases.

Usage
-----
    python compare_long_range_dumps.py results/.../long_range_dump_a \
        results/.../long_range_dump_final_a
"""

from __future__ import (
    annotations,
)

import argparse
import filecmp
import os

import numpy as np


def read_atoms(path: str) -> np.ndarray:
    """Return the rows of a dump sorted by atom id, with shape (N, 8)."""
    with open(path) as handle:
        lines = handle.read().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("ITEM: ATOMS"))
    rows = np.loadtxt(lines[start + 1 :], dtype=np.float64, ndmin=2)
    return rows[np.argsort(rows[:, 0])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference", help="directory of the reference dumps")
    parser.add_argument("candidate", help="directory of the compared dumps")
    args = parser.parse_args()
    names = sorted(n for n in os.listdir(args.reference) if n.endswith(".dump"))
    missing = [name for name in names if not os.path.exists(f"{args.candidate}/{name}")]
    if missing:
        raise SystemExit(f"missing in {args.candidate}: {', '.join(missing)}")
    for name in names:
        reference = os.path.join(args.reference, name)
        candidate = os.path.join(args.candidate, name)
        if filecmp.cmp(reference, candidate, shallow=False):
            print(f"{name:20s} identical")
            continue
        ref, cand = read_atoms(reference), read_atoms(candidate)
        if ref.shape != cand.shape or not np.array_equal(ref[:, :2], cand[:, :2]):
            print(f"{name:20s} different atoms")
            continue
        position = np.abs(cand[:, 2:5] - ref[:, 2:5]).max()
        force = np.abs(cand[:, 5:8] - ref[:, 5:8]).max() / np.abs(ref[:, 5:8]).max()
        print(f"{name:20s} position {position:.3e} A  force {force:.3e} of the largest")
    print(f"{len(names)} dumps compared")


if __name__ == "__main__":
    main()

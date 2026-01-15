# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compare two Newtonian spin runs atom by atom and step by step.

Both runs dump ``id x y z vx vy vz sp spx spy spz fx fy fz fmx fmy fmz`` with
``in_spin_newton.lammps`` (``-var dump N``), sorted by atom ID. Frames are
matched in order; positions are compared under the minimum-image convention,
and the spin is compared as the Cartesian vector ``S = sp * (spx, spy, spz)``,
whose next value also carries the spin velocity of the previous half step. The
thermo blocks of the two logs are compared column by column.

Usage
-----
    python compare_runs.py run_a.dump run_b.dump --logs log_a.lammps log_b.lammps
"""

from __future__ import (
    annotations,
)

import argparse
import re

import numpy as np

QUANTITIES = {
    "x": slice(1, 4),
    "v": slice(4, 7),
    "S": None,
    "f": slice(11, 14),
    "fm": slice(14, 17),
}


def read_frames(path: str) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Return ``(step, box lengths, rows sorted by ID)`` for every dump frame."""
    with open(path, encoding="utf-8") as stream:
        lines = stream.read().splitlines()
    frames, i = [], 0
    while i < len(lines):
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue
        step = int(lines[i + 1])
        count = int(lines[i + 3])
        bounds = np.array(
            [[float(v) for v in lines[i + 5 + k].split()[:2]] for k in range(3)]
        )
        rows = np.array(
            [[float(v) for v in line.split()] for line in lines[i + 9 : i + 9 + count]]
        )
        frames.append((step, bounds[:, 1] - bounds[:, 0], rows[np.argsort(rows[:, 0])]))
        i += 9 + count
    return frames


def read_thermo(path: str) -> np.ndarray:
    """Return every thermo row of a log as one array."""
    with open(path, encoding="utf-8") as stream:
        text = stream.read()
    rows = []
    for header in re.finditer(r"^\s*Step\s+KinEng.*$", text, re.MULTILINE):
        block = text[header.end() :].split("Loop time")[0]
        rows += [
            [float(v) for v in line.split()] for line in block.strip().splitlines()
        ]
    return np.array(rows)


def frame_differences(
    a: np.ndarray, b: np.ndarray, box: np.ndarray
) -> dict[str, float]:
    """Return the largest absolute difference of every compared quantity."""
    if not np.array_equal(a[:, 0], b[:, 0]):
        raise ValueError("the two frames hold different atom IDs")
    result = {}
    for name, columns in QUANTITIES.items():
        if name == "S":
            delta = a[:, 8:11] * a[:, 7:8] - b[:, 8:11] * b[:, 7:8]
        else:
            delta = a[:, columns] - b[:, columns]
            if name == "x":
                delta -= box * np.round(delta / box)
        result[name] = float(np.abs(delta).max())
    return result


def main() -> None:
    """Print per-frame differences and a bitwise verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_a")
    parser.add_argument("dump_b")
    parser.add_argument("--logs", nargs=2, metavar=("LOG_A", "LOG_B"))
    args = parser.parse_args()

    frames_a, frames_b = read_frames(args.dump_a), read_frames(args.dump_b)
    if [f[0] for f in frames_a] != [f[0] for f in frames_b]:
        raise ValueError("the two dumps hold different frame sequences")
    identical = True
    for (step, box, a), (_, _, b) in zip(frames_a, frames_b, strict=True):
        diff = frame_differences(a, b, box)
        identical &= all(value == 0.0 for value in diff.values())
        print(  # noqa: T201
            f"step {step:6d} " + " ".join(f"max|d{k}|={v:.3e}" for k, v in diff.items())
        )
    if args.logs:
        thermo_a, thermo_b = (read_thermo(path) for path in args.logs)
        delta = np.abs(thermo_a - thermo_b).max(axis=0)
        identical &= bool((delta == 0.0).all())
        names = ("step", "ke", "ske", "pe", "etotal", "temp", "press")
        print(  # noqa: T201
            "thermo max|d|: "
            + " ".join(f"{n}={v:.3e}" for n, v in zip(names, delta, strict=False))
        )
    print("bitwise identical" if identical else "NOT bitwise identical")  # noqa: T201


if __name__ == "__main__":
    main()

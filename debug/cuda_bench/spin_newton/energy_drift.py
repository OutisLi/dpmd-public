# SPDX-License-Identifier: LGPL-3.0-or-later
"""Summarize NVE energy conservation of Newtonian spin runs.

Each log holds the thermo output ``step ke ske pe etotal temp press`` of one
``in_spin_newton.lammps`` run. For every log the script reports the largest
deviation of ``etotal`` from its initial value, the end-point drift, the slope
of a least-squares line through ``etotal(t)``, and the amount of energy moved
between the potential and the spin kinetic energy, which sets the scale the
deviation is to be compared with. Logs of one grade at successive halvings of
the timestep also give the ratio of their largest deviations, which is four
for a second-order integrator with consistent forces.

Usage
-----
    python energy_drift.py logs/nve_nano_512_n*.log logs/nve_plus_512_n*.log
"""

from __future__ import (
    annotations,
)

import argparse
import re

import numpy as np


def read_log(path: str) -> tuple[int, float, np.ndarray]:
    """Return the atom count, the timestep in ps and the last run's thermo rows."""
    with open(path, encoding="utf-8") as stream:
        text = stream.read()
    timestep = float(re.findall(r"^timestep\s+(\S+)", text, re.MULTILINE)[-1])
    atoms = int(re.findall(r"with (\d+) atoms", text)[-1])
    header = list(re.finditer(r"^\s*Step\s+KinEng.*$", text, re.MULTILINE))[-1]
    block = text[header.end() :].split("Loop time")[0]
    rows = [
        [float(value) for value in line.split()] for line in block.strip().splitlines()
    ]
    return atoms, timestep, np.array(rows)


def main() -> None:
    """Print one conservation line per log and the timestep-halving ratios."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    args = parser.parse_args()

    previous: dict[str, float] = {}
    for path in args.logs:
        atoms, timestep, rows = read_log(path)
        step, ske, pe, etotal = rows[:, 0], rows[:, 2], rows[:, 3], rows[:, 4]
        time = (step - step[0]) * timestep
        deviation = etotal - etotal[0]
        largest = np.abs(deviation).max()
        slope = np.polyfit(time, etotal, 1)[0] if len(time) > 1 else float("nan")
        moved = max(ske.max() - ske.min(), pe.max() - pe.min())
        grade = re.sub(r"_n\d+\.log$", "", path.split("/")[-1])
        ratio = previous.get(grade, float("nan")) / largest
        previous[grade] = largest
        print(  # noqa: T201
            f"{grade:16s} dt={timestep * 1e3:8.4f} fs steps={int(step[-1]):5d} "
            f"window={time[-1] * 1e3:5.1f} fs | max|dE|={largest:.3e} eV "
            f"({largest / atoms:.2e} eV/atom, {largest / moved:.1e} of moved) "
            f"end dE={deviation[-1]:+.3e} eV slope={slope / atoms:+.2e} eV/atom/ps | "
            f"moved={moved:8.2f} eV | ratio vs 2*dt={ratio:5.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

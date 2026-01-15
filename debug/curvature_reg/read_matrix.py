# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201
"""Read matched-step events, complete validation scans, and full pair surveys."""

from __future__ import (
    annotations,
)

import argparse
import json
import re
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np
from needle_window import (
    ROOT,
    entries,
)

if TYPE_CHECKING:
    from pathlib import (
        Path,
    )


def accuracy(run: Path, step: int) -> tuple[str, str, str]:
    """Read a validated full-set scan; stochastic one-frame validation is excluded."""
    prefix = run / f"accuracy_sub001_{step}"
    if not prefix.with_suffix(".done").exists():
        return "-", "-", "-"
    with np.load(prefix.with_suffix(".npz")) as data:
        weight = data["natoms"]
        force = np.sqrt(np.sum(weight * data["f_rmse"] ** 2) / weight.sum())
        energy = data["e_err_atom"].mean() * 1000
        component = (
            np.sum(weight * data["f_component_mae"]) / weight.sum()
            if "f_component_mae" in data
            else None
        )
    return (
        f"{force:.5f}",
        f"{energy:.3f}",
        "-" if component is None else f"{component:.5f}",
    )


def survey(run: Path, step: int) -> tuple[str, str, str, str]:
    """Read absolute tails, additional extrema, and reference flags outside 0.6 contacts."""
    path = run / f"survey_full_{step}.json"
    if path.exists():
        pairs = json.loads(path.read_text())["pairs"]
        count = len(pairs)
        tails = sum(not p["tail_pass"] for p in pairs)
        extra = sum(bool(p["additional_extrema_beyond_15A"]) for p in pairs)
        flagged = sum(
            any(x["outside_06_contact"] for x in p["reference_flagged_minima"])
            for p in pairs
        )
        hydrogen = next((p for p in pairs if p["pair"] == "H-H"), None)
        well = (
            "-"
            if hydrogen is None
            else f"{hydrogen['principal_energy']:.3g}@{hydrogen['principal_r']:.3f}"
        )
        return f"{tails}/{count}", f"{extra}/{count}", str(flagged), well
    path = run / f"survey_{step}.log"
    if path.exists():
        tail = re.search(r"(\d+) tails above", path.read_text())
        return tail.group(1) + "/legacy" if tail else "-", "-", "-", "-"
    return "-", "-", "-", "-"


def row(run: Path, step: int) -> dict[str, Any]:
    """Read one run without extrapolating its coverage to the requested step."""
    events = entries(run.name)
    selected = [(s, e, d) for s, e, d in events if s < step]
    tears = [(s, e, d) for s, e, d in selected if e > 1000]
    force, energy, component = accuracy(run, step)
    tails, extra, flagged, well = survey(run, step)
    return {
        "run": run.name,
        "reached": events[-1][0] + 1 if events else 0,
        "small": sum(100 < e <= 1000 for _, e, _ in selected),
        "tears": len(tears),
        "tears_over_1A": sum(d != "-" and float(d) >= 1 for _, _, d in tears),
        "force_rmse": force,
        "energy_mae": energy,
        "component_force_mae": component,
        "tails": tails,
        "extra": extra,
        "reference_flags": flagged,
        "H2_principal": well,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", type=int)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    print(
        "All events are counted before the requested checkpoint step; reached must cover that step."
    )
    print(
        "A distance above 1 A is not a covalent-contact classification. Reference flags are diagnostic."
    )
    print(
        f"{'run':51s} {'reached':>8s} {'small/tear':>11s} {'tear>1A':>8s} "
        f"{'F RMSE':>9s} {'E meV/a':>9s} {'F MAE/c':>9s} {'tails':>10s} {'extra':>7s} {'ref>0.6c':>9s} {'H2 E@r':>14s}"
    )
    for name in args.runs:
        data = row(ROOT / name, args.step)
        counts = f"{data['small']}/{data['tears']}"
        print(
            f"{name:51s} {data['reached']:8d} {counts:>11s} {data['tears_over_1A']:8d} "
            f"{data['force_rmse']:>9s} {data['energy_mae']:>9s} {data['component_force_mae']:>9s} "
            f"{data['tails']:>10s} {data['extra']:>7s} {data['reference_flags']:>9s} {data['H2_principal']:>14s}"
        )


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: LGPL-3.0-or-later
"""Plot measured MD time changes for every validated geometry configuration."""

from __future__ import (
    annotations,
)

import argparse
import json
from pathlib import (
    Path,
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import (
    TwoSlopeNorm,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", choices=("auto", "indexed"), default="auto")
    parser.add_argument(
        "--reference", choices=("vector", "before_contract"), default="vector"
    )
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in args.summary]
    grades = ("nano", "mini", "neo", "air", "plus")
    columns = [
        (element, atoms)
        for element in ("carbon", "bcn")
        for atoms in (512, 10296, 100800)
    ]
    panels = [(flavor, zbl) for flavor in ("energy", "spin") for zbl in (False, True)]
    metric = f"{'net_' if args.reference == 'before_contract' else ''}{args.layout}_reduction_percent"
    repeats = {case["repeats"] for report in reports for case in report["cases"]}
    assert len(repeats) == 1, repeats
    extent = max(
        5.0, max(abs(case[metric]) for report in reports for case in report["cases"])
    )
    norm = TwoSlopeNorm(vmin=-extent, vcenter=0.0, vmax=extent)
    figure, axes = plt.subplots(
        len(reports),
        4,
        figsize=(18, 4.1 * len(reports)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, report in enumerate(reports):
        cases = {
            (
                case["grade"],
                case["flavor"],
                case["zbl"],
                case["elements"],
                case["atoms"],
            ): case
            for case in report["cases"]
        }
        for col, (flavor, zbl) in enumerate(panels):
            values = np.array(
                [
                    [
                        cases[(grade, flavor, zbl, element, atoms)][metric]
                        for element, atoms in columns
                    ]
                    for grade in grades
                ]
            )
            axis = axes[row, col]
            artist = axis.imshow(values, cmap="RdBu", norm=norm, aspect="auto")
            axis.set_title(
                f"{report['device']}\n{'Spin' if flavor == 'spin' else 'Energy'} / {'ZBL' if zbl else 'plain'}"
            )
            axis.set_yticks(range(5), [grade.capitalize() for grade in grades])
            axis.set_xticks(
                range(6),
                [
                    f"{'BCN' if element == 'bcn' else 'Carbon'}\n{atoms:,}"
                    for element, atoms in columns
                ],
                fontsize=8,
            )
            axis.axvline(2.5, color="white", linewidth=1.5)
            for index, value in np.ndenumerate(values):
                axis.text(
                    index[1],
                    index[0],
                    f"{value:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if abs(value) > 0.6 * extent else "black",
                )
    reference = (
        "retained vector (implicit FMA)"
        if args.reference == "before_contract"
        else "fixed vector"
    )
    figure.colorbar(artist, ax=axes, shrink=0.7, label="MD time reduction (%)")
    figure.suptitle(
        f"{'Automatic' if args.layout == 'auto' else 'Fixed indexed'} geometry · median of {repeats.pop()} MD runs\nReference: {reference}; positive values are faster; startup calibration is excluded",
        fontsize=14,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()

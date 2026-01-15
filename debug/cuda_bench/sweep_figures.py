# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared figure construction for the DPA4C parameter sweeps.

Both sweeps vary one structural parameter at a fixed channel width and report
the same two quantities, so they share the axis layout, the styling, and the
completeness checks and differ only in the swept column.
"""

from __future__ import (
    annotations,
)

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(HERE, "results", "parameter_sweep_summary.csv")
CHANNELS = (8, 16, 32, 64, 128)
THROUGHPUT_STYLE = ("#1F5FA9", "o")
CAPACITY_STYLE = ("#C43C2B", "s")


def load_summary() -> np.ndarray:
    """Return the sweep summary as a structured array.

    Returns
    -------
    numpy.ndarray
        One record per scanned configuration.

    Raises
    ------
    ValueError
        If any configuration lacks a saturated throughput or an out-of-memory
        bracket, which would make the figure misleading.
    """
    data = np.atleast_1d(
        np.genfromtxt(SUMMARY, delimiter=",", names=True, dtype=None, encoding=None)
    )
    if (
        not np.all(np.isfinite(data["saturated_atoms_per_ms"]))
        or np.any(data["max_atoms"] <= 0)
        or np.any(data["first_failed_atoms"] <= data["max_atoms"])
    ):
        raise ValueError("The sweep contains incomplete throughput/OOM brackets")
    return data


def draw_sweep(
    data: np.ndarray,
    column: str,
    values: tuple[int, ...],
    fixed: dict[str, int],
    label: str,
    title: str,
    stem: str,
) -> None:
    """Write one two-panel figure per channel width.

    Parameters
    ----------
    data
        Complete sweep summary.
    column
        Name of the swept column.
    values
        Expected values of the swept column, in plotting order.
    fixed
        Column/value pairs that select the swept family.
    label
        Axis label of the swept parameter.
    title
        Figure title, formatted with the channel width.
    stem
        Output file stem, formatted with the channel width.
    """
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.linewidth": 0.9,
        }
    )
    selection = data["family"] == "sweep"
    for name, value in fixed.items():
        selection &= data[name] == value
    family = data[selection]
    for channels in CHANNELS:
        rows = family[family["channels"] == channels]
        rows = rows[np.argsort(rows[column])]
        if tuple(int(value) for value in rows[column]) != values:
            raise ValueError(
                f"C{channels} must provide {column} values {values}, got "
                f"{tuple(int(value) for value in rows[column])}"
            )
        position = np.arange(len(values), dtype=float)
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))

        color, marker = THROUGHPUT_STYLE
        axes[0].plot(
            position,
            rows["saturated_atoms_per_ms"],
            color=color,
            marker=marker,
            lw=2.4,
            ms=7,
        )
        axes[0].set_ylabel("Saturated throughput  (atoms / ms)")
        baseline = float(rows["saturated_atoms_per_ms"][0])
        for x, value in zip(position, rows["saturated_atoms_per_ms"], strict=True):
            axes[0].annotate(
                f"{value / baseline:.2f}x",
                (x, value),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
                color=color,
            )

        color, marker = CAPACITY_STYLE
        axes[1].plot(
            position,
            rows["max_atoms"] / 1_000_000,
            color=color,
            marker=marker,
            lw=2.4,
            ms=7,
            label="largest success",
        )
        axes[1].plot(
            position,
            rows["first_failed_atoms"] / 1_000_000,
            color=color,
            marker=marker,
            markerfacecolor="white",
            ls=":",
            lw=1.3,
            ms=6,
            label="first out of memory",
        )
        axes[1].set_ylabel("System size  (million atoms)")
        axes[1].legend(frameon=False, fontsize=9, loc="lower left")

        for axis in axes:
            axis.set_xlabel(label)
            axis.set_xticks(position)
            axis.set_xticklabels([str(value) for value in values])
            axis.grid(True, color="0.9", lw=0.7)
            axis.set_axisbelow(True)
        fig.suptitle(title.format(channels=channels))
        fig.tight_layout()
        output = os.path.join(HERE, stem.format(channels=channels))
        fig.savefig(output, dpi=300)
        plt.close(fig)
        print(f"saved -> {output}")  # noqa: T201

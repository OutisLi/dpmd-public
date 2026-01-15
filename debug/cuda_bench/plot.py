# SPDX-License-Identifier: LGPL-3.0-or-later
"""Plot whole-step MD throughput and capacity for DPA4C, DPA1, and NEP.

The DPA4C curves are the five released grades; ``parameter_sweep.GRADES``
holds their architectures. The DPA1 curves reuse the compact canonical S/M/L
deployment baselines. NEP89 is measured in GPUMD on the same geometries.
"""

from __future__ import (
    annotations,
)

import os

import gen_system
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import (
    LogLocator,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

# Model families use independent line and marker grammars. Color shades then
# distinguish model sizes within each family.
FAMILY_STYLES = {
    "NEP89": {"ls": "-.", "lw": 3.0, "fill": True},
    "DPA1": {"ls": "--", "lw": 2.6, "fill": False},
    "DPA4C": {"ls": "-", "lw": 2.6, "fill": True},
}

# (csv, label, family, color, marker)
CURVES = [
    ("nep.csv", "NEP89", "NEP89", "#222222", "^"),
    (
        "lmp_dpa1_s.csv",
        "DPA1-S/F64",
        "DPA1",
        "#F0A35E",
        "o",
    ),
    (
        "lmp_dpa1_m.csv",
        "DPA1-M/F128",
        "DPA1",
        "#D97706",
        "s",
    ),
    (
        "lmp_dpa1_l.csv",
        "DPA1-L/F256",
        "DPA1",
        "#9A3412",
        "D",
    ),
    ("lmp_dpa4c_nano.csv", "DPA4C-Nano", "DPA4C", "#6BAED6", "o"),
    ("lmp_dpa4c_mini.csv", "DPA4C-Mini", "DPA4C", "#4292C6", "s"),
    ("lmp_dpa4c_neo.csv", "DPA4C-Neo", "DPA4C", "#2171B5", "D"),
    ("lmp_dpa4c_air.csv", "DPA4C-Air", "DPA4C", "#08519C", "P"),
    ("lmp_dpa4c_plus.csv", "DPA4C-Plus", "DPA4C", "#08306B", "X"),
]


def load(name: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return None
    arr = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    expected_small = {
        gen_system.atom_count(target) for target in gen_system.SMALL_TARGETS
    }
    measured = {int(value) for value in arr[:, 0]}
    if not expected_small.issubset(measured):
        raise ValueError(f"Benchmark curve is missing the small-system grid: {name}")
    if not np.any(~np.isfinite(arr[:, 1])):
        raise ValueError(f"Benchmark curve has no OOM capacity bracket: {name}")
    arr = arr[np.isfinite(arr[:, 1])]
    return arr[:, 0], arr[:, 1]


def main() -> None:
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 12, "axes.linewidth": 0.9}
    )
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    for name, label, family, color, marker in CURVES:
        curve = load(name)
        if curve is None:
            print(f"skipping absent curve: {name}", flush=True)  # noqa: T201
            continue
        x, y = curve
        if len(x) == 0:
            raise ValueError(f"Benchmark curve contains no finite values: {name}")
        style = FAMILY_STYLES[family]
        ax.plot(
            x,
            y,
            marker=marker,
            ls=style["ls"],
            color=color,
            lw=style["lw"],
            ms=5.0,
            markerfacecolor=color if style["fill"] else "white",
            markeredgecolor="white" if style["fill"] else color,
            markeredgewidth=0.5 if style["fill"] else 1.0,
            label=label,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Throughput  (atoms / ms)")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.grid(True, which="major", ls="-", lw=0.6, color="0.85")
    ax.grid(True, which="minor", ls="-", lw=0.4, color="0.93")
    ax.set_axisbelow(True)
    ax.legend(
        frameon=False,
        loc="lower right",
        fontsize=8.5,
        ncol=2,
        handlelength=2.4,
        labelspacing=0.35,
    )
    fig.tight_layout()
    out = os.path.join(HERE, "dpa4c_grade_benchmark.png")
    fig.savefig(out, dpi=300)
    print(f"saved -> {out}")  # noqa: T201

    # Saturated throughput (mean over N >= 1M) and largest completed system.
    print(f"\n{'curve':34s} {'plateau':>9s} {'max N':>10s}")  # noqa: T201
    for name, label, *_ in CURVES:
        path = os.path.join(RES, name)
        if not os.path.exists(path):
            continue
        x, y = load(name)
        if len(x) == 0:
            continue
        big = y[x >= 1_000_000]
        plateau = float(np.mean(big)) if len(big) else float(y[-1])
        print(f"{label:34s} {plateau:9.0f} {int(x.max()):10d}")  # noqa: T201


if __name__ == "__main__":
    main()

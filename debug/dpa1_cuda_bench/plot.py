# SPDX-License-Identifier: LGPL-3.0-or-later
"""Figure: whole-step MD throughput vs system size, DPA1 (LAMMPS) vs NEP (GPUMD).

Seven configurations on identical diamond-carbon supercells, scaled to
out-of-memory:

    GPUMD + NEP                       native NEP5 reference (GPUMD)
    LMP + DPA1-L0-{S,M,L}             uncompressed fused mega kernel (LAMMPS/Kokkos)
    LMP + DPA1-L0-{S,M,L} (compress)  geometrically compressed table kernel

DPA1-L0 S / M / L are the MatPES sizes (embedding neuron [16/32/64]^3, fitting
[64/128/256]^3, axis_neuron 16), strip se_atten_v2, attention-free.
Throughput is the whole MD step (LAMMPS Loop time under Kokkos; GPUMD per-step
Speed).
"""

from __future__ import (
    annotations,
)

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import (
    LogLocator,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

# (csv, label, color, marker, linestyle)
CURVES = [
    ("nep.csv", "GPUMD + NEP", "#555555", "^", "-"),
    ("lmp_S_uncompressed.csv", "LMP + DPA1-L0-S", "#1b9e77", "o", "-"),
    (
        "lmp_S_fp32_lowmem.csv",
        "LMP + DPA1-L0-S (compress)",
        "#1b9e77",
        "o",
        "--",
    ),
    ("lmp_M_verified_uncompressed.csv", "LMP + DPA1-L0-M", "#d95f02", "s", "-"),
    (
        "lmp_M_verified_compress.csv",
        "LMP + DPA1-L0-M (compress)",
        "#d95f02",
        "s",
        "--",
    ),
    ("lmp_L_verified_uncompressed.csv", "LMP + DPA1-L0-L", "#5e4fa2", "D", "-"),
    (
        "lmp_L_verified_compress.csv",
        "LMP + DPA1-L0-L (compress)",
        "#5e4fa2",
        "D",
        "--",
    ),
]


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(os.path.join(RES, name), delimiter=",", skiprows=1, ndmin=2)
    arr = arr[np.isfinite(arr[:, 1])]
    return arr[:, 0], arr[:, 1]


def main() -> None:
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 12, "axes.linewidth": 0.9}
    )
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    for name, label, color, marker, ls in CURVES:
        if not os.path.exists(os.path.join(RES, name)):
            print(f"skip missing {name}")  # noqa: T201
            continue
        x, y = load(name)
        if len(x) == 0:
            continue
        ax.plot(
            x,
            y,
            marker=marker,
            ls=ls,
            color=color,
            lw=1.8,
            ms=4.5,
            markeredgecolor="white",
            markeredgewidth=0.4,
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
        loc="upper left",
        fontsize=9,
        ncol=1,
        handlelength=2.4,
        labelspacing=0.35,
    )
    fig.tight_layout()
    out = os.path.join(HERE, "throughput.png")
    fig.savefig(out, dpi=300)
    print(f"saved -> {out}")  # noqa: T201

    # Plateau throughput (mean over N >= 50k) and largest completed system.
    print(f"\n{'curve':34s} {'plateau':>9s} {'max N':>10s}")  # noqa: T201
    for name, label, *_ in CURVES:
        path = os.path.join(RES, name)
        if not os.path.exists(path):
            continue
        x, y = load(name)
        if len(x) == 0:
            continue
        big = y[x >= 50000]
        plateau = float(np.mean(big)) if len(big) else float(y[-1])
        print(f"{label:34s} {plateau:9.0f} {int(x.max()):10d}")  # noqa: T201


if __name__ == "__main__":
    main()

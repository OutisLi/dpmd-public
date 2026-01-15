# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201
"""Read the isolated-pair curves of a run's surveys in a distance window, as values.

For every surveyed pair and checkpoint the script reports, over the window
[r_lo, r_hi] in Angstrom, the largest |F| along the bond in eV/A and the largest
energy above the curve's end value in eV, and counts the pairs whose window force
exceeds the threshold. Usage: pair_window_read.py <run> [--lo 2.5] [--hi 4.5]
[--fmax 0.3] [--steps 40000,100000]
"""

import argparse
import glob
import os

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--lo", type=float, default=2.5)
    parser.add_argument("--hi", type=float, default=4.5)
    parser.add_argument("--fmax", type=float, default=0.3)
    parser.add_argument("--steps", default="")
    args = parser.parse_args()
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", args.run)
    files = sorted(
        glob.glob(os.path.join(root, "survey_full_*.npz")),
        key=lambda f: int(f.rsplit("_", 1)[1].split(".")[0]),
    )
    wanted = {int(s) for s in args.steps.split(",") if s}
    for f in files:
        step = int(f.rsplit("_", 1)[1].split(".")[0])
        if wanted and step not in wanted:
            continue
        z = np.load(f)
        pairs = sorted({k[: -len("_r")] for k in z.files if k.endswith("_r")})
        rows = []
        for pair in pairs:
            r, e, fx = z[pair + "_r"], z[pair + "_energy"], z[pair + "_force"]
            m = (r >= args.lo) & (r <= args.hi)
            if not m.any():
                continue
            rows.append(
                (
                    pair,
                    float(np.abs(fx[m]).max()),
                    float(r[m][np.abs(fx[m]).argmax()]),
                    float((e[m] - e[-1]).max()),
                )
            )
        bad = [x for x in rows if x[1] > args.fmax]
        worst = max(rows, key=lambda x: x[1])
        print(
            f"{step:>7} window {args.lo}-{args.hi} A: pairs |F|>{args.fmax}: {len(bad):2d}/{len(rows)}  "
            f"worst {worst[0]} |F|={worst[1]:.3f} eV/A at {worst[2]:.2f} A  "
            f"largest E above end {max(x[3] for x in rows):.3f} eV  "
            + (
                "["
                + ", ".join(
                    f"{x[0]}:{x[1]:.2f}" for x in sorted(bad, key=lambda x: -x[1])[:6]
                )
                + "]"
                if bad
                else ""
            )
        )


if __name__ == "__main__":
    main()

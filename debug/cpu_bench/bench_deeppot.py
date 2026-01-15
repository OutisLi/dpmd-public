# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: ANN202, T201
"""Whole-call throughput of the ASE-style DeepPot path.

``bench.py`` holds the neighbor graph fixed and measures the model graph.
This script measures what an ASE calculator pays: ``DeepPot.eval`` rebuilds
the neighbor graph from coordinates and a cell on every call, then evaluates
the frozen package. The difference between the two is the graph construction,
which is reported explicitly because it is the dominant term of a small
system and a shared cost of every model.

Usage
-----
    OMP_NUM_THREADS=83 python bench_deeppot.py --grades nano,neo --atoms 8000
"""

from __future__ import (
    annotations,
)

import argparse
import os
import time

import numpy as np
from harness import (
    CARBON,
    HERE,
    THREADS,
    diamond_cell,
)

MODELS = os.path.join(HERE, "models")


def measure(model_file: str, atoms: int, iters: int, warmup: int) -> dict:
    """Time ``DeepPot.eval`` on one diamond supercell."""
    from deepmd.infer.deep_pot import (
        DeepPot,
    )

    coord, box = diamond_cell(atoms, jitter=0.03)
    count = coord.shape[0]
    atype = np.full(count, CARBON, dtype=np.int32)
    potential = DeepPot(model_file)

    def run():
        return potential.eval(
            coord.reshape(1, -1),
            box.reshape(1, 9),
            atype,
            atomic=False,
        )

    for _ in range(warmup):
        run()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        energy, force, virial = run()
        samples.append((time.perf_counter() - start) * 1e3)
    return {
        "atoms": count,
        "ms": float(np.mean(samples)),
        "ms_best": float(np.min(samples)),
        "energy": float(np.asarray(energy).sum()),
        "force_absmax": float(np.abs(np.asarray(force)).max()),
    }


def main() -> None:
    """Scan the requested grades and sizes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grades", default="nano,mini,neo,air,plus")
    parser.add_argument("--variant", default="compress", choices=("compress", "plain"))
    parser.add_argument("--atoms", type=int, nargs="+", default=[8000])
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    print(f"# threads={THREADS} variant={args.variant}")
    print(
        f"{'grade':6s} {'atoms':>8s} {'ms/call':>10s} {'best':>9s} "
        f"{'atoms/ms':>9s} {'|F|max':>10s}"
    )
    for grade in args.grades.split(","):
        model_file = os.path.join(MODELS, f"dpa4c_{grade}_{args.variant}.pt2")
        for atoms in args.atoms:
            row = measure(model_file, atoms, args.iters, args.warmup)
            print(
                f"{grade:6s} {row['atoms']:8d} {row['ms']:10.2f} "
                f"{row['ms_best']:9.2f} {row['atoms'] / row['ms']:9.2f} "
                f"{row['force_absmax']:10.3e}",
                flush=True,
            )


if __name__ == "__main__":
    main()

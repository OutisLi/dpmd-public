# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, ANN001, T201
"""System-size scaling of the compressed DPA4C CPU path.

Reports wall time, throughput and resident-set growth against the atom count.
The baseline is measured only where it is affordable: the uncompressed package
costs tens of seconds per step on the mode-carrying grades.

Thread scaling needs one process per point, because the intra-op thread count
is fixed when a process configures its runtime; ``--quiet`` emits one JSON row
so a shell loop over ``OMP_NUM_THREADS`` can collect the curve.

Usage
-----
    OMP_NUM_THREADS=83 python scaling.py --grades nano,neo,plus
    for t in 1 2 4 8 16 32 64 83; do \
        OMP_NUM_THREADS=$t python scaling.py --quiet --grades neo --sizes 8000; \
    done
"""

from __future__ import (
    annotations,
)

import argparse
import json
import os

import torch
from harness import (
    GRADES,
    HERE,
    THREADS,
    build_lower_inputs,
    load_model,
    timed,
)

MODELS = os.path.join(HERE, "models")

#: Cubic diamond supercells; the builder rounds to ``8 * repeat ** 3``.
SIZES = (216, 1000, 4096, 8000, 16000, 32768, 64000, 128000)


def measure(grade: str, atoms: int, variant: str, iters: int) -> dict:
    """Time one frozen package on one system size."""
    import paths as path_module

    compressed = variant == "compress"
    model = load_model(GRADES[grade])
    if compressed:
        model.get_descriptor().enable_compression(0.0, table_stride_1=0.01)
    sample = build_lower_inputs(
        model,
        atoms,
        edge_dtype=torch.float32 if compressed else None,
    )
    run = path_module.package(os.path.join(MODELS, f"dpa4c_{grade}_{variant}.pt2"))
    result = timed(lambda: run(*sample.args), iters=iters, warmup=2)
    result.update(
        grade=grade, variant=variant, atoms=sample.n_atom, edges=sample.n_edge
    )
    return result


def size_scan(grades: list[str], variant: str, sizes, iters: int) -> list[dict]:
    """Scan the atom count of one variant."""
    rows = []
    print(
        f"{'grade':6s} {'atoms':>7s} {'edges':>10s} {'ms':>9s} {'best':>9s} "
        f"{'atom/ms':>8s} {'rssGB':>7s} {'incrGB':>7s}"
    )
    for grade in grades:
        for atoms in sizes:
            row = measure(grade, atoms, variant, iters)
            rows.append(row)
            print(
                f"{row['grade']:6s} {row['atoms']:7d} {row['edges']:10d} "
                f"{row['ms']:9.2f} {row['ms_best']:9.2f} "
                f"{row['atoms'] / row['ms']:8.2f} {row['rss_gb']:7.2f} "
                f"{row['rss_incr_gb']:7.3f}",
                flush=True,
            )
    return rows


def main() -> None:
    """Run the requested scan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grades", default="nano,mini,neo,air,plus")
    parser.add_argument("--variant", default="compress", choices=("compress", "plain"))
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    if args.quiet:
        rows = [
            measure(grade, atoms, args.variant, args.iters)
            for grade in args.grades.split(",")
            for atoms in args.sizes
        ]
        print(json.dumps(rows))
        return
    print(f"# threads={THREADS} variant={args.variant}")
    rows = size_scan(args.grades.split(","), args.variant, args.sizes, args.iters)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(rows, handle, indent=1)


if __name__ == "__main__":
    main()

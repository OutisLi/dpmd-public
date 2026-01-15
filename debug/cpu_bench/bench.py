# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, T201
"""Benchmark DPA4C inference paths on the CPU graph lower.

Three paths are compared, all evaluating one full energy, force and virial
step on the same system:

``plain``
    the uncompressed model frozen to an AOTInductor package -- the CPU
    deployment as it exists today, and the baseline;
``compress``
    the compressed model frozen the same way, which routes the descriptor,
    the fitting and the force assembly through the hand-written CPU
    operators;
``eager``
    the compressed model without the package, which isolates the tracing
    overhead the package removes.

Usage
-----
    OMP_NUM_THREADS=83 python bench.py --grade all --paths plain,compress \
        --atoms 4096 8000
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
    cpu_topology,
    env_summary,
    load_model,
    timed,
)

MODELS = os.path.join(HERE, "models")


def run_one(
    grade: str,
    path: str,
    atoms: int,
    iters: int,
    warmup: int,
    stride: float,
) -> dict:
    """Time one (grade, path) pair on one system size."""
    import paths as path_module

    compressed = path != "plain"
    model = load_model(GRADES[grade])
    if compressed:
        model.get_descriptor().enable_compression(0.0, table_stride_1=stride)
    sample = build_lower_inputs(
        model,
        atoms,
        edge_dtype=torch.float32 if compressed else None,
    )
    if path == "eager":
        run = path_module.eager(model)
    else:
        variant = "compress" if compressed else "plain"
        run = path_module.package(os.path.join(MODELS, f"dpa4c_{grade}_{variant}.pt2"))
    result = timed(lambda: run(*sample.args), iters=iters, warmup=warmup)
    result.update(
        grade=grade,
        path=path,
        atoms=sample.n_atom,
        edges=sample.n_edge,
        atoms_per_ms=round(sample.n_atom / result["ms"], 2),
    )
    return result


def main() -> None:
    """Parse the requested matrix and print one row per measurement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grade", default="all")
    parser.add_argument("--paths", default="plain,compress")
    parser.add_argument("--atoms", type=int, nargs="+", default=[8000])
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--stride", type=float, default=0.01)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    grades = list(GRADES) if args.grade == "all" else args.grade.split(",")
    topology = cpu_topology()
    print(
        f"# {topology['model']}: {topology['physical']} cores / "
        f"{topology['logical']} threads, {len(topology['nodes'])} NUMA nodes, "
        f"using {THREADS}"
    )
    print(f"# {env_summary()}")
    print(
        f"{'grade':6s} {'path':9s} {'atoms':>7s} {'edges':>9s} {'ms':>9s} "
        f"{'best':>9s} {'atom/ms':>8s} {'rssGB':>7s} {'energy':>16s}"
    )
    rows = []
    for grade in grades:
        for atoms in args.atoms:
            for path in args.paths.split(","):
                row = run_one(grade, path, atoms, args.iters, args.warmup, args.stride)
                rows.append(row)
                print(
                    f"{row['grade']:6s} {row['path']:9s} {row['atoms']:7d} "
                    f"{row['edges']:9d} {row['ms']:9.2f} {row['ms_best']:9.2f} "
                    f"{row['atoms_per_ms']:8.2f} {row['rss_gb']:7.2f} "
                    f"{row['energy']:16.4f}",
                    flush=True,
                )
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(rows, handle, indent=1)


if __name__ == "__main__":
    main()

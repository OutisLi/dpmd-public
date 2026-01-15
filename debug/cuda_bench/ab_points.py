# SPDX-License-Identifier: LGPL-3.0-or-later
r"""A/B whole-step LAMMPS throughput of chosen points under two environments.

Each point (grade, size) runs alternately under the A and B environments on
one device, repeated, so that a few-percent difference between two settings
of the same build is resolved against the run-to-run spread. The settings
are environment assignments, ``NAME=VALUE`` separated by commas; among them
``BENCH_DP_LIB`` swaps the DeePMD library directory and with it the plugin
under it, ``BENCH_PLUGIN_PATH`` names another plugin directory, empty for a
LAMMPS that carries the styles itself, ``BENCH_LMP`` swaps the LAMMPS
executable, and ``BENCH_MODELS`` the directory of the frozen models, which
``--models`` gives otherwise. One process drives one device; several devices
are covered by several processes.

Usage
-----
    python ab_points.py --points air:512,neo:10296 --a "" \
        --b BENCH_DP_LIB=/tmp/variant/lib --gpu 0 --repeats 4
"""

from __future__ import (
    annotations,
)

import argparse
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))

import bench_sizes
import lmp_scan


def parse_env(spec: str) -> dict[str, str]:
    return dict(item.split("=", 1) for item in spec.split(",") if item)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", required=True)
    parser.add_argument("--a", default="")
    parser.add_argument("--b", default="")
    parser.add_argument("--models", default=os.path.join(HERE, "models", "v20260911"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--label", default="ab")
    parser.add_argument("--flavor", default="energy", choices=sorted(lmp_scan.FLAVORS))
    args = parser.parse_args()

    points = [(p.split(":")[0], int(p.split(":")[1])) for p in args.points.split(",")]
    settings = {"A": parse_env(args.a), "B": parse_env(args.b)}
    baseline = dict(os.environ)
    results: dict[tuple[str, int, str], list[float]] = {}
    for grade, n_atoms in points:
        for repeat in range(args.repeats):
            for name in ("A", "B") if repeat % 2 == 0 else ("B", "A"):
                os.environ.clear()
                os.environ.update(baseline)
                os.environ.update(settings[name])
                # A setting may swap the DeePMD library directory, the plugin
                # directory or the LAMMPS executable.
                lmp_scan.DP_LIB = os.environ["BENCH_DP_LIB"]
                # A swapped library directory brings the plugin under it,
                # unless the setting names a plugin directory of its own; a
                # LAMMPS with built-in styles (empty plugin path) loads none.
                plugin = os.environ.get("BENCH_PLUGIN_PATH", "")
                if (
                    "BENCH_PLUGIN_PATH" not in settings[name]
                    and plugin
                    and lmp_scan.DP_LIB != baseline["BENCH_DP_LIB"]
                ):
                    plugin = os.path.join(lmp_scan.DP_LIB, "deepmd_lmp")
                lmp_scan.PLUGIN_PATH = plugin
                lmp_scan.LMP = os.environ["BENCH_LMP"]
                work = os.path.join(
                    HERE,
                    f"work_ab_{args.label}_{args.flavor}",
                    f"gpu{args.gpu}_{name}",
                )
                models = os.environ.get("BENCH_MODELS", args.models)
                value = bench_sizes.run_point(
                    os.path.join(models, f"{grade}.pt2"),
                    n_atoms,
                    args.gpu,
                    work,
                    args.flavor,
                )
                results.setdefault((grade, n_atoms, name), []).append(n_atoms / value)
                print(  # noqa: T201
                    f"{grade:5s} {n_atoms:7d} {name} r{repeat}: {n_atoms / value:.4f} ms/step",
                    flush=True,
                )
    os.environ.clear()
    os.environ.update(baseline)
    for grade, n_atoms in points:
        cells = [
            f"{name} {statistics.median(results[(grade, n_atoms, name)]):.4f} ms"
            for name in ("A", "B")
        ]
        print(f"{grade:5s} {n_atoms:7d}: " + " | ".join(cells), flush=True)  # noqa: T201


if __name__ == "__main__":
    main()

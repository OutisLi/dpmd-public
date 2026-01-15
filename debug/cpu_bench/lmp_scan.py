# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201
"""Whole-step LAMMPS throughput of the compressed DPA4C CPU path.

The model-graph benchmarks in ``bench.py`` hold the neighbor graph fixed, so
they measure the operators alone. This scan measures what a molecular-dynamics
run actually costs: LAMMPS rebuilds its neighbor list, the host adapter turns
it into a NeighborGraph, and the frozen package evaluates one step. It is the
number a deployment decision rests on.

Usage
-----
    OMP_NUM_THREADS=83 python lmp_scan.py --grades nano,neo --atoms 8000 32768
"""

from __future__ import (
    annotations,
)

import argparse
import os
import re
import subprocess
import sys

from harness import (
    HERE,
    THREADS,
)

MODELS = os.path.join(HERE, "models")
SYSTEMS = os.path.join(HERE, "systems")
LAMMPS = os.environ.get(
    "BENCH_LAMMPS", "/aisi-vepfs/outisli/Software/deepmd-kit_cpp/bin/lmp"
)
#: The LAMMPS scan reuses the supercell writer of the CUDA benchmarks so the
#: two curves are measured on identical geometry.
GEN_SYSTEM = os.path.join(os.path.dirname(HERE), "cuda_bench")


def data_file(atoms: int) -> str:
    """Write the diamond supercell nearest ``atoms`` and return its path."""
    sys.path.insert(0, GEN_SYSTEM)
    from gen_system import (
        ensure_data,
    )

    os.makedirs(SYSTEMS, exist_ok=True)
    return ensure_data(atoms, SYSTEMS)[0]


def _release_affinity() -> None:
    """Give the child the whole machine.

    ``OMP_PROC_BIND`` makes libgomp pin this process's main thread to one
    core, and a forked child inherits that mask: LAMMPS would then see a
    single available processor and run the model on one thread regardless of
    ``OMP_NUM_THREADS``, which understates the step by more than an order of
    magnitude. Restoring the full mask in the child is the only way to measure
    a LAMMPS run launched from a process that has already touched an OpenMP
    runtime.
    """
    os.sched_setaffinity(0, range(os.cpu_count() or 1))


def run_lammps(model: str, data: str, steps: int, warmup: int) -> tuple[float, int]:
    """Run one LAMMPS job and return its per-step time and atom count.

    The timing is read from the log rather than the screen, because the second
    ``run`` command is the timed one and only the log keeps both blocks.
    """
    log = os.path.join(HERE, "log.lammps")
    output = subprocess.run(
        [
            LAMMPS,
            "-in",
            os.path.join(HERE, "in.lammps"),
            "-var",
            "model",
            model,
            "-var",
            "datafile",
            data,
            "-var",
            "nsteps",
            str(steps),
            "-var",
            "warmup",
            str(warmup),
            "-log",
            log,
            "-screen",
            "none",
            "-nocite",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=HERE,
        preexec_fn=_release_affinity,
    )
    with open(log) as handle:
        text = handle.read()
    if output.returncode != 0:
        raise RuntimeError((output.stdout + output.stderr + text)[-4000:])
    # The last performance block belongs to the timed run.
    rates = re.findall(r"([0-9.]+) timesteps/s", text)
    atoms = re.findall(r"(\d+) atoms", text)
    if not rates or not atoms:
        raise RuntimeError(f"no timing in the LAMMPS log:\n{text[-4000:]}")
    return 1.0e3 / float(rates[-1]), int(atoms[0])


def main() -> None:
    """Scan the requested grades and sizes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grades", default="nano,mini,neo,air,plus")
    parser.add_argument("--variant", default="compress", choices=("compress", "plain"))
    parser.add_argument("--atoms", type=int, nargs="+", default=[8000])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    print(f"# {LAMMPS} threads={THREADS} variant={args.variant}")
    print(f"{'grade':6s} {'atoms':>8s} {'ms/step':>10s} {'atoms/ms':>9s}")
    for grade in args.grades.split(","):
        model = os.path.join(MODELS, f"dpa4c_{grade}_{args.variant}.pt2")
        for atoms in args.atoms:
            data = data_file(atoms)
            milliseconds, count = run_lammps(model, data, args.steps, args.warmup)
            print(
                f"{grade:6s} {count:8d} {milliseconds:10.2f} "
                f"{count / milliseconds:9.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()

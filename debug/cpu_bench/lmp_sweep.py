# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201
"""LAMMPS capacity and throughput sweep over diamond supercells, CPU path.

Mirrors the CUDA scan's structure: double the system size from a small seed
until the run stops fitting, then bisect the last successful interval to find
the ceiling. The CUDA scan stops at the first out-of-memory failure. A CPU host
has more memory than a job of this shape can usefully occupy, so the ceiling
here is a resident-memory budget rather than an allocation failure -- the
number a deployment actually plans against.

The reported throughput is ``n_atoms / ms_per_step`` from LAMMPS' final timed
run, so it covers the whole molecular-dynamics step: neighbor plumbing, the
model, the force reduction and the integration.

Usage
-----
    OMP_NUM_THREADS=83 python lmp_sweep.py --grades nano,neo --budget-gb 4
"""

from __future__ import (
    annotations,
)

import argparse
import json
import os
import re
import subprocess
import sys
import time

from harness import (
    HERE,
    THREADS,
)

MODELS = os.path.join(HERE, "models")
SYSTEMS = os.path.join(HERE, "systems")
LAMMPS = os.environ.get(
    "BENCH_LAMMPS", "/aisi-vepfs/outisli/Software/deepmd-kit_cpp/bin/lmp"
)
#: The supercell writer is shared with the CUDA benchmarks so the two capacity
#: curves are measured on identical geometry.
GEN_SYSTEM = os.path.join(os.path.dirname(HERE), "cuda_bench")
#: Smallest system of the sweep; the doubling starts here.
SEED_ATOMS = 128
#: Bisection steps taken after the first system that exceeds the budget.
BISECTIONS = 3


def _release_affinity() -> None:
    """Give the child the whole machine.

    ``OMP_PROC_BIND`` makes libgomp pin this process's main thread to one core,
    and a forked child inherits that mask: LAMMPS would then see a single
    processor and run the model on one thread regardless of ``OMP_NUM_THREADS``.
    """
    os.sched_setaffinity(0, range(os.cpu_count() or 1))


def data_file(atoms: int) -> tuple[str, int]:
    """Write the diamond supercell nearest ``atoms`` and return it with its size."""
    sys.path.insert(0, GEN_SYSTEM)
    from gen_system import (
        ensure_data,
    )

    os.makedirs(SYSTEMS, exist_ok=True)
    return ensure_data(atoms, SYSTEMS)


def run_case(
    model: str,
    atoms: int,
    steps: int,
    warmup: int,
    timeout: float,
) -> dict:
    """Run one LAMMPS job and return its step time, atom count and peak memory.

    The peak resident set comes from the kernel's accounting for that exact
    child rather than from sampling ``/proc``, so a transient allocation cannot
    slip between two samples.
    """
    data, realized = data_file(atoms)
    log = os.path.join(HERE, f"log.sweep.{realized}")
    command = [
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
    ]
    started = time.perf_counter()
    with open(os.path.join(HERE, "sweep.stderr"), "w") as sink:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=sink,
            cwd=HERE,
            preexec_fn=_release_affinity,
        )
        outcome = None
        while time.perf_counter() - started < timeout:
            pid, wait_status, usage = os.wait4(process.pid, os.WNOHANG)
            if pid != 0:
                outcome = (wait_status, usage)
                break
            time.sleep(0.05)
        if outcome is None:
            process.kill()
            os.wait4(process.pid, 0)
            return {
                "atoms": realized,
                "ms": float("nan"),
                "peak_gb": float("nan"),
                "status": "timeout",
            }
    wall = time.perf_counter() - started
    wait_status, usage = outcome
    peak_gb = usage.ru_maxrss / (1024.0 * 1024.0)
    if wait_status != 0:
        return {
            "atoms": realized,
            "ms": float("nan"),
            "peak_gb": peak_gb,
            "status": "failed",
        }
    with open(log) as handle:
        text = handle.read()
    rates = re.findall(r"([0-9.]+) timesteps/s", text)
    if not rates:
        return {
            "atoms": realized,
            "ms": float("nan"),
            "peak_gb": peak_gb,
            "status": "no timing",
        }
    return {
        "atoms": realized,
        "ms": 1.0e3 / float(rates[-1]),
        "peak_gb": peak_gb,
        "status": "ok",
        "wall_s": wall,
    }


def sweep_grade(
    grade: str,
    variant: str,
    budget_gb: float,
    steps: int,
    warmup: int,
    timeout: float,
) -> list[dict]:
    """Double until the budget is exceeded, then bisect the last interval."""
    model = os.path.join(MODELS, f"dpa4c_{grade}_{variant}.pt2")
    rows: list[dict] = []
    seen: set[int] = set()

    def measure(target: int) -> dict:
        row = run_case(model, target, steps, warmup, timeout)
        if row["atoms"] in seen:
            return next(r for r in rows if r["atoms"] == row["atoms"])
        seen.add(row["atoms"])
        rows.append(row)
        throughput = (
            row["atoms"] / row["ms"] if row["ms"] == row["ms"] else float("nan")
        )
        print(
            f"{grade:6s} {row['atoms']:9d} {row['ms']:10.2f} {throughput:9.1f} "
            f"{row['peak_gb']:8.2f}  {row['status']}",
            flush=True,
        )
        return row

    # === Doubling until the resident set leaves the budget ===
    target = SEED_ATOMS
    last_ok = 0
    first_over = 0
    while True:
        row = measure(target)
        fits = row["status"] == "ok" and row["peak_gb"] <= budget_gb
        if fits:
            last_ok = row["atoms"]
            target *= 2
            continue
        first_over = row["atoms"]
        break

    # === Bisect the interval between the last fit and the first miss ===
    for _ in range(BISECTIONS):
        if first_over - last_ok <= 1:
            break
        probe = (last_ok + first_over) // 2
        row = measure(probe)
        if row["status"] == "ok" and row["peak_gb"] <= budget_gb:
            last_ok = max(last_ok, row["atoms"])
        else:
            first_over = min(first_over, row["atoms"])
    print(
        f"# {grade}: largest system within {budget_gb:.0f} GiB = {last_ok} atoms",
        flush=True,
    )
    return rows


def main() -> None:
    """Sweep every requested grade."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grades", default="nano,mini,neo,air,plus")
    parser.add_argument("--variant", default="compress", choices=("compress", "plain"))
    parser.add_argument("--budget-gb", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    print(
        f"# {LAMMPS} threads={THREADS} variant={args.variant} "
        f"budget={args.budget_gb:.0f} GiB"
    )
    print(
        f"{'grade':6s} {'atoms':>9s} {'ms/step':>10s} {'atoms/ms':>9s} "
        f"{'peakGiB':>8s}  status"
    )
    results: dict[str, list[dict]] = {}
    for grade in args.grades.split(","):
        results[grade] = sweep_grade(
            grade,
            args.variant,
            args.budget_gb,
            args.steps,
            args.warmup,
            args.timeout,
        )
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: LGPL-3.0-or-later
"""GPUMD NEP throughput scan over the same diamond supercells.

The NEP5 reference potential is run in GPUMD on the identical diamond-carbon
geometry as the LAMMPS deepmd scans (:mod:`gen_system`), so the curves share
x-points. The metric is GPUMD's per-step ``Speed of this run`` converted to
atoms/ms -- the whole-step MD throughput, comparable to the LAMMPS Loop-time
throughput. Environment: ``BENCH_GPU`` selects the device. Resumable CSV.
"""

from __future__ import (
    annotations,
)

import os
import re
import subprocess

import gen_system
import numpy as np
from gen_system import (
    TARGETS,
)

HERE = os.path.dirname(os.path.abspath(__file__))
GPUMD = "/aisi-vepfs/outisli/Software/GPUMD/src/gpumd"
NEP = os.path.join(HERE, "nep.txt")
WARMUP, MEASURE = 10, 100


def write_runin(work: str) -> None:
    with open(os.path.join(work, "run.in"), "w") as f:
        f.write(
            f"potential       {NEP}\n"
            "velocity        300\n"
            "time_step       1\n"
            "ensemble        nvt_nhc 300 300 100\n"
            "dump_thermo     1000\n"
            f"run             {WARMUP}\n"
            f"run             {MEASURE}\n"
        )


def run_case(work: str, gpu: str) -> float:
    """Return whole-step throughput (atoms/ms), or NaN on OOM/failure."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    try:
        proc = subprocess.run(
            [GPUMD], cwd=work, env=env, capture_output=True, text=True, timeout=3600
        )
    except subprocess.TimeoutExpired:
        return float("nan")
    speeds = re.findall(
        r"Speed of this run = ([\d.eE+]+) atom\*step/second", proc.stdout
    )
    if len(speeds) < 2:
        return float("nan")
    return float(speeds[1]) / 1000.0  # atom*step/s -> atoms/ms per step


def _write_results(path: str, values: dict[int, float]) -> None:
    """Persist completed scan points in ascending atom-count order."""
    rows = np.array(sorted(values.items()), dtype=float)
    temporary = path + ".tmp"
    np.savetxt(
        temporary,
        rows,
        delimiter=",",
        header="n_atoms,nep_atoms_per_ms",
        comments="",
    )
    os.replace(temporary, path)


def main() -> None:
    gpu = os.environ.get("BENCH_GPU", "0")
    work = os.path.join(HERE, "work_nep")
    out = os.path.join(HERE, "results", "nep.csv")
    os.makedirs(work, exist_ok=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_runin(work)

    done: dict[int, float] = {}
    if os.path.exists(out):
        prev = np.loadtxt(out, delimiter=",", skiprows=1, ndmin=2)
        done = {int(r[0]): float(r[1]) for r in prev}
    if any(not np.isfinite(v) for v in done.values()):
        print(  # noqa: T201
            "[nep] previous scan already hit its size limit", flush=True
        )
    else:
        seen = set(done)
        for target in TARGETS:
            pos, box = gen_system.build(target)
            n = gen_system.write_xyz(os.path.join(work, "model.xyz"), pos, box)
            if n in seen:
                continue
            seen.add(n)
            tp = run_case(work, gpu)
            done[n] = tp
            _write_results(out, done)
            print(f"[nep] N={n:>9d} tp={tp:9.1f} atoms/ms", flush=True)  # noqa: T201
            if not np.isfinite(tp):
                print(  # noqa: T201
                    f"[nep] stop: size {n} failed (OOM monotonic in N)", flush=True
                )
                break
    _write_results(out, done)
    print(f"[nep] scan done -> {out}", flush=True)  # noqa: T201


if __name__ == "__main__":
    main()

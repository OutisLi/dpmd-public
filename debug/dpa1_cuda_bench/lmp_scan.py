# SPDX-License-Identifier: LGPL-3.0-or-later
"""LAMMPS + Kokkos deepmd throughput scan over diamond supercells.

Drives one frozen graph-lower ``.pt2`` (compressed or uncompressed) through the
Kokkos ``pair_style deepmd/kk`` on increasing diamond-carbon supercells until
the first out-of-memory / failure (memory is monotonic in system size). The
reported throughput is ``n_atoms / loop_ms_per_step`` from LAMMPS' final timed
run and therefore covers the complete MD step.

Environment:
    BENCH_MODEL   absolute path to the .pt2 package
    BENCH_TAG     short label -> results/lmp_<tag>.csv and the work directory
    BENCH_GPU     CUDA device index
    BENCH_DP_LIB  DeePMD shared-library directory (defaults to deepmd-kit_cpp/lib)
    BENCH_LD_PRELOAD  optional colon-separated shared libraries loaded first
    BENCH_TARGETS  optional comma-separated target atom counts
The scan is resumable: sizes already in the CSV are kept, a recorded OOM
terminates the schedule, and the merged result is rewritten sorted.
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
LMP = "/aisi-vepfs/outisli/Software/lammps/lammps-patch_4Jul2026/build_kk/lmp"
DP_LIB = os.environ.get(
    "BENCH_DP_LIB", "/aisi-vepfs/outisli/Software/deepmd-kit_cpp/lib"
)
TORCH_LIB = (
    "/aisi-vepfs/outisli/miniforge3/envs/dpmd/lib/python3.13/site-packages/torch/lib"
)
WARMUP, NSTEPS = 10, 30


def run_case(work: str, model: str, gpu: str) -> float:
    """Return whole-step throughput (atoms/ms), or NaN on OOM/failure.

    The metric is ``n_atoms / loop_ms_per_step`` from LAMMPS' last (timed) run,
    i.e. the complete MD step under Kokkos -- directly comparable to GPUMD's
    per-step ``Speed of this run`` for the NEP reference.
    """
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["OMP_NUM_THREADS"] = "1"
    env["LD_LIBRARY_PATH"] = (
        DP_LIB + ":" + TORCH_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
    )
    preload = os.environ.get("BENCH_LD_PRELOAD")
    if preload:
        env["LD_PRELOAD"] = preload + ":" + env.get("LD_PRELOAD", "")
    try:
        proc = subprocess.run(
            [
                LMP,
                "-k",
                "on",
                "g",
                "1",
                "-sf",
                "kk",
                "-in",
                os.path.join(HERE, "in.lammps"),
                "-var",
                "datafile",
                os.path.join(work, "sys.data"),
                "-var",
                "model",
                model,
                "-var",
                "warmup",
                str(WARMUP),
                "-var",
                "nsteps",
                str(NSTEPS),
            ],
            cwd=work,
            env=env,
            capture_output=True,
            text=True,
            timeout=1200,
        )
    except subprocess.TimeoutExpired:
        return float("nan")
    loops = re.findall(
        r"Loop time of ([\d.eE+-]+) on \d+ procs for (\d+) steps", proc.stdout
    )
    if not loops:
        return float("nan")
    loop_t, steps = float(loops[-1][0]), int(loops[-1][1])
    loop_ms = loop_t * 1e3 / steps
    return _last_natoms(proc.stdout) / loop_ms


def _last_natoms(out: str) -> int:
    return int(re.findall(r"with (\d+) atoms", out)[-1])


def _write_results(path: str, values: dict[int, float], tag: str) -> None:
    """Persist completed scan points in ascending atom-count order."""
    rows = np.array(sorted(values.items()), dtype=float)
    temporary = path + ".tmp"
    np.savetxt(
        temporary,
        rows,
        delimiter=",",
        header=f"n_atoms,{tag}_atoms_per_ms",
        comments="",
    )
    os.replace(temporary, path)


def main() -> None:
    tag = os.environ["BENCH_TAG"]
    gpu = os.environ.get("BENCH_GPU", "0")
    model = os.environ["BENCH_MODEL"]
    work = os.path.join(HERE, f"work_lmp_{tag}")
    out = os.path.join(HERE, "results", f"lmp_{tag}.csv")
    os.makedirs(work, exist_ok=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    done: dict[int, float] = {}
    if os.path.exists(out):
        prev = np.loadtxt(out, delimiter=",", skiprows=1, ndmin=2)
        done = {int(r[0]): float(r[1]) for r in prev}
    if any(not np.isfinite(v) for v in done.values()):
        print(  # noqa: T201
            f"[{tag}] previous scan already hit its size limit", flush=True
        )
    else:
        seen = set(done)
        targets_text = os.environ.get("BENCH_TARGETS")
        targets = (
            [int(value) for value in targets_text.split(",")]
            if targets_text
            else TARGETS
        )
        for target in targets:
            pos, box = gen_system.build(target)
            n = gen_system.write_data(os.path.join(work, "sys.data"), pos, box)
            if n in seen:
                continue
            seen.add(n)
            tp = run_case(work, model, gpu)
            done[n] = tp
            _write_results(out, done, tag)
            print(f"[{tag}] N={n:>9d} tp={tp:9.1f} atoms/ms", flush=True)  # noqa: T201
            if not np.isfinite(tp):
                print(  # noqa: T201
                    f"[{tag}] stop: size {n} failed (OOM monotonic in N)", flush=True
                )
                break
    _write_results(out, done, tag)
    print(f"[{tag}] scan done -> {out}", flush=True)  # noqa: T201


if __name__ == "__main__":
    main()

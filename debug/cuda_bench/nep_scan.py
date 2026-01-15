# SPDX-License-Identifier: LGPL-3.0-or-later
"""GPUMD NEP throughput and capacity scan over diamond supercells.

The NEP89 reference potential (``models/nep89_20250409.txt``, an 89-element
NEP4 model with ZBL) is run in GPUMD on the identical diamond-carbon geometry
as the LAMMPS deepmd scans (:mod:`gen_system`), so the curves share x-points. It retains the logarithmic grid below one million atoms, advances in
two-million increments above one million until failure, then bisects the final
interval at one-million and half-million resolution.

The metric is GPUMD's per-step ``Speed of this run`` converted to atoms/ms,
which is comparable to the LAMMPS whole-step Loop-time throughput.
``BENCH_GPU`` selects the device. ``BENCH_FRESH=1`` ignores existing results.
"""

from __future__ import (
    annotations,
)

import os
import re
import subprocess

import benchmark_manifest
import gen_system
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GPUMD = os.environ.get("BENCH_GPUMD", "/nas/outisli/Software/GPUMD/src/gpumd")
NEP = os.environ.get(
    "BENCH_NEP",
    os.path.join(HERE, "models", "nep89_20250409.txt"),
)
WARMUP, MEASURE = 10, 100
COARSE_START = 1_000_000
COARSE_STEP = 2_000_000
REFINEMENT_STEPS = (1_000_000, 500_000)
SYSTEM_CACHE = os.path.join(HERE, "systems")
OOM_PATTERN = re.compile(
    r"(out of memory|memory allocation|cudaErrorMemoryAllocation|"
    r"failed to allocate|alloc failed|std::bad_alloc)",
    re.IGNORECASE,
)


def _record_failure(work: str, output: str) -> None:
    path = os.path.join(work, "failure.log")
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        stream.write(output)
    os.replace(temporary, path)


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
    except subprocess.TimeoutExpired as error:
        output = (error.stdout or "") + "\n" + (error.stderr or "")
        _record_failure(work, "GPUMD benchmark timed out\n" + output)
        raise RuntimeError(f"GPUMD benchmark timed out in {work}") from error
    combined_output = proc.stdout + "\n" + proc.stderr
    speeds = re.findall(
        r"Speed of this run = ([\d.eE+]+) atom\*step/second", proc.stdout
    )
    if proc.returncode != 0 or len(speeds) < 2:
        _record_failure(work, combined_output)
        if OOM_PATTERN.search(combined_output):
            return float("nan")
        raise RuntimeError(
            f"GPUMD benchmark failed without an OOM signature in {work}; "
            "see failure.log"
        )
    failure_log = os.path.join(work, "failure.log")
    if os.path.exists(failure_log):
        os.unlink(failure_log)
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


def _measure_target(
    target: int,
    *,
    work: str,
    gpu: str,
    output: str,
    completed: dict[int, float],
) -> float:
    """Measure one requested size or reuse its persisted result.

    Parameters
    ----------
    target
        Requested atom count before diamond-supercell rounding.
    work
        Benchmark working directory.
    gpu
        CUDA device index.
    output
        Result CSV path.
    completed
        Results keyed by realized atom count.

    Returns
    -------
    float
        Whole-step throughput in atoms/ms, or NaN on failure.
    """
    n_atoms = gen_system.atom_count(target)
    if n_atoms in completed:
        throughput = completed[n_atoms]
        print(  # noqa: T201
            f"[nep] reuse N={n_atoms:>9d} tp={throughput:9.1f} atoms/ms",
            flush=True,
        )
        return throughput

    cached_xyz, realized = gen_system.ensure_xyz(target, SYSTEM_CACHE)
    if realized != n_atoms:
        raise RuntimeError(
            f"Cached diamond size mismatch: expected {n_atoms}, got {realized}"
        )
    model_xyz = os.path.join(work, "model.xyz")
    temporary_link = f"{model_xyz}.{os.getpid()}.tmp"
    if os.path.lexists(temporary_link):
        os.unlink(temporary_link)
    os.symlink(cached_xyz, temporary_link)
    os.replace(temporary_link, model_xyz)
    throughput = run_case(work, gpu)
    completed[n_atoms] = throughput
    _write_results(output, completed)
    print(  # noqa: T201
        f"[nep] N={n_atoms:>9d} tp={throughput:9.1f} atoms/ms",
        flush=True,
    )
    return throughput


def main() -> None:
    gpu = os.environ.get("BENCH_GPU", "0")
    work = os.path.join(HERE, "work_nep")
    out = os.path.join(HERE, "results", "nep.csv")
    manifest = os.path.join(HERE, "results", "nep.manifest.json")
    os.makedirs(work, exist_ok=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_runin(work)

    done: dict[int, float] = {}
    fresh = os.environ.get("BENCH_FRESH", "0") == "1"
    identity = {
        "protocol": "gpumd-nep-v2",
        "potential": benchmark_manifest.file_identity(NEP, content_hash=True),
        "gpumd": benchmark_manifest.file_identity(GPUMD, content_hash=False),
        "runner": benchmark_manifest.file_identity(__file__, content_hash=True),
        "geometry": benchmark_manifest.file_identity(
            gen_system.__file__,
            content_hash=True,
        ),
        "gpu": benchmark_manifest.gpu_identity(gpu),
        "warmup_steps": WARMUP,
        "measured_steps": MEASURE,
        "coarse_start": COARSE_START,
        "coarse_step": COARSE_STEP,
        "refinement_steps": list(REFINEMENT_STEPS),
    }
    benchmark_manifest.publish(
        manifest,
        identity,
        fresh=fresh,
        has_results=os.path.exists(out),
    )
    if fresh and os.path.exists(out):
        os.unlink(out)
    if os.path.exists(out) and not fresh:
        prev = np.loadtxt(out, delimiter=",", skiprows=1, ndmin=2)
        done = {int(r[0]): float(r[1]) for r in prev}

    # === Step 1. Preserve the small-system throughput curve ===
    for target in gen_system.SMALL_TARGETS:
        throughput = _measure_target(
            target,
            work=work,
            gpu=gpu,
            output=out,
            completed=done,
        )
        if not np.isfinite(throughput):
            print(  # noqa: T201
                "[nep] failed below the adaptive capacity range",
                flush=True,
            )
            _write_results(out, done)
            return

    # === Step 2. Locate a two-million-atom failure bracket ===
    lower_target = 0
    upper_target = COARSE_START
    while True:
        throughput = _measure_target(
            upper_target,
            work=work,
            gpu=gpu,
            output=out,
            completed=done,
        )
        if not np.isfinite(throughput):
            break
        lower_target = upper_target
        upper_target += COARSE_STEP

    # === Step 3. Refine the final bracket to half-million resolution ===
    if lower_target > 0:
        for resolution in REFINEMENT_STEPS:
            target = lower_target + resolution
            throughput = _measure_target(
                target,
                work=work,
                gpu=gpu,
                output=out,
                completed=done,
            )
            if np.isfinite(throughput):
                lower_target = target
            else:
                upper_target = target

    lower_atoms = gen_system.atom_count(lower_target) if lower_target else 0
    upper_atoms = gen_system.atom_count(upper_target)
    print(  # noqa: T201
        f"[nep] capacity bracket: {lower_atoms} successful, {upper_atoms} failed",
        flush=True,
    )
    _write_results(out, done)
    print(f"[nep] scan done -> {out}", flush=True)  # noqa: T201


if __name__ == "__main__":
    main()

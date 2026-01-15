# SPDX-License-Identifier: LGPL-3.0-or-later
"""LAMMPS + Kokkos throughput scan over diamond supercells.

The scan retains the logarithmic small-system grid below one million atoms.
From one million onward it advances by two million until the first failure,
then bisects the final successful interval at one-million and half-million
resolution. This finds the capacity ceiling without imposing a manually
selected maximum system size.

The reported throughput is ``n_atoms / loop_ms_per_step`` from LAMMPS' final
timed run and therefore covers the complete MD step.

Environment:
    BENCH_MODEL   absolute path to the .pt2 package
    BENCH_TAG     short label -> results/lmp_<tag>.csv and the work directory
    BENCH_GPU     CUDA device index
    BENCH_GPUS    optional comma-separated CUDA device list for MPI runs;
                   omitted means devices 0..BENCH_NPROCS-1
    BENCH_NPROCS  MPI process count; ``1`` uses the direct LAMMPS executable
    BENCH_MPIEXEC MPI launcher (defaults to ``mpiexec.gforker``)
    BENCH_RESULT_DIR result directory (defaults to ``results``)
    BENCH_WORK_ROOT work-directory root (defaults to the script directory)
    BENCH_DP_LIB  DeePMD shared-library directory (defaults to deepmd-kit_cpp/lib)
    BENCH_PLUGIN_PATH  directory LAMMPS loads the DeePMD styles from at startup
                   (defaults to the installed plugin; empty for a LAMMPS that
                   carries the styles itself)
    BENCH_LD_PRELOAD  optional colon-separated shared libraries loaded first
    BENCH_FRESH   ignore an existing result CSV when set to 1
    BENCH_INCLUDE_SMALL  retain the sub-million grid; defaults to 1
    BENCH_DENSE_SMALL  use the denser sub-million grid for small-capacity models
    BENCH_TARGET_GRID  comma-separated explicit target sizes; skips adaptive search
    BENCH_LOG_SCALE  force logarithmic capacity search for single-GPU runs
    BENCH_TIMEOUT per-case timeout in seconds; defaults to 1200

The scan is resumable: realized system sizes already present in the CSV are
reused, and the merged result is rewritten in ascending atom-count order.
"""

from __future__ import (
    annotations,
)

import math
import os
import re
import subprocess

import gen_system
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LMP = os.environ.get(
    "BENCH_LMP",
    "/nas/outisli/Software/lammps/lammps-patch_4Jul2026/bin/lmp",
)
MPI_RANK_EXEC = os.path.join(HERE, "mpi_rank_exec.sh")
MPIEXEC = os.environ.get(
    "BENCH_MPIEXEC",
    "/nas/outisli/Software/miniforge3/envs/dpmd/bin/mpiexec.gforker",
)
DP_LIB = os.environ.get("BENCH_DP_LIB", "/nas/outisli/Software/deepmd-kit_cpp/lib")
PLUGIN_PATH = os.environ.get("BENCH_PLUGIN_PATH", os.path.join(DP_LIB, "deepmd_lmp"))
TORCH_LIB = os.environ.get(
    "BENCH_TORCH_LIB",
    "/nas/outisli/Software/miniforge3/envs/dpmd/lib/python3.13/site-packages/torch/lib",
)
WARMUP, NSTEPS = 10, 100
COARSE_START = 1_000_000
COARSE_STEP = 2_000_000
REFINEMENT_STEPS = (1_000_000, 500_000)
MULTI_LOG_FACTOR = 1.5
MULTI_CAPACITY_STEPS = {
    2: 1_000_000,
    4: 2_500_000,
    8: 2_500_000,
}
SYSTEM_CACHE = os.path.join(HERE, "systems")
DENSE_SMALL_TARGETS = (
    512,
    750,
    1_000,
    1_500,
    2_016,
    3_000,
    4_032,
    5_000,
    6_000,
    8_000,
    10_000,
    12_000,
    16_016,
    20_000,
    24_000,
    32_256,
    40_000,
    48_000,
    64_000,
    80_000,
    96_000,
    129_168,
    160_000,
    192_000,
    253_952,
    320_000,
    384_000,
    499_200,
)
#: Serving flavors of the scan. A flavor pairs the LAMMPS input script with the
#: geometry writer it requires, so the two never disagree: the spin script
#: declares ``atom_style spin`` and is served by ``dpa4spin``, which needs a
#: data file carrying per-atom moments. Both flavors place the same atoms in
#: the same box, so their curves are comparable point by point.
FLAVORS = {
    "energy": ("in.lammps", "ensure_data"),
    "spin": ("in_spin.lammps", "ensure_spin_data"),
    "meam": ("in_meam.lammps", "ensure_data"),
    "tersoff": ("in_tersoff.lammps", "ensure_data"),
}
KOKKOS_HALF_FLAVORS = frozenset(("meam", "tersoff"))
OOM_PATTERN = re.compile(
    r"(out of memory|memory allocation|cudaErrorMemoryAllocation|"
    r"failed to allocate|alloc failed|std::bad_alloc|Kokkos[^\n]*alloc)",
    re.IGNORECASE,
)


def _record_failure(work: str, output: str) -> None:
    path = os.path.join(work, "failure.log")
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        stream.write(output)
    os.replace(temporary, path)


def run_case(
    work: str,
    data_file: str,
    model: str,
    visible_gpus: str,
    script: str = "in.lammps",
    nprocs: int = 1,
    mpiexec: str = MPIEXEC,
    use_kokkos: bool = True,
    kokkos_half: bool = False,
    potential: str = "energy",
    lmp: str = LMP,
    plugin_path: str = PLUGIN_PATH,
) -> float:
    """Return whole-step throughput (atoms/ms), or NaN on OOM/failure.

    The metric is ``n_atoms / loop_ms_per_step`` from LAMMPS' last (timed) run,
    i.e. the complete MD step under the selected LAMMPS backend -- directly
    comparable to GPUMD's per-step ``Speed of this run`` for the NEP reference.

    ``lmp`` selects the LAMMPS executable. A non-empty ``plugin_path`` is the
    directory LAMMPS loads its plugins from at startup (``LAMMPS_PLUGIN_PATH``),
    which serves the DeePMD styles to a LAMMPS built without them; an empty one
    removes the variable, so that no plugin overrides the styles built into an
    ``lmp`` that carries them itself.
    """
    os.makedirs(work, exist_ok=True)
    env = dict(os.environ)
    if plugin_path:
        env["LAMMPS_PLUGIN_PATH"] = plugin_path
    else:
        env.pop("LAMMPS_PLUGIN_PATH", None)
    if nprocs == 1 or visible_gpus:
        env["CUDA_VISIBLE_DEVICES"] = visible_gpus
    else:
        # Kokkos and DeepPotPTExpt both map MPI local rank to the logical CUDA
        # device. Leaving the process-wide visibility unchanged makes rank r
        # use cuda:r; remapping a non-contiguous subset here would make the
        # raw Kokkos pointer and the torch device disagree.
        env.pop("CUDA_VISIBLE_DEVICES", None)
    env["OMP_NUM_THREADS"] = "1"
    env["LD_LIBRARY_PATH"] = (
        DP_LIB + ":" + TORCH_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
    )
    preload = os.environ.get("BENCH_LD_PRELOAD")
    if preload:
        env["LD_PRELOAD"] = preload + ":" + env.get("LD_PRELOAD", "")
    command = [lmp]
    if use_kokkos:
        command.extend(["-k", "on", "g", str(nprocs)])
        if kokkos_half:
            command.extend(["-pk", "kokkos", "neigh", "half"])
        command.extend(["-sf", "kk"])
    command.extend(
        [
            "-in",
            os.path.join(HERE, script),
            "-var",
            "datafile",
            data_file,
            "-var",
            "model",
            model,
            "-var",
            "potential",
            potential,
            "-var",
            "warmup",
            str(WARMUP),
            "-var",
            "nsteps",
            str(NSTEPS),
        ]
    )
    if nprocs > 1:
        env["HWLOC_COMPONENTS"] = "-cuda,-nvml,-gl,-opencl,-rsmi"
        env.pop("MPICH_GPU_SUPPORT_ENABLED", None)
        command = [mpiexec, "-n", str(nprocs), MPI_RANK_EXEC, *command]
    try:
        proc = subprocess.run(
            command,
            cwd=work,
            env=env,
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("BENCH_TIMEOUT", "1200")),
        )
    except subprocess.TimeoutExpired as error:
        output = (error.stdout or "") + "\n" + (error.stderr or "")
        _record_failure(work, "LAMMPS benchmark timed out\n" + output)
        raise RuntimeError(f"LAMMPS benchmark timed out in {work}") from error
    combined_output = proc.stdout + "\n" + proc.stderr
    loops = re.findall(
        r"Loop time of ([\d.eE+-]+) on \d+ procs for (\d+) steps", proc.stdout
    )
    if proc.returncode != 0 or len(loops) < 2:
        _record_failure(work, combined_output)
        if OOM_PATTERN.search(combined_output):
            return float("nan")
        raise RuntimeError(
            f"LAMMPS benchmark failed without an OOM signature in {work}; "
            "see failure.log"
        )
    failure_log = os.path.join(work, "failure.log")
    if os.path.exists(failure_log):
        os.unlink(failure_log)
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


def _measure_target(
    target: int,
    *,
    work: str,
    model: str,
    gpu: str,
    output: str,
    tag: str,
    completed: dict[int, float],
    flavor: str,
    nprocs: int,
    mpiexec: str,
    use_kokkos: bool,
    kokkos_half: bool,
    potential: str,
) -> float:
    """Measure one requested size or reuse its persisted result.

    Parameters
    ----------
    target
        Requested atom count before diamond-supercell rounding.
    work
        Per-model benchmark working directory.
    model
        Absolute frozen-model path.
    gpu
        CUDA device index.
    output
        Result CSV path.
    tag
        Benchmark label.
    completed
        Results keyed by realized atom count.
    flavor
        Serving flavor selecting the input script and the geometry writer.

    Returns
    -------
    float
        Whole-step throughput in atoms/ms, or NaN on failure.
    """
    script, provider = FLAVORS[flavor]
    n_atoms = gen_system.atom_count(target)
    if n_atoms in completed:
        throughput = completed[n_atoms]
        print(  # noqa: T201
            f"[{tag}] reuse N={n_atoms:>9d} tp={throughput:9.1f} atoms/ms",
            flush=True,
        )
        return throughput

    data_file, realized = getattr(gen_system, provider)(target, SYSTEM_CACHE)
    if realized != n_atoms:
        raise RuntimeError(
            f"Cached diamond size mismatch: expected {n_atoms}, got {realized}"
        )
    throughput = run_case(
        work,
        data_file,
        model,
        gpu,
        script,
        nprocs=nprocs,
        mpiexec=mpiexec,
        use_kokkos=use_kokkos,
        kokkos_half=kokkos_half,
        potential=potential,
    )
    completed[n_atoms] = throughput
    _write_results(output, completed, tag)
    print(  # noqa: T201
        f"[{tag}] N={n_atoms:>9d} tp={throughput:9.1f} atoms/ms",
        flush=True,
    )
    return throughput


def _next_log_target(target: int, factor: float, resolution: int) -> int:
    """Return the next logarithmically spaced target on a fixed grid."""
    grown = math.ceil(target * factor / resolution) * resolution
    return max(grown, target + resolution)


def _explicit_target_grid(value: str) -> tuple[int, ...]:
    """Parse and validate an explicit target-size grid."""
    targets = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    if not targets or any(target <= 0 for target in targets):
        raise ValueError("BENCH_TARGET_GRID must contain positive target sizes")
    return targets


def main() -> None:
    tag = os.environ["BENCH_TAG"]
    gpu = os.environ.get("BENCH_GPU", "0")
    nprocs = int(os.environ.get("BENCH_NPROCS", "1"))
    use_kokkos = os.environ.get("BENCH_KOKKOS", "1") == "1"
    force_log_scale = os.environ.get("BENCH_LOG_SCALE", "0") == "1"
    visible_gpus = os.environ.get("BENCH_GPUS", "") if nprocs > 1 else gpu
    if nprocs < 1:
        raise ValueError(f"BENCH_NPROCS must be positive, got {nprocs}")
    if nprocs > 1:
        gpu_indices = [index.strip() for index in visible_gpus.split(",")]
        if visible_gpus and (len(gpu_indices) != nprocs or not all(gpu_indices)):
            raise ValueError(
                "BENCH_GPUS must be empty or contain exactly BENCH_NPROCS "
                f"comma-separated device indices, got {visible_gpus!r} "
                f"for {nprocs} processes"
            )
        if not os.path.isfile(MPIEXEC):
            raise FileNotFoundError(f"MPI launcher does not exist: {MPIEXEC}")
        if not os.access(MPI_RANK_EXEC, os.X_OK):
            raise PermissionError(
                f"MPI rank wrapper is not executable: {MPI_RANK_EXEC}"
            )
    model = os.environ["BENCH_MODEL"]
    flavor = os.environ.get("BENCH_FLAVOR", "energy")
    if flavor not in FLAVORS:
        raise ValueError(f"unknown BENCH_FLAVOR {flavor!r}")
    script, _ = FLAVORS[flavor]
    kokkos_half = flavor in KOKKOS_HALF_FLAVORS
    work_root = os.environ.get("BENCH_WORK_ROOT", HERE)
    result_dir = os.environ.get("BENCH_RESULT_DIR", os.path.join(HERE, "results"))
    work = os.path.join(work_root, f"work_lmp_{tag}")
    out = os.path.join(result_dir, f"lmp_{tag}.csv")
    os.makedirs(work, exist_ok=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    done: dict[int, float] = {}
    fresh = os.environ.get("BENCH_FRESH", "0") == "1"
    include_small = os.environ.get("BENCH_INCLUDE_SMALL", "1") == "1"
    dense_small = os.environ.get("BENCH_DENSE_SMALL", "0") == "1"
    target_grid = os.environ.get("BENCH_TARGET_GRID")
    if fresh and os.path.exists(out):
        os.unlink(out)
    if os.path.exists(out) and not fresh:
        prev = np.loadtxt(out, delimiter=",", skiprows=1, ndmin=2)
        done = {int(r[0]): float(r[1]) for r in prev}

    # === Step 1. Preserve the small-system throughput curve ===
    small_targets = DENSE_SMALL_TARGETS if dense_small else gen_system.SMALL_TARGETS
    for target in small_targets if include_small else ():
        throughput = _measure_target(
            target,
            work=work,
            model=model,
            gpu=visible_gpus,
            output=out,
            tag=tag,
            completed=done,
            flavor=flavor,
            nprocs=nprocs,
            mpiexec=MPIEXEC,
            use_kokkos=use_kokkos,
            kokkos_half=kokkos_half,
            potential=flavor,
        )
        if not np.isfinite(throughput):
            print(  # noqa: T201
                f"[{tag}] failed below the adaptive capacity range",
                flush=True,
            )
            _write_results(out, done, tag)
            return

    if target_grid is not None:
        for target in _explicit_target_grid(target_grid):
            throughput = _measure_target(
                target,
                work=work,
                model=model,
                gpu=visible_gpus,
                output=out,
                tag=tag,
                completed=done,
                flavor=flavor,
                nprocs=nprocs,
                mpiexec=MPIEXEC,
                use_kokkos=use_kokkos,
                kokkos_half=kokkos_half,
                potential=flavor,
            )
            if not np.isfinite(throughput):
                break
        _write_results(out, done, tag)
        print(f"[{tag}] explicit grid done -> {out}", flush=True)  # noqa: T201
        return

    if nprocs > 1 or force_log_scale:
        log_factor = float(os.environ.get("BENCH_LOG_FACTOR", str(MULTI_LOG_FACTOR)))
        default_resolution = MULTI_CAPACITY_STEPS.get(nprocs, 500_000)
        resolution = int(os.environ.get("BENCH_CAPACITY_STEP", str(default_resolution)))
        if log_factor <= 1.0 or resolution <= 0:
            raise ValueError(
                "BENCH_LOG_FACTOR must exceed one and BENCH_CAPACITY_STEP "
                "must be positive"
            )

        # === Step 2. Locate a logarithmic multi-GPU failure bracket ===
        lower_target = 0
        upper_target = COARSE_START
        while True:
            throughput = _measure_target(
                upper_target,
                work=work,
                model=model,
                gpu=visible_gpus,
                output=out,
                tag=tag,
                completed=done,
                flavor=flavor,
                nprocs=nprocs,
                mpiexec=MPIEXEC,
                use_kokkos=use_kokkos,
                kokkos_half=kokkos_half,
                potential=flavor,
            )
            if not np.isfinite(throughput):
                break
            lower_target = upper_target
            upper_target = _next_log_target(upper_target, log_factor, resolution)

        # === Step 3. Bisect to the requested multi-GPU resolution ===
        while upper_target - lower_target > resolution:
            midpoint = ((lower_target + upper_target) // 2) // resolution
            midpoint *= resolution
            if midpoint <= lower_target:
                midpoint = lower_target + resolution
            throughput = _measure_target(
                midpoint,
                work=work,
                model=model,
                gpu=visible_gpus,
                output=out,
                tag=tag,
                completed=done,
                flavor=flavor,
                nprocs=nprocs,
                mpiexec=MPIEXEC,
                use_kokkos=use_kokkos,
            )
            if np.isfinite(throughput):
                lower_target = midpoint
            else:
                upper_target = midpoint

        lower_atoms = gen_system.atom_count(lower_target) if lower_target else 0
        upper_atoms = gen_system.atom_count(upper_target)
        print(  # noqa: T201
            f"[{tag}] capacity bracket: {lower_atoms} successful, {upper_atoms} failed",
            flush=True,
        )
        _write_results(out, done, tag)
        print(f"[{tag}] scan done -> {out}", flush=True)  # noqa: T201
        return

    # === Step 2. Locate a two-million-atom failure bracket ===
    lower_target = 0
    upper_target = COARSE_START
    while True:
        throughput = _measure_target(
            upper_target,
            work=work,
            model=model,
            gpu=visible_gpus,
            output=out,
            tag=tag,
            completed=done,
            flavor=flavor,
            nprocs=nprocs,
            mpiexec=MPIEXEC,
            use_kokkos=use_kokkos,
            kokkos_half=kokkos_half,
            potential=flavor,
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
                model=model,
                gpu=visible_gpus,
                output=out,
                tag=tag,
                completed=done,
                flavor=flavor,
                nprocs=nprocs,
                mpiexec=MPIEXEC,
                use_kokkos=use_kokkos,
                kokkos_half=kokkos_half,
                potential=flavor,
            )
            if np.isfinite(throughput):
                lower_target = target
            else:
                upper_target = target

    lower_atoms = gen_system.atom_count(lower_target) if lower_target else 0
    upper_atoms = gen_system.atom_count(upper_target)
    print(  # noqa: T201
        f"[{tag}] capacity bracket: {lower_atoms} successful, {upper_atoms} failed",
        flush=True,
    )
    _write_results(out, done, tag)
    print(f"[{tag}] scan done -> {out}", flush=True)  # noqa: T201


if __name__ == "__main__":
    main()

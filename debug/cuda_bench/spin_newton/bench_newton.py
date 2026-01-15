# SPDX-License-Identifier: LGPL-3.0-or-later
"""Whole-step timing of Newtonian spin dynamics for the DPA4C grades.

Every point runs ``in_spin_newton.lammps`` through the Kokkos ``dpa4spin/kk``
of the DeePMD plugin in the OutisLi LAMMPS fork on one GPU and records the loop time
of the timed run together with its LAMMPS timing breakdown (Pair, Neigh, Comm,
Output, Modify, Other) per step. The sizes and step counts match
``bench_sizes.py``: about 500 atoms (launch- and host-bound), 10,000 atoms
(mixed) and 100,000 atoms (device-throughput bound). Repeats are the outer
loop, so the grades alternate and a slow drift of the device does not bias
one grade.

``--system bcn`` runs boron, carbon and nitrogen at random on the sites and
moments of the carbon system (``gen_system.write_bcn_spin_data``), which
differs from it in the atom types alone, and ``--system five`` five elements
(``gen_system.write_five_spin_data``).

``--newton 0`` runs the reference step without Newtonian spin integration:
the same pair style and timestep with fixed moments and the device integrator
``nve/kk``, so the difference to ``--newton 1`` is the cost of the host
``fix nve/spin``.

Usage
-----
    python bench_newton.py --label baseline --repeats 3
    python bench_newton.py --label base_dev --newton 0 --grades nano,plus

``--lib`` puts library directories (a colon-separated list) first on the
library path, which runs a saved build of the same executable or another
DeePMD-kit operator library for A/B comparisons; ``--models`` selects the
directory of the ``<grade>.pt2`` models;
``--var NAME=VALUE`` passes further input variables (for example ``sort``,
``rebuild`` and ``check``), ``--pk`` a package setting such as
``"kokkos comm/forward host"``, and ``--steps`` replaces the per-size step count.
"""

from __future__ import (
    annotations,
)

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import gen_system
import machine

machine.load()
LMP = os.environ["BENCH_SPIN_LMP"]
DP_LIB = os.environ["BENCH_DP_LIB"]
TORCH_LIB = os.environ["BENCH_TORCH_LIB"]
INPUT = os.path.join(HERE, "in_spin_newton.lammps")
MODELS = os.path.join(os.path.dirname(HERE), "models", "spin_random")
SYSTEMS = os.path.join(HERE, "systems")
#: Data writer and LAMMPS elements of every benchmark system.
SYSTEM_DATA = {
    "carbon": gen_system.ensure_spin_data,
    "bcn": gen_system.ensure_bcn_spin_data,
    "five": gen_system.ensure_five_spin_data,
}
SYSTEM_ELEMENTS = {
    "carbon": "C",
    "bcn": " ".join(name for name, _ in gen_system.BCN_ELEMENTS),
    "five": " ".join(name for name, _ in gen_system.FIVE_ELEMENTS),
}
GRADES = ("nano", "mini", "neo", "air", "plus")
SIZES = (512, 10_296, 100_800)
#: Timed steps per size; the warm-up is a tenth of it.
STEPS = {512: 2000, 10_296: 1000, 100_800: 200}
SECTIONS = ("Pair", "Neigh", "Comm", "Output", "Modify", "Other")
GPU_IDLE_TIMEOUT = 600.0


def _steps(n_atoms: int, override: int | None = None) -> int:
    if override:
        return override
    return STEPS.get(n_atoms, max(50, int(2e7 // max(n_atoms, 1))))


def wait_gpu_idle(gpu: str) -> None:
    """Block until the GPU runs no compute process, or fail after a timeout."""
    deadline = time.monotonic() + GPU_IDLE_TIMEOUT
    while subprocess.run(
        ["bash", os.path.join(HERE, "gpu_idle.sh"), gpu], capture_output=True
    ).returncode:
        if time.monotonic() > deadline:
            raise RuntimeError(f"GPU {gpu} stayed busy for {GPU_IDLE_TIMEOUT} s")
        time.sleep(5.0)


def environment(gpu: str, lib: str | None) -> dict[str, str]:
    """Return the run environment: one GPU, the plugin of the fork, dev libraries.

    ``lib`` replaces the DeePMD library directory alone; the plugin is the one
    of ``BENCH_SPIN_PLUGIN_PATH``.
    """
    env = dict(os.environ)
    env["LAMMPS_PLUGIN_PATH"] = os.environ["BENCH_SPIN_PLUGIN_PATH"]
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["OMP_NUM_THREADS"] = "1"
    env["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torchinductor_claude_dpa4c"
    paths = [lib] if lib else []
    paths += [DP_LIB, TORCH_LIB, env.get("LD_LIBRARY_PATH", "")]
    env["LD_LIBRARY_PATH"] = ":".join(path for path in paths if path)
    return env


def parse_timed_run(output: str) -> dict[str, float]:
    """Return loop time, atom count, step count and per-section times of the last run.

    Parameters
    ----------
    output
        LAMMPS screen output.

    Returns
    -------
    dict[str, float]
        ``loop_s``, ``steps``, ``n_atoms``, ``neigh_builds`` and the average
        time in seconds of every breakdown section.
    """
    loops = list(
        re.finditer(
            r"Loop time of ([\d.eE+-]+) on \d+ procs for (\d+) steps with (\d+) atoms",
            output,
        )
    )
    if len(loops) < 2:
        raise RuntimeError("LAMMPS output holds fewer than two timed runs")
    tail = output[loops[-1].end() :]
    result = {
        "loop_s": float(loops[-1].group(1)),
        "steps": float(loops[-1].group(2)),
        "n_atoms": float(loops[-1].group(3)),
        "neigh_builds": float(
            re.search(r"Neighbor list builds = (\d+)", tail).group(1)
        ),
    }
    for section in SECTIONS:
        match = re.search(
            rf"^{section}\s*\|\s*[\d.eE+-]*\s*\|\s*([\d.eE+-]+)\s*\|",
            tail,
            re.MULTILINE,
        )
        result[section] = float(match.group(1)) if match else 0.0
    return result


def run_point(
    grade: str,
    n_atoms: int,
    newton: int,
    gpu: str,
    work: str,
    lmp: str,
    lib: str | None,
    extra: list[str] | None = None,
    steps: int | None = None,
    package: list[str] | None = None,
    models: str = MODELS,
    system: str = "carbon",
) -> dict[str, float]:
    """Run one timed point and return its parsed timings."""
    data_file, realized = SYSTEM_DATA[system](n_atoms, SYSTEMS)
    if realized != n_atoms:
        raise RuntimeError(f"expected {n_atoms} atoms, got {realized}")
    steps = _steps(n_atoms, steps)
    os.makedirs(work, exist_ok=True)
    log = os.path.join(work, f"log_{grade}_{n_atoms}_newton{newton}.lammps")
    command = [
        lmp,
        "-k",
        "on",
        "g",
        "1",
        "-sf",
        "kk",
        "-in",
        INPUT,
        "-log",
        log,
        "-var",
        "datafile",
        data_file,
        "-var",
        "model",
        os.path.join(models, f"{grade}.pt2"),
        "-var",
        "warmup",
        str(max(10, steps // 10)),
        "-var",
        "nsteps",
        str(steps),
        "-var",
        "newton",
        str(newton),
        "-var",
        "elements",
        SYSTEM_ELEMENTS[system],
    ]
    for pair in extra or []:
        command += ["-var", *pair.split("=", 1)]
    for setting in package or []:
        command += ["-pk", *setting.split()]
    wait_gpu_idle(gpu)
    proc = subprocess.run(
        command,
        cwd=work,
        env=environment(gpu, lib),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"LAMMPS failed for {grade} N={n_atoms}:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
        )
    return parse_timed_run(proc.stdout)


def main() -> None:
    """Run every (repeat, size, grade) point and write per-run and summary CSVs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--grades", default=",".join(GRADES))
    parser.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--newton", type=int, choices=(0, 1), default=1)
    parser.add_argument("--system", choices=sorted(SYSTEM_DATA), default="carbon")
    parser.add_argument("--gpu", default=os.environ["BENCH_SPIN_GPU"])
    parser.add_argument("--lmp", default=LMP)
    parser.add_argument(
        "--lib", default=None, help="library directories put first, colon-separated"
    )
    parser.add_argument(
        "--models", default=MODELS, help="directory of the <grade>.pt2 models"
    )
    parser.add_argument("--var", action="append", help="input variable NAME=VALUE")
    parser.add_argument("--steps", type=int, default=None, help="timed steps per point")
    parser.add_argument(
        "--pk", action="append", help='package setting, e.g. "kokkos comm/forward host"'
    )
    args = parser.parse_args()

    grades = args.grades.split(",")
    sizes = [int(s) for s in args.sizes.split(",")]
    work = os.path.join("/tmp/spin_newton/bench", args.label)
    results_dir = os.path.join(HERE, "results")
    os.makedirs(results_dir, exist_ok=True)
    runs: dict[tuple[str, int], list[dict[str, float]]] = {}
    per_run = os.path.join(results_dir, f"newton_{args.label}_runs.csv")
    with open(per_run, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["grade", "n_atoms", "repeat", "ms_per_step", "atoms_per_ms"]
            + [f"{s.lower()}_ms" for s in SECTIONS]
            + ["neigh_builds"]
        )
        for repeat in range(args.repeats):
            for n_atoms in sizes:
                for grade in grades:
                    timing = run_point(
                        grade,
                        n_atoms,
                        args.newton,
                        args.gpu,
                        work,
                        args.lmp,
                        args.lib,
                        args.var,
                        args.steps,
                        args.pk,
                        args.models,
                        args.system,
                    )
                    runs.setdefault((grade, n_atoms), []).append(timing)
                    step_ms = timing["loop_s"] * 1e3 / timing["steps"]
                    writer.writerow(
                        [
                            grade,
                            n_atoms,
                            repeat,
                            f"{step_ms:.4f}",
                            f"{n_atoms / step_ms:.1f}",
                        ]
                        + [f"{timing[s] * 1e3 / timing['steps']:.4f}" for s in SECTIONS]
                        + [int(timing["neigh_builds"])]
                    )
                    stream.flush()
                    print(  # noqa: T201
                        f"{args.label} {grade:5s} {n_atoms:7d} r{repeat}: "
                        f"{step_ms:8.3f} ms/step {n_atoms / step_ms:9.1f} atoms/ms",
                        flush=True,
                    )

    summary = os.path.join(results_dir, f"newton_{args.label}.csv")
    with open(summary, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["grade", "n_atoms", "ms_per_step", "atoms_per_ms"]
            + [f"{s.lower()}_ms" for s in SECTIONS]
            + ["runs"]
        )
        print(  # noqa: T201
            "| Grade | Atoms | ms/step | atoms/ms | "
            + " | ".join(SECTIONS)
            + " |\n|---|---:|---:|---:|"
            + "---:|" * len(SECTIONS)
        )
        for grade in grades:
            for n_atoms in sizes:
                points = runs[(grade, n_atoms)]
                # The median run by loop time supplies the breakdown, so the
                # sections of one row add up to its step time.
                median = sorted(points, key=lambda p: p["loop_s"])[len(points) // 2]
                steps = median["steps"]
                step_ms = median["loop_s"] * 1e3 / steps
                sections = [median[s] * 1e3 / steps for s in SECTIONS]
                spread = [p["loop_s"] * 1e3 / p["steps"] for p in points]
                writer.writerow(
                    [grade, n_atoms, f"{step_ms:.4f}", f"{n_atoms / step_ms:.1f}"]
                    + [f"{v:.4f}" for v in sections]
                    + [" ".join(f"{v:.4f}" for v in spread)]
                )
                print(  # noqa: T201
                    f"| {grade} | {n_atoms} | {step_ms:.3f} | {n_atoms / step_ms:.0f} | "
                    + " | ".join(f"{v:.3f}" for v in sections)
                    + f" |  (runs: {', '.join(f'{v:.3f}' for v in spread)}; "
                    f"stdev {statistics.pstdev(spread):.3f})",
                    flush=True,
                )
    print(f"wrote {summary} and {per_run}", flush=True)  # noqa: T201


if __name__ == "__main__":
    main()

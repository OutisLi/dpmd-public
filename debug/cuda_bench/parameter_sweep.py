# SPDX-License-Identifier: LGPL-3.0-or-later
"""Scan DPA4C configurations on LAMMPS + Kokkos.

Two independent scans share this driver.

The model grades Nano through Plus are the released architectures, each pairing
its own channel width, angular degree, mode rank and fitting width. They form
the DPA4C half of the nine-curve capacity figure.

The structural sweeps hold the fitting width fixed per channel width and vary
one kernel parameter at a time: the radial mode rank at the production angular
degree, and the angular degree without radial modes. Their shared corner is
scanned once and consumed by both sweep figures.

Every GPU worker freezes and scans its assigned configurations sequentially.
Workers share immutable cached diamond systems, while each model retains an
independent result CSV and LAMMPS work directory.
"""

from __future__ import (
    annotations,
)

import argparse
import concurrent.futures
import csv
import os
import queue
import subprocess
import threading
from dataclasses import (
    dataclass,
)
from typing import (
    TYPE_CHECKING,
)

import benchmark_runtime
import numpy as np

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.environ.get(
    "BENCH_PYTHON", "/nas/outisli/Software/miniforge3/envs/dpmd/bin/python"
)
CHANNELS = (8, 16, 32, 64, 128)
RADIAL_MODES = (0, 2, 4, 8)
ANGULAR_DEGREES = (2, 3, 4)
BASE_DEGREE = 2
BASE_MODES = 0
FITTING_WIDTH = {8: 64, 16: 64, 32: 128, 64: 256, 128: 256}
FITTING_DEPTH = 3
SUMMARY = os.path.join(HERE, "results", "parameter_sweep_summary.csv")


@dataclass(frozen=True)
class Configuration:
    """Identify one benchmarked architecture and its result tag.

    The family separates the two scans in the shared summary. A grade and a
    sweep point can agree on every descriptor parameter while differing in
    fitting width, so the figures select on the family rather than on the
    architecture.
    """

    tag: str
    family: str
    channels: int
    lmax: int
    radial_modes: int
    fitting_width: int


def _sweep_point(channels: int, lmax: int, radial_modes: int) -> Configuration:
    """Build a structural-sweep point at the fixed fitting width."""
    return Configuration(
        f"dpa4c_c{channels}_l{lmax}_r{radial_modes}",
        "sweep",
        channels,
        lmax,
        radial_modes,
        FITTING_WIDTH[channels],
    )


#: Released model grades, in ascending cost. Each grade fixes every parameter
#: that the compiled kernel and the fitting width depend on.
GRADES = (
    Configuration("dpa4c_nano", "grade", 8, 2, 0, 96),
    Configuration("dpa4c_mini", "grade", 32, 2, 0, 192),
    Configuration("dpa4c_neo", "grade", 64, 2, 0, 256),
    Configuration("dpa4c_air", "grade", 64, 3, 4, 256),
    Configuration("dpa4c_plus", "grade", 128, 3, 4, 384),
)


def configurations(grades_only: bool = False) -> list[Configuration]:
    """Enumerate the configurations to freeze and scan.

    Parameters
    ----------
    grades_only
        Restrict the run to the five released grades. A change that leaves the
        mode and degree kernels untouched only needs those rescanned.

    Returns
    -------
    list[Configuration]
        Configurations to freeze and scan, without repeated tags.
    """
    if grades_only:
        return list(GRADES)
    grid = [
        _sweep_point(channels, BASE_DEGREE, modes)
        for channels in CHANNELS
        for modes in RADIAL_MODES
    ]
    grid.extend(
        _sweep_point(channels, lmax, BASE_MODES)
        for channels in CHANNELS
        for lmax in ANGULAR_DEGREES
        if lmax != BASE_DEGREE
    )
    return list(GRADES) + grid


def _model_path(configuration: Configuration) -> str:
    return os.path.join(HERE, "models", "sweep", configuration.tag + ".pt2")


def _result_path(configuration: Configuration) -> str:
    return os.path.join(HERE, "results", "lmp_" + configuration.tag + ".csv")


def _freeze(configuration: Configuration, model_path: str, gpu: str) -> None:
    env = dict(os.environ)
    env["DP_CUDA_INFER"] = "2"
    env["CUDA_VISIBLE_DEVICES"] = gpu
    temporary = model_path + ".tmp.pt2"
    if os.path.exists(temporary):
        os.unlink(temporary)
    subprocess.run(
        [
            PYTHON,
            os.path.join(HERE, "freeze_dpa4c.py"),
            "--channels",
            str(configuration.channels),
            "--lmax",
            str(configuration.lmax),
            "--radial-modes",
            str(configuration.radial_modes),
            "--fitting-width",
            str(configuration.fitting_width),
            "--fitting-depth",
            str(FITTING_DEPTH),
            "--out",
            temporary,
        ],
        cwd=HERE,
        env=env,
        check=True,
    )
    os.replace(temporary, model_path)


def _scan(configuration: Configuration, gpu: str, fresh: bool) -> None:
    env = dict(os.environ)
    env.update(
        {
            "BENCH_MODEL": _model_path(configuration),
            "BENCH_TAG": configuration.tag,
            "BENCH_GPU": gpu,
            "BENCH_INCLUDE_SMALL": "1",
            "BENCH_FRESH": "1" if fresh else "0",
        }
    )
    subprocess.run(
        [PYTHON, os.path.join(HERE, "lmp_scan.py")],
        cwd=HERE,
        env=env,
        check=True,
    )


def _freeze_worker(
    gpu: str,
    assigned: list[Configuration],
    fresh: bool,
) -> None:
    for configuration in assigned:
        model_path = _model_path(configuration)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        if fresh or not os.path.exists(model_path):
            print(f"[gpu {gpu}] freeze {configuration.tag}", flush=True)  # noqa: T201
            _freeze(configuration, model_path, gpu)
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model freeze did not produce {model_path}")


def _scan_worker(
    gpu: str,
    assigned: list[Configuration],
    fresh: bool,
) -> None:
    for configuration in assigned:
        print(f"[gpu {gpu}] scan {configuration.tag}", flush=True)  # noqa: T201
        _scan(configuration, gpu, fresh)


def _run_parallel(
    gpus: list[str],
    grid: list[Configuration],
    worker: Callable[[str, list[Configuration], bool], None],
    fresh: bool,
) -> None:
    """Run the grid across the devices, one configuration at a time per device.

    The devices pull from a shared queue rather than from a fixed share of the
    grid, so a device that finishes early takes the next configuration instead
    of idling while another device works through a backlog. The queue is
    ordered by descending expected cost, which keeps the longest runs from
    starting last.

    Parameters
    ----------
    gpus
        CUDA device indices to occupy.
    grid
        Configurations to process.
    worker
        Callable applied to one configuration on one device.
    fresh
        Whether to ignore cached models and results.
    """
    pending = queue.SimpleQueue()
    for configuration in sorted(
        grid,
        key=lambda item: (
            item.channels,
            item.lmax,
            item.radial_modes,
            item.fitting_width,
        ),
        reverse=True,
    ):
        pending.put(configuration)
    stop = threading.Event()

    def consume(gpu: str) -> None:
        while not stop.is_set():
            try:
                configuration = pending.get_nowait()
            except queue.Empty:
                return
            try:
                worker(gpu, [configuration], fresh)
            except (OSError, RuntimeError, subprocess.SubprocessError, ValueError):
                stop.set()
                raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(consume, gpu) for gpu in gpus]
        for future in futures:
            future.result()


def summarize(grid: list[Configuration]) -> str:
    """Write the saturated throughput and capacity of every configuration.

    Parameters
    ----------
    grid
        Configurations whose result CSVs are complete.

    Returns
    -------
    str
        Path of the summary CSV.
    """
    temporary = SUMMARY + ".tmp"
    os.makedirs(os.path.dirname(SUMMARY), exist_ok=True)
    with open(temporary, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "tag",
                "family",
                "channels",
                "lmax",
                "radial_modes",
                "fitting_width",
                "saturated_atoms_per_ms",
                "max_atoms",
                "first_failed_atoms",
                "result_csv",
            ]
        )
        for configuration in grid:
            result_path = _result_path(configuration)
            values = np.loadtxt(result_path, delimiter=",", skiprows=1, ndmin=2)
            finite = np.isfinite(values[:, 1])
            saturated = finite & (values[:, 0] >= 1_000_000)
            throughput = (
                float(np.mean(values[saturated, 1]))
                if np.any(saturated)
                else float("nan")
            )
            max_atoms = int(np.max(values[finite, 0])) if np.any(finite) else 0
            failed = values[~finite, 0]
            writer.writerow(
                [
                    configuration.tag,
                    configuration.family,
                    configuration.channels,
                    configuration.lmax,
                    configuration.radial_modes,
                    configuration.fitting_width,
                    f"{throughput:.9g}",
                    max_atoms,
                    int(np.min(failed)) if failed.size else 0,
                    os.path.relpath(result_path, HERE),
                ]
            )
    os.replace(temporary, SUMMARY)
    return SUMMARY


def main() -> None:
    """Parse worker settings and execute the complete parameter grid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="re-freeze models and overwrite every result CSV",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="assume the installed DeePMD-kit and LAMMPS binaries are current",
    )
    parser.add_argument(
        "--grades-only",
        action="store_true",
        help="scan only the five released model grades",
    )
    args = parser.parse_args()
    benchmark_runtime.preflight(
        args.gpus, require_nep=False, require_build_trees=not args.skip_build
    )
    if not args.skip_build:
        benchmark_runtime.build_runtime()

    grid = configurations(args.grades_only)
    if args.fresh:
        stale = [SUMMARY, os.path.join(HERE, "dpa4c_grade_benchmark.png")]
        if not args.grades_only:
            stale.extend(
                os.path.join(HERE, f"dpa4c_{name}_sweep_c{channels}.png")
                for name in ("rank", "degree")
                for channels in CHANNELS
            )
        for path in stale:
            if os.path.exists(path):
                os.unlink(path)

    _run_parallel(args.gpus, grid, _freeze_worker, args.fresh)
    _run_parallel(args.gpus, grid, _scan_worker, args.fresh)

    summary = summarize(grid)
    scripts = ["plot.py"]
    if not args.grades_only:
        scripts += ["plot_rank_sweep.py", "plot_degree_sweep.py"]
    for script in scripts:
        subprocess.run([PYTHON, os.path.join(HERE, script)], cwd=HERE, check=True)
    print(f"parameter sweep complete -> {summary}", flush=True)  # noqa: T201


if __name__ == "__main__":
    main()

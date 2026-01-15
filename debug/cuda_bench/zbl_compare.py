# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compare every DPA4C grade with and without ZBL zone bridging on LAMMPS + Kokkos.

Each released grade is frozen twice from one architecture, as a plain model and
as a zone-bridging composition with the ZBL repulsion, and both are timed on the
same diamond supercells.

The two variants of a grade alternate on one GPU, size by size and repeat by
repeat, in an order reversed on every other repeat, so that device and drift
affect them alike. The reported difference is the median over the repeats of
the per-repeat step-time ratio.

The models carry untrained weights, which is immaterial for the cost of a step
but not for the trajectory: see ``in_zbl.lammps`` for how the protocol keeps
both variants on the same configurations.

Usage
-----
    python zbl_compare.py --gpus 3 4 5 6 7
"""

from __future__ import (
    annotations,
)

import argparse
import concurrent.futures
import csv
import os
import statistics
import subprocess

import benchmark_runtime
import gen_system
import lmp_scan
from parameter_sweep import (
    FITTING_DEPTH,
    GRADES,
    PYTHON,
    Configuration,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models", "zbl_comparison")
RESULTS = os.path.join(HERE, "results", "zbl_comparison")
SCRIPT = "in_zbl.lammps"
#: Requested atom counts: two points of the rising curve and two saturated ones.
TARGETS = (32_000, 256_000, 1_000_000, 3_000_000)
VARIANTS = ("plain", "zbl")
#: One timing: atom count, repeat index, variant and throughput in atoms/ms of
#: the whole MD step.
Row = tuple[int, int, str, float]


def _model_path(grade: Configuration, variant: str) -> str:
    suffix = "" if variant == "plain" else "_zbl"
    return os.path.join(MODELS, grade.tag + suffix + ".pt2")


def _freeze(grade: Configuration, variant: str, gpu: str) -> None:
    """Freeze one variant of a grade unless its archive exists."""
    model_path = _model_path(grade, variant)
    if os.path.exists(model_path):
        return
    os.makedirs(MODELS, exist_ok=True)
    env = dict(os.environ)
    env["DP_CUDA_INFER"] = "2"
    env["CUDA_VISIBLE_DEVICES"] = gpu
    temporary = model_path + ".tmp.pt2"
    subprocess.run(
        [
            PYTHON,
            os.path.join(HERE, "freeze_dpa4c.py"),
            "--channels",
            str(grade.channels),
            "--lmax",
            str(grade.lmax),
            "--radial-modes",
            str(grade.radial_modes),
            "--fitting-width",
            str(grade.fitting_width),
            "--fitting-depth",
            str(FITTING_DEPTH),
            *(["--zbl"] if variant == "zbl" else []),
            "--out",
            temporary,
        ],
        cwd=HERE,
        env=env,
        check=True,
    )
    os.replace(temporary, model_path)


def _measure_grade(
    grade: Configuration,
    gpu: str,
    repeats: int,
    work_root: str,
) -> list[Row]:
    """Time both variants of one grade on one GPU."""
    for variant in VARIANTS:
        _freeze(grade, variant, gpu)
    rows = []
    for target in TARGETS:
        data_file, atoms = gen_system.ensure_data(target, lmp_scan.SYSTEM_CACHE)
        for repeat in range(repeats):
            throughput = {}
            for variant in VARIANTS if repeat % 2 == 0 else VARIANTS[::-1]:
                throughput[variant] = lmp_scan.run_case(
                    os.path.join(work_root, f"work_zbl_{grade.tag}_{variant}"),
                    data_file,
                    _model_path(grade, variant),
                    gpu,
                    SCRIPT,
                )
            rows.extend(
                (atoms, repeat, variant, throughput[variant]) for variant in VARIANTS
            )
            readings = "  ".join(
                f"{variant} {throughput[variant]:9.1f}" for variant in VARIANTS
            )
            print(  # noqa: T201
                f"[gpu {gpu}] {grade.tag} N={atoms:>8d} repeat {repeat}: "
                f"{readings} atoms/ms",
                flush=True,
            )
    return rows


def _write(grade: Configuration, rows: list[Row]) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, grade.tag + ".csv")
    with open(path + ".tmp", "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["n_atoms", "repeat", "variant", "atoms_per_ms"])
        writer.writerows(rows)
    os.replace(path + ".tmp", path)


def _step_time_increase(
    rows: list[Row], atoms: int, variant: str, reference: str
) -> float:
    """Median over the repeats of the step-time increase of a variant, in percent.

    Step time is the reciprocal of throughput; the ratio is taken per repeat so
    that drift between repeats cancels.
    """
    readings = {(row[1], row[2]): row[3] for row in rows if row[0] == atoms}
    repeats = sorted({row[1] for row in rows if row[0] == atoms})
    return statistics.median(
        100.0 * (readings[repeat, reference] / readings[repeat, variant] - 1.0)
        for repeat in repeats
    )


def summarize(results: dict[str, list[Row]]) -> str:
    """Write the median throughput and the relative cost per grade and size.

    ``zbl_cost_percent`` is the step-time increase of the bridged model over the
    plain one.
    """
    path = os.path.join(RESULTS, "summary.csv")
    with open(path + ".tmp", "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["tag", "n_atoms", "variant", "atoms_per_ms", "zbl_cost_percent"]
        )
        for tag, rows in results.items():
            for atoms in sorted({row[0] for row in rows}):
                for variant in VARIANTS:
                    throughput = statistics.median(
                        row[3] for row in rows if row[0] == atoms and row[2] == variant
                    )
                    zbl_cost = (
                        f"{_step_time_increase(rows, atoms, variant, 'plain'):.3f}"
                        if variant == "zbl"
                        else ""
                    )
                    writer.writerow(
                        [tag, atoms, variant, f"{throughput:.6g}", zbl_cost]
                    )
    os.replace(path + ".tmp", path)
    return path


def main() -> None:
    """Freeze and time every grade, one grade per GPU at a time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", default=["4", "5", "6", "7"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--work-root",
        default=HERE,
        help="directory that receives the LAMMPS work directories",
    )
    args = parser.parse_args()
    benchmark_runtime.preflight(args.gpus, require_nep=False, require_build_trees=False)
    if not os.access(lmp_scan.LMP, os.X_OK):
        raise FileNotFoundError(f"LAMMPS executable: {lmp_scan.LMP}")
    plugin = os.path.join(lmp_scan.PLUGIN_PATH, "dpplugin.so")
    if lmp_scan.PLUGIN_PATH and not os.path.exists(plugin):
        raise FileNotFoundError(f"DeePMD LAMMPS plugin: {plugin}")

    # The costliest grades start first so that no GPU idles behind them.
    grades = sorted(GRADES, key=lambda grade: grade.channels * grade.lmax, reverse=True)
    results: dict[str, list[Row]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        free = list(args.gpus)
        pending = {}
        for grade in grades:
            if not free:
                finished = next(concurrent.futures.as_completed(pending))
                finished_grade, gpu = pending.pop(finished)
                results[finished_grade.tag] = finished.result()
                _write(finished_grade, results[finished_grade.tag])
                free.append(gpu)
            gpu = free.pop(0)
            future = pool.submit(
                _measure_grade, grade, gpu, args.repeats, args.work_root
            )
            pending[future] = (grade, gpu)
        for future, (grade, _gpu) in pending.items():
            results[grade.tag] = future.result()
            _write(grade, results[grade.tag])
    ordered = {grade.tag: results[grade.tag] for grade in GRADES}
    print(f"ZBL comparison complete -> {summarize(ordered)}", flush=True)  # noqa: T201


if __name__ == "__main__":
    main()

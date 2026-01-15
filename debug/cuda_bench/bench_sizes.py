# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Whole-step LAMMPS + Kokkos throughput of the DPA4C grades at fixed sizes.

The capacity scan of ``lmp_scan.py`` measures one 100-step run per size. This
driver instead targets the three sizes that bracket the inference regimes --
about 500 atoms (launch- and host-bound), 10,000 atoms (mixed) and 100,000
atoms (device-throughput bound) -- and repeats every point, alternating the
models on each device, so that a few-percent change is resolved rather than
lost in the run-to-run spread. The step counts fall with the size (``STEPS``),
so that no size dominates the time of a campaign.

Environment (beyond those of ``lmp_scan.py``):
    BENCH_LMP      LAMMPS executable (defaults to the plugin-mode build of the
                   public release)
    BENCH_DP_LIB   DeePMD C++ library directory (defaults to the development
                   install of the working tree)

Usage
-----
    python bench_sizes.py --label baseline --models models/v20260911_trained_rcf \
        --flavor bcn --gpus 0,1,2,3 --repeats 3
"""

from __future__ import (
    annotations,
)

import argparse
import csv
import os
import queue
import statistics
import threading

HERE = os.path.dirname(os.path.abspath(__file__))

import gen_system
import lmp_scan

GRADES = ("nano", "mini", "neo", "air", "plus")
SIZES = (512, 10_296, 100_800)
#: Timed steps per size; the warm-up is a tenth of it.
STEPS = {512: 2000, 10_296: 1000, 100_800: 200}


def _steps(n_atoms: int) -> int:
    return STEPS.get(n_atoms, max(50, int(2e7 // max(n_atoms, 1))))


def run_point(model: str, n_atoms: int, gpu: str, work: str, flavor: str) -> float:
    """Return the whole-step throughput of one run in atoms/ms."""
    script, writer = lmp_scan.FLAVORS[flavor]
    data_file, realized = getattr(gen_system, writer)(
        n_atoms, os.path.join(HERE, "systems")
    )
    if realized != n_atoms:
        raise RuntimeError(f"expected {n_atoms} atoms, got {realized}")
    steps = _steps(n_atoms)
    return lmp_scan.run_case(
        work,
        data_file,
        model,
        gpu,
        script=script,
        warmup=max(10, steps // 10),
        nsteps=steps,
    )


def main() -> None:
    """Run every (grade, size, repeat) point and write one summary CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--models", default=os.path.join(HERE, "models", "v20260911"))
    parser.add_argument("--grades", default=",".join(GRADES))
    parser.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    parser.add_argument("--gpus", default=",".join(os.environ["BENCH_GPUS"].split()))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--flavor", default="energy", choices=sorted(lmp_scan.FLAVORS))
    args = parser.parse_args()

    grades = args.grades.split(",")
    sizes = [int(s) for s in args.sizes.split(",")]
    suffix = "" if args.flavor == "energy" else f"_{args.flavor}"
    # The spin flavor serves the spin variant of every grade; every other
    # flavor serves the energy model of the grade on its own system.
    model_suffix = "_spin" if args.flavor == "spin" else ""
    jobs: queue.Queue = queue.Queue()
    # Repeats are the outer loop, so the models alternate on every device.
    for repeat in range(args.repeats):
        for n_atoms in sizes:
            for grade in grades:
                jobs.put((grade, n_atoms, repeat))
    results: dict[tuple[str, int], list[float]] = {}
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                grade, n_atoms, repeat = jobs.get_nowait()
            except queue.Empty:
                return
            model = os.path.join(args.models, f"{grade}{model_suffix}.pt2")
            work = os.path.join(HERE, f"work_sizes_{args.label}{suffix}", f"gpu{gpu}")
            value = run_point(model, n_atoms, gpu, work, args.flavor)
            with lock:
                results.setdefault((grade, n_atoms), []).append(value)
                print(  # noqa: T201
                    f"{args.label} {grade:5s} {n_atoms:7d} r{repeat} gpu{gpu}: "
                    f"{value:9.1f} atoms/ms",
                    flush=True,
                )

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus.split(",")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    path = os.path.join(HERE, "results", f"sizes_{args.label}{suffix}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["grade", "n_atoms", "median_atoms_per_ms", "step_ms", "runs"])
        for grade in grades:
            for n_atoms in sizes:
                values = results.get((grade, n_atoms), [])
                median = statistics.median(values) if values else float("nan")
                writer.writerow(
                    [
                        grade,
                        n_atoms,
                        f"{median:.1f}",
                        f"{n_atoms / median:.4f}",
                        " ".join(f"{v:.1f}" for v in values),
                    ]
                )
    print(f"wrote {path}", flush=True)  # noqa: T201
    for grade in grades:
        row = []
        for n_atoms in sizes:
            values = results.get((grade, n_atoms), [])
            median = statistics.median(values) if values else float("nan")
            row.append(f"{n_atoms}: {median:9.1f} ({n_atoms / median:7.3f} ms)")
        print(f"{grade:5s} " + " | ".join(row), flush=True)  # noqa: T201


if __name__ == "__main__":
    main()

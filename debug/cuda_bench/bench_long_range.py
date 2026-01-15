# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Whole-step LAMMPS + Kokkos time of compressed DPA4C with and without long range.

Every case runs the trained Air archive of ``freeze_long_range.py`` without a
module (``sr``), with an LES module (``les``) or with an SOG module (``sog``)
on the BCN diamond systems of ``gen_system.write_bcn_data``, periodic
(``p_*``, archives whose module serves periodic systems) or as a
shrink-wrapped cluster of the same atoms (``s_*``). Every point is repeated,
the cases alternating within a repeat and rotating between repeats, and the
median step time is recorded with the spread of the repeats.

With ``--dump`` every case instead runs once for a short trajectory and
writes the positions and forces after it, in full precision, for the
reproducibility check of ``device_suite.sh``.

Environment: those of ``lmp_scan.py`` (machine profile of ``machine.sh``).

Usage
-----
    python bench_long_range.py --label h20 --models results/.../models/long_range \
        --gpus 0,1 --repeats 3
    python bench_long_range.py --label h20 --models ... --dump results/.../dump_a
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

import gen_system
import lmp_scan

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = (512, 10_296, 100_800)
#: Timed steps per size; the warm-up is a tenth of it.
STEPS = {512: 1000, 10_296: 400, 100_800: 100}
#: Steps of the trajectory of a reproducibility dump.
DUMP_STEPS = 20
#: Case name -> (archive, boundary of every dimension).
CASES = {
    "s_sr": ("air.pt2", "s"),
    "s_les": ("air_les.pt2", "s"),
    "s_sog": ("air_sog.pt2", "s"),
    "p_sr": ("air.pt2", "p"),
    "p_les": ("air_les_p.pt2", "p"),
    "p_sog": ("air_sog_p.pt2", "p"),
}


def run_point(
    case: str, models: str, n_atoms: int, gpu: str, dumpfile: str | None
) -> float:
    """Return the whole-step time of one run in ms."""
    archive, boundary = CASES[case]
    data_file, realized = gen_system.ensure_bcn_data(
        n_atoms, os.path.join(HERE, "systems")
    )
    if realized != n_atoms:
        raise RuntimeError(f"expected {n_atoms} atoms, got {realized}")
    steps = DUMP_STEPS if dumpfile else STEPS.get(n_atoms, 100)
    throughput = lmp_scan.run_case(
        os.path.join(HERE, "work_long_range", f"{case}_{n_atoms}_gpu{gpu}"),
        data_file,
        os.path.join(models, archive),
        gpu,
        script="in_long_range.lammps",
        warmup=10 if dumpfile else max(10, steps // 10),
        nsteps=steps,
        variables={
            "boundary": " ".join(boundary * 3),
            "dump": "1" if dumpfile else "0",
            "dumpfile": dumpfile or "none",
        },
    )
    if not throughput > 0.0:
        raise RuntimeError(f"{case} at {n_atoms} atoms ran out of device memory")
    return n_atoms / throughput


def main() -> None:
    """Run every (case, size, repeat) point and write one summary CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    parser.add_argument("--gpus", default=",".join(os.environ["BENCH_GPUS"].split()))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dump", help="directory of the reproducibility dumps")
    args = parser.parse_args()
    cases = args.cases.split(",")
    sizes = [int(s) for s in args.sizes.split(",")]
    repeats = 1 if args.dump else args.repeats
    if args.dump:
        os.makedirs(args.dump, exist_ok=True)

    # Each device takes the next point, so that the devices alternate the
    # cases of a repeat rather than each running one case throughout.
    points: queue.Queue = queue.Queue()
    for n_atoms in sizes:
        for repeat in range(repeats):
            shift = repeat % len(cases)
            for case in cases[shift:] + cases[:shift]:
                points.put((case, n_atoms))
    times: dict[tuple[str, int], list[float]] = {}
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                case, n_atoms = points.get_nowait()
            except queue.Empty:
                return
            dumpfile = (
                os.path.join(os.path.abspath(args.dump), f"{case}_{n_atoms}.dump")
                if args.dump
                else None
            )
            try:
                elapsed = run_point(case, args.models, n_atoms, gpu, dumpfile)
            except BaseException as error:  # reported by the main thread
                with lock:
                    errors.append(error)
                return
            with lock:
                times.setdefault((case, n_atoms), []).append(elapsed)
                print(  # noqa: T201
                    f"{case:6s} {n_atoms:7d} atoms gpu {gpu}: {elapsed:8.4f} ms/step",
                    flush=True,
                )

    threads = [
        threading.Thread(target=worker, args=(gpu,)) for gpu in args.gpus.split(",")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]

    rows = []
    for n_atoms in sizes:
        for case in cases:
            values = times[(case, n_atoms)]
            median = statistics.median(values)
            rows.append(
                {
                    "case": case,
                    "atoms": n_atoms,
                    "ms_per_step": f"{median:.4f}",
                    "spread": f"{(max(values) - min(values)) / median:.4f}",
                    "repeats": len(values),
                }
            )
    out = os.path.join(HERE, "results", f"long_range_{args.label}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(  # noqa: T201
            f"{row['case']:6s} {row['atoms']:7d} atoms  {float(row['ms_per_step']):8.4f}"
            f" ms/step  spread {row['spread']}"
        )
    print(f"-> {out}")  # noqa: T201


if __name__ == "__main__":
    main()

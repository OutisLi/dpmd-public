# SPDX-License-Identifier: LGPL-3.0-or-later
"""Alternate two LAMMPS builds while preserving every run's input and output."""

from __future__ import (
    annotations,
)

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import (
    Path,
)

BENCH = Path(__file__).resolve().parents[1] / "cuda_bench"
sys.path.insert(0, str(BENCH))

import bench_sizes
import lmp_scan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--points", default="nano:512,nano:10296,nano:100800,plus:512,plus:100800"
    )
    parser.add_argument("--flavor", choices=("energy", "bcn", "spin"), default="energy")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--steps", type=int, help="Override measured steps per point")
    parser.add_argument("--plugin-a")
    parser.add_argument("--plugin-b")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    libraries = {"A": args.a.resolve(), "B": args.b.resolve()}
    plugins = {
        "A": args.plugin_a or os.environ["BENCH_SPIN_PLUGIN_PATH"],
        "B": args.plugin_b or os.environ["BENCH_SPIN_PLUGIN_PATH"],
    }
    original_path = os.environ.get("LD_LIBRARY_PATH", "")
    if args.flavor == "spin":
        sys.path.insert(0, str(BENCH / "spin_newton"))
        import bench_newton
    summaries = []
    for point in args.points.split(","):
        grade, size = point.split(":")
        atoms = int(size)
        if args.steps is not None:
            bench_sizes.STEPS[atoms] = args.steps
        samples = {"A": [], "B": []}
        for repeat in range(args.repeats):
            for side in ("A", "B") if repeat % 2 == 0 else ("B", "A"):
                lmp_scan.LMP = str(libraries[side] / "lmp")
                lmp_scan.PLUGIN_PATH = plugins[side]
                os.environ["BENCH_SPIN_PLUGIN_PATH"] = plugins[side]
                os.environ["LD_LIBRARY_PATH"] = (
                    str(libraries[side]) + ":" + original_path
                )
                work = args.output / f"{grade}_{atoms}_{repeat}_{side}"
                if args.flavor == "spin":
                    timing = bench_newton.run_point(
                        grade,
                        atoms,
                        1,
                        args.gpu,
                        str(work),
                        lmp_scan.LMP,
                        str(libraries[side]),
                        steps=args.steps,
                    )
                    throughput = atoms / (1000 * timing["loop_s"] / timing["steps"])
                else:
                    throughput = bench_sizes.run_point(
                        str(
                            BENCH / "models" / "v20260911_trained_rcf" / f"{grade}.pt2"
                        ),
                        atoms,
                        args.gpu,
                        str(work),
                        args.flavor,
                    )
                if not math.isfinite(throughput) or throughput <= 0:
                    raise RuntimeError(f"Invalid throughput {throughput}: {work}")
                milliseconds = atoms / throughput
                samples[side].append(milliseconds)
                record = {
                    "grade": grade,
                    "atoms": atoms,
                    "repeat": repeat,
                    "side": side,
                    "step_ms": milliseconds,
                }
                with (args.output / "runs.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(record, flush=True)  # noqa: T201
        medians = {side: statistics.median(values) for side, values in samples.items()}
        summary = {
            "grade": grade,
            "atoms": atoms,
            "medians_ms": medians,
            "samples_ms": samples,
            "reduction_percent": 100 * (1 - medians["B"] / medians["A"]),
        }
        summaries.append(summary)
        (args.output / "summary.json").write_text(
            json.dumps(summaries, indent=2) + "\n"
        )
        print(summary, flush=True)  # noqa: T201


if __name__ == "__main__":
    main()

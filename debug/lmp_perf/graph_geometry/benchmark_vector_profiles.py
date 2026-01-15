# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compare unchanged model archives on the installed and working vector paths."""

from __future__ import (
    annotations,
)

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import (
    Path,
)

BENCH = Path(__file__).resolve().parents[2] / "cuda_bench"
sys.path.insert(0, str(BENCH))

import machine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--grades", default="mini,neo,air,plus")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    machine.load()
    root = args.artifacts.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline = Path(os.environ["BENCH_DP_LIB"])
    library = {"before": baseline, "after": root / "integrated"}
    plugins = {"before": root / "base", "after": root / "integrated"}
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES=args.gpu,
        OMP_NUM_THREADS="1",
        DP_INTER_OP_PARALLELISM_THREADS="0",
        DP_INTRA_OP_PARALLELISM_THREADS="0",
    )
    summaries = []
    for grade in args.grades.split(","):
        samples: dict[str, list[float]] = {"before": [], "after": []}
        for repeat in range(args.repeats):
            order = ("before", "after") if repeat % 2 == 0 else ("after", "before")
            for side in order:
                work = output / f"{grade}_{repeat}_{side}"
                work.mkdir()
                (work / "input.lmp").write_text((BENCH / "in_bcn.lammps").read_text())
                environment["DP_BACKEND_PLUGIN_PATH"] = str(library[side])
                environment["LAMMPS_PLUGIN_PATH"] = str(plugins[side])
                environment["LD_LIBRARY_PATH"] = ":".join(
                    [
                        str(root / "lammps"),
                        str(library[side]),
                        os.environ["BENCH_TORCH_LIB"],
                        str(baseline),
                        os.environ.get("LD_LIBRARY_PATH", ""),
                    ]
                )
                command = [
                    str(root / "lammps/lmp"),
                    "-k",
                    "on",
                    "g",
                    "1",
                    "-sf",
                    "kk",
                    "-in",
                    "input.lmp",
                    "-var",
                    "datafile",
                    str(BENCH / "systems/diamond_v1_100800.bcn.data"),
                    "-var",
                    "model",
                    str(BENCH / "models/v20260911_trained_rcf" / f"{grade}.pt2"),
                    "-var",
                    "warmup",
                    str(max(10, args.steps // 10)),
                    "-var",
                    "nsteps",
                    str(args.steps),
                ]
                process = subprocess.run(
                    command,
                    cwd=work,
                    env=environment,
                    text=True,
                    capture_output=True,
                    timeout=600,
                )
                (work / "screen.txt").write_text(process.stdout + process.stderr)
                (work / "command.json").write_text(json.dumps(command, indent=2) + "\n")
                (work / "libraries.json").write_text(
                    json.dumps(
                        {
                            key: environment[key]
                            for key in (
                                "LD_LIBRARY_PATH",
                                "DP_BACKEND_PLUGIN_PATH",
                                "LAMMPS_PLUGIN_PATH",
                            )
                        },
                        indent=2,
                    )
                    + "\n"
                )
                process.check_returncode()
                loops = list(
                    re.finditer(
                        r"Loop time of ([\d.eE+-]+) on \d+ procs for (\d+) steps with (\d+) atoms",
                        process.stdout,
                    )
                )
                assert len(loops) == 2
                last = loops[-1]
                assert int(last.group(2)) == args.steps and int(last.group(3)) == 100800
                tail = process.stdout[last.end() :]
                builds = int(re.search(r"Neighbor list builds = (\d+)", tail).group(1))
                danger = int(re.search(r"Dangerous builds = (\d+)", tail).group(1))
                assert danger == 0, work
                milliseconds = float(last.group(1)) * 1000 / args.steps
                samples[side].append(milliseconds)
                record = {
                    "grade": grade,
                    "side": side,
                    "repeat": repeat,
                    "step_ms": milliseconds,
                    "builds": builds,
                    "danger": danger,
                }
                with (output / "runs.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(record, flush=True)  # noqa: T201
        medians = {side: statistics.median(values) for side, values in samples.items()}
        summary = {
            "grade": grade,
            "atoms": 100800,
            "steps": args.steps,
            "medians_ms": medians,
            "samples_ms": samples,
            "reduction_percent": 100 * (1 - medians["after"] / medians["before"]),
        }
        summaries.append(summary)
        (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        print(summary, flush=True)  # noqa: T201


if __name__ == "__main__":
    main()

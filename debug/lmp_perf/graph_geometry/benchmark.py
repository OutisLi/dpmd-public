# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compare canonical geometry layouts on one fixed LAMMPS runtime and archive."""

from __future__ import (
    annotations,
)

import argparse
import json
import math
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
    parser.add_argument("--flavors", default="bcn,energy,spin")
    parser.add_argument("--sizes", default="100800")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--spin-dt", type=float, default=0.000002)
    args = parser.parse_args()
    machine.load()
    root = args.artifacts.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES=args.gpu,
        OMP_NUM_THREADS="1",
        DP_INTER_OP_PARALLELISM_THREADS="0",
        DP_INTRA_OP_PARALLELISM_THREADS="0",
        DP_BACKEND_PLUGIN_PATH=str(root / "integrated"),
    )
    environment["LD_LIBRARY_PATH"] = ":".join(
        [
            str(root / "integrated"),
            str(root / "lammps"),
            environment["BENCH_DP_LIB"],
            environment["BENCH_TORCH_LIB"],
            environment.get("LD_LIBRARY_PATH", ""),
        ]
    )
    summaries = []
    for flavor in args.flavors.split(","):
        script = {
            "bcn": "in_bcn.lammps",
            "energy": "in.lammps",
            "spin": "in_spin.lammps",
        }[flavor]
        suffix = {"bcn": ".bcn", "energy": "", "spin": ".spin"}[flavor]
        system_directory = "spin_newton/systems" if flavor == "spin" else "systems"
        model_name = "spin_nano.pt2" if flavor == "spin" else "nano.pt2"
        model = root / "models_shape_restored" / model_name
        inputs = (BENCH / script).read_text()
        if flavor == "spin":
            inputs = inputs.replace(
                "timestep        0.001", f"timestep        {args.spin_dt}"
            )
        for atoms in map(int, args.sizes.split(",")):
            data = BENCH / system_directory / f"diamond_v1_{atoms}{suffix}.data"
            if not data.is_file():
                raise FileNotFoundError(data)
            samples: dict[str, list[float]] = {"vector": [], "indexed": []}
            for repeat in range(args.repeats):
                order = (
                    ("vector", "indexed") if repeat % 2 == 0 else ("indexed", "vector")
                )
                for side in order:
                    work = output / f"{flavor}_{atoms}_{repeat}_{side}"
                    work.mkdir()
                    (work / "input.lmp").write_text(inputs)
                    plugin = "vector_plugin" if side == "vector" else "integrated"
                    environment["LAMMPS_PLUGIN_PATH"] = str(root / plugin)
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
                        str(data),
                        "-var",
                        "model",
                        str(model),
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
                    (work / "command.json").write_text(
                        json.dumps(command, indent=2) + "\n"
                    )
                    process.check_returncode()
                    loops = list(
                        re.finditer(
                            r"Loop time of ([\d.eE+-]+) on \d+ procs for (\d+) steps with (\d+) atoms",
                            process.stdout,
                        )
                    )
                    if len(loops) != 2:
                        raise RuntimeError(f"Expected warm-up and timed loops: {work}")
                    last = loops[-1]
                    milliseconds = float(last.group(1)) * 1000 / int(last.group(2))
                    if int(last.group(3)) != atoms or not math.isfinite(milliseconds):
                        raise RuntimeError(f"Invalid timed loop: {work}")
                    tail = process.stdout[last.end() :]
                    builds = int(
                        re.search(r"Neighbor list builds = (\d+)", tail).group(1)
                    )
                    danger = int(re.search(r"Dangerous builds = (\d+)", tail).group(1))
                    if danger:
                        raise RuntimeError(
                            f"Dangerous neighbor builds invalidate timing: {work}"
                        )
                    samples[side].append(milliseconds)
                    record = {
                        "flavor": flavor,
                        "atoms": atoms,
                        "repeat": repeat,
                        "side": side,
                        "step_ms": milliseconds,
                        "builds": builds,
                        "danger": danger,
                    }
                    with (output / "runs.jsonl").open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
                    print(record, flush=True)  # noqa: T201
            medians = {
                side: statistics.median(values) for side, values in samples.items()
            }
            summary = {
                "flavor": flavor,
                "atoms": atoms,
                "steps": args.steps,
                "medians_ms": medians,
                "samples_ms": samples,
                "reduction_percent": 100 * (1 - medians["indexed"] / medians["vector"]),
            }
            summaries.append(summary)
            (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
            print(summary, flush=True)  # noqa: T201


if __name__ == "__main__":
    main()

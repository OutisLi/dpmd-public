# SPDX-License-Identifier: LGPL-3.0-or-later
"""Alternate LAMMPS builds on frozen BCN with a rebuild on every step."""

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

BENCH = Path(__file__).resolve().parents[1] / "cuda_bench"
sys.path.insert(0, str(BENCH))

import machine


def main() -> None:
    machine.load()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--repeats", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    libraries = {"A": args.a.resolve(), "B": args.b.resolve()}
    source = (BENCH / "in_frozen.lammps").read_text()
    source = source.replace("every 1 delay 0 check yes", "every 1 delay 0 check no")
    original = dict(os.environ)
    records = []
    for repeat in range(args.repeats):
        for side in ("A", "B") if repeat % 2 == 0 else ("B", "A"):
            work = args.output / f"{repeat}_{side}"
            work.mkdir()
            (work / "input.lmp").write_text(source)
            env = dict(original)
            env.update(
                CUDA_VISIBLE_DEVICES=args.gpu,
                OMP_NUM_THREADS="1",
                LAMMPS_PLUGIN_PATH=env["BENCH_SPIN_PLUGIN_PATH"],
            )
            env["LD_LIBRARY_PATH"] = ":".join(
                (
                    str(libraries[side]),
                    env["BENCH_DP_LIB"],
                    env["BENCH_TORCH_LIB"],
                    env.get("LD_LIBRARY_PATH", ""),
                )
            )
            command = [
                str(libraries[side] / "lmp"),
                "-k", "on", "g", "1", "-sf", "kk", "-in", "input.lmp",
                "-var", "datafile", str(BENCH / "systems/diamond_v1_100800.bcn.data"),
                "-var", "model", str(BENCH / "models/v20260911_trained_rcf/nano.pt2"),
                "-var", "elements", "B C N",
                "-var", "warmup", "20", "-var", "nsteps", "100",
            ]
            proc = subprocess.run(
                command, cwd=work, env=env, text=True, capture_output=True, timeout=240
            )
            (work / "screen.txt").write_text(proc.stdout + proc.stderr)
            if proc.returncode:
                raise RuntimeError(f"Rebuild benchmark failed; see {work / 'screen.txt'}")
            loops = list(re.finditer(
                r"Loop time of ([\d.eE+-]+) on \d+ procs for (\d+) steps", proc.stdout
            ))
            tail = proc.stdout[loops[-1].end():]
            row = {
                "side": side,
                "repeat": repeat,
                "ms": float(loops[-1][1]) * 1000 / int(loops[-1][2]),
                "builds": int(re.search(r"Neighbor list builds = (\d+)", tail)[1]),
            }
            if row["builds"] != 100:
                raise RuntimeError(f"Expected 100 rebuilds, got {row}: {work}")
            records.append(row)
            (args.output / "runs.json").write_text(json.dumps(records, indent=2) + "\n")
            print(row, flush=True)  # noqa: T201
    medians = {
        side: statistics.median(row["ms"] for row in records if row["side"] == side)
        for side in libraries
    }
    summary = {
        "median_ms": medians,
        "reduction_percent": 100 * (1 - medians["B"] / medians["A"]),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary, flush=True)  # noqa: T201


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: LGPL-3.0-or-later
r"""CUDA runtime calls and memory copies per MD step of a LAMMPS input.

LAMMPS emits no NVTX ranges, so the script profiles the same input twice under
Nsight Systems, with ``nsteps`` timed steps and with twice as many, and divides
the difference of the per-process totals by ``nsteps``: setup, warm-up and
teardown cancel, and what remains is the cost of one step. It reports, per
step, the count and host time of every CUDA runtime call that blocks or copies
(synchronizations and memory copies) and the bytes copied in each direction.

Usage
-----
    python nsys_api_per_step.py --input ../in.lammps --model ../models/v20260911_trained/nano.pt2 \
        --data ../systems/diamond_v1_512.data --steps 1000 --tag nvt_nano512
"""

from __future__ import (
    annotations,
)

import argparse
import collections
import os
import re
import sqlite3
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from bench_newton import (
    LMP,
    environment,
    wait_gpu_idle,
)

#: Runtime calls that block the host or move memory.
BLOCKING = re.compile(r"Synchronize|Memcpy|Memset|cudaFree|cudaMalloc|EventQuery")
COPY_KIND = {1: "HtoD", 2: "DtoH", 8: "DtoD"}


def profile(args: argparse.Namespace, steps: int) -> str:
    """Profile one run with ``steps`` timed steps and return its sqlite export."""
    work = os.path.join("/tmp/spin_newton/nsys_api", f"{args.tag}_{steps}")
    os.makedirs(work, exist_ok=True)
    report = os.path.join(work, "report")
    command = [
        "nsys",
        "profile",
        "--trace=cuda",
        "--sample=none",
        "--cpuctxsw=none",
        "--force-overwrite=true",
        "-o",
        report,
        args.lmp,
        "-k",
        "on",
        "g",
        "1",
        "-sf",
        "kk",
        "-in",
        args.input,
        "-log",
        "none",
        "-var",
        "datafile",
        args.data,
        "-var",
        "model",
        args.model,
        "-var",
        "warmup",
        str(args.warmup),
        "-var",
        "nsteps",
        str(steps),
    ]
    wait_gpu_idle(args.gpu)
    proc = subprocess.run(
        command,
        cwd=work,
        env=environment(args.gpu, None),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"nsys/LAMMPS failed in {work}:\n{proc.stdout[-2000:]}{proc.stderr[-2000:]}"
        )
    subprocess.run(
        [
            "nsys",
            "export",
            "--type=sqlite",
            "--force-overwrite=true",
            "-o",
            report + ".sqlite",
            report + ".nsys-rep",
        ],
        check=True,
        capture_output=True,
    )
    return report + ".sqlite"


def totals(
    database: str,
) -> tuple[collections.Counter, collections.Counter, collections.Counter]:
    """Return call counts and host time per runtime call, and bytes per copy direction."""
    connection = sqlite3.connect(database)
    strings = dict(connection.execute("SELECT id, value FROM StringIds"))
    counts, times, copied = (
        collections.Counter(),
        collections.Counter(),
        collections.Counter(),
    )
    for start, end, name in connection.execute(
        "SELECT start, end, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"
    ):
        label = re.sub(r"_v\d+$", "", strings[name])
        counts[label] += 1
        times[label] += end - start
    for nbytes, kind in connection.execute(
        "SELECT bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY"
    ):
        copied[COPY_KIND.get(kind, str(kind))] += nbytes
        copied[COPY_KIND.get(kind, str(kind)) + " copies"] += 1
    return counts, times, copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--lmp", default=LMP)
    args = parser.parse_args()
    short = totals(profile(args, args.steps))
    long = totals(profile(args, 2 * args.steps))
    lines = [
        f"{args.tag}: per step, from runs of {args.steps} and {2 * args.steps} steps"
    ]
    counts = {k: (long[0][k] - short[0][k]) / args.steps for k in long[0]}
    for label in sorted(counts, key=lambda k: -(long[1][k] - short[1][k])):
        if BLOCKING.search(label) and abs(counts[label]) >= 0.01:
            elapsed = (long[1][label] - short[1][label]) / args.steps / 1e3
            lines.append(f"  {label:32s} {counts[label]:6.2f} calls  {elapsed:9.2f} us")
    for key in ("DtoH", "HtoD", "DtoD"):
        nbytes = (long[2][key] - short[2][key]) / args.steps
        ncopies = (long[2][key + " copies"] - short[2][key + " copies"]) / args.steps
        lines.append(
            f"  copies {key}: {ncopies:5.2f} per step, {nbytes / 1e3:9.1f} kB per step"
        )
    report = "\n".join(lines)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    with open(
        os.path.join(HERE, "logs", f"nsys_api_{args.tag}.txt"), "w", encoding="utf-8"
    ) as stream:
        stream.write(report + "\n")
    print(report)  # noqa: T201


if __name__ == "__main__":
    main()

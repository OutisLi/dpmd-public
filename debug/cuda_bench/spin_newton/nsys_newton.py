# SPDX-License-Identifier: LGPL-3.0-or-later
"""Per-step anatomy of a Newtonian spin LAMMPS step under Nsight Systems.

LAMMPS emits no NVTX ranges, so the timed window is recovered from the kernel
timeline: the patched ``dpa4spin/kk`` launches its force scatter once per step,
and the window spans the last ``steps`` intervals between those launches. For
that window the report gives, per step, the wall span, the device busy time
(union of kernels and memory operations), kernel time by group, every memory
copy by direction together with its bytes and host memory kind, and the host
time of the CUDA runtime calls, which shows where the host waits for the
device.

Usage
-----
    python nsys_newton.py --grade nano --atoms 512 --steps 500 --tag nano512
    python nsys_newton.py --grade plus --atoms 100800 --steps 50 --newton 0 ...
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
sys.path.insert(0, os.path.dirname(HERE))

import gen_system
from bench_newton import (
    INPUT,
    LMP,
    MODELS,
    SYSTEMS,
    environment,
    wait_gpu_idle,
)

#: Kernel-name patterns, checked in order, that group the step budget.
GROUPS = (
    ("dpa4c forward", r"deepmd_dpa4c::forward_kernel"),
    ("dpa4c node bwd", r"deepmd_dpa4c::node_backward_kernel"),
    ("dpa4c edge bwd", r"deepmd_dpa4c::edge_backward_kernel"),
    ("dpa4c other", r"deepmd_dpa4c::"),
    (
        "fitting",
        r"gemm|cutlass|layer_forward|layer_backward|layer_epilogue|backward_epilogue|head_",
    ),
    ("force/virial", r"edge_force_virial|frame_segment"),
    ("graph build", r"CompactCanonicalGraph"),
    ("dpa4spin pair", r"PairDPA4SpinKokkos"),
    ("neighbor", r"NPair|NeighborKokkos|NBin|neigh"),
    ("integrate (device)", r"FixNVESpinKokkos|FixNVEKokkos|FixNVE"),
    ("comm", r"CommKokkos|AtomVec|pack|unpack|Pack|Unpack"),
    ("inductor", r"triton_"),
    ("aten", r"at::native|at::cuda|elementwise|reduce_kernel|vectorized"),
)
#: CUPTI copy kinds and memory kinds.
COPY_KIND = {1: "HtoD", 2: "DtoH", 8: "DtoD", 10: "PtoP"}
MEMORY_KIND = {
    0: "unknown",
    1: "pageable",
    2: "pinned",
    3: "device",
    4: "array",
    5: "managed",
    6: "device_static",
    7: "managed_static",
}


def group_of(name: str) -> str:
    for label, pattern in GROUPS:
        if re.search(pattern, name):
            return label
    return "other"


def union_length(intervals: list[tuple[int, int]]) -> int:
    """Return the total length covered by possibly overlapping intervals."""
    total, start, end = 0, None, None
    for first, last in sorted(intervals):
        if end is None or first > end:
            if end is not None:
                total += end - start
            start, end = first, last
        else:
            end = max(end, last)
    if end is not None:
        total += end - start
    return total


def profile(args: argparse.Namespace) -> tuple[str, str]:
    """Run LAMMPS under nsys and return the sqlite export and the screen output."""
    work = os.path.join("/tmp/spin_newton/nsys", args.tag)
    os.makedirs(work, exist_ok=True)
    data_file, realized = gen_system.ensure_spin_data(args.atoms, SYSTEMS)
    if realized != args.atoms:
        raise RuntimeError(f"expected {args.atoms} atoms, got {realized}")
    report = os.path.join(work, "report")
    for suffix in (".nsys-rep", ".sqlite"):
        if os.path.exists(report + suffix):
            os.unlink(report + suffix)
    command = [
        "nsys",
        "profile",
        "--trace=cuda,osrt",
        "--cuda-graph-trace=node",
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
        INPUT,
        "-var",
        "datafile",
        data_file,
        "-var",
        "model",
        os.path.join(args.models, f"{args.grade}.pt2"),
        "-var",
        "warmup",
        str(args.warmup),
        "-var",
        "nsteps",
        str(args.steps),
        "-var",
        "newton",
        str(args.newton),
    ]
    for pair in args.var or []:
        command += ["-var", *pair.split("=", 1)]
    for setting in args.pk or []:
        command += ["-pk", *setting.split()]
    wait_gpu_idle(args.gpu)
    proc = subprocess.run(
        command,
        cwd=work,
        env=environment(args.gpu, args.lib),
        capture_output=True,
        text=True,
    )
    output = proc.stdout + "\n" + proc.stderr
    with open(os.path.join(work, "lammps.out"), "w", encoding="utf-8") as stream:
        stream.write(output)
    if proc.returncode != 0:
        raise RuntimeError(f"nsys/LAMMPS failed in {work}:\n{output[-3000:]}")
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
    return report + ".sqlite", output


def analyze(database: str, steps: int, output: str) -> str:
    """Return the per-step report of the last ``steps`` steps."""
    connection = sqlite3.connect(database)
    strings = dict(connection.execute("SELECT id, value FROM StringIds"))
    kernels = sorted(
        (start, end, strings[name])
        for start, end, name in connection.execute(
            "SELECT start, end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"
        )
    )
    # The force scatter of the pair style, a lambda of
    # PairDPA4SpinKokkos::compute, runs once per force evaluation.
    counts = collections.Counter(
        k[2]
        for k in kernels
        if re.search(
            r"PairDPA4SpinKokkos<.*>::compute\(int, int\)::\[lambda\(int\)", k[2]
        )
    )
    if not counts:
        raise RuntimeError("no PairDPA4SpinKokkos::compute kernel in the profile")
    marker = counts.most_common(1)[0][0]
    starts = [k[0] for k in kernels if k[2] == marker]
    if len(starts) < steps + 1:
        raise RuntimeError(f"found {len(starts)} steps, need {steps + 1}")
    begin, end = starts[-steps - 1], starts[-1]
    window = [k for k in kernels if begin <= k[0] < end]
    copies = list(
        connection.execute(
            "SELECT start, end, bytes, copyKind, srcKind, dstKind FROM CUPTI_ACTIVITY_KIND_MEMCPY "
            "WHERE start >= ? AND start < ?",
            (begin, end),
        )
    )
    try:
        sets = list(
            connection.execute(
                "SELECT start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE start >= ? AND start < ?",
                (begin, end),
            )
        )
    except sqlite3.OperationalError:
        sets = []
    span = (end - begin) / steps / 1e6
    busy = union_length(
        [(s, e) for s, e, _ in window]
        + [(c[0], c[1]) for c in copies]
        + [(m[0], m[1]) for m in sets]
    )
    kernel_busy = union_length([(s, e) for s, e, _ in window])
    lines = []
    loop = re.findall(r"Loop time of ([\d.eE+-]+) on \d+ procs for (\d+) steps", output)
    if loop:
        lines.append(
            f"LAMMPS loop under nsys: {float(loop[-1][0]) * 1e3 / int(loop[-1][1]):.4f} ms/step"
        )
    lines.append(
        f"window: {steps} steps, {span:.4f} ms/step; device busy {busy / steps / 1e6:.4f} ms/step "
        f"({100 * busy / (end - begin):.1f}%), kernels alone {kernel_busy / steps / 1e6:.4f} ms/step; "
        f"{len(window) / steps:.1f} kernels/step"
    )
    per_group, launches = collections.Counter(), collections.Counter()
    per_kernel = collections.Counter()
    for start, stop, name in window:
        per_group[group_of(name)] += stop - start
        launches[group_of(name)] += 1
        per_kernel[re.sub(r"\(.*", "", name)[:120]] += stop - start
    lines.append("kernel group             ms/step  launches/step")
    for label, value in per_group.most_common():
        lines.append(
            f"  {label:22s} {value / steps / 1e6:8.4f}  {launches[label] / steps:6.1f}"
        )
    lines.append("memory copies per step (kind, src->dst memory):")
    by_kind = collections.defaultdict(lambda: [0, 0, 0])
    for start, stop, nbytes, kind, src, dst in copies:
        key = f"{COPY_KIND.get(kind, kind)} {MEMORY_KIND.get(src, src)}->{MEMORY_KIND.get(dst, dst)}"
        by_kind[key][0] += 1
        by_kind[key][1] += nbytes
        by_kind[key][2] += stop - start
    for key, (count, nbytes, duration) in sorted(by_kind.items()):
        rate = nbytes / duration if duration else 0.0
        lines.append(
            f"  {key:28s} {count / steps:6.2f} copies {nbytes / steps / 1e6:9.4f} MB "
            f"{duration / steps / 1e6:8.4f} ms  ({rate:5.2f} GB/s)"
        )
    lines.append(
        f"  memsets {len(sets) / steps:.2f}/step {sum(m[2] for m in sets) / steps / 1e6:.4f} MB"
    )
    api, api_count = collections.Counter(), collections.Counter()
    for start, stop, name in connection.execute(
        "SELECT start, end, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start >= ? AND start < ?",
        (begin, end),
    ):
        label = re.sub(r"_v\d+$", "", strings[name])
        api[label] += stop - start
        api_count[label] += 1
    lines.append("host CUDA runtime calls per step:")
    for label, value in api.most_common(12):
        lines.append(
            f"  {label:36s} {value / steps / 1e6:8.4f} ms  {api_count[label] / steps:6.2f} calls"
        )
    lines.append("top kernels (ms/step):")
    for name, value in per_kernel.most_common(14):
        lines.append(f"  {value / steps / 1e6:8.4f}  {name}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grade", required=True)
    parser.add_argument("--atoms", type=int, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--newton", type=int, choices=(0, 1), default=1)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--gpu", default=os.environ["BENCH_SPIN_GPU"])
    parser.add_argument("--lmp", default=LMP)
    parser.add_argument(
        "--lib", default=None, help="library directories put first, colon-separated"
    )
    parser.add_argument(
        "--models", default=MODELS, help="directory of the <grade>.pt2 models"
    )
    parser.add_argument(
        "--var", action="append", help="extra LAMMPS variable NAME=VALUE"
    )
    parser.add_argument(
        "--pk", action="append", help='package setting, e.g. "kokkos comm/forward host"'
    )
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    database = os.path.join("/tmp/spin_newton/nsys", args.tag, "report.sqlite")
    if args.analyze_only:
        with open(
            os.path.join(os.path.dirname(database), "lammps.out"), encoding="utf-8"
        ) as stream:
            output = stream.read()
    else:
        database, output = profile(args)
    report = analyze(database, args.steps, output)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    with open(
        os.path.join(HERE, "logs", f"nsys_{args.tag}.txt"), "w", encoding="utf-8"
    ) as stream:
        stream.write(report + "\n")
    print(report)  # noqa: T201


if __name__ == "__main__":
    main()

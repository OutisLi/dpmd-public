# SPDX-License-Identifier: LGPL-3.0-or-later
"""Per-step kernel budget of one LAMMPS + Kokkos run under Nsight Systems.

LAMMPS emits no NVTX ranges, so the timed window is recovered from the kernel
timeline: every MD step ends with one force scatter of the Kokkos pair style,
and the window spans the last ``steps`` steps between those scatters.
Inside it the report gives, per step, the device time of every kernel group,
the union of kernel intervals (device busy time), the launch and memory
operation counts, and the host time of the dominant CUDA API calls.

Usage
-----
    python lmp_nsys.py --model models/v20260911/nano.pt2 --atoms 512 --gpu 0
"""

from __future__ import (
    annotations,
)

import argparse
import collections
import itertools
import os
import re
import sqlite3
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))

import gen_system
import lmp_scan

#: LAMMPS elements of the systems of the flavors that carry several types.
FLAVOR_ELEMENTS = {
    "bcn": " ".join(name for name, _ in gen_system.BCN_ELEMENTS),
    "five": " ".join(name for name, _ in gen_system.FIVE_ELEMENTS),
}

#: Kernel-name patterns, checked in order, that group the step budget.
GROUPS = (
    ("dpa4c forward", r"deepmd_dpa4c::forward_kernel"),
    ("dpa4c node bwd", r"deepmd_dpa4c::node_backward_kernel"),
    ("dpa4c edge bwd", r"deepmd_dpa4c::edge_backward_kernel"),
    ("dpa4c padding", r"zero_padding_kernel"),
    ("dpa4c readout", r"deepmd_dpa4c::forward_readout_kernel"),
    ("fitting fused", r"fused_network_kernel|transpose_weights_kernel"),
    (
        "fitting gemm",
        r"gemm|sgemm|cutlass|Kernel2|ampere|sm\d+_xmma|layer_forward|layer_backward",
    ),
    (
        "fitting epilogue",
        r"layer_epilogue|backward_epilogue|seed_backward|head_kernel|head_finalize",
    ),
    ("force/virial", r"edge_force_virial|frame_segment"),
    ("graph build", r"CompactCanonicalGraphKokkos"),
    ("pair other", r"LAMMPS_NS::Pair\w+Kokkos<"),
    ("neighbor", r"NPair|Neighbor|BinFull|NBin|neigh"),
    ("integrate", r"FixNH|NVT|Verlet|integrate"),
    ("comm", r"Comm|AtomVec|pack|unpack|Pack|Unpack"),
    ("inductor", r"triton_"),
    ("aten", r"at::native|at::cuda|elementwise|reduce_kernel|vectorized"),
)


#: Force scatter of a Kokkos pair style, the kernel that ends every MD step.
PAIR_SCATTER = re.compile(r"ParallelFor<LAMMPS_NS::Pair\w+Kokkos<")


def group_of(name: str) -> str:
    for label, pattern in GROUPS:
        if re.search(pattern, name):
            return label
    return "other"


def union_length(intervals: list[tuple[int, int]]) -> int:
    total, current_start, current_end = 0, None, None
    for start, end in sorted(intervals):
        if current_end is None or start > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        total += current_end - current_start
    return total


def profile(
    model: str,
    atoms: int,
    gpu: str,
    warmup: int,
    steps: int,
    tag: str,
    flavor: str = "energy",
    script: str | None = None,
) -> tuple[str, str]:
    """Run LAMMPS under nsys and return the sqlite export and the log.

    The serving flavor of ``lmp_scan.FLAVORS`` selects the input script and the
    system; ``script`` replaces the input script of the flavor and receives the
    elements of the system as the variable ``elements``.
    """
    work = os.path.join(HERE, "work_nsys", tag)
    os.makedirs(work, exist_ok=True)
    flavor_script, writer = lmp_scan.FLAVORS[flavor]
    script = script or flavor_script
    data_file, realized = getattr(gen_system, writer)(
        atoms, os.path.join(HERE, "systems")
    )
    if realized != atoms:
        raise RuntimeError(f"expected {atoms} atoms, got {realized}")
    env = dict(os.environ)
    if lmp_scan.PLUGIN_PATH:
        env["LAMMPS_PLUGIN_PATH"] = lmp_scan.PLUGIN_PATH
    else:
        env.pop("LAMMPS_PLUGIN_PATH", None)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["OMP_NUM_THREADS"] = "1"
    env["LD_LIBRARY_PATH"] = (
        lmp_scan.DP_LIB
        + ":"
        + lmp_scan.TORCH_LIB
        + ":"
        + env.get("LD_LIBRARY_PATH", "")
    )
    preload = os.environ.get("BENCH_LD_PRELOAD")
    if preload:
        env["LD_PRELOAD"] = preload + ":" + env.get("LD_PRELOAD", "")
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
        lmp_scan.LMP,
        "-k",
        "on",
        "g",
        "1",
        "-sf",
        "kk",
        "-in",
        os.path.join(HERE, script),
        "-var",
        "datafile",
        data_file,
        "-var",
        "model",
        model,
        "-var",
        "potential",
        "energy",
        "-var",
        "warmup",
        str(warmup),
        "-var",
        "nsteps",
        str(steps),
        "-var",
        "elements",
        FLAVOR_ELEMENTS.get(flavor, "C"),
    ]
    proc = subprocess.run(command, cwd=work, env=env, capture_output=True, text=True)
    log = proc.stdout + "\n" + proc.stderr
    with open(os.path.join(work, "lammps.out"), "w", encoding="utf-8") as stream:
        stream.write(log)
    if proc.returncode != 0:
        raise RuntimeError(f"nsys/LAMMPS failed in {work}:\n{log[-3000:]}")
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
    return report + ".sqlite", log


def analyze(database: str, steps: int, log: str) -> str:
    connection = sqlite3.connect(database)
    strings = dict(connection.execute("SELECT id, value FROM StringIds"))
    kernels = [
        (start, end, strings[name])
        for start, end, name in connection.execute(
            "SELECT start, end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"
        )
    ]
    kernels.sort()
    marks = [k[0] for k in kernels if "deepmd_dpa4c::forward_kernel" in k[2]]
    # Every step ends with one force scatter of the Kokkos pair style, so
    # consecutive scatters delimit the steps.
    tile_starts = [k[0] for k in kernels if PAIR_SCATTER.search(k[2])]
    if len(tile_starts) < steps + 1:
        raise RuntimeError(f"found {len(tile_starts)} steps, need {steps + 1}")
    begin, end = tile_starts[-steps - 1], tile_starts[-1]
    window = [k for k in kernels if begin <= k[0] < end]
    # Steps that rebuild the neighbor list carry the LAMMPS build kernel; the
    # others show the cost of a step between rebuilds.
    builds = [k[0] for k in window if "NPairKokkosBuildFunctor" in k[2]]
    bounds = tile_starts[-steps - 1 :]
    plain, rebuild = [], []
    for first, last in itertools.pairwise(bounds):
        has_build = any(first <= t < last for t in builds)
        (rebuild if has_build else plain).append((last - first) / 1e6)
    per_group = collections.Counter()
    per_kernel = collections.Counter()
    launches = collections.Counter()
    for start, stop, name in window:
        per_group[group_of(name)] += stop - start
        short = re.sub(r"\(.*", "", name)[:110]
        per_kernel[short] += stop - start
        launches[group_of(name)] += 1
    busy = union_length([(s, e) for s, e, _ in window])
    span = end - begin
    lines = []
    loop = re.findall(r"Loop time of ([\d.eE+-]+) on \d+ procs for (\d+) steps", log)
    if loop:
        lines.append(
            f"LAMMPS loop: {float(loop[-1][0]) * 1e3 / int(loop[-1][1]):.4f} ms/step"
        )
    lines.append(
        f"profiled window: {span / steps / 1e6:.4f} ms/step, device busy "
        f"{busy / steps / 1e6:.4f} ms/step ({100 * busy / span:.1f}%), "
        f"{len(window) / steps:.1f} kernels/step, {len(marks)} forward launches total"
    )
    if plain:
        plain.sort()
        rebuild.sort()
        lines.append(
            f"steps between rebuilds: {len(plain)}, median {plain[len(plain) // 2]:.4f} ms; "
            f"rebuild steps: {len(rebuild)}"
            + (f", median {rebuild[len(rebuild) // 2]:.4f} ms" if rebuild else "")
        )
    lines.append("group                   ms/step   share   launches/step")
    total = sum(per_group.values())
    for label, value in per_group.most_common():
        lines.append(
            f"  {label:20s} {value / steps / 1e6:8.4f}  {100 * value / total:5.1f}%"
            f"  {launches[label] / steps:6.1f}"
        )
    lines.append("top kernels (ms/step):")
    for name, value in per_kernel.most_common(25):
        lines.append(f"  {value / steps / 1e6:8.4f}  {name}")
    for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        try:
            rows = list(
                connection.execute(
                    f"SELECT start, end FROM {table} WHERE start >= ? AND start < ?",
                    (begin, end),
                )
            )
        except sqlite3.OperationalError:
            rows = []
        lines.append(
            f"{table.split('_')[-1].lower()}: {len(rows) / steps:.1f}/step, "
            f"{sum(e - s for s, e in rows) / steps / 1e6:.4f} ms/step"
        )
    api = collections.Counter()
    api_count = collections.Counter()
    for start, stop, name in connection.execute(
        "SELECT start, end, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME "
        "WHERE start >= ? AND start < ?",
        (begin, end),
    ):
        label = strings[name]
        api[label] += stop - start
        api_count[label] += 1
    lines.append("host CUDA API (ms/step, calls/step):")
    for name, value in api.most_common(12):
        lines.append(
            f"  {value / steps / 1e6:8.4f}  {api_count[name] / steps:6.1f}  {name}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--atoms", type=int, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--flavor", default="energy", choices=sorted(lmp_scan.FLAVORS))
    parser.add_argument("--script", default=None)
    args = parser.parse_args()
    tag = args.tag or (
        os.path.splitext(os.path.basename(args.model))[0] + f"_{args.atoms}"
    )
    database, log = profile(
        os.path.abspath(args.model),
        args.atoms,
        args.gpu,
        args.warmup,
        args.steps,
        tag,
        args.flavor,
        args.script,
    )
    report = analyze(database, args.steps, log)
    with open(
        os.path.join(HERE, "work_nsys", tag, "summary.txt"), "w", encoding="utf-8"
    ) as stream:
        stream.write(report + "\n")
    print(f"== {tag}\n{report}", flush=True)  # noqa: T201


if __name__ == "__main__":
    main()

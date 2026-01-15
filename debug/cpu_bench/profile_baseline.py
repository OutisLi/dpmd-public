# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, T201
"""Attribute the Inductor CPU baseline to its generated kernels.

Freezing with ``cpp.enable_kernel_profile`` wraps every generated kernel in a
profiler scope, so a ``torch.profiler`` trace of the frozen package reports
one row per kernel. The kernel names carry the fused aten operations, which
is enough to map a row back to a stage of the descriptor.

Usage
-----
    DEVICE=cpu python profile_baseline.py --grade nano --atoms 4096
"""

from __future__ import (
    annotations,
)

import argparse
import os

import torch
from harness import (
    GRADES,
    HERE,
    build_lower_inputs,
    load_model,
)

OUT = os.path.join(HERE, "models")

#: Options that make the Inductor CPU lowering of the graph parallel. The
#: export sizes every dynamic axis from the tiny synthetic trace system, so
#: the default chunk threshold keeps almost every loop serial.
PARALLEL_OPTIONS = {"cpp.min_chunk_size": 1}


def freeze_profiled(grade: str, threads: int) -> str:
    """Freeze one grade with per-kernel profiling enabled."""
    import deepmd.pt.utils.compile_compat as compile_compat
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file,
    )

    path = os.path.join(OUT, f"profiled_{grade}.pt2")
    if os.path.exists(path):
        return path
    torch.set_num_threads(threads)
    model = load_model(GRADES[grade])
    original = compile_compat.build_inductor_compile_options

    def patched(*args: object, **kwargs: object) -> dict:
        merged = original(*args, **kwargs)
        merged.update(PARALLEL_OPTIONS)
        merged["cpp.enable_kernel_profile"] = True
        return merged

    compile_compat.build_inductor_compile_options = patched
    try:
        deserialize_to_file(
            path,
            {"model": model.serialize()},
            lower_kind="auto",
            do_atomic_virial=True,
        )
    finally:
        compile_compat.build_inductor_compile_options = original
    return path


def main() -> None:
    """Freeze, profile and print the dominant generated kernels."""
    import paths as path_module

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grade", default="nano")
    parser.add_argument("--atoms", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=83)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    path = freeze_profiled(args.grade, args.threads)
    torch.set_num_threads(args.threads)
    model = load_model(GRADES[args.grade])
    sample = build_lower_inputs(model, args.atoms)
    run = path_module.package(path)
    for _ in range(3):
        run(*sample.args)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=False,
    ) as profiler:
        for _ in range(3):
            run(*sample.args)
    events = profiler.key_averages()
    total = sum(event.self_cpu_time_total for event in events)
    print(
        f"# {args.grade} atoms={sample.n_atom} edges={sample.n_edge} "
        f"threads={args.threads}"
    )
    print(f"{'self ms/iter':>12s} {'%':>6s}  kernel")
    for event in sorted(events, key=lambda e: -e.self_cpu_time_total)[: args.top]:
        share = 100.0 * event.self_cpu_time_total / max(total, 1)
        print(
            f"{event.self_cpu_time_total / 3e3:12.2f} {share:6.1f}  {event.key[:110]}"
        )


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201, TID253
"""Benchmark the SeZM / DPA4 inference path on a periodic diamond supercell.

Two measurement scopes are available. ``lower`` drives
``forward_common_lower`` on a pre-built edge schema and therefore reports the
model graph alone. ``deeppot`` drives ``DeepPot.eval`` and includes
neighbor-list construction and the host glue, matching an ASE step.

TF32 is disabled in both scopes because the production potential-energy surface
requires it off. Warmup iterations are discarded.
"""

from __future__ import (
    annotations,
)

import argparse
import os
import time

import numpy as np
import torch
from sezm_harness import (
    CKPT_MINI,
    build_lower_inputs,
    diamond_cell,
    env_summary,
    load_model,
    timed,
)


def bench_lower(args: argparse.Namespace) -> None:
    """Measure the compiled lower graph on a fixed edge schema."""
    model, params = load_model(args.ckpt)
    carbon = params["type_map"].index("C") if "C" in params["type_map"] else 0
    inputs, info = build_lower_inputs(model, args.atoms, atom_type=carbon)
    print(f"[lower] {env_summary()}", flush=True)
    print(f"[lower] system {info}", flush=True)
    t0 = time.perf_counter()
    out0 = model.forward_common_lower(*inputs)
    torch.cuda.synchronize()
    print(
        f"[lower] first call {time.perf_counter() - t0:.1f} s "
        f"E={float(out0['energy'].sum()):.6f}",
        flush=True,
    )
    stats = timed(model, inputs, iters=args.iters, warmup=args.warmup)
    print(
        f"[lower] {stats['ms']:.2f} ms  peak {stats['peak_gb']:.2f} GiB  "
        f"E={stats['energy']:.6f}  |F|max={stats['force_absmax']:.6f}",
        flush=True,
    )


def bench_deeppot(args: argparse.Namespace) -> None:
    """Measure the full ``DeepPot.eval`` path including the neighbor list."""
    from deepmd.infer import (
        DeepPot,
    )

    coord, box = diamond_cell(args.atoms)
    nloc = coord.shape[0]
    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    params = raw["model"]["_extra_state"]["model_params"]
    if "Default" in params.get("model_dict", {}):
        params = params["model_dict"]["Default"]
    type_map = params["type_map"]
    carbon = type_map.index("C") if "C" in type_map else 0
    atype = np.full((nloc,), carbon, dtype=np.int32)
    print(f"[deeppot] {env_summary()} atoms={nloc}", flush=True)
    dp = DeepPot(args.ckpt)
    coord_b = coord.reshape(1, -1)
    box_b = box.reshape(1, -1)

    t0 = time.perf_counter()
    e0 = dp.eval(coord_b, box_b, atype)[0]
    torch.cuda.synchronize()
    print(
        f"[deeppot] first call {time.perf_counter() - t0:.1f} s "
        f"E={float(np.asarray(e0).reshape(-1)[0]):.6f}",
        flush=True,
    )
    for _ in range(args.warmup):
        dp.eval(coord_b, box_b, atype)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t1 = time.perf_counter()
    for _ in range(args.iters):
        e, f, v = dp.eval(coord_b, box_b, atype)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t1) * 1e3 / args.iters
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"[deeppot] {ms:.2f} ms  peak {peak:.2f} GiB  "
        f"E={float(np.asarray(e).reshape(-1)[0]):.6f}  |F|max={float(np.abs(f).max()):.6f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("lower", "deeppot", "both"), default="both")
    parser.add_argument("--atoms", type=int, default=8000)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--ckpt", default=CKPT_MINI)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.mode in ("lower", "both"):
        bench_lower(args)
    if args.mode in ("deeppot", "both"):
        bench_deeppot(args)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("DP_INTER_OP_PARALLELISM_THREADS", "0")
    os.environ.setdefault("DP_INTRA_OP_PARALLELISM_THREADS", "0")
    main()

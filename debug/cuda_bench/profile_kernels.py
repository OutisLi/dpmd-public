# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201, TID253
"""Per-kernel CUDA time of the compiled SeZM lower graph, joined with the
Inductor source-node annotation so every pointwise fragment is attributed to
the model operations that produced it.
"""

from __future__ import (
    annotations,
)

import argparse
import os
import re
import sys
from collections import (
    defaultdict,
)

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sezm_harness import (
    CKPT_MINI,
    build_lower_inputs,
    env_summary,
    load_model,
)


def kernel_origin_map(path: str) -> dict[str, str]:
    """Map each Inductor kernel name to its 'Topologically Sorted Source Nodes'."""
    out: dict[str, str] = {}
    pending = None
    with open(path) as fh:
        for line in fh:
            m = re.match(r"^# Topologically Sorted Source Nodes: \[(.*)\]", line)
            if m:
                pending = m.group(1)
                continue
            m = re.match(r"^(triton_\w+) = async_compile", line)
            if m and pending is not None:
                out[m.group(1)] = pending
                pending = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atoms", type=int, default=8000)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--ckpt", default=CKPT_MINI)
    ap.add_argument("--origins", default="/tmp/dpa4c2/inductor_out.py")
    ap.add_argument("--out", default="/tmp/dpa4c2/kernels.txt")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, params = load_model(args.ckpt)
    carbon = params["type_map"].index("C") if "C" in params["type_map"] else 0
    inputs, info = build_lower_inputs(model, args.atoms, atom_type=carbon)
    print(f"{env_summary()}  {info}", flush=True)
    for _ in range(4):
        model.forward_common_lower(*inputs)
    torch.cuda.synchronize()

    from torch.profiler import (
        ProfilerActivity,
        profile,
    )

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(args.iters):
            model.forward_common_lower(*inputs)
        torch.cuda.synchronize()

    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for evt in prof.key_averages():
        if evt.device_type.name != "CUDA" and evt.self_device_time_total <= 0:
            continue
        if evt.self_device_time_total <= 0:
            continue
        totals[evt.key] += evt.self_device_time_total
        counts[evt.key] += evt.count
    origins = kernel_origin_map(args.origins) if os.path.exists(args.origins) else {}

    rows = sorted(totals.items(), key=lambda kv: -kv[1])
    total = sum(totals.values()) / args.iters / 1e3
    lines = [f"total GPU self time {total:.2f} ms/step over {args.iters} iters", ""]
    for name, us in rows:
        ms = us / args.iters / 1e3
        if ms < 0.005:
            continue
        origin = origins.get(name, "")
        lines.append(
            f"{ms:8.3f} ms  n={counts[name] // args.iters:4d}  {name[:70]:70s}  {origin[:150]}"
        )
    text = "\n".join(lines)
    with open(args.out, "w") as fh:
        fh.write(text + "\n")
    print(text[:4000])
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()

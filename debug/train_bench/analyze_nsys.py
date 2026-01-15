#!/usr/bin/env python
# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201
"""Split a one-step nsys trace into forward/backward windows and bucket kernels.

The forward window is the GPU projection of the ``SeZM/forward_common`` NVTX
range; everything after it until the last kernel is the loss/optimizer backward.
Kernels are bucketed by name family, and the top entries of each window are
printed with their summed durations.
"""

import re
import sqlite3
import sys
from collections import (
    defaultdict,
)


def family(name: str) -> str:
    if name.startswith("triton_poi_fused"):
        return "inductor pointwise"
    if name.startswith("triton_red_fused") or name.startswith("triton_per_fused"):
        return "inductor reduction"
    if name.startswith("triton_tem_fused") or name.startswith("triton_mm"):
        return "inductor gemm"
    if "cutlass" in name or "gemm" in name.lower() or "gemv" in name.lower():
        return "library gemm"
    if name.startswith("_") and name.endswith("kernel"):
        return "sezm fused triton"
    if "indexFunc" in name or "index_elementwise" in name:
        return "aten scatter/gather"
    if "elementwise_kernel" in name or "vectorized_elementwise" in name:
        return "aten elementwise"
    if "reduce_kernel" in name or "Reduce" in name:
        return "aten reduction"
    if "CUDA memcpy" in name or "CUDA memset" in name or "Memcpy" in name:
        return "memcpy/memset"
    return "other"


def main() -> None:
    db = sqlite3.connect(sys.argv[1])
    cur = db.cursor()
    row = cur.execute(
        "SELECT ne.start, ne.end FROM NVTX_EVENTS ne "
        "WHERE ne.text LIKE '%forward_common%' ORDER BY ne.start LIMIT 1"
    ).fetchone()
    fwd_start, fwd_end = row
    kernels = cur.execute(
        "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.demangledName = s.id"
    ).fetchall()
    print(
        f"forward nvtx: {(fwd_end - fwd_start) / 1e6:.2f} ms  kernels: {len(kernels)}"
    )

    for title, lo, hi in (
        ("forward window", fwd_start, fwd_end),
        ("backward window", fwd_end, 2**63),
    ):
        fam_us: dict[str, float] = defaultdict(float)
        fam_n: dict[str, int] = defaultdict(int)
        top: dict[str, float] = defaultdict(float)
        top_n: dict[str, int] = defaultdict(int)
        span_lo, span_hi = None, None
        for st, en, name in kernels:
            if not (lo <= st < hi):
                continue
            dur = (en - st) / 1e3
            f = family(name)
            fam_us[f] += dur
            fam_n[f] += 1
            short = re.sub(r"<.*", "", name)[:80]
            top[short] += dur
            top_n[short] += 1
            span_lo = st if span_lo is None else min(span_lo, st)
            span_hi = en if span_hi is None else max(span_hi, en)
        total = sum(fam_us.values())
        span = (span_hi - span_lo) / 1e6 if span_lo is not None else 0.0
        print(f"\n=== {title}: busy {total / 1e3:.2f} ms, span {span:.2f} ms ===")
        for name, v in sorted(fam_us.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<24}{v / 1e3:>9.3f} ms {fam_n[name]:>6} launches")
        print("  -- top kernels --")
        for name, v in sorted(top.items(), key=lambda kv: -kv[1])[:18]:
            print(f"  {v / 1e3:>8.3f} ms {top_n[name]:>5}x  {name}")


if __name__ == "__main__":
    main()

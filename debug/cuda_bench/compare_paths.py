# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201, TID253, ANN001, ANN201, ANN202
"""Compare the Triton-L3 baseline against the fused CUDA path on one process.

Both models are constructed in the same process with different gates, so the
comparison covers time, peak memory and the force deviation on identical inputs.
"""

from __future__ import (
    annotations,
)

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build(atoms: int, cuda_level: str, ckpt: str | None):
    os.environ["DP_CUDA_INFER"] = cuda_level
    import importlib

    import deepmd.pt_expt.kernels.utils as ku

    importlib.reload(ku)
    from sezm_harness import (
        CKPT_MINI,
        build_lower_inputs,
        load_model,
    )

    model, params = load_model(ckpt or CKPT_MINI)
    carbon = params["type_map"].index("C") if "C" in params["type_map"] else 0
    inputs, info = build_lower_inputs(model, atoms, atom_type=carbon)
    return model, inputs, info


def measure(model, inputs, iters: int, warmup: int):
    def run():
        return model.forward_common_lower(*inputs)

    out = run()
    for _ in range(warmup):
        out = run()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        out = run()
    stop.record()
    torch.cuda.synchronize()
    key = next(k for k in out if "force" in k or "derv_r" in k)
    return (
        start.elapsed_time(stop) / iters,
        torch.cuda.max_memory_allocated() / 2**30,
        float(out["energy"].sum()),
        out[key].detach().clone(),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atoms", type=int, nargs="+", default=[8000])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--level", default="2", help="DP_CUDA_INFER of the fused path")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for atoms in args.atoms:
        rows = {}
        for tag, level in (("triton", "0"), ("cuda", args.level)):
            t0 = time.perf_counter()
            model, inputs, info = build(atoms, level, args.ckpt)
            ms, peak, energy, force = measure(model, inputs, args.iters, args.warmup)
            rows[tag] = (ms, peak, energy, force)
            print(
                f"atoms={info['nloc']} edges={info['n_edge']} {tag:6s} "
                f"{ms:7.2f} ms  peak {peak:5.2f} GiB  E={energy:.6f}  "
                f"(build {time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
            del model, inputs
            torch.cuda.empty_cache()
        (t_ms, _, t_e, t_f), (c_ms, _, c_e, c_f) = rows["triton"], rows["cuda"]
        df = (t_f - c_f).abs().max().item()
        print(
            f"  speedup {t_ms / c_ms:.3f}x   dE={abs(t_e - c_e):.4f} eV "
            f"({abs(t_e - c_e) / max(abs(t_e), 1e-30):.2e} rel)   "
            f"dF_max={df:.3e} eV/A   |F|max={t_f.abs().max().item():.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

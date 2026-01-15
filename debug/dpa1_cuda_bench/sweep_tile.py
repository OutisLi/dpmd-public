# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backward tiling sweep for the DPA1 graph-lower CUDA kernel.

The backward's per-thread edge fragment (EPT) and its spill prefetch (PIPE) are
compile-time template parameters resolved per width by ``backward_edges_per_thread``
/ ``backward_prefetch`` in ``source/op/pt/dpa1_graph_descriptor.cu``. Those rules
were measured on the NVIDIA H20; a different GPU re-derives them. This tool
forces each ``(EPT, PIPE)`` on a chosen width and times the isolated backward,
so the fastest configuration per width is measurable without editing the
production dispatch.

Method: the production ``.cu`` sources are copied to a scratch build directory,
their ``TORCH_LIBRARY`` namespace is renamed (to coexist with the loaded
production ops) and the two policy functions are pinned to the swept values;
each variant is JIT-built and its backward timed against the same inputs. The
production files are never modified.

Usage
-----
    DP_CUDA_INFER=1 python sweep_tile.py --shape L
    DP_CUDA_INFER=1 python sweep_tile.py --neuron 32 64 128 --axis 16
"""

from __future__ import (
    annotations,
)

import argparse
import os
import re
import shutil
import tempfile
import time
from typing import (
    Any,
)

import numpy as np
import torch  # noqa: TID253
from freeze import (
    SHAPES,
    model_params,
)
from torch.utils.cpp_extension import load  # noqa: TID253

from deepmd.kernels.triton.dpa1.activation import (
    ACT_CODES,
)
from deepmd.pt_expt.model.get_model import get_model  # noqa: TID253
from deepmd.pt_expt.train.wrapper import ModelWrapper  # noqa: TID253

OP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "source",
    "op",
    "pt",
)
SOURCES = ["dpa1_graph_descriptor.cu", "graph_fitting.cu", "edge_force_virial.cu"]
CONFIGS = [(8, 0), (8, 1), (4, 0), (4, 1)]  # (edges_per_thread, prefetch)
A0 = 3.567
_FRAC = np.array(
    [
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0],
        [0.25, 0.25, 0.25],
        [0.25, 0.75, 0.75],
        [0.75, 0.25, 0.75],
        [0.75, 0.75, 0.25],
    ],
    dtype=np.float64,
)


def build_variant(
    ept: int, pipe: int, width: tuple[int, int, int], workroot: str
) -> Any:
    """JIT-build the kernel with the backward policy pinned to ``(ept, pipe)``.

    The namespace is suffixed so the variant coexists with the loaded
    production ``deepmd`` ops, and the width dispatch table is narrowed to the
    swept stack so the build stays fast. Returns the ``torch.ops`` namespace
    handle.
    """
    ns = f"deepmd_sweep_{ept}_{pipe}"
    d = os.path.join(workroot, ns)
    os.makedirs(d, exist_ok=True)

    def keep_only_width(match: re.Match) -> str:
        n1, n2, ng = int(match[1]), int(match[2]), int(match[3])
        return match[0] if (n1, n2, ng) == width else ""

    for src in SOURCES:
        text = open(os.path.join(OP_DIR, src)).read()
        text = text.replace(
            "TORCH_LIBRARY_FRAGMENT(deepmd,", f"TORCH_LIBRARY_FRAGMENT({ns},"
        )
        if src == "dpa1_graph_descriptor.cu":
            text = re.sub(
                r"backward_edges_per_thread\(int[^)]*\)\s*\{[^}]*\}",
                f"backward_edges_per_thread(int) {{ return {ept}; }}",
                text,
            )
            text = re.sub(
                r"backward_prefetch\(int[^)]*\)\s*\{[^}]*\}",
                f"backward_prefetch(int) {{ return {pipe}; }}",
                text,
            )
            text = re.sub(
                r"DPA1_DISPATCH_ONE\(LAUNCH, scalar_t, (\d+), (\d+), (\d+)\)",
                keep_only_width,
                text,
            )
        open(os.path.join(d, src), "w").write(text)
    load(
        name=ns,
        sources=[os.path.join(d, s) for s in SOURCES],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-gencode",
            "arch=compute_90,code=sm_90",
        ],
        extra_ldflags=["-lcublas"],
        with_cuda=True,
        is_python_module=False,
    )
    return getattr(torch.ops, ns)


def descriptor_args(
    desc: Any, graph: Any, atype: torch.Tensor, tebd: torch.Tensor
) -> tuple:
    """Assemble the forward op arguments from a descriptor module.

    Mirrors the pt_expt wrapper deepmd.kernels.cuda.dpa1.graph_descriptor.
    """
    se = desc.se_atten
    layers = se.embeddings[0].layers
    empty = layers[0].w.new_empty(0)

    def opt(t: torch.Tensor | None) -> torch.Tensor:
        return t.contiguous() if t is not None else empty

    gate = empty.reshape(0, 0)
    w1, w2, w3 = (layer.w.contiguous() for layer in layers)
    return (
        graph.edge_vec.contiguous(),
        graph.edge_index.contiguous(),
        graph.edge_mask.contiguous(),
        atype.contiguous(),
        tebd.contiguous(),
        se.mean[:, 0, :].contiguous(),
        se.stddev[:, 0, :].contiguous(),
        w1,
        opt(layers[0].b),
        opt(layers[0].idt),
        w2,
        opt(layers[1].b),
        opt(layers[1].idt),
        w3,
        opt(layers[2].b),
        opt(layers[2].idt),
        gate,
        ACT_CODES[str(layers[0].activation_function).lower()],
        int(se.type_one_side),
        int(desc.concat_output_tebd),
        1,
        0,
        int(se.axis_neuron),
        int(layers[1].resnet),
        int(layers[2].resnet),
        float(se.rcut),
        float(se.rcut_smth),
        float(se.env_protection),
        float(se.nnei),
    )


def timed(fn: Any, reps: int = 50, warm: int = 20, rounds: int = 6) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / reps * 1e3)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shape", default="L", choices=list(SHAPES))
    ap.add_argument("--neuron", type=int, nargs="+", default=None)
    ap.add_argument("--axis", type=int, default=None)
    ap.add_argument("--n", type=int, default=2000, help="probe atom count")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda:0")

    neuron, axis, fitting = SHAPES[args.shape]
    neuron, axis = args.neuron or neuron, args.axis or axis
    params = model_params(neuron, axis, fitting, "concat", "silu")
    model = ModelWrapper(get_model(params)).model["Default"].to(dev).eval()

    n = max(4, int(np.ceil((args.n / 8.0) ** (1.0 / 3.0))))
    cells = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1)
    pos = (_FRAC[None] + cells.reshape(-1, 3)[:, None]).reshape(
        -1, 3
    ) * A0 + 0.03 * np.random.default_rng(0).standard_normal((n**3 * 8, 3))
    from deepmd.pt_expt.utils.nv_graph_builder import (
        build_neighbor_graph_nv,
    )

    cc = torch.tensor(pos.reshape(1, -1, 3), dtype=torch.float64, device=dev)
    atype = torch.full((1, pos.shape[0]), 5, dtype=torch.int64, device=dev)
    box = torch.diag(torch.full((3,), n * A0, dtype=torch.float64, device=dev))
    graph = build_neighbor_graph_nv(
        cc, atype.clone(), box.reshape(1, 3, 3), model.get_rcut()
    )
    desc = model.atomic_model.descriptor
    fa = descriptor_args(
        desc, graph, atype.reshape(-1), desc.type_embedding.call().to(torch.float32)
    )

    work = tempfile.mkdtemp(prefix="dpa1_sweep_")
    e_edges = int(graph.edge_mask.sum())
    print(f"shape neuron={neuron} axis={axis}  N={pos.shape[0]}  E={e_edges}")  # noqa: T201
    results = {}
    try:
        width = (neuron[0], neuron[1], neuron[2])
        for ept, pipe in CONFIGS:
            ns = build_variant(ept, pipe, width, work)
            out = ns.dpa1_graph_descriptor(*fa)
            d_grrg = torch.randn_like(out[0])
            aux = out[2:]
            (
                ev,
                ei,
                em,
                at,
                _t,
                dv,
                ds,
                w1,
                b1,
                i1,
                w2,
                b2,
                i2,
                w3,
                b3,
                i3,
                gt,
                ac,
                os_,
                ct,
                sm,
                ax,
                r2,
                r3,
                rc,
                rs,
                pr,
                nn,
            ) = fa
            ba = (
                d_grrg,
                None,
                *aux,
                ev,
                ei,
                em,
                at,
                dv,
                ds,
                w1,
                b1,
                i1,
                w2,
                b2,
                i2,
                w3,
                b3,
                i3,
                gt,
                ac,
                os_,
                sm,
                ax,
                r2,
                r3,
                rc,
                rs,
                pr,
                nn,
            )
            bwd = ns.dpa1_graph_descriptor_backward
            ms = timed(lambda: bwd(*ba))
            results[(ept, pipe)] = ms
            print(f"  EPT={ept} PIPE={pipe}  backward={ms:.3f} ms")  # noqa: T201
    finally:
        shutil.rmtree(work, ignore_errors=True)
    best = min(results, key=results.get)
    print(f"best: EPT={best[0]} PIPE={best[1]}  ({results[best]:.3f} ms)")  # noqa: T201


if __name__ == "__main__":
    main()

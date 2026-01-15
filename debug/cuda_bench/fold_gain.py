# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Edge-kernel time of a single-type system against the same system with two types.

A single-type system lets the edge scans fold: they apply the pair
coefficients to reduced sums instead of to every edge. Changing the type of
one atom of the jittered diamond ends the single-type system without changing
the work of any other row, so the edge kernels then scan pairwise. The
difference between the two kernel times is the gain of the fold. With
``--spin`` the models are the spin-conditioned grades, evaluated with random
moments of magnitude two, whose scans do not fold, so the two times agree.

Usage
-----
    BENCH_OP_LIB=.../libdeepmd_op_pt.so DP_CUDA_INFER=2 CUDA_VISIBLE_DEVICES=2 \
    python spin_newton/run_with_oplib.py fold_gain.py --grades air plus [--spin]
"""

from __future__ import (
    annotations,
)

import argparse
import collections
import re

import torch  # noqa: TID253
from bench_ops import (
    COMPRESS_STRIDE,
    jittered_diamond_graph,
)
from freeze_dpa4c import (
    model_config,
)

from deepmd.pt_expt.model.get_model import (  # noqa: TID253
    get_model,
)


def build(grade: str, spin: bool, device: torch.device) -> torch.nn.Module:
    """Construct and compress the random-weight model of one released grade."""
    config = model_config(
        0, 0, 0, 2, 0, spin, False, [0.0, 1.0], None, f"dpa4c-{grade}-v20260911"
    )
    model = get_model(config).eval()
    descriptor = model.atomic_model.fused_decomposition()[0].descriptor
    if spin:
        # A fresh spin branch is gated shut; opening the gate makes the spin
        # families contribute, as the benchmark models do.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.numel() and not bool((parameter != 0).any()):
                    parameter.fill_(0.3)
    descriptor.enable_compression(1.4, table_stride_1=COMPRESS_STRIDE)
    return model.to(device)


def edge_kernel_ms(
    model: torch.nn.Module,
    graph: object,
    atype: torch.Tensor,
    spin: torch.Tensor | None,
    repeats: int,
) -> dict[str, float]:
    """Return the device milliseconds per evaluation of the DPA4C kernels."""
    from torch.profiler import (
        ProfilerActivity,
        profile,
    )

    def run() -> None:
        kwargs = {"spin": spin} if spin is not None else {}
        model.forward_lower_canonical_graph(
            atype,
            graph.n_node,
            graph.n_local,
            graph.source,
            graph.edge_vec,
            graph.destination_row_ptr,
            graph.source_row_ptr.new_zeros(2),
            graph.source_order.new_zeros(2),
            do_atomic_virial=False,
            **kwargs,
        )

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as trace:
        for _ in range(repeats):
            run()
        torch.cuda.synchronize()
    totals: collections.Counter = collections.Counter()
    for event in trace.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            match = re.search(r"deepmd_dpa4c::(\w+)", event.name)
            name = match.group(1) if match else "other"
            totals[name] += event.device_time / 1e3 / repeats
    return dict(totals)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grades", nargs="+", default=["air", "plus"])
    parser.add_argument("--atoms", type=int, default=100_800)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--spin", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda")
    graph, atype = jittered_diamond_graph(args.atoms, 0.05, device)
    mixed = atype.clone()
    mixed[0] = 1
    generator = torch.Generator(device=device).manual_seed(5)
    moments = None
    if args.spin:
        direction = torch.randn(atype.shape[0], 3, generator=generator, device=device)
        moments = 2.0 * direction / direction.norm(dim=1, keepdim=True)
    for grade in args.grades:
        model = build(grade, args.spin, device)
        single = edge_kernel_ms(model, graph, atype, moments, args.repeats)
        paired = edge_kernel_ms(model, graph, mixed, moments, args.repeats)
        for kernel in ("forward_kernel", "edge_backward_kernel"):
            one, two = single.get(kernel, 0.0), paired.get(kernel, 0.0)
            print(  # noqa: T201
                f"{grade:5s} {kernel:22s} single type {one:7.3f} ms, two types "
                f"{two:7.3f} ms, fold gain {1 - one / two:+.1%}",
                flush=True,
            )
        print(  # noqa: T201
            f"{grade:5s} all kernels             single type "
            f"{sum(single.values()):7.3f} ms, two types {sum(paired.values()):7.3f} ms",
            flush=True,
        )


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, ANN001, ANN201, ANN202, T201
"""Time the compressed DPA4C operator against the same math under Inductor.

The comparison isolates the kernel from the compression: both sides evaluate
the identical compressed equations on identical inputs, one through the
hand-written C++ operator and one through the portable reference lowered by
Inductor. The end-to-end comparison against the uncompressed baseline lives
in ``bench.py``.

Usage
-----
    OMP_NUM_THREADS=83 python bench_kernel.py --grade neo --atoms 8000
"""

from __future__ import (
    annotations,
)

import argparse
import time

import numpy as np
import torch
from harness import (
    GRADES,
    THREADS,
    build_lower_inputs,
    load_model,
)


def build_case(grade: str, atoms: int, stride: float):
    """Build a compressed model, its graph and the operator arguments."""
    from deepmd.dpmodel.utils.neighbor_graph import (
        NeighborGraph,
    )
    from deepmd.pt_expt.kernels.dpa4c.graph_compress import (
        compressed_operator_arguments,
        ensure_registered,
    )

    ensure_registered()
    model = load_model(GRADES[grade])
    descriptor = model.get_descriptor()
    descriptor.enable_compression(0.0, table_stride_1=stride)
    sample = build_lower_inputs(model, atoms)
    (
        atype,
        n_node,
        _n_local,
        edge_index,
        edge_vec,
        edge_mask,
        destination_order,
        destination_row_ptr,
        source_order,
        source_row_ptr,
        destination_sorted,
        _virial,
    ) = sample.args
    graph = NeighborGraph(
        n_node=n_node,
        edge_index=edge_index,
        edge_vec=edge_vec.to(torch.float32),
        edge_mask=edge_mask,
        n_local=n_node,
        destination_order=destination_order,
        destination_row_ptr=destination_row_ptr,
        source_order=source_order,
        source_row_ptr=source_row_ptr,
        destination_sorted=destination_sorted,
    )
    arguments = (
        graph.edge_index.contiguous(),
        graph.edge_mask.contiguous(),
        graph.destination_order.contiguous(),
        graph.destination_row_ptr.contiguous(),
        atype.contiguous(),
        *compressed_operator_arguments(descriptor, None),
        bool(graph.destination_sorted),
        int(descriptor.lmax),
        *descriptor._compression_scalars,
    )
    return model, sample, graph, arguments


def measure(run, iters: int, warmup: int) -> dict[str, float]:
    """Time a callable, reporting the mean and the achieved parallelism."""
    for _ in range(warmup):
        run()
    samples = []
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    for _ in range(iters):
        start = time.perf_counter()
        run()
        samples.append((time.perf_counter() - start) * 1e3)
    wall = (time.perf_counter() - wall_start) * 1e3
    cpu = (time.process_time() - cpu_start) * 1e3
    return {
        "ms": float(np.mean(samples)),
        "ms_best": float(np.min(samples)),
        "parallelism": cpu / max(wall, 1e-9),
    }


def main() -> None:
    """Time the operator forward and backward against the reference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grade", default="neo")
    parser.add_argument("--atoms", type=int, default=8000)
    parser.add_argument("--stride", type=float, default=0.01)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()

    _model, sample, graph, arguments = build_case(args.grade, args.atoms, args.stride)
    descriptor_out, state = torch.ops.deepmd.dpa4c_graph_compress(
        graph.edge_vec, *arguments
    )
    seed = torch.randn_like(descriptor_out)
    print(
        f"# {args.grade} atoms={sample.n_atom} edges={sample.n_edge} "
        f"D={descriptor_out.shape[1]} threads={THREADS} stride={args.stride}"
    )

    forward = measure(
        lambda: torch.ops.deepmd.dpa4c_graph_compress(graph.edge_vec, *arguments),
        args.iters,
        args.warmup,
    )
    # The descriptor alone is differentiated: no analytical pair potential.
    no_pair = graph.edge_vec.new_empty(0)
    backward = measure(
        lambda: torch.ops.deepmd.dpa4c_graph_compress_backward(
            seed, state, graph.edge_vec, *arguments, no_pair, no_pair
        ),
        args.iters,
        args.warmup,
    )
    print(
        f"kernel   forward {forward['ms']:8.2f} ms "
        f"(best {forward['ms_best']:7.2f}, parallelism {forward['parallelism']:5.1f}x)"
    )
    print(
        f"kernel   backward{backward['ms']:8.2f} ms "
        f"(best {backward['ms_best']:7.2f}, parallelism {backward['parallelism']:5.1f}x)"
    )
    print(
        f"kernel   total   {forward['ms'] + backward['ms']:8.2f} ms "
        f"-> {sample.n_atom / (forward['ms'] + backward['ms']):8.2f} atoms/ms"
    )

    if args.reference:
        from deepmd.pt_expt.kernels.dpa4c.graph_compress import (
            _reference_descriptor,
        )

        def reference_call():
            return _reference_descriptor(
                graph.edge_vec,
                *arguments[:15],
                *arguments[18:],
                spin=arguments[15],
                spin_pair=arguments[16],
                spin_type=arguments[17],
            )

        eager = measure(reference_call, max(args.iters // 2, 1), 1)
        compiled_call = torch.compile(reference_call, dynamic=True)
        compiled = measure(compiled_call, max(args.iters // 2, 1), 2)
        print(
            f"reference eager  {eager['ms']:8.2f} ms   "
            f"inductor {compiled['ms']:8.2f} ms   "
            f"kernel speedup {compiled['ms'] / forward['ms']:5.2f}x (forward only)"
        )


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: LGPL-3.0-or-later
"""Dump the compact DPA4C descriptor operators on a benchmark graph.

Runs the canonical forward and backward operators of one benchmark model
(``bench_ops.build_model``) on the jittered diamond graph and saves their
outputs, so that two builds of the operator library can be compared bit for
bit. ``--legacy`` drops the low-rank table and its expansion from the
arguments, for a library built before the operators took them. ``--trained``
compresses the trained checkpoint of the grade instead of the random-weight
preset, which is the model that adopts the low-rank table.

Usage
-----
    BENCH_OP_LIB=... python freeze_dev.py op_parity.py --grade air --atoms 10296 --out a.npz
"""

from __future__ import (
    annotations,
)

import argparse

import numpy as np
import torch  # noqa: TID253
from bench_ops import (
    build_model,
    jittered_diamond_graph,
)
from lowrank_accuracy import (
    compress,
    load_trained,
)

from deepmd.pt_expt.kernels.dpa4c.graph_compress import (  # noqa: TID253
    compressed_operator_arguments,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grade", required=True)
    parser.add_argument("--atoms", type=int, default=10296)
    parser.add_argument("--out", required=True)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--trained", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda")
    graph, atype = jittered_diamond_graph(
        args.atoms, 0.05, device, "lattice", 0.0, 0.0, None
    )
    torch.manual_seed(0)
    if args.trained:
        model = load_trained(args.grade, device, "float32")
        compress(model, 0.01)
        atype = torch.full_like(atype, model.get_type_map().index("C"))
    else:
        model = build_model(args.grade, device)
    descriptor = model.atomic_model.fused_decomposition()[0].descriptor
    artifacts = compressed_operator_arguments(descriptor)
    if args.legacy:
        artifacts = (artifacts[0], *artifacts[3:])
    operands = (
        graph.source,
        graph.destination_row_ptr,
        atype,
        *artifacts,
        int(descriptor.lmax),
        *descriptor._compression_scalars,
    )
    with torch.no_grad():
        value, state = torch.ops.deepmd.dpa4c_canonical_compress(
            graph.edge_vec, *operands
        )
        generator = torch.Generator(device=device).manual_seed(1)
        cotangent = torch.randn(value.shape, generator=generator, device=device)
        gradient = torch.ops.deepmd.dpa4c_canonical_compress_backward(
            cotangent, state, graph.edge_vec, *operands
        )[0]
    np.savez(
        args.out,
        value=value.cpu().numpy(),
        state=state.cpu().numpy(),
        gradient=gradient.cpu().numpy(),
    )
    print(f"wrote {args.out}", flush=True)  # noqa: T201


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Outputs of the native-spin grades on a single-type and a two-type system.

The jittered diamond of ``bench_ops.py`` is evaluated with random moments of
magnitude two, once as one type and once with the first atom changed to the
second type, a single-type and a two-type system.
Energy, force, virial and magnetic force of both systems are written to an
``.npz`` file; ``--compare`` reports the deviation from another such file, so
two operator libraries can be compared on the same inputs.

Usage
-----
    BENCH_OP_LIB=.../libdeepmd_op_pt.so DP_CUDA_INFER=2 CUDA_VISIBLE_DEVICES=2 \
    python spin_newton/run_with_oplib.py spin_parity.py --out new.npz [--compare old.npz]
"""

from __future__ import (
    annotations,
)

import argparse

import numpy as np
import torch  # noqa: TID253
from bench_ops import (
    jittered_diamond_graph,
)
from fold_gain import (
    build,
)

KEYS = ("energy", "force", "virial", "force_mag")


def evaluate(
    model: torch.nn.Module,
    graph: object,
    atype: torch.Tensor,
    spin: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Return energy, force, virial and magnetic force of one evaluation."""
    output = model.forward_lower_canonical_graph(
        atype,
        graph.n_node,
        graph.n_local,
        graph.source,
        graph.edge_vec,
        graph.destination_row_ptr,
        graph.source_row_ptr,
        graph.source_order,
        do_atomic_virial=False,
        spin=spin,
    )
    return {key: output[key].detach().double().cpu().numpy() for key in KEYS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grades", nargs="+", default=["nano", "mini", "neo", "air", "plus"]
    )
    parser.add_argument("--atoms", type=int, default=512)
    parser.add_argument("--out", required=True)
    parser.add_argument("--compare", default=None)
    args = parser.parse_args()
    device = torch.device("cuda")
    graph, atype = jittered_diamond_graph(args.atoms, 0.05, device)
    mixed = atype.clone()
    mixed[0] = 1
    generator = torch.Generator(device=device).manual_seed(5)
    direction = torch.randn(atype.shape[0], 3, generator=generator, device=device)
    spin = 2.0 * direction / direction.norm(dim=1, keepdim=True)
    results: dict[str, np.ndarray] = {}
    for grade in args.grades:
        model = build(grade, True, device)
        for system, types in (("single", atype), ("mixed", mixed)):
            for key, value in evaluate(model, graph, types, spin).items():
                results[f"{grade}/{system}/{key}"] = value
    np.savez(args.out, **results)
    if args.compare is None:
        return
    reference = dict(np.load(args.compare))
    for grade in args.grades:
        for system in ("single", "mixed"):
            report = []
            for key in KEYS:
                name = f"{grade}/{system}/{key}"
                new, old = results[name], reference[name]
                if np.array_equal(new, old):
                    report.append(f"{key} equal")
                else:
                    scale = max(float(np.abs(old).max()), 1e-30)
                    report.append(f"{key} {float(np.abs(new - old).max()) / scale:.1e}")
            print(f"{grade:5s} {system:6s} " + ", ".join(report), flush=True)  # noqa: T201


if __name__ == "__main__":
    main()

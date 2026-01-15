# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, ANN001, ANN201, ANN202, T201
"""Check the compressed DPA4C CPU operator against the portable reference.

The operator and the reference evaluate the same compressed equations, so the
forward is compared directly and the analytical edge cotangent against the
autograd of the reference. Both are run over the released grades and over a
synthetic sweep of the structural parameters the kernel specializes on.

Usage
-----
    python check_kernel.py --grades nano,mini,neo,air,plus
    python check_kernel.py --sweep
"""

from __future__ import (
    annotations,
)

import argparse

import torch
from harness import (
    GRADES,
    build_lower_inputs,
    load_model,
)


def _operator_arguments(descriptor, graph, atype):
    from deepmd.pt_expt.kernels.dpa4c.graph_compress import (
        compressed_operator_arguments,
    )

    return (
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


def compare(model, sample, label: str) -> bool:
    """Compare one model against the reference and report the deviations."""
    from deepmd.dpmodel.utils.neighbor_graph import (
        NeighborGraph,
    )
    from deepmd.pt_expt.kernels.dpa4c.graph_compress import (
        _reference_descriptor,
        ensure_registered,
    )

    ensure_registered()
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
    descriptor = model.get_descriptor()
    arguments = _operator_arguments(descriptor, graph, atype)

    kernel, state = torch.ops.deepmd.dpa4c_graph_compress(graph.edge_vec, *arguments)
    leaf = graph.edge_vec.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        reference = _reference_descriptor(
            leaf,
            *arguments[:15],
            *arguments[18:],
            spin=arguments[15],
            spin_pair=arguments[16],
            spin_type=arguments[17],
        )
    seed = torch.randn_like(reference)
    (reference_gradient,) = torch.autograd.grad((reference * seed).sum(), leaf)
    # The descriptor alone is differentiated: no analytical pair potential.
    no_pair = graph.edge_vec.new_empty(0)
    kernel_gradient = torch.ops.deepmd.dpa4c_graph_compress_backward(
        seed, state, graph.edge_vec, *arguments, no_pair, no_pair
    )[0]

    scale = reference.abs().max().clamp(min=1e-12)
    forward_error = (kernel - reference).abs().max() / scale
    gradient_scale = reference_gradient.abs().max().clamp(min=1e-12)
    gradient_error = (kernel_gradient - reference_gradient).abs().max() / gradient_scale
    passed = forward_error < 3e-5 and gradient_error < 3e-4
    print(
        f"{label:28s} D={reference.shape[1]:4d} "
        f"forward={forward_error:.3e} backward={gradient_error:.3e} "
        f"{'ok' if passed else 'FAIL'}"
    )
    return bool(passed)


def synthetic(channels: int, lmax: int, modes: int, atoms: int, stride: float):
    """Build a small synthetic model of one structural configuration."""
    from deepmd.pt_expt.model.get_model import (
        get_model,
    )

    model = get_model(
        {
            "type_map": ["H", "C", "O", "Fe"],
            "descriptor": {
                "type": "dpa4c",
                "rcut": 6.0,
                "channels": channels,
                "lmax": lmax,
                "basis_type": "bessel",
                "n_radial": 16,
                "radial_modes": modes,
                "precision": "float32",
                "seed": 7,
            },
            "fitting_net": {
                "type": "ener",
                "neuron": [32, 32],
                "resnet_dt": False,
                "activation_function": "silu",
                "precision": "float32",
                "seed": 7,
            },
        }
    ).eval()
    model.get_descriptor().enable_compression(0.0, table_stride_1=stride)
    return model, build_lower_inputs(model, atoms, atom_type=1)


def main() -> None:
    """Run the requested comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grades", default="nano,mini,neo,air,plus")
    parser.add_argument("--atoms", type=int, default=216)
    parser.add_argument("--stride", type=float, default=0.01)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    passed = True
    if args.sweep:
        for channels in (8, 16, 32, 64, 128):
            for lmax in (2, 3, 4):
                for modes in (0, 4):
                    model, sample = synthetic(
                        channels, lmax, modes, args.atoms, args.stride
                    )
                    passed &= compare(model, sample, f"C{channels} lmax{lmax} R{modes}")
    else:
        for grade in args.grades.split(","):
            model = load_model(GRADES[grade])
            model.get_descriptor().enable_compression(0.0, table_stride_1=args.stride)
            sample = build_lower_inputs(model, args.atoms)
            passed &= compare(model, sample, grade)
    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()

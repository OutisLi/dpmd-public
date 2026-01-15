# SPDX-License-Identifier: LGPL-3.0-or-later
"""Accuracy of a low-rank compression of the DPA4C scalar-only radial channels.

Channels at index ``C1`` and above feed degree zero only, so their moments are
linear in the channel values and may be accumulated on ``K`` basis functions
and reconstructed per node. The basis is built at compression time from the
network itself: the radial map is the linear output layer ``W`` applied to the
shared hidden state ``h(r)``, so the best rank-``K`` approximation of the
scalar-only channels in a weighted value-and-slope norm is the projection
``W_Q <- W_Q D V V^T D^-1`` onto the leading right singular vectors ``V`` of
the sampled, per-channel scaled function matrix (Eckart-Young). The basis
functions ``b(r) = V^T D g_Q(r)`` are then outputs of the same network and are
tabulated by the same quintic spline as the channels they replace.

The script reports, per grade,

1. the function-level error of the projection: the largest error of each
   scalar-only channel and of its radial derivative relative to the largest
   magnitude of the channel over ``[r_lo, rcut]``, next to the fp32 noise of
   the current table in the same metric;
2. the end-to-end error of energy, forces and virial on a jittered diamond
   supercell of (b) the current compression and (c) the compression of the
   rank-``K`` projected network, both evaluated as deployed, against the
   uncompressed fp32 model of production and against the same weights
   evaluated in fp64, next to (a) the fp32 model itself against fp64.

Usage
-----
    python lowrank_accuracy.py --grades neo air plus --ranks 24 28 32 40
"""

from __future__ import (
    annotations,
)

import argparse
import copy
from typing import (
    TYPE_CHECKING,
)

import numpy as np
import torch  # noqa: TID253
from gen_system import (
    build,
)
from table_accuracy import (
    TRAINED,
    network,
    table_eval,
)

from deepmd.pt_expt.kernels.dpa4c.graph_compress import (  # noqa: TID253
    build_radial_table,
)
from deepmd.pt_expt.model.get_model import (  # noqa: TID253
    get_model,
)
from deepmd.pt_expt.train.wrapper import (  # noqa: TID253
    ModelWrapper,
)
from deepmd.pt_expt.utils.canonical_graph import (  # noqa: TID253
    CanonicalGraph,
    canonical_graph_from_pairs,
)

if TYPE_CHECKING:
    from deepmd.pt_expt.descriptor.dpa4c import (
        DescrptDPA4C,
    )


def degree_one_width(channels: int) -> int:
    """Degree-one width of the kernels' channel profile."""
    exponent = max(channels.bit_length() - 1, 0)
    return max(1 << ((exponent + 1) // 2), 4)


def load_trained(grade: str, device: torch.device, precision: str) -> torch.nn.Module:
    """Load the trained EMA checkpoint of one released grade at ``precision``."""
    state = torch.load(
        TRAINED.format(grade.capitalize()), map_location=device, weights_only=True
    )
    state = state.get("model", state)
    params = copy.deepcopy(state["_extra_state"]["model_params"])
    params["descriptor"]["precision"] = precision
    params["fitting_net"]["precision"] = precision
    model = get_model(params)
    ModelWrapper(model).load_state_dict(state)
    return model.to(device).eval()


def output_layer(descriptor: DescrptDPA4C) -> torch.Tensor:
    """Weight ``(H, C)`` of the final linear layer of the radial map."""
    return descriptor.radial_embedding.layers[-1].w


def projection_basis(
    descriptor: DescrptDPA4C,
    first: int,
    r_lo: float,
    slope_weight: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right singular vectors and scales of the scaled scalar-only channels.

    The function matrix stacks the values and the length-weighted radial
    derivatives on a dense grid of ``[r_lo, rcut]``, every row weighted by the
    square root of ``r^2``, the density of neighbors at that length, and every
    channel divided by its largest magnitude.
    """
    rcut = float(descriptor.rcut)
    grid = torch.arange(r_lo, rcut, 2.0e-4, dtype=torch.float64, device=device)
    values, slopes = network(descriptor, grid)
    channels = int(descriptor.channels)
    g, dg = values[:, first:channels], slopes[:, first:channels]
    scale = g.abs().amax(0).clamp_min(1e-30)
    weight = grid[:, None]
    rows = torch.cat([weight * g / scale, slope_weight * weight * dg / scale], dim=0)
    _, s, vh = torch.linalg.svd(rows, full_matrices=False)
    return vh.T, s / s[0], scale


def project(
    descriptor: DescrptDPA4C, first: int, v: torch.Tensor, scale: torch.Tensor
) -> None:
    """Replace the scalar-only output columns by their rank-``K`` projection."""
    w = output_layer(descriptor)
    with torch.no_grad():
        block = w[:, first:].to(torch.float64)
        projector = (v @ v.T) * scale[None, :] / scale[:, None]
        w[:, first:] = (block @ projector).to(w.dtype)


def function_errors(
    descriptor: DescrptDPA4C,
    reference: DescrptDPA4C,
    first: int,
    r_lo: float,
    device: torch.device,
) -> tuple[float, float]:
    """Largest relative value and slope error of the scalar-only table columns.

    The table is built from ``descriptor`` and compared against the network of
    ``reference``, so a projected descriptor reports the whole error of its
    compression against the network it approximates.
    """
    rcut = float(reference.rcut)
    grid = torch.arange(r_lo, rcut, 1.0e-4, dtype=torch.float64, device=device)
    values, slopes = network(reference, grid)
    table, info = build_radial_table(descriptor, 0.01)
    value32, slope32 = table_eval(table, float(info[0]), grid, torch.float32)
    channels = int(descriptor.channels)
    g, dg = values[:, first:channels], slopes[:, first:channels]
    value_scale = g.abs().amax(0).clamp_min(1e-30)
    slope_scale = dg.abs().amax(0).clamp_min(1e-30)
    value_error = (
        (value32[:, first:channels].double() - g).abs().amax(0) / value_scale
    ).max()
    slope_error = (
        (slope32[:, first:channels].double() - dg).abs().amax(0) / slope_scale
    ).max()
    return float(value_error), float(slope_error)


def minimum_image_graph(
    positions: np.ndarray, box: np.ndarray, rcut: float, device: torch.device
) -> CanonicalGraph:
    """Canonical graph of every minimum-image pair closer than ``rcut``.

    The box must be at least twice the cutoff, so that the minimum image is the
    only image of a pair inside the cutoff.
    """
    if float(np.min(box)) < 2.0 * rcut:
        raise ValueError(f"box {box} is shorter than twice the cutoff {rcut}")
    pos = torch.as_tensor(positions, dtype=torch.float64, device=device)
    cell = torch.as_tensor(box, dtype=torch.float64, device=device)
    vector = pos[None, :, :] - pos[:, None, :]
    vector -= torch.round(vector / cell) * cell
    length2 = (vector * vector).sum(-1)
    count = pos.shape[0]
    inside = (length2 < rcut * rcut) & ~torch.eye(
        count, dtype=torch.bool, device=device
    )
    destination, source = inside.nonzero(as_tuple=True)
    return canonical_graph_from_pairs(
        destination,
        source,
        vector[destination, source],
        torch.tensor([count], dtype=torch.int64, device=device),
        count,
    )


def reference_outputs(
    model: torch.nn.Module,
    positions: np.ndarray,
    box: np.ndarray,
    carbon: int,
    device: torch.device,
) -> dict:
    """Energy, forces and virial of the uncompressed model by autograd."""
    dtype = next(model.parameters()).dtype
    coord = torch.as_tensor(positions, dtype=dtype, device=device).reshape(1, -1, 3)
    cell = torch.as_tensor(np.diag(box), dtype=dtype, device=device).reshape(1, 9)
    atype = torch.full((1, coord.shape[1]), carbon, dtype=torch.int64, device=device)
    out = model(coord, atype, box=cell)
    return {
        "energy": out["energy"].reshape(-1).double(),
        "force": out["force"].reshape(-1, 3).double(),
        "virial": out["virial"].reshape(-1).double(),
    }


def compressed_outputs(
    model: torch.nn.Module, graph: CanonicalGraph, atype: torch.Tensor
) -> dict:
    """Energy, forces and virial of a compressed model on the canonical graph."""
    out = model.forward_lower_canonical_graph(
        atype,
        graph.n_node,
        graph.n_local,
        graph.source,
        graph.edge_vec,
        graph.destination_row_ptr,
        graph.source_row_ptr,
        graph.source_order,
        do_atomic_virial=False,
    )
    return {
        "energy": out["energy"].reshape(-1).double(),
        "force": out["force"].reshape(-1, 3).double(),
        "virial": out["virial"].reshape(-1).double(),
    }


def compare(test: dict, ref: dict, atoms: int) -> str:
    """Errors of ``test`` against ``ref``: energy per atom, forces, virial."""
    energy = float((test["energy"] - ref["energy"]).abs().max()) / atoms
    delta = test["force"] - ref["force"]
    force_max = float(delta.abs().max())
    force_rms = float(delta.pow(2).mean().sqrt())
    force_scale = float(ref["force"].pow(2).mean().sqrt())
    virial = float((test["virial"] - ref["virial"]).abs().max()) / atoms
    return (
        f"dE/atom {energy:.2e} eV | dF max {force_max:.2e} rms {force_rms:.2e} eV/A "
        f"(rel rms {force_rms / force_scale:.2e}) | dW/atom {virial:.2e} eV"
    )


def compress(model: torch.nn.Module, stride: float) -> None:
    """Compress ``model`` in place as ``dp compress`` does: fold the vacuum
    reference into the energy bias, then tabulate the descriptor.
    """
    model.fold_vacuum_reference()
    descriptor = model.atomic_model.fused_decomposition()[0].descriptor
    descriptor.enable_compression(1.4, table_stride_1=stride)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grades", nargs="+", default=["mini", "neo", "air", "plus"])
    parser.add_argument(
        "--ranks", type=int, nargs="+", default=[20, 24, 28, 32, 40, 48]
    )
    parser.add_argument("--atoms", type=int, default=512)
    parser.add_argument("--jitter", type=float, default=0.1)
    parser.add_argument("--r-lo", type=float, default=0.5)
    parser.add_argument("--slope-weight", type=float, default=1.0)
    parser.add_argument("--stride", type=float, default=0.01)
    parser.add_argument("--no-e2e", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda")
    positions, box = build(args.atoms, jitter=args.jitter)
    graph = None
    for grade in args.grades:
        model32 = load_trained(grade, device, "float32")
        type_map = model32.get_type_map()
        carbon = type_map.index("C")
        descriptor = model32.atomic_model.fused_decomposition()[0].descriptor
        if graph is None:
            graph = minimum_image_graph(positions, box, float(descriptor.rcut), device)
        channels = int(descriptor.channels)
        first = degree_one_width(channels)
        v_all, spectrum, scale = projection_basis(
            descriptor, first, args.r_lo, args.slope_weight, device
        )
        base_value, base_slope = function_errors(
            descriptor, descriptor, first, args.r_lo, device
        )
        print(  # noqa: T201
            f"== {grade}: C={channels}, scalar-only {channels - first}; current table "
            f"fp32 error value {base_value:.2e} slope {base_slope:.2e}; "
            "sigma_k/sigma_0 at k=16,24,32,40: "
            + ", ".join(
                f"{float(spectrum[k]):.1e}"
                for k in (16, 24, 32, 40)
                if k < len(spectrum)
            ),
            flush=True,
        )
        atype = torch.full((args.atoms,), carbon, dtype=torch.int64, device=device)
        if not args.no_e2e:
            ref64 = reference_outputs(
                load_trained(grade, device, "float64"), positions, box, carbon, device
            )
            ref32 = reference_outputs(model32, positions, box, carbon, device)
            print(  # noqa: T201
                f"  uncompressed fp32 vs fp64        {compare(ref32, ref64, args.atoms)}",
                flush=True,
            )
            current = copy.deepcopy(model32)
            compress(current, args.stride)
            out = compressed_outputs(current, graph, atype)
            print(  # noqa: T201
                f"  current compression vs fp32      {compare(out, ref32, args.atoms)}",
                flush=True,
            )
            print(  # noqa: T201
                f"  current compression vs fp64      {compare(out, ref64, args.atoms)}",
                flush=True,
            )
        for rank in args.ranks:
            if rank >= channels - first:
                continue
            reduced = copy.deepcopy(model32)
            reduced_descriptor = reduced.atomic_model.fused_decomposition()[
                0
            ].descriptor
            project(reduced_descriptor, first, v_all[:, :rank], scale)
            value_error, slope_error = function_errors(
                reduced_descriptor, descriptor, first, args.r_lo, device
            )
            line = (
                f"  K={rank:3d} (columns {first + rank:3d} of {channels}): table vs network "
                f"value {value_error:.2e} slope {slope_error:.2e}"
            )
            print(line, flush=True)  # noqa: T201
            if not args.no_e2e:
                compress(reduced, args.stride)
                out = compressed_outputs(reduced, graph, atype)
                print(  # noqa: T201
                    f"    K={rank:3d} vs fp32                   {compare(out, ref32, args.atoms)}",
                    flush=True,
                )
                print(  # noqa: T201
                    f"    K={rank:3d} vs fp64                   {compare(out, ref64, args.atoms)}",
                    flush=True,
                )


if __name__ == "__main__":
    main()

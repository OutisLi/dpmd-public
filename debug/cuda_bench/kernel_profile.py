# SPDX-License-Identifier: LGPL-3.0-or-later
"""Time the compressed DPA4C descriptor kernels on a diamond neighborhood.

The harness builds one canonical CSR graph from a diamond supercell and times
the forward and backward operators in isolation, which is the fast inner loop
for kernel work. End-to-end MD throughput is measured separately by the LAMMPS
scan.

Usage
-----
    python kernel_profile.py --channels 8 16 32 64 128 --atoms 32768
"""

from __future__ import (
    annotations,
)

import argparse
import dataclasses
import time
from typing import (
    TYPE_CHECKING,
)

import torch  # noqa: TID253

from deepmd.pt_expt.descriptor.dpa4c import (  # noqa: TID253
    DescrptDPA4C,
)
from deepmd.pt_expt.kernels.dpa4c.graph_compress import (  # noqa: TID253
    build_compression_artifacts,
    contact_radius_input,
    ensure_registered,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

DIAMOND_LATTICE = 3.567
ARTIFACT_ORDER = (
    "data",
    "pair_film",
    "pair_mixing",
    "type_embedding",
    "readout_matrices",
    "coupling_meta",
    "coupling_entry",
    "coupling_value",
    "output_mean",
    "output_inv_std",
)
SPIN_ARTIFACT_ORDER = ("spin_pair", "spin_type")
_BASIS = (
    (0.00, 0.00, 0.00),
    (0.00, 0.50, 0.50),
    (0.50, 0.00, 0.50),
    (0.50, 0.50, 0.00),
    (0.25, 0.25, 0.25),
    (0.25, 0.75, 0.75),
    (0.75, 0.25, 0.75),
    (0.75, 0.75, 0.25),
)


@dataclasses.dataclass(frozen=True)
class CanonicalGraph:
    """Compact destination-sorted CSR topology of a fully periodic lattice."""

    edge_vec: torch.Tensor
    source: torch.Tensor
    destination_row_ptr: torch.Tensor


def _neighbor_shell(rcut: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Enumerate the translation-invariant diamond neighbor shell.

    Returns
    -------
    cell_offset
        Integer cell displacement of each neighbor with shape ``(B, M, 3)``.
    basis_index
        Basis index of each neighbor with shape ``(B, M)``.
    displacement
        Cartesian displacement of each neighbor with shape ``(B, M, 3)`` in Å.
    """
    basis = torch.tensor(_BASIS, dtype=torch.float64)
    reach = int(rcut / DIAMOND_LATTICE) + 1
    index = torch.arange(-reach, reach + 1, dtype=torch.float64)
    cells = torch.stack(
        torch.meshgrid(index, index, index, indexing="ij"),
        dim=-1,
    ).reshape(-1, 3)
    offsets = []
    indices = []
    vectors = []
    for center in basis:
        delta = (cells[:, None, :] + basis[None, :, :] - center[None, None, :]).reshape(
            -1, 3
        )
        distance = delta.norm(dim=-1) * DIAMOND_LATTICE
        keep = (distance > 1.0e-6) & (distance < rcut)
        cell = cells.repeat_interleave(basis.shape[0], dim=0)[keep]
        basis_of = torch.arange(basis.shape[0]).repeat(cells.shape[0])[keep]
        offsets.append(cell.to(torch.int64))
        indices.append(basis_of)
        vectors.append(delta[keep] * DIAMOND_LATTICE)
    return (
        torch.stack(offsets),
        torch.stack(indices),
        torch.stack(vectors).to(torch.float32),
    )


def build_graph(rcut: float, atoms: int) -> tuple[CanonicalGraph, torch.Tensor]:
    """Build one canonical CSR graph of a fully periodic diamond supercell.

    Every site of a perfect lattice carries the same neighbor shell, so the
    topology follows from index arithmetic instead of a spatial search. The
    resulting edge count and memory access pattern match the production LAMMPS
    graph, which is what the kernel timing depends on.

    Parameters
    ----------
    rcut
        Cutoff radius in Å.
    atoms
        Requested atom count; rounded to a cubic supercell.

    Returns
    -------
    graph
        Canonical destination-sorted CSR graph on the current CUDA device.
    atype
        Flat node types with shape ``(N,)``.
    """
    repeat = max(1, round((atoms / len(_BASIS)) ** (1.0 / 3.0)))
    cell_offset, basis_index, displacement = _neighbor_shell(rcut)
    shell = cell_offset.shape[1]
    cells = repeat**3
    node_count = cells * len(_BASIS)

    grid = torch.arange(repeat)
    coordinate = torch.stack(
        torch.meshgrid(grid, grid, grid, indexing="ij"),
        dim=-1,
    ).reshape(-1, 3)
    # Node index ((cx * R + cy) * R + cz) * B + b, so the destination axis is
    # already sorted by construction.
    neighbor_cell = (coordinate[:, None, None, :] + cell_offset[None]) % repeat
    linear = (
        neighbor_cell[..., 0] * repeat + neighbor_cell[..., 1]
    ) * repeat + neighbor_cell[..., 2]
    source = (linear * len(_BASIS) + basis_index[None]).reshape(-1)
    edge_vec = displacement[None].expand(cells, -1, -1, -1).reshape(-1, 3)
    row_ptr = torch.arange(node_count + 1, dtype=torch.int64) * shell
    graph = CanonicalGraph(
        edge_vec=edge_vec.contiguous().cuda(),
        source=source.to(torch.int32).cuda(),
        destination_row_ptr=row_ptr.cuda(),
    )
    atype = torch.zeros(node_count, dtype=torch.int64, device="cuda")
    return graph, atype


def time_kernels(
    channels: int,
    lmax: int,
    radial_modes: int,
    graph: CanonicalGraph,
    atype: torch.Tensor,
    repeats: int,
    stride: float,
    use_spin: bool = False,
) -> tuple[float, float]:
    """Return mean forward and backward milliseconds of one configuration."""
    descriptor = (
        DescrptDPA4C(
            rcut=6.0,
            ntypes=1,
            channels=channels,
            lmax=lmax,
            n_radial=16,
            radial_modes=radial_modes,
            precision="float32",
            seed=42,
            use_spin=[True] if use_spin else None,
        )
        .cuda()
        .eval()
    )
    artifacts = build_compression_artifacts(descriptor, stride)
    ensure_registered()
    spin = (
        torch.randn(atype.shape[0], 3, dtype=torch.float32, device="cuda")
        if use_spin
        else artifacts["spin_type"][:0]
    )
    tail = (
        *[artifacts[name] for name in ARTIFACT_ORDER],
        spin,
        *[artifacts[name] for name in SPIN_ARTIFACT_ORDER],
        contact_radius_input(descriptor),
        int(lmax),
        *[float(value) for value in artifacts["info"].tolist()],
        # No bridging window: equal switch radii.
        0.0,
        0.0,
    )
    arguments = (graph.source, graph.destination_row_ptr, atype, *tail)
    edge_vec = graph.edge_vec

    forward = torch.ops.deepmd.dpa4c_canonical_compress
    backward = torch.ops.deepmd.dpa4c_canonical_compress_backward
    value, state = forward(edge_vec, *arguments)
    cotangent = torch.randn_like(value)
    backward(cotangent, state, edge_vec, *arguments)
    torch.cuda.synchronize()

    def measure(function: Callable[[], object]) -> float:
        """Return the fastest of several timed batches.

        The minimum is reported because every perturbation of a fixed kernel
        launch sequence is additive noise from clock and scheduling jitter.
        """
        for _ in range(8):
            function()
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(repeats):
                function()
            torch.cuda.synchronize()
            best = min(best, (time.perf_counter() - start) * 1.0e3 / repeats)
        return best

    forward_ms = measure(lambda: forward(edge_vec, *arguments))
    backward_ms = measure(lambda: backward(cotangent, state, edge_vec, *arguments))
    return forward_ms, backward_ms


def main() -> None:
    """Time every requested configuration on one shared graph."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--lmax", type=int, nargs="+", default=[2])
    parser.add_argument("--radial-modes", type=int, nargs="+", default=[0])
    parser.add_argument("--atoms", type=int, default=32768)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--stride", type=float, nargs="+", default=[0.002])
    parser.add_argument(
        "--spin",
        action="store_true",
        help="also time the native spin specialization against the baseline",
    )
    args = parser.parse_args()

    graph, atype = build_graph(6.0, args.atoms)
    nodes = atype.shape[0]
    edges = int(graph.destination_row_ptr[-1].item())
    print(  # noqa: T201
        f"graph: {nodes} nodes, {edges} edges, "
        f"{edges / max(nodes, 1):.1f} edges per node",
        flush=True,
    )
    for stride in args.stride:
        for lmax in args.lmax:
            for radial_modes in args.radial_modes:
                for channels in args.channels:
                    for use_spin in (False, True) if args.spin else (False,):
                        forward_ms, backward_ms = time_kernels(
                            channels,
                            lmax,
                            radial_modes,
                            graph,
                            atype,
                            args.repeats,
                            stride,
                            use_spin,
                        )
                        label = "spin" if use_spin else "base"
                        print(  # noqa: T201
                            f"C={channels:3d} lmax={lmax} R={radial_modes} "
                            f"h={stride:g} {label}: "
                            f"forward {forward_ms:7.3f} ms  "
                            f"backward {backward_ms:7.3f} ms  "
                            f"total {forward_ms + backward_ms:7.3f} ms",
                            flush=True,
                        )


if __name__ == "__main__":
    main()

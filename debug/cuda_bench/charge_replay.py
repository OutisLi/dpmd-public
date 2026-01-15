# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check that a runtime charge state reaches a replayed compact canonical model.

The C++ backend captures the compact canonical evaluation of a graph without
halo nodes into a CUDA graph on its second call and replays it afterwards.
The captured graph reads the model constants where they resided at capture,
so ``set_charge_spin``, which rebuilds the charge-state constants, must
release the graph before it applies the new state.

Two subcommands, run in separate processes so that the second one loads the
operator library of the C++ build alone:

    python charge_replay.py graph --model model.pt2 --out graph.npz [--atoms 64]
    python charge_replay.py run --lib LIBDIR --model model.pt2 --graph graph.npz

``graph`` writes a canonical graph of a random cluster without halo nodes, at
the cutoff that the model archive records, since the compact operator reads
its radial table at every bond length the graph holds.
``run`` drives the C API of ``LIBDIR/libdeepmd_c.so``: three calls under the
default state (eager, capture and replay, replay), ``set_charge_spin``, three
more calls, and fresh instances evaluated eagerly under each state as the
references. The model must be a compressed DPA4C frozen with a charge state,
for example ``freeze_dpa4c.py --preset dpa4c-nano-v20260911 --charge-state
--default-charge-spin 2 3``.
"""

from __future__ import (
    annotations,
)

import argparse
import ctypes
import json
import sys
import zipfile
from typing import (
    TYPE_CHECKING,
)

import numpy as np

if TYPE_CHECKING:
    import torch


def model_cutoff(model: str) -> float:
    """Return the cutoff radius in Å that the metadata of a ``.pt2`` records."""
    with zipfile.ZipFile(model) as archive:
        return float(json.loads(archive.read("model/extra/metadata.json"))["rcut"])


def build_graph(out: str, atoms: int, rcut: float) -> None:
    """Write the canonical graph of a random open cluster of ``atoms`` atoms."""
    import dataclasses

    import torch

    from deepmd.dpmodel.utils.neighbor_graph import (
        attach_edge_csr,
        graph_from_dense_quartet,
    )
    from deepmd.pt.utils.nlist import (
        extend_input_and_build_neighbor_list,
    )
    from deepmd.pt_expt.utils.canonical_graph import (
        canonical_graph_from_neighbor_graph,
    )

    generator = torch.Generator(device="cuda").manual_seed(7)
    # A density of 0.1 atoms per cubic Å, a typical condensed-phase value.
    side = (atoms / 0.1) ** (1.0 / 3.0)
    coordinate = (
        torch.rand(1, atoms, 3, generator=generator, device="cuda", dtype=torch.float32)
        * side
    )
    atype = (torch.arange(atoms, device="cuda") % 2).reshape(1, -1)
    coord_ext, atype_ext, mapping, nlist = extend_input_and_build_neighbor_list(
        coordinate, atype, rcut, [160], mixed_types=True, box=None
    )
    graph, flat_type = graph_from_dense_quartet(coord_ext, atype_ext, nlist, mapping)
    graph = attach_edge_csr(graph, flat_type.shape[0], canonicalize=True)
    canonical = canonical_graph_from_neighbor_graph(
        dataclasses.replace(graph, n_local=graph.n_node)
    )
    np.savez(
        out,
        atype=flat_type.cpu().numpy().astype(np.int64),
        source=canonical.source.cpu().numpy(),
        edge_vec=canonical.edge_vec.float().cpu().numpy(),
        destination_row_ptr=canonical.destination_row_ptr.cpu().numpy(),
        source_row_ptr=canonical.source_row_ptr.cpu().numpy(),
        source_order=canonical.source_order.cpu().numpy(),
    )
    edges = int(canonical.destination_row_ptr[-1])
    print(f"graph -> {out}: {atoms} nodes, {edges} edges")  # noqa: T201


class Model:
    """One ``DP_DeepPot`` instance evaluating a fixed device-resident graph."""

    def __init__(
        self, lib: ctypes.CDLL, model: str, graph: dict, stream: torch.cuda.Stream
    ) -> None:
        import torch

        self.lib = lib
        self.stream = stream
        self.handle = lib.DP_NewDeepPot(model.encode())
        self.check()
        device = torch.device("cuda")
        self.inputs = {
            "atype": torch.as_tensor(graph["atype"], device=device),
            "source": torch.from_numpy(np.ascontiguousarray(graph["source"])).to(
                device
            ),
            "edge_vec": torch.as_tensor(graph["edge_vec"], device=device).contiguous(),
            "destination_row_ptr": torch.as_tensor(
                graph["destination_row_ptr"], device=device
            ),
        }
        self.nodes = int(graph["atype"].shape[0])
        self.storage = int(graph["edge_vec"].shape[0])
        self.atom_energy = torch.zeros(self.nodes, dtype=torch.float64, device=device)
        self.force = torch.zeros(self.nodes * 3, dtype=torch.float64, device=device)
        self.virial = torch.zeros(9, dtype=torch.float64, device=device)
        torch.cuda.synchronize()

    def check(self) -> None:
        message = ctypes.string_at(self.lib.DP_DeepPotCheckOK(self.handle)).decode()
        if message:
            raise RuntimeError(message)

    def set_charge_spin(self, state: list[float]) -> None:
        values = (ctypes.c_double * len(state))(*state)
        self.lib.DP_DeepPotSetChargeSpin(self.handle, values, len(state))
        self.check()

    def evaluate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate once without the source CSR or a per-atom virial."""
        inputs = self.inputs
        self.lib.DP_DPA4ComputeCanonicalGraphGPU(
            self.handle,
            self.atom_energy.data_ptr(),
            self.force.data_ptr(),
            self.virial.data_ptr(),
            None,
            inputs["atype"].data_ptr(),
            inputs["source"].data_ptr(),
            inputs["source"].element_size(),
            inputs["edge_vec"].data_ptr(),
            inputs["destination_row_ptr"].data_ptr(),
            None,
            None,
            None,
            None,
            self.nodes,
            self.nodes,
            self.storage,
            self.stream.cuda_stream,
        )
        self.check()
        self.stream.synchronize()
        return (
            self.atom_energy.cpu().numpy().copy(),
            self.force.cpu().numpy().copy(),
            self.virial.cpu().numpy().copy(),
        )


def deviation(
    actual: tuple[np.ndarray, ...], expected: tuple[np.ndarray, ...]
) -> float:
    """Largest absolute difference over the atom energy, force and virial."""
    return max(
        float(np.abs(a - e).max()) for a, e in zip(actual, expected, strict=True)
    )


def run(libdir: str, model: str, graph_path: str, state: list[float]) -> int:
    import torch

    lib = ctypes.CDLL(f"{libdir}/libdeepmd_c.so")
    lib.DP_NewDeepPot.restype = ctypes.c_void_p
    lib.DP_NewDeepPot.argtypes = [ctypes.c_char_p]
    lib.DP_DeepPotCheckOK.restype = ctypes.c_void_p
    lib.DP_DeepPotCheckOK.argtypes = [ctypes.c_void_p]
    lib.DP_DeepPotSetChargeSpin.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
    ]
    lib.DP_DPA4ComputeCanonicalGraphGPU.restype = None
    lib.DP_DPA4ComputeCanonicalGraphGPU.argtypes = (
        [ctypes.c_void_p] * 7
        + [ctypes.c_int]
        + [ctypes.c_void_p] * 6
        + [ctypes.c_int64] * 3
        + [ctypes.c_void_p]
    )
    graph = dict(np.load(graph_path))
    stream = torch.cuda.Stream()

    # === Step 1. One instance through both states, replayed between them ===
    model_ab = Model(lib, model, graph, stream)
    before = [model_ab.evaluate() for _ in range(3)]
    model_ab.set_charge_spin(state)
    after = [model_ab.evaluate() for _ in range(3)]

    # === Step 2. Fresh instances evaluated eagerly as the references ===
    reference_a = Model(lib, model, graph, stream).evaluate()
    model_b = Model(lib, model, graph, stream)
    model_b.set_charge_spin(state)
    reference_b = model_b.evaluate()

    print(f"states differ by {deviation(reference_a, reference_b):.3e}")  # noqa: T201
    failures = 0
    for phase, results, reference in (
        ("default state", before, reference_a),
        (f"state {state}", after, reference_b),
    ):
        for call, result in enumerate(results, start=1):
            gap = deviation(result, reference)
            failures += gap != 0.0
            print(f"{phase}, call {call}: max |deviation| {gap:.3e}")  # noqa: T201
    print("PASS" if failures == 0 else f"FAIL ({failures} calls)")  # noqa: T201
    return 0 if failures == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    graph = commands.add_parser("graph", help="write a canonical test graph")
    graph.add_argument("--model", required=True, help="model whose cutoff applies")
    graph.add_argument("--out", required=True)
    graph.add_argument("--atoms", type=int, default=64)
    check = commands.add_parser("run", help="drive the C API through two states")
    check.add_argument("--lib", required=True, help="directory of libdeepmd_c.so")
    check.add_argument("--model", required=True)
    check.add_argument("--graph", required=True)
    check.add_argument(
        "--state",
        type=float,
        nargs=2,
        default=(0.0, 1.0),
        metavar=("CHARGE", "MULTIPLICITY"),
        help="charge state installed after the first three calls",
    )
    args = parser.parse_args()
    if args.command == "graph":
        build_graph(args.out, args.atoms, model_cutoff(args.model))
        return 0
    return run(args.lib, args.model, args.graph, list(args.state))


if __name__ == "__main__":
    sys.exit(main())

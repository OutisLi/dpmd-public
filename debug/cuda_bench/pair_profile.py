# SPDX-License-Identifier: LGPL-3.0-or-later
"""Time the zone-bridging specialization of the compressed DPA4C edge scans.

The harness reuses the diamond graph of ``kernel_profile.py`` and runs the
generic operators on its canonical topology, because they are the ones that
take the analytical pair potential. Each channel width is timed three ways:
the plain scans, the bridged scans with a window, and the bridged backward
with the ZBL pair table in addition. The differences are the cost of the
window and of the pair term.

Kernel timings are reproducible to a fraction of a percent only across
processes of one configuration, so a comparison runs one width per process
and takes the median of several processes.

Usage
-----
    python pair_profile.py --channels 64 --atoms 32768
"""

from __future__ import (
    annotations,
)

import argparse
import time
from typing import (
    TYPE_CHECKING,
)

import torch  # noqa: TID253
from kernel_profile import (
    ARTIFACT_ORDER,
    SPIN_ARTIFACT_ORDER,
    CanonicalGraph,
    build_graph,
)

from deepmd.dpmodel.atomic_model.inner_potential import (
    InnerPotential,
)
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

#: Bridging window of the windowed measurement, in Å.
WINDOW = (0.5, 0.8)


def measure(function: Callable[[], object], repeats: int) -> float:
    """Return the fastest mean milliseconds of several timed batches."""
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


def time_scans(
    channels: int,
    lmax: int,
    graph: CanonicalGraph,
    atype: torch.Tensor,
    window: tuple[float, float] | None,
    repeats: int,
) -> tuple[float, float, float]:
    """Time the forward, the backward and the backward with the pair term.

    Parameters
    ----------
    channels
        Scalar width of the descriptor.
    lmax
        Angular degree of the descriptor.
    graph
        Canonical destination-sorted CSR graph on the CUDA device.
    atype
        Flat node types with shape ``(N,)``.
    window
        Inner and outer radius of the bridging window in Å, or ``None`` for a
        descriptor without one, which runs the plain scans.
    repeats
        Launches per timed batch.

    Returns
    -------
    forward_ms, backward_ms, pair_backward_ms
        Milliseconds per launch. The last one attaches the ZBL pair table and
        therefore always runs the bridged backward.
    """
    bridging = (
        {}
        if window is None
        else {
            "inner_clamp_f_inner": window[0],
            "inner_clamp_f_outer": window[1],
            "inner_clamp_scale": "absolute",
        }
    )
    descriptor = (
        DescrptDPA4C(
            rcut=6.0,
            ntypes=1,
            channels=channels,
            lmax=lmax,
            n_radial=16,
            radial_modes=0,
            precision="float32",
            seed=42,
            **bridging,
        )
        .cuda()
        .eval()
    )
    artifacts = build_compression_artifacts(descriptor, 0.002)
    ensure_registered()
    nodes = atype.shape[0]
    row_ptr = graph.destination_row_ptr
    destination = torch.repeat_interleave(
        torch.arange(nodes, device="cuda"), row_ptr[1:] - row_ptr[:-1]
    )
    edge_index = torch.stack((graph.source.to(torch.int64), destination))
    arguments = (
        edge_index.contiguous(),
        torch.empty(0, dtype=torch.bool, device="cuda"),
        torch.empty(0, dtype=torch.int64, device="cuda"),
        row_ptr,
        atype,
        *[artifacts[name] for name in ARTIFACT_ORDER],
        artifacts["spin_type"][:0],
        *[artifacts[name] for name in SPIN_ARTIFACT_ORDER],
        contact_radius_input(descriptor),
        True,
        int(lmax),
        *[float(value) for value in artifacts["info"].tolist()],
        *((0.0, 0.0) if window is None else window),
    )
    edge_vec = graph.edge_vec
    forward = torch.ops.deepmd.dpa4c_graph_compress
    backward = torch.ops.deepmd.dpa4c_graph_compress_backward
    value, state = forward(edge_vec, *arguments)
    cotangent = torch.randn_like(value)
    no_pair = edge_vec.new_empty(0)
    pair_table = torch.as_tensor(InnerPotential(["C"]).pair_table, device="cuda")
    pair_seed = torch.ones(nodes, dtype=torch.float64, device="cuda")
    return (
        measure(lambda: forward(edge_vec, *arguments), repeats),
        measure(
            lambda: backward(cotangent, state, edge_vec, *arguments, no_pair, no_pair),
            repeats,
        ),
        measure(
            lambda: backward(
                cotangent, state, edge_vec, *arguments, pair_table, pair_seed
            ),
            repeats,
        ),
    )


def main() -> None:
    """Time every requested width on one shared graph."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", type=int, nargs="+", default=[64])
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument("--atoms", type=int, default=32768)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    graph, atype = build_graph(6.0, args.atoms)
    edges = int(graph.destination_row_ptr[-1].item())
    print(  # noqa: T201
        f"graph: {atype.shape[0]} nodes, {edges} edges",
        flush=True,
    )
    for channels in args.channels:
        plain = time_scans(channels, args.lmax, graph, atype, None, args.repeats)
        bridged = time_scans(channels, args.lmax, graph, atype, WINDOW, args.repeats)
        plain_total = plain[0] + plain[1]
        print(  # noqa: T201
            f"C={channels:3d} lmax={args.lmax}: "
            f"plain forward {plain[0]:.3f} backward {plain[1]:.3f} ms | "
            f"bridged forward {bridged[0]:.3f} backward {bridged[1]:.3f} "
            f"with pair {bridged[2]:.3f} ms | "
            f"window {100.0 * (bridged[0] + bridged[1] - plain_total) / plain_total:+.1f}% "
            f"pair {100.0 * (bridged[2] - bridged[1]) / plain_total:+.1f}% "
            f"of the plain forward and backward",
            flush=True,
        )


if __name__ == "__main__":
    main()

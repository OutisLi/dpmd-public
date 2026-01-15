# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Operator time of the latent-charge long range across system sizes.

The direct sum of a cluster picks its tile and its partner chunks from the
node count and the resident block slots of the device (direct_sum.cu,
pair_launch), and the mesh sum of a periodic frame its launches from the mesh
and the multiprocessor count. This sweep times ``deepmd::latent_charge_sum``
on random clusters of a dense size grid and on periodic cubes of the same
density, with the package-default LES and SOG kernels and one constrained
channel, so that a launch choice that does not fit the device shows as a step
of the time per pair (clusters) or per node (periodic) between neighbouring
sizes.

Every point is the device time of one call: the call is captured into a CUDA
graph after eager warm-up calls, and the median over ``--repeats`` blocks of
ten replays, timed with CUDA events, is divided by ten, so that the host work
of the eager path, which dominates small systems, does not mask the launches.

Usage
-----
    python bench_long_range_ops.py --out results/long_range_ops_<device>.csv
"""

from __future__ import (
    annotations,
)

import argparse
import csv
import math
import statistics

import numpy as np
import torch  # noqa: TID253

# Importing the module loads the operator library.
import deepmd.pt.cxx_op  # noqa: TID253, F401
from deepmd.dpmodel.long_range import (
    COULOMB_CONSTANT,
)
from deepmd.dpmodel.long_range.latent_charge import (
    PME_SPACING_FRACTION,
)
from deepmd.pt_expt.kernels.long_range import (  # noqa: TID253
    latent_charge_sum,
)
from deepmd.pt_expt.long_range.kernel import (  # noqa: TID253
    LESKernel,
    SOGKernel,
)

#: Atoms per cubic angstrom of the systems, about that of Air.
DENSITY = 0.1
KERNELS = {"les": lambda: LESKernel(), "sog": lambda: SOGKernel(precision="float32")}


def system(n_atoms: int, periodic: bool) -> tuple[torch.Tensor, ...]:
    """Positions, charges, mask, node count, total charge and cell of one frame."""
    rng = np.random.default_rng(5)
    side = (n_atoms / DENSITY) ** (1 / 3)
    coord = torch.tensor(
        rng.uniform(0.0, side, (n_atoms, 3)), dtype=torch.float64, device="cuda"
    )
    charge = torch.tensor(
        rng.normal(0.0, 0.3, (n_atoms, 1)), dtype=torch.float32, device="cuda"
    )
    mask = torch.ones(n_atoms, dtype=torch.bool, device="cuda")
    count = torch.tensor([n_atoms], dtype=torch.int64, device="cuda")
    total = torch.zeros(1, dtype=torch.float64, device="cuda")
    # A captured call reads the cell of every replay from pinned memory.
    box = (
        torch.tensor(np.eye(3).reshape(1, 9) * side, dtype=torch.float64).pin_memory()
        if periodic
        else None
    )
    return coord, charge, mask, count, total, box


def time_call(n_atoms: int, kernel_name: str, periodic: bool, repeats: int) -> float:
    """Device time of one call in microseconds."""
    coord, charge, mask, count, total, box = system(n_atoms, periodic)
    kernel = KERNELS[kernel_name]()
    # The default mesh of a frozen module: a fraction of the Nyquist spacing
    # of the wave-number cutoff of the kernel (LatentChargeLongRange.mesh_spacing).
    spacing = (
        PME_SPACING_FRACTION * math.pi / kernel.kspace_cutoff() if periodic else 1.0
    )

    def call() -> None:
        latent_charge_sum(
            coord,
            charge,
            mask,
            count,
            total,
            kernel,
            0,
            COULOMB_CONSTANT,
            box=box,
            spacing=spacing,
        )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e2)
    return statistics.median(times)


def main() -> None:
    """Run the sweep and write one CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--cluster-sizes",
        default=",".join(str(int(256 * 1.15**k)) for k in range(30)),
    )
    parser.add_argument("--periodic-sizes", default="512,2000,10296,40000,100800")
    args = parser.parse_args()
    rows = []
    for periodic, sizes in ((False, args.cluster_sizes), (True, args.periodic_sizes)):
        for n_atoms in map(int, sizes.split(",")):
            for name in KERNELS:
                micro = time_call(n_atoms, name, periodic, args.repeats)
                per_unit = (
                    micro * 1e3 / (n_atoms * n_atoms if not periodic else n_atoms)
                )
                rows.append(
                    {
                        "kind": "periodic" if periodic else "cluster",
                        "kernel": name,
                        "atoms": n_atoms,
                        "us": f"{micro:.2f}",
                        "ns_per_unit": f"{per_unit:.5f}",
                    }
                )
                print(  # noqa: T201
                    f"{rows[-1]['kind']:8s} {name} {n_atoms:7d} atoms {micro:10.2f} us"
                    f"  {per_unit:.5f} ns per {'node' if periodic else 'pair'}",
                    flush=True,
                )
    with open(args.out, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"-> {args.out}")  # noqa: T201


if __name__ == "__main__":
    main()

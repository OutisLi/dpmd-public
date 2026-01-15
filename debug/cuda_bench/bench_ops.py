# SPDX-License-Identifier: LGPL-3.0-or-later
"""Time and check the compact canonical DPA4C inference step in isolation.

The step is the model computation of one MD step on a prebuilt graph: the
fused descriptor-fitting operator and the force/virial assembly, exactly the
sequence a frozen canonical ``.pt2`` executes, evaluated eagerly. The graph is
built from a thermally jittered diamond supercell, so the radial-table
accesses spread over the intervals the way they do in a real trajectory; a
perfect lattice would collapse them onto a handful of shells and hide every
cache effect.

``--dump`` writes the outputs of every grade to an ``.npz`` file and
``--compare`` reports the deviation of the current build from such a file,
which is how a kernel change is checked against the code it replaces.

Usage
-----
    python bench_ops.py --grades nano neo --atoms 100800 --repeats 20
    python bench_ops.py --grades nano --atoms 10296 --dump ref.npz
    python bench_ops.py --grades nano --atoms 10296 --compare ref.npz
"""

from __future__ import (
    annotations,
)

import argparse
import collections
import os
import re

import numpy as np
import torch  # noqa: TID253

# A development build of the operator library is loaded ahead of the package,
# whose loader then finds the operators registered and keeps the installed
# library out of the process.
if os.environ.get("BENCH_OP_LIB"):
    torch.ops.load_library(os.environ["BENCH_OP_LIB"])

from freeze_dpa4c import (
    COMPRESS_STRIDE,
    model_config,
)
from gen_system import (
    build,
)
from kernel_profile import (
    _neighbor_shell,
)

from deepmd.pt_expt.model.get_model import (  # noqa: TID253
    get_model,
)
from deepmd.pt_expt.utils.canonical_graph import (  # noqa: TID253
    CanonicalGraph,
    canonical_graph_from_pairs,
)

RCUT = 6.0
GRADES = ("nano", "mini", "neo", "air", "plus")
OUTPUTS = ("energy", "atom_energy", "force", "virial", "atom_virial")


def read_dump(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read the ``id x y z`` rows of a LAMMPS dump in file order."""
    with open(path, encoding="utf-8") as stream:
        lines = stream.readlines()
    start = (
        next(i for i, line in enumerate(lines) if line.startswith("ITEM: ATOMS")) + 1
    )
    rows = np.loadtxt(lines[start:])
    return rows[:, 0].astype(np.int64) - 1, rows[:, 1:4]


def jittered_diamond_graph(
    target: int,
    jitter: float,
    device: torch.device,
    order: str = "lattice",
    stale: float = 0.0,
    bucket: float = 0.0,
    dump: str | None = None,
) -> tuple[CanonicalGraph, torch.Tensor]:
    """Build the canonical graph of a jittered periodic diamond supercell.

    Candidate pairs come from the perfect-lattice neighbor shell within the
    cutoff plus a margin, which bounds every pair the jitter can bring inside
    the cutoff; the minimum-image vectors of the displaced atoms then decide
    which candidates are edges.
    """
    positions, box = build(target, jitter=jitter)
    node_count = positions.shape[0]
    # A LAMMPS dump replaces the jittered sites by a trajectory frame and
    # numbers the nodes in its file order, the order of the LAMMPS arrays.
    relabel = None
    if dump is not None:
        ids, frame = read_dump(dump)
        positions = np.empty_like(positions)
        positions[ids] = frame
        relabel = np.empty(node_count, dtype=np.int64)
        relabel[ids] = np.arange(node_count)
    cells = node_count // 8
    # gen_system orders atoms as ((cx * ny + cy) * nz + cz) * 8 + basis.
    from gen_system import (
        reps,
    )

    nx, ny, nz = reps(target)
    offsets, basis_index, _ = _neighbor_shell(RCUT + 0.5)
    shell = offsets.shape[1]
    cell_coordinate = torch.stack(
        torch.meshgrid(
            torch.arange(nx), torch.arange(ny), torch.arange(nz), indexing="ij"
        ),
        dim=-1,
    ).reshape(-1, 3)
    extent = torch.tensor([nx, ny, nz])
    center = torch.arange(node_count, dtype=torch.int64).reshape(cells, 8)
    # (cells, 8, shell, 3): neighbor cell of every candidate of every site.
    neighbor_cell = (cell_coordinate[:, None, None, :] + offsets[None]) % extent
    linear = (neighbor_cell[..., 0] * ny + neighbor_cell[..., 1]) * nz + neighbor_cell[
        ..., 2
    ]
    source = linear * 8 + basis_index[None]
    destination = center[:, :, None].expand(-1, -1, shell)
    source = source.reshape(-1).to(device)
    destination = destination.reshape(-1).to(device)
    pos = torch.as_tensor(positions, dtype=torch.float64, device=device)
    box_t = torch.as_tensor(box, dtype=torch.float64, device=device)
    vector = pos[source] - pos[destination]
    vector -= torch.round(vector / box_t) * box_t
    keep = (vector * vector).sum(-1) < RCUT * RCUT
    destination, source, vector = destination[keep], source[keep], vector[keep]
    if relabel is not None:
        relabel_t = torch.as_tensor(relabel, device=device)
        destination, source = relabel_t[destination], relabel_t[source]
    if order == "distance":
        # The builder sorts stably by destination, so a prior sort by length
        # leaves every row in increasing length. A stale order sorts by the
        # lengths of a configuration displaced by ``stale`` Å per component,
        # as a row sorted at the last neighbor rebuild sees them later.
        key = vector
        if stale > 0.0:
            shift = stale * torch.randn(
                pos.shape,
                dtype=pos.dtype,
                device=device,
                generator=torch.Generator(device=device).manual_seed(1),
            )
            key = vector + shift[source] - shift[destination]
        length = (key * key).sum(-1).sqrt()
        if bucket > 0.0:
            # Stable bucketing by length keeps the candidate order inside a
            # bucket, as a counting sort over radial shells does.
            length = torch.floor(length / bucket)
        permutation = torch.argsort(length, stable=True)
    elif order == "random":
        permutation = torch.randperm(destination.shape[0], device=device)
    else:
        permutation = None
    if permutation is not None:
        destination = destination[permutation]
        source = source[permutation]
        vector = vector[permutation]
    graph = canonical_graph_from_pairs(
        destination,
        source,
        vector,
        torch.tensor([node_count], dtype=torch.int64, device=device),
        node_count,
    )
    atype = torch.zeros(node_count, dtype=torch.int64, device=device)
    return graph, atype


def build_model(
    grade: str,
    device: torch.device,
    stride: float = COMPRESS_STRIDE,
    trained: bool = False,
) -> torch.nn.Module:
    """Construct and compress the benchmark model of one released grade.

    With ``trained`` the model is the trained checkpoint of the grade
    (``table_accuracy.TRAINED``), compressed as ``dp compress`` does, which
    adopts the low-rank radial table where the trained radial map allows it;
    otherwise it is the preset with its initial weights.
    """
    if trained:
        from lowrank_accuracy import (
            compress,
            load_trained,
        )

        model = load_trained(grade, device, "float32")
        compress(model, stride)
        return model.to(device)
    config = model_config(
        0, 0, 0, 2, 0, False, False, [0.0, 1.0], None, f"dpa4c-{grade}-v20260911"
    )
    model = get_model(config).eval()
    descriptor = model.atomic_model.fused_decomposition()[0].descriptor
    descriptor.enable_compression(1.4, table_stride_1=stride)
    return model.to(device)


#: Two-entry source CSR placeholders, which decline the per-atom virial.
PLACEHOLDER: dict[str, torch.Tensor] = {}


def evaluate(
    model: torch.nn.Module,
    graph: CanonicalGraph,
    atype: torch.Tensor,
    atom_virial: bool = True,
) -> dict[str, torch.Tensor]:
    if atom_virial:
        source_row_ptr, source_order = graph.source_row_ptr, graph.source_order
    else:
        if not PLACEHOLDER:
            PLACEHOLDER["row"] = graph.source_row_ptr.new_zeros(2)
            PLACEHOLDER["order"] = graph.source_order.new_zeros(2)
        source_row_ptr, source_order = PLACEHOLDER["row"], PLACEHOLDER["order"]
    return model.forward_lower_canonical_graph(
        atype,
        graph.n_node,
        graph.n_local,
        graph.source,
        graph.edge_vec,
        graph.destination_row_ptr,
        source_row_ptr,
        source_order,
        do_atomic_virial=True,
    )


def time_step(
    model: torch.nn.Module,
    graph: CanonicalGraph,
    atype: torch.Tensor,
    repeats: int,
    atom_virial: bool,
) -> float:
    """Return the median milliseconds of one evaluation over timed batches."""
    for _ in range(5):
        evaluate(model, graph, atype, atom_virial)
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            evaluate(model, graph, atype, atom_virial)
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) / repeats)
    return float(np.median(samples))


def kernel_breakdown(
    model: torch.nn.Module,
    graph: CanonicalGraph,
    atype: torch.Tensor,
    repeats: int,
    atom_virial: bool,
) -> list[tuple[str, float]]:
    """Return the device milliseconds per evaluation of every kernel name."""
    from torch.profiler import (
        ProfilerActivity,
        profile,
    )

    evaluate(model, graph, atype, atom_virial)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(repeats):
            evaluate(model, graph, atype, atom_virial)
        torch.cuda.synchronize()
    totals: collections.Counter = collections.Counter()
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            name = event.name.replace("(anonymous namespace)", "anon")
            name = re.sub(r"\(.*", "", name)[:100]
            totals[name] += event.device_time / 1e3 / repeats
    return totals.most_common()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grades", nargs="+", default=list(GRADES))
    parser.add_argument("--atoms", type=int, default=100_800)
    parser.add_argument("--jitter", type=float, default=0.05)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--kernels", action="store_true")
    parser.add_argument("--stride", type=float, default=COMPRESS_STRIDE)
    parser.add_argument(
        "--order", choices=("lattice", "distance", "random"), default="lattice"
    )
    parser.add_argument("--stale", type=float, default=0.0)
    parser.add_argument("--bucket", type=float, default=0.0)
    parser.add_argument(
        "--frame", default=None, help="LAMMPS dump whose frame and order to use"
    )
    parser.add_argument(
        "--no-atom-virial",
        action="store_true",
        help="time the step without the per-atom virial (source CSR placeholders)",
    )
    parser.add_argument(
        "--trained",
        action="store_true",
        help="compress the trained checkpoints instead of the initial presets",
    )
    parser.add_argument("--dump", default=None)
    parser.add_argument("--compare", default=None)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    graph, atype = jittered_diamond_graph(
        args.atoms, args.jitter, device, args.order, args.stale, args.bucket, args.frame
    )
    edges = int(graph.destination_row_ptr[-1])
    print(  # noqa: T201
        f"graph: {atype.shape[0]} nodes, {edges} edges "
        f"({edges / atype.shape[0]:.1f} per node)",
        flush=True,
    )
    reference = dict(np.load(args.compare)) if args.compare else None
    dumped = {}
    for grade in args.grades:
        torch.manual_seed(0)
        model = build_model(grade, device, args.stride, args.trained)
        with torch.no_grad():
            output = evaluate(model, graph, atype)
        values = {key: output[key].detach().double().cpu().numpy() for key in OUTPUTS}
        atom_virial = not args.no_atom_virial
        line = (
            f"{grade:5s} step "
            f"{time_step(model, graph, atype, args.repeats, atom_virial):8.3f} ms"
        )
        if reference is not None:
            parts = []
            for key in OUTPUTS:
                truth = reference[f"{grade}/{key}"]
                scale = max(float(np.abs(truth).max()), 1e-30)
                parts.append(
                    f"{key} {float(np.abs(values[key] - truth).max()) / scale:.1e}"
                )
            line += "  max rel dev: " + ", ".join(parts)
        print(line, flush=True)  # noqa: T201
        if args.kernels:
            for name, value in kernel_breakdown(model, graph, atype, 5, atom_virial)[
                :16
            ]:
                print(f"    {value:8.4f} ms  {name}", flush=True)  # noqa: T201
        for key in OUTPUTS:
            dumped[f"{grade}/{key}"] = values[key]
        del model
        torch.cuda.empty_cache()
    if args.dump:
        np.savez(args.dump, **dumped)
        print(f"wrote {args.dump}", flush=True)  # noqa: T201


if __name__ == "__main__":
    main()

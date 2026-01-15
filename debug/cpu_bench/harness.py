# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253
"""Shared harness for the DPA4C CPU inference benchmarks.

The harness builds one periodic supercell, converts it to the graph lower ABI
once, and then repeatedly evaluates the model, so a measurement covers the
model graph (descriptor, fitting and the force backward) rather than the
neighbor-list construction. Memory is reported as the resident-set growth of
the process, which is the quantity a CPU deployment is actually bounded by.
"""

from __future__ import (
    annotations,
)

import dataclasses
import gc
import os
import platform
import re
import time
from dataclasses import (
    dataclass,
)
from typing import (
    Any,
)

import numpy as np


def _configure_process() -> int:
    """Pin the device and the thread count before anything reads them.

    The pt_expt device and the intra-op thread count are both resolved once,
    when their environment module is first imported, and DeePMD-kit derives
    the torch thread count from its own variables rather than from
    ``OMP_NUM_THREADS``. The benchmark therefore drives all of them from one
    knob: ``OMP_NUM_THREADS``, defaulting to one thread per physical core.

    Returns
    -------
    int
        The resolved thread count.
    """
    os.environ.setdefault("DEVICE", "cpu")
    threads = int(os.environ.get("OMP_NUM_THREADS", "0")) or _physical_cores()
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["DP_INTRA_OP_PARALLELISM_THREADS"] = str(threads)
    os.environ["DP_INTER_OP_PARALLELISM_THREADS"] = "1"
    return threads


def _physical_cores() -> int:
    """Return the number of online physical cores."""
    cores = set()
    for cpu in _parse_cpu_list(_read("/sys/devices/system/cpu/online")):
        base = f"/sys/devices/system/cpu/cpu{cpu}/topology"
        cores.add((_read(f"{base}/physical_package_id"), _read(f"{base}/core_id")))
    return len(cores)


HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE = "/aisi-vepfs/outisli/Research/aisi_dp_script/Release"
#: Resolved before ``torch`` is imported so the runtime sees a consistent
#: thread count everywhere.
THREADS = _configure_process()

import torch

DPA4C_RELEASE = f"{RELEASE}/DPA4C-OMat24/v20260819"

#: Released DPA4C grades in ascending cost order. Every entry is a training
#: checkpoint of the pt_expt backend.
GRADES: dict[str, str] = {
    "nano": f"{DPA4C_RELEASE}/DPA4C-Nano-OMat24-v20260819.pt",
    "mini": f"{DPA4C_RELEASE}/DPA4C-Mini-OMat24-v20260819.pt",
    "neo": f"{DPA4C_RELEASE}/DPA4C-Neo-OMat24-v20260819.pt",
    "air": f"{DPA4C_RELEASE}/DPA4C-Air-OMat24-v20260819.pt",
    "plus": f"{DPA4C_RELEASE}/DPA4C-Plus-OMat24-v20260819.pt",
}

DIAMOND_LATTICE = 3.567
_DIAMOND_BASIS = np.array(
    [
        (0.00, 0.00, 0.00),
        (0.00, 0.50, 0.50),
        (0.50, 0.00, 0.50),
        (0.50, 0.50, 0.00),
        (0.25, 0.25, 0.25),
        (0.25, 0.75, 0.75),
        (0.75, 0.25, 0.75),
        (0.75, 0.75, 0.25),
    ],
    dtype=np.float64,
)

#: Type index of carbon in the 118-element OMat24 type map.
CARBON = 5


# === System construction ===


def diamond_cell(atoms: int, jitter: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Build the cubic diamond supercell whose atom count is nearest ``atoms``.

    Parameters
    ----------
    atoms
        Requested atom count; the returned cell holds ``8 * repeat ** 3``.
    jitter
        Standard deviation of an isotropic random displacement in Å, drawn
        from a fixed seed. The ideal lattice has zero force by symmetry, so an
        accuracy comparison needs one.

    Returns
    -------
    coord
        Cartesian coordinates with shape ``(N, 3)`` in Å.
    box
        Cell matrix with shape ``(3, 3)`` in Å.
    """
    repeat = max(1, round((atoms / len(_DIAMOND_BASIS)) ** (1.0 / 3.0)))
    grid = np.arange(repeat, dtype=np.float64)
    cells = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1).reshape(
        -1, 3
    )
    frac = (cells[:, None, :] + _DIAMOND_BASIS[None]).reshape(-1, 3) / repeat
    box = np.eye(3) * (repeat * DIAMOND_LATTICE)
    coord = frac @ box
    if jitter > 0.0:
        coord = coord + np.random.default_rng(20260821).normal(
            scale=jitter, size=coord.shape
        )
    return coord, box


# === Model construction ===


def load_model(ckpt: str, compress: bool = False) -> Any:
    """Load a pt_expt checkpoint into an eval-mode CPU model.

    Parameters
    ----------
    ckpt
        Path of a pt_expt training checkpoint.
    compress
        Whether to build the compressed-inference artifacts of the descriptor.

    Returns
    -------
    Any
        The eval-mode model on the CPU device.
    """
    from deepmd.pt_expt.model.get_model import (
        get_model,
    )
    from deepmd.pt_expt.train.wrapper import (
        ModelWrapper,
    )

    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = raw["model"]
    params = state["_extra_state"]["model_params"]
    if "Default" in params.get("model_dict", {}):
        params = params["model_dict"]["Default"]
    model = get_model(params)
    ModelWrapper(model).load_state_dict(state)
    model.eval()
    if compress:
        model.get_descriptor().enable_compression(0.0)
    return model


# === Graph lower inputs ===


@dataclass(frozen=True)
class LowerInputs:
    """Positional inputs of ``forward_common_lower_graph`` and their sizes."""

    args: tuple
    n_atom: int
    n_node: int
    n_edge: int

    @property
    def edges_per_atom(self) -> float:
        """Return the mean neighbor count of a local atom."""
        return self.n_edge / max(self.n_atom, 1)


def build_lower_inputs(
    model: Any,
    atoms: int,
    atom_type: int = CARBON,
    do_atomic_virial: bool = False,
    rcut: float | None = None,
    jitter: float = 0.0,
    edge_dtype: torch.dtype | None = None,
) -> LowerInputs:
    """Build the canonical graph-lower inputs of one diamond supercell.

    The graph is canonicalized so that the payload is destination-major and
    both CSR views are present, which is the layout a deployed artifact and
    the fused operators consume.

    Parameters
    ----------
    model
        Loaded pt_expt model, read for its cutoff.
    atoms
        Requested atom count of the supercell.
    atom_type
        Type index assigned to every atom.
    do_atomic_virial
        Whether the lower is asked for per-atom virials.
    rcut
        Graph construction cutoff in Å, defaulting to the model's own. A
        shorter cutoff builds the same node axis with fewer edges, which
        separates the per-node cost of a kernel from its per-edge cost.
    jitter
        Standard deviation of a random atomic displacement in Å.
    edge_dtype
        Element type of the edge vectors, defaulting to float64. A compressed
        artifact declares float32 geometry and halves the edge traffic.

    Returns
    -------
    LowerInputs
        Positional argument tuple and the resulting system size.
    """
    from deepmd.dpmodel.utils.neighbor_graph import (
        canonicalize_neighbor_graph,
    )
    from deepmd.pt_expt.utils.graph_builder import (
        build_neighbor_graph_for_method,
    )

    coord_np, box_np = diamond_cell(atoms, jitter)
    n_atom = coord_np.shape[0]
    coord = torch.tensor(coord_np, dtype=torch.float64).reshape(1, n_atom, 3)
    box = torch.tensor(box_np, dtype=torch.float64).reshape(1, 3, 3)
    atype = torch.full((1, n_atom), atom_type, dtype=torch.long)

    # The dense builder is all-pairs; a production-sized supercell needs the
    # cell list. Both emit the same neighbor set, so the choice is only a
    # construction cost and never changes what is measured.
    graph = build_neighbor_graph_for_method(
        "vesin",
        coord,
        atype,
        box,
        float(model.get_rcut()) if rcut is None else float(rcut),
        None,
        with_csr=True,
    )
    flat_atype = atype.reshape(-1)
    graph = canonicalize_neighbor_graph(graph, int(flat_atype.shape[0]))
    if edge_dtype is not None:
        graph = dataclasses.replace(graph, edge_vec=graph.edge_vec.to(edge_dtype))
    args = (
        flat_atype,
        graph.n_node,
        graph.n_node,
        graph.edge_index,
        graph.edge_vec,
        graph.edge_mask,
        graph.destination_order,
        graph.destination_row_ptr,
        graph.source_order,
        graph.source_row_ptr,
        bool(graph.destination_sorted),
        do_atomic_virial,
    )
    return LowerInputs(
        args=args,
        n_atom=n_atom,
        n_node=int(flat_atype.shape[0]),
        n_edge=int(graph.edge_mask.sum().item()),
    )


# === Measurement ===


def _resident_bytes() -> int:
    """Return the resident set size of this process in bytes."""
    with open("/proc/self/statm") as handle:
        fields = handle.read().split()
    return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")


def timed(
    run: Any,
    iters: int = 20,
    warmup: int = 5,
) -> dict[str, float]:
    """Time a callable and report its wall time and resident-set growth.

    Parameters
    ----------
    run
        Zero-argument callable returning the model output dictionary.
    iters
        Number of measured iterations.
    warmup
        Number of discarded iterations preceding the measurement.

    Returns
    -------
    dict
        Mean and best iteration time in milliseconds, resident-set growth in
        GiB, and reference outputs used for parity checks.
    """
    out = None
    for _ in range(warmup):
        out = run()
    gc.collect()
    base = _resident_bytes()

    samples = []
    peak = base
    for _ in range(iters):
        start = time.perf_counter()
        out = run()
        samples.append((time.perf_counter() - start) * 1e3)
        peak = max(peak, _resident_bytes())
    # The eager lower returns the internal output-definition keys while a
    # frozen package returns the public model keys; both name the same two
    # tensors.
    force_key = next(k for k in out if k in ("force", "energy_derv_r"))
    energy_key = next(k for k in out if k in ("energy", "energy_redu"))
    return {
        "ms": float(np.mean(samples)),
        "ms_best": float(np.min(samples)),
        "ms_std": float(np.std(samples)),
        "rss_gb": peak / 2**30,
        "rss_incr_gb": (peak - base) / 2**30,
        "energy": float(out[energy_key].double().sum().item()),
        "force_absmax": float(out[force_key].double().abs().max().item()),
    }


# === Machine description ===


def cpu_topology() -> dict[str, Any]:
    """Return the online-core topology this machine exposes.

    Returns
    -------
    dict
        Physical and logical core counts, per-NUMA-node online CPU lists, and
        the CPU model string.
    """
    online = _parse_cpu_list(_read("/sys/devices/system/cpu/online"))
    cores: dict[tuple[int, int], list[int]] = {}
    node_of: dict[int, int] = {}
    for cpu in online:
        base = f"/sys/devices/system/cpu/cpu{cpu}"
        package = int(_read(f"{base}/topology/physical_package_id"))
        core = int(_read(f"{base}/topology/core_id"))
        cores.setdefault((package, core), []).append(cpu)
        node_of[cpu] = _node_of_cpu(cpu)
    nodes: dict[int, list[int]] = {}
    for cpu, node in node_of.items():
        nodes.setdefault(node, []).append(cpu)
    return {
        "model": _cpu_model(),
        "logical": len(online),
        "physical": len(cores),
        "smt": max(len(v) for v in cores.values()) if cores else 1,
        "nodes": {node: sorted(cpus) for node, cpus in sorted(nodes.items())},
        "first_siblings": sorted(min(v) for v in cores.values()),
    }


def _read(path: str) -> str:
    with open(path) as handle:
        return handle.read().strip()


def _parse_cpu_list(text: str) -> list[int]:
    values: list[int] = []
    for chunk in text.split(","):
        if "-" in chunk:
            low, high = chunk.split("-")
            values.extend(range(int(low), int(high) + 1))
        elif chunk:
            values.append(int(chunk))
    return values


def _node_of_cpu(cpu: int) -> int:
    base = f"/sys/devices/system/cpu/cpu{cpu}"
    for entry in os.listdir(base):
        if entry.startswith("node"):
            return int(entry[4:])
    return 0


def _cpu_model() -> str:
    with open("/proc/cpuinfo") as handle:
        for line in handle:
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def isa_support() -> dict[str, bool]:
    """Return the vector extensions advertised by the running CPU."""
    with open("/proc/cpuinfo") as handle:
        flags = set(re.split(r"\s+", handle.read().split("flags")[1].split("\n")[0]))
    return {
        name: name in flags
        for name in (
            "avx2",
            "fma",
            "avx512f",
            "avx512dq",
            "avx512bw",
            "avx512vl",
            "avx512_vnni",
            "avx512_bf16",
            "amx_tile",
            "amx_bf16",
        )
    }


def env_summary() -> str:
    """Return a one-line summary of the environment that selects a path."""
    keys = ("OMP_PROC_BIND", "OMP_PLACES", "DP_NODE_TILE")
    return " ".join(
        [f"threads={THREADS}", *(f"{k}={os.environ.get(k, '-')}" for k in keys)]
    )

# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, ANN202
"""Shared harness for SeZM / DPA4 inference benchmarks.

The harness builds one diamond supercell, converts it to the ``edge_vec`` lower
ABI once, and then repeatedly calls ``forward_common_lower``, so a measurement
covers only the model graph (descriptor, fitting and the force backward) and not
the neighbor-list construction. ``bench_deeppot`` measures the full ASE-style
path instead, where the neighbor list is rebuilt on every call.
"""

from __future__ import (
    annotations,
)

import os
import time
from typing import (
    Any,
)

import numpy as np
import torch

DIAMOND_LATTICE = 3.567
_BASIS = np.array(
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

CKPT_MINI = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models/checkpoints/dpa4-mini.pt"
)


def diamond_cell(atoms: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the cubic diamond supercell whose atom count is nearest ``atoms``.

    Parameters
    ----------
    atoms : int
        Requested atom count; the returned cell holds ``8 * repeat ** 3`` atoms.

    Returns
    -------
    tuple of numpy.ndarray
        Cartesian coordinates with shape (N, 3) in Angstrom and the cell matrix
        with shape (3, 3) in Angstrom.
    """
    repeat = max(1, round((atoms / len(_BASIS)) ** (1.0 / 3.0)))
    grid = np.arange(repeat, dtype=np.float64)
    cells = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1).reshape(
        -1, 3
    )
    frac = (cells[:, None, :] + _BASIS[None]).reshape(-1, 3) / repeat
    box = np.eye(3) * (repeat * DIAMOND_LATTICE)
    return frac @ box, box


def load_model(ckpt: str = CKPT_MINI, device: str = "cuda") -> tuple[Any, dict]:
    """
    Load a SeZM checkpoint into an eval-mode module on ``device``.

    Parameters
    ----------
    ckpt : str
        Path of the PyTorch checkpoint.
    device : str
        Target device string.

    Returns
    -------
    tuple
        The eval-mode model and its stored model parameter dictionary.
    """
    from deepmd.pt.model.model import (
        get_model,
    )
    from deepmd.pt.train.wrapper import (
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
    return model.to(device), params


def build_lower_inputs(
    model: Any, atoms: int, atom_type: int = 0, device: str = "cuda"
) -> tuple[tuple, dict]:
    """
    Build the ``forward_common_lower`` edge-schema inputs for a diamond cell.

    Parameters
    ----------
    model : Any
        The loaded SeZM model, used for its cutoff and selection settings.
    atoms : int
        Requested atom count of the supercell.
    atom_type : int
        Type index assigned to every atom.
    device : str
        Target device string.

    Returns
    -------
    tuple
        The positional argument tuple for ``forward_common_lower`` and a
        dictionary describing the resulting system size.
    """
    from deepmd.pt.utils.nlist import (
        build_neighbor_list,
        extend_coord_with_ghosts,
    )
    from deepmd.pt.utils.region import (
        normalize_coord,
    )
    from deepmd.pt_expt.utils.edge_schema import (
        edge_schema_from_extended,
    )

    coord_np, box_np = diamond_cell(atoms)
    nloc = coord_np.shape[0]
    coord = torch.tensor(coord_np, dtype=torch.float64, device=device).reshape(
        1, nloc, 3
    )
    box = torch.tensor(box_np, dtype=torch.float64, device=device).reshape(1, 3, 3)
    atype = torch.full((1, nloc), atom_type, dtype=torch.long, device=device)

    rcut = float(model.get_rcut())
    sel = list(model.get_sel())
    coord = normalize_coord(coord, box)
    ext_coord, ext_atype, mapping = extend_coord_with_ghosts(
        coord, atype, box.reshape(1, 9), rcut
    )
    nlist = build_neighbor_list(
        ext_coord, ext_atype, nloc, rcut, sel, distinguish_types=not model.mixed_types()
    )
    ext_coord = ext_coord.reshape(1, -1, 3)
    nlist = model.format_nlist(ext_coord, ext_atype, nlist)
    schema = edge_schema_from_extended(ext_coord, atype, nlist, mapping)
    return (
        schema.coord,
        schema.atype,
        schema.edge_index,
        schema.edge_vec,
        schema.edge_scatter_index,
        schema.edge_mask,
    ), {
        "nloc": nloc,
        "nall": int(ext_coord.shape[1]),
        "n_edge": int(schema.edge_mask.sum().item()),
        "n_edge_pad": int(schema.edge_vec.shape[0]),
    }


def timed(
    model: Any, inputs: tuple, iters: int = 20, warmup: int = 5
) -> dict[str, float]:
    """
    Time ``forward_common_lower`` and report peak allocated memory.

    Parameters
    ----------
    model : Any
        The loaded SeZM model.
    inputs : tuple
        Positional arguments produced by :func:`build_lower_inputs`.
    iters : int
        Number of measured iterations.
    warmup : int
        Number of discarded iterations preceding the measurement.

    Returns
    -------
    dict
        Timing in milliseconds, peak memory in GiB, and reference outputs.
    """

    def run():
        return model.forward_common_lower(*inputs)

    out = None
    for _ in range(warmup):
        out = run()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    start.record()
    for _ in range(iters):
        out = run()
    stop.record()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1e3 / iters
    peak = torch.cuda.max_memory_allocated()
    force_key = next(k for k in out if "force" in k or "derv_r" in k)
    return {
        "ms": start.elapsed_time(stop) / iters,
        "wall_ms": wall,
        "peak_gb": peak / 2**30,
        "incr_gb": (peak - base) / 2**30,
        "energy": float(out["energy"].sum().item()),
        "force_absmax": float(out[force_key].abs().max().item()),
    }


def env_summary() -> str:
    """Return a one-line summary of the inference-path environment gates."""
    keys = (
        "DP_COMPILE_INFER",
        "DP_TRITON_INFER",
        "DP_CUDA_INFER",
        "DP_TF32_INFER",
        "DP_CUTILE_INFER",
    )
    return " ".join(f"{k}={os.environ.get(k, '-')}" for k in keys)

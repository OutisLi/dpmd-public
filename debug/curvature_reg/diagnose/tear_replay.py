# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201, TID253
"""Replay a captured training-frame tear on saved checkpoints.

The captured batch stores the live model's forces at the step of the tear.
Replaying the same frame on the raw and averaged checkpoints around that step
separates a persistent defect of the trained function from a transient of the
live weights, and the parameter sensitivity of the force on the torn atoms
measures how close each checkpoint's function is to the same defect.
"""

from __future__ import (
    annotations,
)

import argparse
import json
from pathlib import (
    Path,
)

import numpy as np
import torch
from ase.geometry import (
    find_mic,
)
from curvature_split import (
    build_model,
)

ISOLATED_BOX_A = 40.0


def load_frame(batch: Path) -> dict[str, np.ndarray]:
    """Return the frame of the batch holding the largest per-atom force error."""
    with np.load(batch) as data:
        error = data["per_atom_error"].reshape(data["atype"].shape)
        frame = int(np.unravel_index(np.argmax(error), error.shape)[0])
        real = data["atype"][frame] >= 0
        return {
            "frame": frame,
            "coord": data["coord"][frame, real].copy(),
            "atype": data["atype"][frame, real].copy(),
            "box": data["box"][frame].reshape(3, 3).copy(),
            "label": data["label_force"][frame, real].copy(),
            "recorded": data["model_force"][frame, real].copy(),
            "error": error[frame, real].copy(),
        }


def evaluate(
    model: torch.nn.Module,
    coord: np.ndarray,
    atype: np.ndarray,
    box: np.ndarray,
    create_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate energy and forces of one periodic frame in float64."""
    coordinates = torch.tensor(
        coord[None], dtype=torch.float64, requires_grad=create_graph
    )
    types = torch.tensor(atype[None], dtype=torch.long)
    cells = torch.tensor(box.reshape(1, 9), dtype=torch.float64)
    output = model(coordinates, types, box=cells, do_atomic_virial=False)
    return output["energy"], output["force"][0], coordinates


def radial_force(
    force: np.ndarray, coord: np.ndarray, box: np.ndarray, i: int, j: int
) -> float:
    """Return the force component along the minimum-image vector from i to j (eV/A)."""
    vector, _ = find_mic((coord[j] - coord[i])[None], box, pbc=True)
    unit = vector[0] / np.linalg.norm(vector[0])
    return float((force[j] - force[i]) @ unit / 2.0)


def parameter_sensitivity(
    model: torch.nn.Module,
    coord: np.ndarray,
    atype: np.ndarray,
    box: np.ndarray,
    atoms: list[int],
) -> dict[str, float]:
    """Return the Frobenius norm of d(force on atom)/d(parameters) for each atom.

    The SeZM model in evaluation mode differentiates the energy along the
    edges without retaining a graph for the force, so the edge derivative is
    asked for one while the forces are computed; the evaluation-mode kernels
    are otherwise unchanged.
    """
    import importlib

    sezm_model = importlib.import_module("deepmd.pt.model.model.sezm_model")
    parameters = [p for p in model.parameters() if p.requires_grad]
    coordinates = torch.tensor(coord[None], dtype=torch.float64, requires_grad=True)
    types = torch.tensor(atype[None], dtype=torch.long)
    cells = torch.tensor(box.reshape(1, 9), dtype=torch.float64)
    native = sezm_model.edge_energy_deriv

    def with_graph(*call_args, **call_kwargs):  # noqa: ANN002, ANN003, ANN202
        return native(*call_args, **{**call_kwargs, "create_graph": True})

    sezm_model.edge_energy_deriv = with_graph
    try:
        output = model(coordinates, types, box=cells, do_atomic_virial=False)
    finally:
        sezm_model.edge_energy_deriv = native
    force = output["force"][0]
    if not force.requires_grad:
        raise RuntimeError(
            "Force has no graph; the create_graph override did not take effect"
        )
    result = {}
    for atom in atoms:
        total = 0.0
        for component in range(3):
            grads = torch.autograd.grad(
                force[atom, component], parameters, retain_graph=True, allow_unused=True
            )
            total += sum(float(g.square().sum()) for g in grads if g is not None)
        result[str(atom)] = float(np.sqrt(total))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--atoms",
        type=int,
        nargs="*",
        default=[],
        help="atoms whose errors, isolated-pair forces and sensitivities are reported",
    )
    parser.add_argument(
        "--pair",
        type=int,
        nargs=2,
        default=None,
        help="the compressed pair evaluated in the frame and alone",
    )
    parser.add_argument("--sensitivity", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    frame = load_frame(args.batch)
    coord, atype, box, label = (
        frame["coord"],
        frame["atype"],
        frame["box"],
        frame["label"],
    )
    n_atoms = len(atype)
    median_atom = int(np.argsort(frame["error"])[n_atoms // 2])
    report_atoms = sorted(set(args.atoms) | {median_atom})
    record = {
        "batch": str(args.batch.resolve()),
        "frame": frame["frame"],
        "n_atoms": n_atoms,
        "recorded_max_error_eV_A": float(frame["error"].max()),
        "median_atom": median_atom,
        "checkpoints": [],
    }
    if args.pair is not None:
        i, j = args.pair
        vector, distance = find_mic((coord[j] - coord[i])[None], box, pbc=True)
        record["pair"] = {
            "atoms": [i, j],
            "distance_A": float(distance[0]),
            "label_radial_force_eV_A": radial_force(label, coord, box, i, j),
            "recorded_radial_force_eV_A": radial_force(
                frame["recorded"], coord, box, i, j
            ),
        }
    for checkpoint in args.checkpoints:
        model, type_map = build_model(checkpoint, "float64")
        energy, force, _ = evaluate(model, coord, atype, box)
        force = force.detach().numpy()
        error = np.linalg.norm(force - label, axis=-1)
        row = {
            "checkpoint": str(checkpoint.resolve()),
            "energy_eV": float(energy.sum()),
            "max_error_eV_A": float(error.max()),
            "argmax_atom": int(error.argmax()),
            "atoms_above_100": int((error > 100).sum()),
            "atoms_above_1000": int((error > 1000).sum()),
            "atom_errors_eV_A": {str(a): float(error[a]) for a in report_atoms},
        }
        if args.pair is not None:
            i, j = args.pair
            row["pair_radial_force_in_frame_eV_A"] = radial_force(
                force, coord, box, i, j
            )
            # The same pair alone in a large empty cell, at the frame's distance.
            pair_coord = np.zeros((2, 3))
            pair_coord[1, 0] = record["pair"]["distance_A"]
            pair_coord += ISOLATED_BOX_A / 2
            pair_box = np.eye(3) * ISOLATED_BOX_A
            pair_energy, pair_force, _ = evaluate(
                model, pair_coord, atype[[i, j]], pair_box
            )
            pair_force = pair_force.detach().numpy()
            row["pair_alone_energy_eV"] = float(pair_energy.sum())
            row["pair_alone_radial_force_eV_A"] = radial_force(
                pair_force, pair_coord, pair_box, 0, 1
            )
        if args.sensitivity:
            row["force_parameter_sensitivity"] = parameter_sensitivity(
                model, coord, atype, box, report_atoms
            )
        print(json.dumps(row), flush=True)
        record["checkpoints"].append(row)
    args.out.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

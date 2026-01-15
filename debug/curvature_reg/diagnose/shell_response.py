# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201, TID253
"""Energy per neighbour of a central atom surrounded by symmetric shells.

A central atom receives ``k`` neighbours of one element on a sphere of radius
``r`` (one neighbour, a pair on opposite sides, a triangle, a tetrahedron, an
octahedron). The binding energy per neighbour,

    [E(centre + k neighbours) - E(centre alone) - k E(neighbour alone)] / k,

compares the single-neighbour limit that the isolated-pair survey measures
with the same neighbour distance inside a symmetric environment. A feature
present at k = 1 only is a property of the sparse limit; a feature present at
every k is a per-neighbour rule the model applies in any environment. The
neighbours of a shell also interact with one another; their mutual distances
are reported so that readings inside the radial support are recognized.
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
from curvature_split import (
    build_model,
)

BOX_A = 40.0
SHELLS: dict[int, np.ndarray] = {
    1: np.array([[1.0, 0.0, 0.0]]),
    2: np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
    3: np.array(
        [[1.0, 0.0, 0.0], [-0.5, np.sqrt(3) / 2, 0.0], [-0.5, -np.sqrt(3) / 2, 0.0]]
    ),
    4: np.array(
        [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
    )
    / np.sqrt(3),
    6: np.array(
        [
            [1.0, 0, 0],
            [-1.0, 0, 0],
            [0, 1.0, 0],
            [0, -1.0, 0],
            [0, 0, 1.0],
            [0, 0, -1.0],
        ]
    ),
}


def energy(model: torch.nn.Module, coord: np.ndarray, atype: np.ndarray) -> float:
    """Total energy (eV) of a finite cluster in an empty cubic cell."""
    coordinates = torch.tensor(coord[None] + BOX_A / 2, dtype=torch.float64)
    types = torch.tensor(atype[None], dtype=torch.long)
    cells = torch.tensor((np.eye(3) * BOX_A).reshape(1, 9), dtype=torch.float64)
    # The model differentiates its energy internally for the forces, so the
    # evaluation must run with autograd enabled.
    return float(model(coordinates, types, box=cells)["energy"].sum().detach())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--centre", required=True, help="element symbol of the central atom"
    )
    parser.add_argument(
        "--neighbour", required=True, help="element symbol of the shell atoms"
    )
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=[1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0],
    )
    parser.add_argument("--shells", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    model, type_map = build_model(args.checkpoint, "float64")
    centre, neighbour = type_map.index(args.centre), type_map.index(args.neighbour)
    e_centre = energy(model, np.zeros((1, 3)), np.array([centre]))
    e_neighbour = energy(model, np.zeros((1, 3)), np.array([neighbour]))
    table = {}
    for k in args.shells:
        directions = SHELLS[k]
        mutual = None
        if k > 1:
            separations = [
                np.linalg.norm(directions[i] - directions[j])
                for i in range(k)
                for j in range(i + 1, k)
            ]
            mutual = float(min(separations))
        rows = []
        for r in args.radii:
            coord = np.vstack([np.zeros((1, 3)), r * directions])
            atype = np.array([centre] + [neighbour] * k)
            e_total = energy(model, coord, atype)
            rows.append(
                {
                    "r_A": r,
                    "per_neighbour_eV": (e_total - e_centre - k * e_neighbour) / k,
                    "neighbour_spacing_A": None if mutual is None else mutual * r,
                }
            )
        table[str(k)] = rows
        print(
            f"k={k}: "
            + "  ".join(
                f"{row['r_A']:.2f}:{row['per_neighbour_eV']:+.3f}" for row in rows
            ),
            flush=True,
        )
    args.out.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "centre": args.centre,
                "neighbour": args.neighbour,
                "isolated_atom_energies_eV": [e_centre, e_neighbour],
                "shells": table,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

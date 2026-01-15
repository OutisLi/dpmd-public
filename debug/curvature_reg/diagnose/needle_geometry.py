# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201
"""Classify captured force-error atoms using exact periodic neighbour distances."""

from __future__ import (
    annotations,
)

import argparse
import json
import sys
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import numpy as np
from ase import (
    Atoms,
)
from ase.data import (
    atomic_numbers,
    covalent_radii,
)
from ase.neighborlist import (
    neighbor_list,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from needle_window import (
    entries,
)


def frame_geometry(
    coord: np.ndarray,
    atype: np.ndarray,
    error: np.ndarray,
    predicted: np.ndarray,
    label: np.ndarray,
    box: np.ndarray | None,
    type_map: list[str],
    threshold: float,
    cutoff: float,
) -> list[dict[str, Any]]:
    """Read close contacts for every real atom whose force error exceeds the threshold.

    Coordinates and the cell are in A; force magnitudes are in eV/A.
    Periodic images of the central atom are included. The nearest neighbour
    by absolute distance and the most compressed covalent contact need not
    be the same atom. A missing neighbour means all contacts exceed cutoff.
    """
    indices = np.flatnonzero(atype >= 0)
    numbers = np.asarray([atomic_numbers[type_map[int(atype[i])]] for i in indices])
    atoms = Atoms(
        numbers=numbers, positions=coord[indices], cell=box, pbc=box is not None
    )
    src, dst, distance = neighbor_list("ijd", atoms, cutoff, self_interaction=False)
    ratios = distance / (covalent_radii[numbers[src]] + covalent_radii[numbers[dst]])
    records = []
    for local in np.flatnonzero(error[indices] > threshold):
        atom = int(indices[local])
        edges = np.flatnonzero(src == local)
        nearest = int(edges[np.argmin(distance[edges])]) if len(edges) else None
        compressed = int(edges[np.argmin(ratios[edges])]) if len(edges) else None
        separation = None if nearest is None else float(distance[nearest])
        compression = None if compressed is None else float(ratios[compressed])
        records.append(
            {
                "atom": atom,
                "type": type_map[int(atype[atom])],
                "force_error_eV_A": float(error[atom]),
                "predicted_force_eV_A": float(np.linalg.norm(predicted[atom])),
                "label_force_eV_A": float(np.linalg.norm(label[atom])),
                "nearest_distance_A": separation,
                "nearest_type": None
                if nearest is None
                else type_map[int(atype[indices[dst[nearest]]])],
                "minimum_contact_ratio": compression,
                "most_compressed_type": None
                if compressed is None
                else type_map[int(atype[indices[dst[compressed]]])],
                "outside_1A": separation is None or separation >= 1.0,
                "outside_06_contact": compression is None or compression >= 0.6,
            }
        )
    return records


def audit_run(run: Path, threshold: float) -> dict[str, Any]:
    """Audit available snapshots and report which logged events lack a unique snapshot."""
    config = json.loads((run / "input.json").read_text())
    type_map = config["model"]["type_map"]
    cutoff = float(config["model"]["descriptor"]["rcut"])
    observed = [
        (step, error) for step, error, _ in entries(run.name) if error > threshold
    ]
    snapshots = []
    matched = set()
    for path in sorted(run.glob("needle_rank0_step*.npz")):
        with np.load(path) as data:
            types = data["atype"]
            errors = data["per_atom_error"].reshape(types.shape)
            peak = float(errors.max())
            if peak <= threshold:
                continue
            candidates = [
                i
                for i, (_, error) in enumerate(observed)
                if abs(error - peak) <= 0.0006
            ]
            if len(candidates) == 1:
                matched.add(candidates[0])
            coordinates = data["coord"].reshape(*types.shape, 3)
            forces = data["model_force"].reshape(*types.shape, 3)
            labels = data["label_force"].reshape(*types.shape, 3)
            boxes = data["box"].reshape(-1, 3, 3) if "box" in data else None
            records = []
            for frame in np.flatnonzero((errors > threshold).any(axis=1)):
                cell = None if boxes is None else boxes[frame]
                for item in frame_geometry(
                    coordinates[frame],
                    types[frame],
                    errors[frame],
                    forces[frame],
                    labels[frame],
                    cell,
                    type_map,
                    threshold,
                    cutoff,
                ):
                    records.append({"frame": int(frame), **item})
            snapshots.append(
                {
                    "file": str(path.resolve()),
                    "recorded_step": int(data["step"]),
                    "possible_optimizer_steps": [observed[i][0] for i in candidates],
                    "peak_error_eV_A": peak,
                    "bad_atoms": records,
                    "outside_1A": sum(item["outside_1A"] for item in records),
                    "outside_06_contact": sum(
                        item["outside_06_contact"] for item in records
                    ),
                    "outside_both": sum(
                        item["outside_1A"] and item["outside_06_contact"]
                        for item in records
                    ),
                }
            )
    return {
        "run": run.name,
        "threshold_eV_A": threshold,
        "logged_events": len(observed),
        "uniquely_matched_events": len(matched),
        "events_without_unique_snapshot": [
            observed[i] for i in range(len(observed)) if i not in matched
        ],
        "snapshots": snapshots,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--threshold", type=float, default=1000.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    results = []
    for run in args.runs:
        result = audit_run(run, args.threshold)
        results.append(result)
        print(
            run.name,
            "logged",
            result["logged_events"],
            "matched",
            result["uniquely_matched_events"],
        )
        for snapshot in result["snapshots"]:
            worst = max(
                snapshot["bad_atoms"], key=lambda item: item["force_error_eV_A"]
            )
            print(
                snapshot["possible_optimizer_steps"],
                f"peak={snapshot['peak_error_eV_A']:.3f}",
                "worst",
                worst["type"],
                worst["nearest_type"],
                worst["nearest_distance_A"],
                "min_ratio",
                worst["minimum_contact_ratio"],
                "outside_abs/relative/both",
                snapshot["outside_1A"],
                snapshot["outside_06_contact"],
                snapshot["outside_both"],
                flush=True,
            )
    args.out.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

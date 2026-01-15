# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201, TID253
"""Separate Gaussian-basis force derivatives and inspect centre projection.

Detaching the bare radial features preserves forward energies and removes their
coordinate-derivative path. Projecting trained centres into their initial
interval changes the function while retaining all other weights. Both operations
are diagnostics; neither represents a retrained or accepted potential.
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
from ase.data import (
    atomic_numbers,
    covalent_radii,
)
from curvature_split import (
    build_model,
)
from dimer_survey import (
    curve_summary,
)


def scan(
    model: torch.nn.Module,
    type_map: list[str],
    pair: str,
    radius: np.ndarray,
    device: torch.device,
    detach: bool,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return energies and radial forces for two-atom frames in vacuum."""
    radial = model.atomic_model.descriptor.radial_basis
    calls = 0

    def radial_output(
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return output.detach() if detach else output

    handle = radial.register_forward_hook(radial_output)
    energies, forces = [], []
    atom_types = torch.tensor(
        [type_map.index(x) for x in pair.split("-")], device=device
    )
    try:
        for start in range(0, len(radius), batch_size):
            distances = torch.as_tensor(
                radius[start : start + batch_size], dtype=torch.float64, device=device
            )
            coord = torch.zeros(
                (len(distances), 2, 3), dtype=torch.float64, device=device
            )
            coord[:, 1, 0] = distances
            types = atom_types.unsqueeze(0).expand(len(distances), -1)
            box = (
                (20 * torch.eye(3, dtype=torch.float64, device=device))
                .reshape(1, 9)
                .repeat(len(distances), 1)
            )
            prediction = model(coord, types, box=box)
            energies.append(prediction["energy"].detach().reshape(-1).cpu().numpy())
            forces.append(prediction["force"].detach()[:, 1, 0].cpu().numpy())
    finally:
        handle.remove()
    if calls == 0:
        raise RuntimeError("The radial-basis intervention was not reached")
    return np.concatenate(energies), np.concatenate(forces), calls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--centre-limit", type=float, default=4.0)
    parser.add_argument("--pairs", default="H-H,O-O,Cu-Cu,Fe-O")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--spacing", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.spacing <= 0 or args.batch_size <= 0 or args.centre_limit <= 0:
        parser.error("spacing, batch size, and centre limit must be positive")
    if args.out.exists() or args.out.with_suffix(".json").exists():
        raise FileExistsError(args.out)
    device = torch.device(args.device)
    model, type_map = build_model(args.checkpoint, "float64")
    model = model.to(device)
    descriptor = model.atomic_model.descriptor
    radial = descriptor.radial_basis
    if radial.basis_type != "gaussian" or not radial._experiment_single_envelope:
        raise ValueError(
            "The diagnostic requires a Gaussian model with a bare radial basis"
        )
    assert descriptor._cuda_radial_fn is None
    original = radial.adam_freqs.detach().clone()
    projected = original.clamp(0.0, args.centre_limit)
    curves = {}
    summaries = []
    for pair in args.pairs.split(","):
        left, right = pair.split("-")
        contact = float(
            covalent_radii[atomic_numbers[left]] + covalent_radii[atomic_numbers[right]]
        )
        radius = np.unique(
            np.round(
                np.concatenate(
                    (
                        np.arange(0.25 * contact, 7.0, args.spacing),
                        [0.6 * contact, 1.0, 4.0, radial.rcut, 7.0],
                    )
                ),
                12,
            )
        )
        prefix = pair.replace("-", "_")
        curves[f"{prefix}_r"] = radius
        tail = (radius >= 4) & (radius <= 7)
        for label, centres in (("trained", original), ("projected", projected)):
            with torch.no_grad():
                radial.adam_freqs.copy_(centres)
            energy, force, calls = scan(
                model, type_map, pair, radius, device, False, args.batch_size
            )
            detached_energy, remaining, detached_calls = scan(
                model, type_map, pair, radius, device, True, args.batch_size
            )
            np.testing.assert_array_equal(energy, detached_energy)
            assert calls == detached_calls
            radial_force = force - remaining
            for name, values in (
                ("energy", energy - energy[-1]),
                ("force", force),
                ("force_detach_rbf", remaining),
                ("force_radial_path", radial_force),
            ):
                assert np.isfinite(values).all(), (pair, label, name)
                curves[f"{prefix}_{label}_{name}"] = values
            summary = curve_summary(pair, contact, radius, energy, force)
            peak = int(np.flatnonzero(tail)[np.argmax(np.abs(force[tail]))])
            row = {
                "pair": pair,
                "centres": label,
                "energy_bit_identical_under_detach": True,
                "radial_calls": calls,
                "tail_max_force": summary["tail_max_force"],
                "tail_max_detach_rbf": float(np.abs(remaining[tail]).max()),
                "tail_max_radial_path": float(np.abs(radial_force[tail]).max()),
                "peak_r": float(radius[peak]),
                "peak_force": float(force[peak]),
                "peak_radial_path": float(radial_force[peak]),
                "peak_remaining": float(remaining[peak]),
                "principal_energy": summary["principal_energy"],
                "principal_r": summary["principal_r"],
                "extra_extrema": summary["additional_extrema_outside_06contact"],
            }
            summaries.append(row)
            print(json.dumps(row), flush=True)
    with torch.no_grad():
        radial.adam_freqs.copy_(original)
    with args.out.open("xb") as stream:
        np.savez_compressed(stream, **curves)
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "precision": "float64",
                "device": str(device),
                "maximum_spacing_A": args.spacing,
                "diagnostic_only": True,
                "original_centres_A": original.cpu().reshape(-1).tolist(),
                "projected_centres_A": projected.cpu().reshape(-1).tolist(),
                "gaussian_coeff": float(radial.gaussian_coeff),
                "pairs": summaries,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

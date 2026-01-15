# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201, TID253
"""Split a captured frame's force into fitting-input amplitude and direction paths."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(".npz").exists():
        raise FileExistsError(args.out)
    with np.load(args.batch) as batch:
        errors = batch["per_atom_error"].reshape(batch["atype"].shape)
        frame = int(np.unravel_index(np.argmax(errors), errors.shape)[0])
        real = batch["atype"][frame] >= 0
        coord = batch["coord"].reshape(*errors.shape, 3)[frame, real].copy()
        atype = batch["atype"][frame, real].copy()
        box = batch["box"][frame].reshape(1, 9).copy()
        label = batch["label_force"].reshape(*errors.shape, 3)[frame, real].copy()
    model, _ = build_model(args.checkpoint, "float64")
    coordinates = torch.tensor(coord[None], dtype=torch.float64)
    types = torch.tensor(atype[None], dtype=torch.long)
    cells = torch.tensor(box, dtype=torch.float64)
    fitting = model.atomic_model.fitting_net
    energies = {}
    forces = {}
    rows = []
    amplitude_range = None
    for mode in ("base", "amplitude", "direction", "rmsnorm"):
        observed = []

        def project(
            module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
        ) -> tuple[torch.Tensor, ...]:
            x = inputs[0]
            mean_square = x.square().mean(dim=-1, keepdim=True)
            magnitude = mean_square.sqrt()
            if not bool((magnitude > 0).all()):
                raise ValueError(
                    "Amplitude/direction decomposition requires nonzero fitting inputs"
                )
            observed.append(magnitude.detach())
            direction = x / magnitude
            if mode == "amplitude":
                x = x.detach() + direction.detach() * (magnitude - magnitude.detach())
            elif mode == "direction":
                x = x + direction.detach() * (magnitude.detach() - magnitude)
            elif mode == "rmsnorm":
                x = x / (mean_square + 1e-5).sqrt()
            return (x, *inputs[1:])

        hook = fitting.register_forward_pre_hook(project)
        try:
            output = model(coordinates.clone(), types, box=cells)
        finally:
            hook.remove()
        if len(observed) != 1:
            raise RuntimeError(
                f"Expected one fitting call, observed {len(observed)} in {mode}"
            )
        if amplitude_range is None:
            amplitude_range = [float(observed[0].min()), float(observed[0].max())]
        energies[mode] = output["energy"].detach()
        forces[mode] = output["force"].detach()[0]
        if mode in ("amplitude", "direction"):
            torch.testing.assert_close(
                energies[mode], energies["base"], atol=0.0, rtol=0.0
            )
        force = forces[mode].numpy()
        row = {
            "mode": mode,
            "energy_eV": float(energies[mode].sum()),
            "maximum_force_eV_A": float(np.linalg.norm(force, axis=-1).max()),
            "maximum_force_error_eV_A": float(
                np.linalg.norm(force - label, axis=-1).max()
            ),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    torch.testing.assert_close(
        forces["amplitude"] + forces["direction"], forces["base"], atol=1e-8, rtol=1e-9
    )
    record = {
        "checkpoint": str(args.checkpoint.resolve()),
        "batch": str(args.batch.resolve()),
        "frame": frame,
        "precision": "float64",
        "fitting_input_rms_range": amplitude_range,
        "force_decomposition_verified": True,
        "interventions": rows,
    }
    args.out.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(
        args.out.with_suffix(".npz"),
        coord=coord,
        atype=atype,
        box=box,
        label=label,
        **{key: value.numpy() for key, value in forces.items()},
    )


if __name__ == "__main__":
    main()

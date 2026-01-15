# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201, TID253
"""Measure the environment seed's coordinate path with actual and zero radial inputs."""

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
from radial_response import (
    scan,
)


def detach_output(
    module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
) -> torch.Tensor:
    """Remove the seed output's derivative while retaining its exact value."""
    return output.detach()


def zero_output(
    module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
) -> torch.Tensor:
    """Replace the raw radial values by zero for a separate counterfactual."""
    return torch.zeros_like(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", nargs="+", default=["H-H", "Fe-O"])
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(".json").exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model, type_map = build_model(args.checkpoint, "float64")
    model = model.to("cuda")
    descriptor = model.atomic_model.descriptor
    assert descriptor.use_env_seed and descriptor._cuda_radial_fn is None
    assert not descriptor.radial_embedding.radial_norm
    centers = descriptor.radial_basis.adam_freqs.detach().cpu().numpy()
    rows, curves = [], {}
    baseline = {}
    for mode in ("base", "detach_seed", "zero_basis", "zero_basis_detach_seed"):
        hooks = []
        if "zero_basis" in mode:
            hooks.append(descriptor.radial_basis.register_forward_hook(zero_output))
        if "detach_seed" in mode:
            hooks.append(
                descriptor.env_seed_embedding.register_forward_hook(detach_output)
            )
        for pair in args.pairs:
            contact = float(
                sum(covalent_radii[atomic_numbers[t]] for t in pair.split("-"))
            )
            radii = np.unique(
                np.round(
                    np.concatenate(
                        (
                            np.arange(0.25 * contact, 7, 0.005),
                            [0.6 * contact, 1, 4, 6, 7],
                        )
                    ),
                    12,
                )
            )
            energy, force, _ = scan(
                model, type_map, pair, radii, torch.device("cuda"), False, 128
            )
            family = "zero_basis" if "zero_basis" in mode else "base"
            if "detach_seed" in mode:
                np.testing.assert_array_equal(energy, baseline[(family, pair)])
            else:
                baseline[(family, pair)] = energy.copy()
            summary = curve_summary(pair, contact, radii, energy, force)
            rows.append({"mode": mode, **summary})
            prefix = f"{mode}_{pair.replace('-', '_')}"
            curves.update(
                {
                    f"{prefix}_r": radii,
                    f"{prefix}_energy": energy - energy[-1],
                    f"{prefix}_force": force,
                }
            )
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "pair": pair,
                        "tail": summary["tail_max_force"],
                        "tail_peak_r": summary["tail_peak_r"],
                        "well": summary["principal_energy"],
                    }
                ),
                flush=True,
            )
        for hook in hooks:
            hook.remove()
    with args.out.open("xb") as stream:
        np.savez_compressed(stream, **curves)
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "precision": "float64",
                "diagnostic_only": True,
                "center_min_A": float(centers.min()),
                "center_max_A": float(centers.max()),
                "radial_mlp_normalized": descriptor.radial_embedding.radial_norm,
                "seed_detach_energy_bit_identical": True,
                "pairs": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

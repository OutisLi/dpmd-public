# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201, TID253
"""Measure the force response to per-edge message amplitude interventions."""

from __future__ import (
    annotations,
)

import argparse
import json
from pathlib import (
    Path,
)
from types import (
    MethodType,
)
from typing import (
    Any,
)

import numpy as np
import torch
from curvature_split import (
    build_model,
)


def configure(
    model: torch.nn.Module, mode: str, selected: int | None
) -> list[dict[str, Any]]:
    """Intervene on global-frame messages before their attention-weighted sum.

    The scalar RMS over all coefficients and channels is rotation invariant.
    Detaching only its derivative preserves forward messages and measures the
    amplitude path; actual normalization also changes their forward values.
    Multiple-block interventions are causal probes, not an additive force split.
    """
    records = []
    for index, block in enumerate(model.atomic_model.descriptor.blocks):
        convolution = block.so2_conv
        original = convolution.so2_message

        def message(
            self: torch.nn.Module,
            x: torch.Tensor,
            edge_cache: Any,
            radial_feat: torch.Tensor,
            return_local: bool = False,
            *,
            method: Any = original,
            block_index: int = index,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if return_local:
                raise ValueError("The diagnostic requires the CPU global-message path")
            value, radial = method(x, edge_cache, radial_feat, return_local=False)
            mean_square = value.square().mean(dim=(1, 2), keepdim=True)
            rms = mean_square.sqrt()
            active = rms.detach().reshape(-1)
            records.append(
                {
                    "block": block_index,
                    "edges": len(active),
                    "rms_min": float(active.min()) if len(active) else None,
                    "rms_median": float(active.median()) if len(active) else None,
                    "rms_max": float(active.max()) if len(active) else None,
                    "below_floor": int((active < 1e-5**0.5).sum()),
                }
            )
            if selected is None or selected == block_index:
                if mode == "detach_amplitude":
                    denominator = (mean_square + 1e-5).sqrt()
                    value = value + value.detach() / denominator.detach() * (
                        denominator.detach() - denominator
                    )
                elif mode == "rmsnorm":
                    value = value / (mean_square + 1e-5).sqrt()
            return value, radial

        convolution.so2_message = MethodType(message, convolution)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--pairs", nargs="+", default=["H-H", "Fe-O"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(".npz").exists():
        raise FileExistsError(args.out)
    model, type_map = build_model(args.checkpoint, "float64")
    modes = [("base", None)] + [
        (mode, selected)
        for mode in ("detach_amplitude", "rmsnorm")
        for selected in (*range(len(model.atomic_model.descriptor.blocks)), None)
    ]
    inputs = []
    if args.batch:
        with np.load(args.batch) as data:
            errors = data["per_atom_error"].reshape(data["atype"].shape)
            frame = int(np.unravel_index(errors.argmax(), errors.shape)[0])
            real = data["atype"][frame] >= 0
            coord = torch.tensor(
                data["coord"].reshape(*errors.shape, 3)[frame, real][None],
                dtype=torch.float64,
            )
            atype = torch.tensor(data["atype"][frame, real][None], dtype=torch.long)
            box = torch.tensor(data["box"][frame].reshape(1, 9), dtype=torch.float64)
        inputs.append(("captured_frame", coord, atype, box, None))
    else:
        separation = np.unique(np.r_[np.arange(0.4, 6.0001, 0.005), 7.0])
        for pair in args.pairs:
            coord = torch.zeros((len(separation), 2, 3), dtype=torch.float64)
            coord[:, 1, 0] = torch.from_numpy(separation)
            atype = torch.tensor([type_map.index(x) for x in pair.split("-")]).expand(
                len(separation), -1
            )
            inputs.append((pair, coord, atype, None, separation))
    rows, arrays = [], {}
    for name, coord, atype, box, separation in inputs:
        baseline = None
        for mode, selected in modes:
            model, _ = build_model(args.checkpoint, "float64")
            records = configure(model, mode, selected)
            output = model(coord.clone(), atype, box=box)
            energy = output["energy"].detach().reshape(-1)
            force = output["force"].detach()
            if mode == "base":
                baseline = energy.clone()
            elif mode == "detach_amplitude":
                torch.testing.assert_close(energy, baseline, atol=0.0, rtol=0.0)
            row = {
                "input": name,
                "mode": mode,
                "block": selected,
                "max_force_eV_A": float(force.norm(dim=-1).max()),
                "message_rms": records,
            }
            if separation is not None:
                energy = energy - energy[-1]
                minimum = int(energy.argmin())
                row.update(
                    minimum_eV=float(energy[minimum]),
                    minimum_r_A=float(separation[minimum]),
                    tail_max_force_eV_A=float(force[separation >= 4, 1, 0].abs().max()),
                )
                arrays["separation"] = separation
            key = f"{name}_{mode}_{selected}"
            arrays[key + "_energy"] = energy.numpy()
            arrays[key + "_force"] = force.numpy()
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.out.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "precision": "float64",
                "interventions": rows,
            },
            indent=2,
        )
        + "\n"
    )
    np.savez_compressed(args.out.with_suffix(".npz"), **arrays)


if __name__ == "__main__":
    main()

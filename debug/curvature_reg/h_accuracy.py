# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201, TID253
"""Evaluate every hydrogen validation frame with identical atom and frame weights."""

from __future__ import (
    annotations,
)

import argparse
import json
import sys
from pathlib import (
    Path,
)

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "diagnose"))
from repro_spike import (
    load_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(".json").exists():
        raise FileExistsError(args.out)
    root = args.checkpoint.parent
    config = json.loads((root / "input.json").read_text())
    dataset = Path(config["training"]["validation_data"]["systems"]) / "set.000"
    values = {
        key: np.load(dataset / f"{key}.npy")
        for key in ("coord", "box", "energy", "force")
    }
    frames = len(values["coord"])
    atoms = values["coord"].shape[-1] // 3
    model, type_map = load_model(args.checkpoint, "cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    (
        indices,
        energy_errors,
        component_errors,
        vector_errors,
        force_rmse,
        maximum_errors,
    ) = [], [], [], [], [], []
    for start in range(0, frames, args.batch):
        end = min(start + args.batch, frames)
        coord = torch.tensor(
            values["coord"][start:end].reshape(-1, atoms, 3),
            dtype=torch.float64,
            device="cuda",
        )
        box = torch.tensor(
            values["box"][start:end].reshape(-1, 9), dtype=torch.float64, device="cuda"
        )
        atype = torch.full(
            (end - start, atoms), type_map.index("H"), dtype=torch.long, device="cuda"
        )
        out = model(coord, atype, box=box)
        energy = out["energy"].detach().cpu().numpy().reshape(-1)
        force = out["force"].detach().cpu().numpy()
        error = force - values["force"][start:end].reshape(-1, atoms, 3)
        norms = np.linalg.norm(error, axis=-1)
        indices.extend(range(start, end))
        energy_errors.extend(
            abs(energy - values["energy"][start:end].reshape(-1)) / atoms
        )
        component_errors.extend(abs(error).mean(axis=(1, 2)))
        vector_errors.extend(norms.mean(axis=1))
        force_rmse.extend(np.sqrt(np.square(norms).mean(axis=1)))
        maximum_errors.extend(norms.max(axis=1))
        if start % (args.batch * 100) == 0:
            print(f"frames {end}/{frames}", flush=True)
    arrays = {
        "idx": np.asarray(indices),
        "e_err_atom": np.asarray(energy_errors),
        "f_component_mae": np.asarray(component_errors),
        "f_mae": np.asarray(vector_errors),
        "f_rmse": np.asarray(force_rmse),
        "f_maxerr": np.asarray(maximum_errors),
    }
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Non-finite hydrogen validation result")
    np.testing.assert_array_equal(arrays["idx"], np.arange(frames))
    np.savez_compressed(args.out, **arrays)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "frames": frames,
        "atoms_per_frame": atoms,
        "energy_mae_eV_atom": float(arrays["e_err_atom"].mean()),
        "force_component_mae_eV_A": float(arrays["f_component_mae"].mean()),
        "force_vector_mae_eV_A": float(arrays["f_mae"].mean()),
        "force_vector_rmse_eV_A": float(np.sqrt(np.square(arrays["f_rmse"]).mean())),
        "maximum_force_error_eV_A": float(arrays["f_maxerr"].max()),
    }
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

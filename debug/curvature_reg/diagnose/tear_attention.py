# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201, TID253
"""Replay a captured training-frame tear and detach the attention gradient paths."""

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
from message_envelope import (
    configure,
)

from deepmd.pt.model.descriptor.sezm_nn import (
    so2,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    with np.load(args.batch) as batch:
        error = batch["per_atom_error"].reshape(batch["atype"].shape)
        frame, atom = np.unravel_index(np.argmax(error), error.shape)
        real = batch["atype"][frame] >= 0
        indices = np.flatnonzero(real)
        worst = int(np.flatnonzero(indices == atom)[0])
        coord = batch["coord"][frame, real].copy()
        atype = batch["atype"][frame, real].copy()
        box = batch["box"][frame].reshape(3, 3).copy()
        label = batch["label_force"][frame, real].copy()
        recorded = batch["model_force"][frame, real].copy()
    _, distances = find_mic(coord - coord[worst], box, pbc=True)
    distances[worst] = np.inf
    neighbour = int(distances.argmin())
    model, type_map = build_model(args.checkpoint, "float64")
    coordinates = torch.tensor(coord[None], dtype=torch.float64)
    types = torch.tensor(atype[None], dtype=torch.long)
    cells = torch.tensor(box.reshape(1, 9), dtype=torch.float64)
    original = so2.segment_envelope_gated_softmax
    reference_energy = None
    rows = []
    forces = {"recorded": recorded, "label": label}
    for mode in (
        "base",
        "detach_envelope",
        "detach_logits",
        "detach_attention",
        "output_envelope",
    ):
        calls = []

        def attention(
            logits: torch.Tensor,
            edge_env: torch.Tensor,
            dst: torch.Tensor,
            n_nodes: int,
            z_bias_raw: torch.Tensor,
            eps: float,
            src_weight: torch.Tensor | None = None,
        ) -> torch.Tensor:
            calls.append(n_nodes)
            alpha = original(
                logits=logits.detach() if mode == "detach_logits" else logits,
                edge_env=edge_env.detach() if mode == "detach_envelope" else edge_env,
                dst=dst,
                n_nodes=n_nodes,
                z_bias_raw=z_bias_raw,
                eps=eps,
                src_weight=src_weight,
            )
            return alpha.detach() if mode == "detach_attention" else alpha

        configure(model.atomic_model.descriptor, mode == "output_envelope")
        so2.segment_envelope_gated_softmax = attention
        try:
            output = model(coordinates.clone(), types, box=cells)
        finally:
            so2.segment_envelope_gated_softmax = original
        if not calls:
            raise RuntimeError(f"No attention calls observed in {mode}")
        energy = output["energy"].detach()
        if reference_energy is None:
            reference_energy = energy.clone()
        elif mode != "output_envelope":
            torch.testing.assert_close(energy, reference_energy, atol=0.0, rtol=0.0)
        force = output["force"].detach().numpy()[0]
        forces[mode] = force
        row = {
            "mode": mode,
            "energy": float(energy.sum()),
            "max_force_error": float(np.linalg.norm(force - label, axis=-1).max()),
            "max_predicted_force": float(np.linalg.norm(force, axis=-1).max()),
            "original_worst_atom_error": float(
                np.linalg.norm(force[worst] - label[worst])
            ),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    record = {
        "checkpoint": str(args.checkpoint.resolve()),
        "batch": str(args.batch.resolve()),
        "frame": int(frame),
        "atom": int(atom),
        "atom_type": type_map[int(atype[worst])],
        "nearest_type": type_map[int(atype[neighbour])],
        "nearest_distance": float(distances[neighbour]),
        "recorded_max_force_error": float(
            np.linalg.norm(recorded - label, axis=-1).max()
        ),
        "precision": "float64",
        "interventions": rows,
    }
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    np.savez_compressed(
        args.out.with_suffix(".npz"), coord=coord, atype=atype, box=box, **forces
    )


if __name__ == "__main__":
    main()

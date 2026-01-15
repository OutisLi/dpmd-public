# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201, TID253
"""Measure attention masses and force changes under gradient-path detachment.

Every intervention preserves the forward energy. Differences between interventions
describe the removed gradient paths; they are not an additive decomposition because
paths through the logits and earlier interaction blocks can overlap.
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

from deepmd.pt.model.descriptor.sezm_nn import (
    so2,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--pair", default="H-H")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dr", type=float, default=0.02)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(".json").exists():
        raise FileExistsError(args.out)
    model, type_map = build_model(args.checkpoint, "float64")
    r = np.unique(
        np.concatenate(
            (
                np.arange(3.0, 6.001, args.dr),
                [1.0, 1.5, 2.0, 2.5, 4.0, 5.705, 5.875, 6.01, 7.0],
            )
        )
    )
    r = np.unique(np.round(r, 10))
    positions = torch.zeros((len(r), 2, 3), dtype=torch.float64)
    positions[:, 1, 0] = torch.from_numpy(r)
    types = torch.tensor(
        [[type_map.index(x) for x in args.pair.split("-")]], dtype=torch.long
    ).repeat(len(r), 1)
    boxes = (torch.eye(3, dtype=torch.float64) * 20.0).reshape(1, 9).repeat(len(r), 1)
    original = so2.segment_envelope_gated_softmax
    data = {"r": r}
    profile = []
    energies = None
    for mode in ("base", "detach_envelope", "detach_logits", "detach_attention"):
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
            alpha = original(
                logits=logits.detach() if mode == "detach_logits" else logits,
                edge_env=edge_env.detach() if mode == "detach_envelope" else edge_env,
                dst=dst,
                n_nodes=n_nodes,
                z_bias_raw=z_bias_raw,
                eps=eps,
                src_weight=src_weight,
            )
            calls.append(n_nodes)
            if mode == "base":
                profile.append(
                    (
                        dst.detach().numpy(),
                        edge_env.detach().numpy(),
                        logits.detach().numpy(),
                        alpha.detach().numpy(),
                        (torch.nn.functional.softplus(z_bias_raw) + eps)
                        .detach()
                        .numpy(),
                    )
                )
            return alpha.detach() if mode == "detach_attention" else alpha

        so2.segment_envelope_gated_softmax = attention
        try:
            output = model(positions.clone(), types, box=boxes)
        finally:
            so2.segment_envelope_gated_softmax = original
        if not calls or any(n != 2 * len(r) for n in calls):
            raise RuntimeError(
                f"Attention probe did not cover the expected pair batch: {calls}"
            )
        energy = output["energy"].detach().reshape(-1)
        if energies is None:
            energies = energy.clone()
        else:
            torch.testing.assert_close(energy, energies, rtol=0.0, atol=0.0)
        data[f"force_{mode}"] = output["force"].detach().numpy()[:, 1, 0]
    data["energy"] = energies.numpy() - energies[-1].item()
    for block, (dst, envelope, logits, alpha, null) in enumerate(profile):
        frames = dst // 2
        for name, values in (
            ("envelope", envelope),
            ("logit", logits),
            ("alpha", alpha),
        ):
            flat = values.reshape(len(frames), -1).max(axis=1)
            result = np.full(len(r), -np.inf)
            np.maximum.at(result, frames, flat)
            result[~np.isfinite(result)] = 0.0
            data[f"block{block}_{name}"] = result
        data[f"block{block}_null_mass"] = null
    tail = (r >= 4.0) & (r <= 7.0)
    summaries = {
        mode: float(np.abs(data[f"force_{mode}"][tail]).max())
        for mode in ("base", "detach_envelope", "detach_logits", "detach_attention")
    }
    print(args.checkpoint, args.pair, "tail maximum eV/A", summaries)
    points = []
    for value in (4.0, 4.5, 5.0, 5.705, 5.875):
        index = int(np.argmin(abs(r - value)))
        row = {"r": float(r[index]), "force": float(data["force_base"][index])}
        for block in range(len(profile)):
            envelope = float(data[f"block{block}_envelope"][index])
            alpha = float(data[f"block{block}_alpha"][index])
            row[f"block{block}"] = {
                "envelope": envelope,
                "alpha": alpha,
                "alpha_over_envelope": alpha / envelope if envelope else None,
                "logit": float(data[f"block{block}_logit"][index]),
                "null_mass_min": float(data[f"block{block}_null_mass"].min()),
            }
        points.append(row)
        print(json.dumps(row))
    np.savez_compressed(args.out, **data)
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "pair": args.pair,
                "precision": "float64",
                "energy_bit_identical": True,
                "radial_step": args.dr,
                "profile_reduction": "maximum over both atoms, foci, and heads",
                "tail_maxima": summaries,
                "points": points,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

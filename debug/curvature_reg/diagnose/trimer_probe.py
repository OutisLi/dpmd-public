# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201
"""
Compressed-trimer probe: the force on a compressed pair with one neighbour present.

The isolated-pair survey reads the model where every other-neighbour channel
vanishes. The tearing state of §11.36/§11.39 lives one neighbour away: the same
compressed pair, normal in isolation, reads 10^2-10^4 eV/A as soon as a third
atom lies inside the cutoff. This probe places the pair A-B at a fraction of its
covalent contact along x and a third atom of type A at a fixed distance from A
at 90 degrees, and reports the radial force on B and the largest force in the
cluster, beside the same pair alone. One checkpoint, many pair types, one line of
JSON per checkpoint appended to the output file.
"""

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
from ase.data import (
    atomic_numbers,
    covalent_radii,
)

PAIRS = (
    "H-H",
    "Li-H",
    "O-H",
    "C-C",
    "N-N",
    "O-O",
    "Si-O",
    "Ti-O",
    "Fe-O",
    "Cu-Cu",
    "Zn-S",
    "Ni-Te",
    "Cs-H",
    "Ga-As",
    "Mo-Mo",
    "Kr-Kr",
    "Ar-Ar",
    "Pb-H",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("label")
    parser.add_argument("--out", type=Path, required=True, help="JSONL file, appended")
    parser.add_argument("--pairs", default=",".join(PAIRS))
    parser.add_argument(
        "--ratio", type=float, default=0.45, help="pair separation in covalent contacts"
    )
    parser.add_argument(
        "--third",
        type=float,
        default=2.0,
        help="distance of the third atom from atom A in A",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch

    directory = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(directory))
    sys.path.insert(0, str(directory / "diagnose"))
    from heal_spike import (
        batch,
    )
    from repro_spike import (
        load_model,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, type_map = load_model(args.checkpoint, args.device)
    cell = np.eye(3) * 20.0
    record = {
        "checkpoint": str(args.checkpoint.resolve()),
        "label": args.label,
        "ratio": args.ratio,
        "third_A": args.third,
        "pairs": {},
    }
    for pair in args.pairs.split(","):
        a, b = pair.split("-")
        contact = float(
            covalent_radii[atomic_numbers[a]] + covalent_radii[atomic_numbers[b]]
        )
        r = args.ratio * contact
        pair_pos = np.zeros((1, 2, 3))
        pair_pos[0, 1, 0] = r
        tri_pos = np.zeros((1, 3, 3))
        tri_pos[0, 1, 0] = r
        tri_pos[0, 2, 1] = args.third
        types2 = np.array([type_map.index(a), type_map.index(b)])
        types3 = np.array([type_map.index(a), type_map.index(b), type_map.index(a)])
        e2, _, f2 = batch(model, pair_pos, cell, types2, args.device)
        e3, _, f3 = batch(model, tri_pos, cell, types3, args.device)
        f2 = f2.detach().double().cpu().numpy()[0]
        f3 = f3.detach().double().cpu().numpy()[0]
        record["pairs"][pair] = {
            "r_A": r,
            "pair_radial_force": float(f2[1, 0]),
            "pair_energy": float(e2.detach().double().cpu().numpy().reshape(-1)[0]),
            "trimer_radial_force_on_B": float(f3[1, 0]),
            "trimer_max_force": float(np.abs(f3).max()),
            "trimer_energy": float(e3.detach().double().cpu().numpy().reshape(-1)[0]),
        }
    worst = max(record["pairs"].items(), key=lambda kv: kv[1]["trimer_max_force"])
    record["worst_pair"] = worst[0]
    record["worst_trimer_max_force"] = worst[1]["trimer_max_force"]
    with args.out.open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(
        f"{args.label}: worst trimer {worst[0]} {worst[1]['trimer_max_force']:.1f} eV/A (pair alone {worst[1]['pair_radial_force']:.1f}); "
        + ", ".join(
            f"{k} {v['trimer_max_force']:.0f}/{abs(v['pair_radial_force']):.0f}"
            for k, v in record["pairs"].items()
        )
    )


if __name__ == "__main__":
    main()

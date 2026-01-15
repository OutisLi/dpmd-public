# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, T201
"""
Spike evaluation of a hydrogen checkpoint at the conditions of the real failure.

Reports, for one checkpoint: the curvature ratio and the largest force on the
eight archived pathological frames (steps 17090-17097 of the failed run), the
largest force a hill climb finds within 0.05 A of s17096, the radial profile
of atom 49 along the 49-48 bond of s17097, the accuracy on the model's own
validation frames, and Langevin MD at 1000 K started from healthy frames of
the failed run's density (1.579 A^3/atom), which is where the original model
failed after 3.4 ps.
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "diagnose"))
from eval_model import (
    langevin,
)
from heal_spike import (
    batch,
    bond_scan,
    penalty_stats,
)
from repro_spike import (
    load_model,
    read_xyz,
)
from spike_density import (
    adversarial_search,
    random_search,
)
from traj_curvature import (
    read_dump,
)

ROOT = Path("/nas/outisli/Software/deepmd-kit/temp/normal")
BAD = [
    "last_normal_s17091",
    "last_normal_s17092",
    "last_normal_s17093",
    "last_normal_s17094",
    "last_normal_s17095",
    "last_normal_s17096",
    "onset_1_s17097",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--state-dict", type=Path, default=None)
    ap.add_argument(
        "--valid",
        type=Path,
        default=Path(
            "/nas/outisli/Software/deepmd-kit/temp/H_data/AIMD_PCA_SELECT_RAW_251112/H64/validation_data"
        ),
    )
    ap.add_argument("--n-valid", type=int, default=200)
    ap.add_argument("--n-md", type=int, default=6)
    ap.add_argument("--md-steps", type=int, default=5000)
    ap.add_argument("--k0", type=float, default=300.0)
    ap.add_argument("--rho0", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, type_map = load_model(args.ckpt, args.device)
    if args.state_dict is not None:
        model.load_state_dict(torch.load(args.state_dict, map_location=args.device))
    res = {}
    frames = [read_xyz(ROOT / f"normal_{t}.xyz") for t in BAD]
    cell = frames[0][1]
    atype = np.array([type_map.index(s) for s in frames[0][2]])
    bad_pos = [f[0] for f in frames]
    stats = penalty_stats(
        model, bad_pos, cell, atype, 0.01, args.k0, args.rho0, args.device
    )
    res["bad_ratio_max"] = float(stats[:, 1].max())
    res["bad_ratio"] = stats[:, 1].round(2).tolist()
    res["bad_fmax"] = stats[:, 2].round(1).tolist()
    print(
        f"pathological frames: ratio {np.round(stats[:, 1], 2)}, Fmax {np.round(stats[:, 2], 1)}"
    )
    s0, f, _ = adversarial_search(model, bad_pos[5], cell, atype, 0.05, 40, args.device)
    gen = torch.Generator().manual_seed(0)
    rnd = random_search(model, bad_pos[5], cell, atype, 0.02, 60, args.device, gen)
    res["hill_climb_s17096"] = float(f)
    res["rand002_max"] = float(rnd.max())
    res["rand002_frac_gt100"] = float((rnd > 100).mean())
    print(
        f"s17096 hill climb 0.05 A: {s0:.1f} -> {f:.1f}; random 0.02 A: median {np.median(rnd):.1f} max {rnd.max():.1f} frac>100 {(rnd > 100).mean():.2f}"
    )
    rows = bond_scan(
        model, bad_pos[6], cell, atype, 49, 48, args.device, np.arange(0.6, 1.16, 0.075)
    )
    res["bond_49_48_peak"] = float(max(r[1] for r in rows))
    res["bond_49_48_fmax"] = float(max(abs(r[2]) for r in rows))
    print(
        "s17097 atom 49 along 49-48: "
        + "  ".join(f"r={r:.3f}:E={e:.1f},F={fr:.0f}" for r, e, fr in rows)
    )
    # Accuracy on the model's own validation frames.
    s = args.valid / "set.000"
    coord = np.load(s / "coord.npy")
    box = np.load(s / "box.npy")
    force = np.load(s / "force.npy")
    energy = np.load(s / "energy.npy")
    n = coord.shape[1] // 3
    idx = np.linspace(0, len(coord) - 1, args.n_valid).astype(int)
    ee, fe = [], []
    for i in idx:
        e0, _, f0 = batch(
            model,
            coord[i].reshape(1, n, 3),
            box[i].reshape(3, 3),
            atype[:n] if n == len(atype) else np.zeros(n, dtype=int),
            args.device,
        )
        ee.append(abs(float(e0[0]) - energy[i]) / n)
        fe.append(
            float(
                (f0[0].double().cpu().numpy() - force[i].reshape(n, 3)).__abs__().mean()
            )
        )
    res["valid_mae_e"] = float(np.mean(ee))
    res["valid_mae_f"] = float(np.mean(fe))
    print(
        f"validation: MAE_E {np.mean(ee) * 1000:.2f} meV/atom, MAE_F {np.mean(fe) * 1000:.1f} meV/A"
    )
    # MD at the failing density from healthy frames of the failed run.
    starts = [int(v) for v in np.linspace(2000, 16000, args.n_md)]
    dumped = read_dump(ROOT / "dump.traj", set(starts))
    runs = [
        langevin(
            model,
            dumped[st].copy(),
            cell,
            atype,
            ["H"] * 64,
            1000.0,
            0.2,
            args.md_steps,
            0.01,
            args.device,
            k,
            100.0,
        )
        for k, st in enumerate(starts)
    ]
    fm = np.array([r["fmax"] for r in runs])
    res["md_exploded"] = float(np.mean([r["exploded"] for r in runs]))
    res["md_fmax"] = fm.round(1).tolist()
    print(
        f"MD 1000 K, {args.md_steps} x 0.2 fs from {len(starts)} frames: exploded {res['md_exploded']:.2f}, fmax per run {fm.round(1)}"
    )
    out = args.out or (args.state_dict or args.ckpt).with_suffix(".spike_eval.json")
    out.write_text(json.dumps(res, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()

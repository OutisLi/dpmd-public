# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, T201
"""
Smoothness of a model along the dimer coordinate.

Two atoms in a large cubic cell are pulled apart from ``r_min`` to ``r_max``;
the energy and the radial force are recorded on a fine grid and the
curvature is read from consecutive force differences. A kink — the artefact a
normalization with a small epsilon produces where the SO(2) messages vanish
near the cutoff — shows as a spike of the curvature at a distance where the
force itself is small. The largest curvature outside the repulsive wall
(``r > r_wall``) and the distance at which it occurs are reported, together
with the physical scale ``k0 = 300 eV/Å²``.

Usage: dimer_scan.py ckpt.pt --element H [--out scan.npz]
"""

from __future__ import (
    annotations,
)

import argparse
import sys
from pathlib import (
    Path,
)

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "diagnose"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from heal_spike import (
    batch,
)
from repro_spike import (
    load_model,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--element", default="H")
    ap.add_argument("--r-min", type=float, default=0.4)
    ap.add_argument("--r-max", type=float, default=7.0)
    ap.add_argument("--dr", type=float, default=0.005)
    ap.add_argument(
        "--r-wall",
        type=float,
        default=1.2,
        help="distances below this are the repulsive wall and are excluded from the kink statistic",
    )
    ap.add_argument(
        "--tail-min",
        type=float,
        default=4.0,
        help="start of the outer window in which the tail kink is measured",
    )
    ap.add_argument("--cell", type=float, default=20.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    model, type_map = load_model(args.ckpt, args.device)
    atype = np.array([type_map.index(args.element)] * 2)
    cell = np.eye(3) * args.cell
    r = np.arange(args.r_min, args.r_max + 1e-9, args.dr)
    pos = np.zeros((len(r), 2, 3))
    pos[:, 1, 0] = r
    e, f = [], []
    for i in range(0, len(r), 64):
        e_, _, f_ = batch(model, pos[i : i + 64], cell, atype, args.device)
        e.append(e_.detach().double().cpu().numpy())
        f.append(f_[:, 1, 0].detach().double().cpu().numpy())
    e, f = np.concatenate(e), np.concatenate(f)
    # A model in the ``dens`` mode predicts its forces with a direct head; the
    # readings below describe the energy surface, so its radial force is the
    # numerical derivative of the energy on the fine grid, and the direct
    # head's force at 0.6 Å is reported alongside.
    direct = hasattr(model, "get_active_mode") and model.get_active_mode() == "dens"
    if direct:
        f_head = f
        f = -np.gradient(e, r)
    # Curvature d²E/dr² = -dF_r/dr from consecutive radial forces.
    curv = -np.gradient(f, r)
    outside = r > args.r_wall
    i = np.argmax(np.abs(curv[outside]))
    r_k, c_k = r[outside][i], curv[outside][i]
    # Tail kink: the largest curvature in the outer window, where the true
    # dimer is nearly flat and a normalization acting on vanishing messages
    # leaves a step in the force.
    tail = (r >= args.tail_min) & (r <= args.r_max)
    j = np.argmax(np.abs(curv[tail]))
    # Wall stiffness: the energy at 0.45 Å above the well and the repulsive
    # force at 0.6 Å, the two readings of criterion A2.
    e_wall = e[np.argmin(np.abs(r - 0.45))] - e.min()
    f_wall = f[np.argmin(np.abs(r - 0.60))]
    # Inner band: the largest force inside 0.62 Å and the number of sign changes of the
    # force there. A smooth wall has at most one; an oscillating short-range pair function
    # (the needle band of a run's late storm) shows several and a force far above the wall's.
    inner = r < 0.62
    f_in = f[inner]
    n_flip = int(np.count_nonzero(np.sign(f_in[1:]) != np.sign(f_in[:-1])))
    print(
        f"{args.element}2 dimer: E(r_max) {e[-1]:.4f} eV, well depth {e.min() - e[-1]:.3f} eV at {r[np.argmin(e)]:.3f} Å; "
        f"wall: E(0.45 Å) - E(well) {e_wall:.2f} eV, F(0.6 Å) {f_wall:.2f} eV/Å; "
        f"inner band [{r[0]:.2f}, 0.62) Å: max |F| {np.abs(f_in).max():.1f} eV/Å, {n_flip} sign changes; "
        f"largest |curvature| beyond {args.r_wall} Å: {abs(c_k):.1f} eV/Å² at {r_k:.3f} Å (k0 = 300); |F| there {abs(f[outside][i]):.3f} eV/Å; "
        f"tail [{args.tail_min}, {args.r_max}] Å: max |curvature| {abs(curv[tail][j]):.3f} eV/Å² at {r[tail][j]:.3f} Å, |F| max {np.abs(f[tail]).max():.4f} eV/Å; "
        f"|F| at r_max {abs(f[-1]):.2e}"
        + (
            f"; direct-head F(0.6 Å) {f_head[np.argmin(np.abs(r - 0.60))]:.2f} eV/Å"
            if direct
            else ""
        )
    )
    if args.out is not None:
        np.savez(args.out, r=r, energy=e, force=f, curvature=curv)


if __name__ == "__main__":
    main()

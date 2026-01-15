# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, T201, ANN001, ANN201, B905
"""
Force-direction curvature along an entire MD trajectory.

The archived LAMMPS dump holds every step of the hydrogen run that failed at
step 17097.  Evaluating the curvature estimators on thousands of ordinary
frames, rather than on a handful of archived ones, answers three questions a
regularizer design depends on:

* what the distribution of ``|c|`` looks like on healthy MD frames, and hence
  how much headroom a threshold has before it touches healthy configurations;
* how early the signal rises before the failure;
* whether the cheap one-sided energy probe, the force-difference probe and the
  consecutive-frame estimator agree, and whether a per-atom force-difference
  resolves a localized defect better than the global scalar.

Estimators, with ``u = F/|F|`` the unit force direction in ``R^{3N}``:

    c_E  = 2 [E(x + eps u) - E(x) + eps |F|] / eps^2         (energy probe)
    g    = [F(x) - F(x + eps u)] / eps  ~ H u                (force probe)
    c_F  = u . g
    L_i  = |g_i|                                             (per-atom)
    c_pair = (F_t - F_{t+1}) . dx / |dx|^2,  dx = x_{t+1} - x_t
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
from repro_spike import (
    load_model,
)


def read_dump(path: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    """
    Read selected frames of a LAMMPS text dump.

    Parameters
    ----------
    path : Path
        Dump file with ``ITEM: ATOMS id type x y z`` records.
    wanted : set[int]
        Timesteps to keep.

    Returns
    -------
    dict[int, np.ndarray]
        Timestep -> positions with shape (N, 3), sorted by atom id.
    """
    frames: dict[int, np.ndarray] = {}
    with path.open() as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step = int(fh.readline())
            fh.readline()
            natoms = int(fh.readline())
            for _ in range(5):
                fh.readline()
            rows = [fh.readline().split() for _ in range(natoms)]
            if step not in wanted:
                continue
            arr = np.array(
                [[float(r[0]), float(r[2]), float(r[3]), float(r[4])] for r in rows]
            )
            frames[step] = arr[np.argsort(arr[:, 0]), 1:]
    return frames


@torch.no_grad()
def _noop() -> None:
    return None


def evaluate_batch(
    model, pos: np.ndarray, cell: np.ndarray, atype: np.ndarray, device: str
):
    """
    Evaluate energy, atomic energy and force on a batch of frames.

    Parameters
    ----------
    pos : np.ndarray
        Positions with shape (nf, N, 3).
    cell : np.ndarray
        Cell with shape (3, 3), shared by all frames.
    atype : np.ndarray
        Types with shape (N,).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Energies (nf,), atomic energies (nf, N), forces (nf, N, 3), float64.
    """
    nf = pos.shape[0]
    c = torch.tensor(pos, dtype=torch.float32, device=device)
    b = (
        torch.tensor(cell, dtype=torch.float32, device=device)
        .expand(nf, 3, 3)
        .contiguous()
    )
    t = torch.tensor(atype, dtype=torch.long, device=device).expand(nf, -1).contiguous()
    out = model(c, t, box=b)
    e = out["energy"].detach().reshape(nf).double().cpu().numpy()
    ae = out["atom_energy"].detach().reshape(nf, -1).double().cpu().numpy()
    f = out["force"].detach().reshape(nf, -1, 3).double().cpu().numpy()
    return e, ae, f


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dump",
        type=Path,
        default=Path("/nas/outisli/Software/deepmd-kit/temp/normal/dump.traj"),
    )
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=Path("/nas/outisli/Software/deepmd-kit/temp/model_ema.ckpt.pt"),
    )
    ap.add_argument("--cell", type=float, default=4.65817)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--dense-from", type=int, default=16900)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument(
        "--out", type=Path, default=Path(__file__).with_name("traj_curvature.npz")
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, type_map = load_model(args.ckpt, args.device)

    steps = sorted(
        set(range(0, 17103, args.stride)) | set(range(args.dense_from, 17103))
    )
    wanted = set(steps) | {s + 1 for s in steps}
    frames = read_dump(args.dump, wanted)
    steps = [s for s in steps if s in frames and s + 1 in frames]
    print(f"{len(steps)} frames to evaluate")

    cell = np.eye(3) * args.cell
    natoms = frames[steps[0]].shape[0]
    atype = np.full(natoms, type_map.index("H"))
    eps = args.eps

    rec = {
        k: []
        for k in [
            "step",
            "E",
            "Fmax",
            "Frms",
            "cE",
            "cE_atom",
            "cF",
            "Lmax",
            "Larg",
            "cpair",
            "Fmax_next",
        ]
    }
    for i0 in range(0, len(steps), args.batch):
        chunk = steps[i0 : i0 + args.batch]
        x0 = np.stack([frames[s] for s in chunk])
        x1 = np.stack([frames[s + 1] for s in chunk])
        e0, ae0, f0 = evaluate_batch(model, x0, cell, atype, args.device)
        e1, _, f1 = evaluate_batch(model, x1, cell, atype, args.device)
        fn = np.linalg.norm(f0.reshape(len(chunk), -1), axis=-1)
        u = f0 / fn[:, None, None]
        ep, aep, fp = evaluate_batch(model, x0 + eps * u, cell, atype, args.device)
        c_e = 2.0 * (ep - e0 + eps * fn) / eps**2
        c_e_atom = 2.0 * ((aep - ae0).sum(-1) + eps * fn) / eps**2
        g = (f0 - fp) / eps
        c_f = np.einsum("bij,bij->b", u, g)
        lnorm = np.linalg.norm(g, axis=-1)
        # Consecutive-frame estimator with minimum-image displacement.
        dx = x1 - x0
        dx -= args.cell * np.round(dx / args.cell)
        dn2 = (dx.reshape(len(chunk), -1) ** 2).sum(-1)
        c_pair = np.einsum("bij,bij->b", f0 - f1, dx) / dn2
        fmax = np.linalg.norm(f0, axis=-1).max(-1)
        frms = np.sqrt((f0**2).sum(-1).mean(-1))
        for k, v in zip(
            rec,
            [
                chunk,
                e0,
                fmax,
                frms,
                c_e,
                c_e_atom,
                c_f,
                lnorm.max(-1),
                lnorm.argmax(-1),
                c_pair,
                np.linalg.norm(f1, axis=-1).max(-1),
            ],
        ):
            rec[k].extend(np.asarray(v).tolist())
        if (i0 // args.batch) % 10 == 0:
            print(
                f"  {chunk[-1]:6d}  E={e0[-1]:9.3f} Fmax={fmax[-1]:7.2f} cE={c_e[-1]:10.1f} cF={c_f[-1]:10.1f} Lmax={lnorm.max(-1)[-1]:9.1f} cpair={c_pair[-1]:9.1f}"
            )
    out = {k: np.asarray(v) for k, v in rec.items()}
    np.savez(args.out, **out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

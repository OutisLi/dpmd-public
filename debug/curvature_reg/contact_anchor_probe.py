# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, T201, ANN001, ANN201
"""
Force on a hard contact inside the frame and inside its sparse anchors.

A frame with a hard contact (a pair far below its equilibrium distance, with
a labelled force of tens of eV/A) is transformed the way the pretraining
safeguard transforms it -- a random fraction of the other atoms deleted, or
the cell dilated -- while the contact pair itself is kept.  For every
configuration the radial force on the pair (positive repulsive) is read from
each model.  A pair repulsion at fixed distance should barely depend on how
many other atoms surround it; a model whose force on the contact collapses as
the environment is thinned out has learned the environment dependence the
zero-force reference imposes on the sparse anchors.
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
from ase import (
    Atoms,
)
from ase.neighborlist import (
    neighbor_list,
)

sys.path.insert(0, str(Path(__file__).resolve().parent / "diagnose"))
from repro_spike import (
    load_model,
)


def forces(model, coord, box, atype, device):
    c = torch.tensor(coord[None], dtype=torch.float32, device=device)
    b = torch.tensor(box[None], dtype=torch.float32, device=device)
    t = torch.tensor(atype[None], dtype=torch.long, device=device)
    out = model(c, t, box=b)
    return out["force"].detach().reshape(-1, 3).double().cpu().numpy()


def radial(f, coord, box, i, j):
    """Radial force on the pair (i, j), positive when repulsive, eV/A."""
    atoms = Atoms(positions=coord, cell=box, pbc=True)
    d = atoms.get_distance(i, j, mic=True, vector=True)  # from i to j
    u = d / np.linalg.norm(d)
    return 0.5 * float(np.dot(f[j] - f[i], u))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frame", type=int)
    ap.add_argument("ckpts", nargs="+", type=Path)
    ap.add_argument("--lmdb", default="/data/Datasets/LMDB/OMat24/sub001val.lmdb")
    ap.add_argument("--drop", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    ap.add_argument("--dilate", type=float, nargs="+", default=[1.3, 1.6, 2.0])
    ap.add_argument("--n-draws", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models = [load_model(p, args.device) for p in args.ckpts]
    type_map = models[0][1]
    from deepmd.dpmodel.utils.lmdb_data import (
        LmdbDataReader,
    )

    reader = LmdbDataReader(args.lmdb, type_map, batch_size=1)
    fr = reader[args.frame]
    coord = fr["coord"].reshape(-1, 3).astype(np.float64)
    box = fr["box"].reshape(3, 3).astype(np.float64)
    atype = fr["atype"].reshape(-1)
    label = fr["force"].reshape(-1, 3)
    n = len(atype)
    ii, jj, dd = neighbor_list("ijd", Atoms(positions=coord, cell=box, pbc=True), 6.0)
    k = int(np.argmin(dd))
    i, j = int(ii[k]), int(jj[k])
    print(
        f"frame {args.frame}: {n} atoms, closest pair {type_map[atype[i]]}{i}-{type_map[atype[j]]}{j} at {dd[k]:.3f} A; "
        f"labelled radial force {radial(label, coord, box, i, j):.2f} eV/A, |F| label {np.linalg.norm(label[i]):.1f} / {np.linalg.norm(label[j]):.1f}"
    )
    names = [p.parent.name + "/" + p.name for p in args.ckpts]
    print(f"{'configuration':>34} " + " ".join(f"{nm:>28}" for nm in names))

    def row(tag: str, cases: list) -> None:
        cells = []
        for model, _ in models:
            vals = [
                radial(forces(model, c, b, t, args.device), c, b, i2, j2)
                for c, b, t, i2, j2 in cases
            ]
            cells.append(
                f"{np.mean(vals):9.2f} ({min(vals):7.2f}..{max(vals):7.2f})"
                if len(vals) > 1
                else f"{vals[0]:9.2f}{'':21}"
            )
        print(f"{tag:>34} " + " ".join(f"{c:>28}" for c in cells))

    row("full frame", [(coord, box, atype, i, j)])
    for scale in args.dilate:
        row(f"dilated x{scale:.1f}", [(coord * scale, box * scale, atype, i, j)])
    rng = np.random.default_rng(0)
    others = np.array([a for a in range(n) if a not in (i, j)])
    for p in args.drop:
        cases = []
        for _ in range(args.n_draws):
            keep = others[rng.random(len(others)) >= p]
            sel = np.concatenate([[i, j], keep])
            cases.append((coord[sel], box, atype[sel], 0, 1))
        row(f"deleted {p:.0%} of the other atoms", cases)
    row("the pair alone", [(coord[[i, j]], box, atype[[i, j]], 0, 1)])


if __name__ == "__main__":
    main()

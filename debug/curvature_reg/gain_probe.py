# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201
"""Compressed-probe gain (force sensitivity to a random weight perturbation, ratio to the data's) along checkpoint series."""

from __future__ import (
    annotations,
)

import math
import sys
import time
from typing import (
    TYPE_CHECKING,
)

import numpy as np

if TYPE_CHECKING:
    import torch

sys.path.insert(0, "/nas/outisli/Software/deepmd-kit/debug/curvature_reg")
import common
from ase.data import (
    atomic_numbers,
    covalent_radii,
)
from common import (
    build,
    forces,
    load_needle,
    subset,
)

R = "/nas/outisli/Software/deepmd-kit/debug/curvature_reg/runs"


def dimer(z: int, rs: list[float], cell: float = 20.0) -> dict[str, torch.Tensor]:
    import torch

    n = len(rs)
    c = torch.zeros(n, 2, 3)
    c[:, 0, 0] = cell / 2
    c[:, 1, 0] = cell / 2 + torch.tensor(rs)
    c[:, :, 1] = cell / 2
    c[:, :, 2] = cell / 2
    return {
        "coord": c,
        "atype": torch.full((n, 2), z, dtype=torch.long),
        "box": (torch.eye(3) * cell).reshape(1, 9).repeat(n, 1),
    }


def compressed_pairs(
    bi: dict[str, torch.Tensor], radii: torch.Tensor, ratio: float = 0.45
) -> dict[str, torch.Tensor]:
    """Each frame's closest pair (minimum image), alone in its cell, compressed to ``ratio`` of the covalent contact."""
    import torch

    coord, atype, box = bi["coord"], bi["atype"], bi["box"]
    n = atype.shape[0]
    out_c, out_t, out_b = [], [], []
    for k in range(n):
        x = coord[k].double()
        cell = box[k].reshape(3, 3).double()
        inv = torch.linalg.inv(cell)
        dvec = x[:, None, :] - x[None, :, :]
        frac = dvec @ inv
        frac = frac - torch.round(frac)
        dvec = frac @ cell
        dist = dvec.norm(dim=-1) + torch.eye(x.shape[0], dtype=torch.float64) * 1e9
        i, j = np.unravel_index(int(dist.argmin()), dist.shape)
        rc = radii[atype[k, i]] + radii[atype[k, j]]
        u = dvec[i, j] / dist[i, j]
        mid = x[j] + 0.5 * dvec[i, j]
        c = torch.stack(
            [mid + 0.5 * ratio * rc * u, mid - 0.5 * ratio * rc * u]
        ).float()
        out_c.append(c)
        out_t.append(atype[k, [i, j]])
        out_b.append(box[k])
    return {
        "coord": torch.stack(out_c),
        "atype": torch.stack(out_t),
        "box": torch.stack(out_b),
    }


def rms(F1: torch.Tensor, F0: torch.Tensor, atype: torch.Tensor) -> float:
    real = atype >= 0
    dF = (F1 - F0).norm(dim=-1)[real]
    return float(dF.square().mean().sqrt())


def main() -> None:
    import torch

    jobs = [tuple(a.split(":")) for a in sys.argv[1:]]
    inp, lab, err, d = load_needle("needle_rank0_step28666.npz")
    nf, nloc = inp["atype"].shape
    fi = 221
    rng = np.random.default_rng(0)
    ord_idx = sorted(
        rng.choice([i for i in range(nf) if i != fi], size=64, replace=False).tolist()
    )
    bi, bl = subset(inp, lab, ord_idx)
    pair_atoms = torch.tensor([3, 31])
    print(
        "run step MAE_d dF_data | gain: O2_.465 O2_.6 O2_.8 H2_.5 ndl-pair cmp45 cmp60 coll | F_O2_.465 ndl-err Fmax_cmp45 kind  -- gain = dF RMS at probe over dF RMS on data for a random weight direction of one step norm, median of 3 draws; iso = isotropic, rel = E8-style relative"
    )
    for run, step in jobs:
        common.CKPT_RUN = f"{R}/{run}"
        t0 = time.time()
        model, ck = build(int(step))
        model.train()
        params = list(model.parameters())
        tm = model.get_type_map()
        radii = torch.tensor(
            [covalent_radii[atomic_numbers[e]] for e in tm], dtype=torch.float64
        )
        P = {
            "data": bi,
            "O2 .465": dimer(7, [0.465]),
            "O2 .6": dimer(7, [0.6]),
            "O2 .8": dimer(7, [0.8]),
            "H2 .5": dimer(0, [0.5]),
            "ndl-pair": {
                "coord": inp["coord"][[fi]][:, pair_atoms],
                "atype": inp["atype"][[fi]][:, pair_atoms],
                "box": inp["box"][[fi]],
            },
            "cmp45": compressed_pairs(bi, radii, 0.45),
            "cmp60": compressed_pairs(bi, radii, 0.60),
        }
        f = bl["force"]
        u = f / (f.norm(dim=-1, keepdim=True) + 1e-9)
        P["coll"] = dict(bi)
        P["coll"]["coord"] = bi["coord"] - 0.15 * u
        F0 = {k: forces(model, v)[0] for k, v in P.items()}
        theta0 = [p.detach().clone() for p in params]
        mae = float((F0["data"] - bl["force"].double()).norm(dim=-1).mean())
        ndl_err = float(
            (
                forces(model, {k: v[[fi]] for k, v in inp.items()})[0][0]
                - lab["force"][fi].double()
            ).norm(dim=-1)[3]
        )
        out = {}
        for kind in ("iso", "rel"):
            rows = []
            for seed in range(3):
                g = torch.Generator().manual_seed(100 + seed)
                xi = [
                    torch.randn(p.shape, generator=g)
                    * (1.0 if kind == "iso" else p.detach().abs())
                    for p in params
                ]
                s = 0.115 / math.sqrt(sum(float(x.double().square().sum()) for x in xi))
                with torch.no_grad():
                    for p, x in zip(params, xi, strict=True):
                        p.add_(x * s)
                F1 = {k: forces(model, v)[0] for k, v in P.items()}
                with torch.no_grad():
                    for p, t in zip(params, theta0, strict=True):
                        p.copy_(t)
                rows.append({k: rms(F1[k], F0[k], P[k]["atype"]) for k in P})
            out[kind] = {k: float(np.median([r[k] for r in rows])) for k in P}
        for kind in ("iso", "rel"):
            o = out[kind]
            sd = o["data"]
            print(
                f"{run[:30]:30s} {int(step):6d} {mae:6.3f} {sd:9.2e}  "
                + " ".join(
                    f"{o[k] / sd:7.0f}"
                    for k in (
                        "O2 .465",
                        "O2 .6",
                        "O2 .8",
                        "H2 .5",
                        "ndl-pair",
                        "cmp45",
                        "cmp60",
                        "coll",
                    )
                )
                + f"   || {float(F0['O2 .465'][0, 1, 0]):+9.2e} {ndl_err:8.1e} {float(F0['cmp45'].norm(dim=-1).max()):9.2e}  {kind}  ({time.time() - t0:.0f}s)"
            )


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, T201, ANN001, ANN201, B905
"""
Whether the label-free curvature criterion identifies frames on which a
pretrained model's force error explodes.

A production DPA4-Pro checkpoint shows a force MAE of 0.1 eV/Å but a force
RMSE of 61 eV/Å on the 1% OMat24 validation subset: a handful of frames carry
enormous errors.  Every frame of that subset is evaluated here with the model
in float32, recording the force error against the DFT label (which needs the
label) and the per-atom force-probe curvature ratio against the physical
ceiling ``k0 + |F_i|/rho0`` (which does not).  If the high-error frames are the
flagged ones, the criterion locates in-distribution defects without labels and
a training-time hinge would have acted on them.
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


def probe(model, pos, cell, atype, eps, device):
    c = torch.tensor(pos, dtype=torch.float32, device=device).reshape(1, -1, 3)
    b = torch.tensor(cell, dtype=torch.float32, device=device).reshape(1, 3, 3)
    t = torch.tensor(atype, dtype=torch.long, device=device).reshape(1, -1)
    out = model(c, t, box=b)
    e0 = float(out["energy"].detach().reshape(-1)[0])
    f0 = out["force"].detach().reshape(-1, 3)
    ae0 = out["atom_energy"].detach().reshape(-1)
    fn = f0.norm()
    u = f0 / fn.clamp_min(1e-12)
    out2 = model(c + eps * u, t, box=b)
    de = float((out2["atom_energy"].detach().reshape(-1) - ae0).sum())
    c_e = 2.0 * (de + eps * float(fn)) / eps**2
    g = (f0 - out2["force"].detach().reshape(-1, 3)) / eps
    fatom = f0.norm(dim=-1)
    return (
        e0,
        f0.double().cpu().numpy(),
        c_e,
        g.norm(dim=-1).double().cpu().numpy(),
        fatom.double().cpu().numpy(),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--lmdb", default="/data/Datasets/LMDB/OMat24/sub001val.lmdb")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--k0", type=float, default=300.0)
    ap.add_argument("--rho0", type=float, default=0.1)
    ap.add_argument(
        "--state-dict",
        type=Path,
        default=None,
        help="optional state dict loaded over the checkpoint's model",
    )
    ap.add_argument(
        "--indices",
        type=Path,
        default=None,
        help="optional .npy of frame indices to scan instead of the whole subset",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, type_map = load_model(args.ckpt, args.device)
    if args.state_dict is not None:
        model.load_state_dict(torch.load(args.state_dict, map_location=args.device))
    from deepmd.dpmodel.utils.lmdb_data import (
        LmdbDataReader,
    )

    reader = LmdbDataReader(args.lmdb, type_map, batch_size=1)
    idxs = (
        list(range(0, len(reader), args.stride))
        if args.indices is None
        else [int(i) for i in np.load(args.indices)]
    )
    keys = [
        "idx",
        "natoms",
        "e_err_atom",
        "f_rmse",
        "f_maxerr",
        "fmax_label",
        "fmax_pred",
        "ratio_E",
        "ratio_L",
        "ratio_L_at_maxerr",
    ]
    rec = {k: [] for k in keys}
    print(f"{len(idxs)} frames")
    for n, i in enumerate(idxs):
        fr = reader[i]
        pos = fr["coord"].reshape(-1, 3)
        cell = fr["box"].reshape(3, 3)
        atype = fr["atype"].reshape(-1)
        flabel = fr["force"].reshape(-1, 3)
        e0, f0, c_e, lnorm, fatom = probe(
            model, pos, cell, atype, args.eps, args.device
        )
        ferr = np.linalg.norm(f0 - flabel, axis=-1)
        kappa_atom = args.k0 + fatom / args.rho0
        ratio_l = lnorm / kappa_atom
        vals = [
            i,
            len(atype),
            abs(e0 - float(fr["energy"].reshape(-1)[0])) / len(atype),
            float(np.sqrt((ferr**2).mean())),
            float(ferr.max()),
            float(np.linalg.norm(flabel, axis=-1).max()),
            float(fatom.max()),
            abs(c_e) / (args.k0 + fatom.max() / args.rho0),
            float(ratio_l.max()),
            float(ratio_l[ferr.argmax()]),
        ]
        for k, v in zip(keys, vals):
            rec[k].append(v)
        if n % 1000 == 0 or ferr.max() > 5.0:
            print(
                f"  {i:6d} N={len(atype):3d} f_rmse={vals[3]:8.3f} f_maxerr={vals[4]:8.2f} Fmax_label={vals[5]:6.1f} Fmax_pred={vals[6]:8.1f} ratio_E={vals[7]:6.2f} ratio_L={vals[8]:6.2f}",
                flush=True,
            )
    out = args.out or args.ckpt.with_suffix(".scan.npz")
    np.savez(out, **{k: np.asarray(v) for k, v in rec.items()})
    r = {k: np.asarray(v) for k, v in rec.items()}
    print(
        f"\nforce MAE-like mean err {np.mean(r['f_rmse']):.4f}, force RMSE over all atoms ~ {np.sqrt(np.average(r['f_rmse'] ** 2, weights=r['natoms'])):.3f} eV/A"
    )
    for thr in [1.0, 5.0, 20.0]:
        bad = r["f_maxerr"] > thr
        print(
            f"frames with max force error > {thr:5.1f} eV/A: {bad.sum():5d}; flagged by ratio_L>1 among them: {(r['ratio_L'][bad] > 1).mean() if bad.any() else float('nan'):.2f}; flagged among the rest: {(r['ratio_L'][~bad] > 1).mean():.4f}; ratio_L median bad {np.median(r['ratio_L'][bad]) if bad.any() else float('nan'):.2f} vs rest {np.median(r['ratio_L'][~bad]):.3f}"
        )
    o = np.argsort(-r["f_maxerr"])[:25]
    print(
        "\nworst frames: idx N f_maxerr Fmax_label Fmax_pred ratio_E ratio_L ratio_L@maxerr"
    )
    for i in o:
        print(
            f"  {r['idx'][i]:6d} {r['natoms'][i]:4d} {r['f_maxerr'][i]:9.2f} {r['fmax_label'][i]:8.1f} {r['fmax_pred'][i]:9.1f} {r['ratio_E'][i]:7.2f} {r['ratio_L'][i]:7.2f} {r['ratio_L_at_maxerr'][i]:7.2f}"
        )
    print(f"saved {out}")


if __name__ == "__main__":
    main()

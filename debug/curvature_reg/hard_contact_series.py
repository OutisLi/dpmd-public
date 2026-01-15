# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, T201
"""
Force error on the hard-contact validation frames along a checkpoint series.

The frames of the 1% validation subset whose closest pair lies below a
chosen fraction of the covalent contact distance (from the census written by
the contact analysis) are evaluated with every checkpoint of a run, and the
force RMSE and the largest per-atom force error of the group are printed per
checkpoint, so that the step at which a recipe starts to damage the hardest
contacts is read directly from the training run's own saved checkpoints.
"""

from __future__ import (
    annotations,
)

import argparse
import sys
from itertools import (
    pairwise,
)
from pathlib import (
    Path,
)

import numpy as np
import torch
from ase.data import (
    chemical_symbols,
)

sys.path.insert(0, str(Path(__file__).resolve().parent / "diagnose"))
from repro_spike import (
    load_model,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt_dir", type=Path, help="directory holding the checkpoints")
    ap.add_argument(
        "--pattern",
        default="model_ema.ckpt-{step}.pt",
        help="checkpoint file name with a {step} placeholder",
    )
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument(
        "--census",
        type=Path,
        default=Path(__file__).with_name("runs")
        / "neo_safe"
        / "contact_census_sub001val.npz",
    )
    ap.add_argument("--lmdb", default="/data/Datasets/LMDB/OMat24/sub001val.lmdb")
    ap.add_argument("--edges", type=float, nargs="+", default=[0.0, 0.5, 0.55])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from deepmd.dpmodel.utils.lmdb_data import (
        LmdbDataReader,
    )

    census = np.load(args.census)
    rel = census["rel_min"]
    groups = [
        (lo, hi, census["idx"][(rel >= lo) & (rel < hi)])
        for lo, hi in pairwise(args.edges)
    ]
    type_map = chemical_symbols[1:119]
    reader = LmdbDataReader(args.lmdb, type_map, batch_size=1)
    frames = {int(i): reader[int(i)] for _, _, idx in groups for i in idx}
    header = "  ".join(
        f"[{lo:.2f},{hi:.2f}) n={len(idx)}: RMSE / max err" for lo, hi, idx in groups
    )
    print(f"{'step':>7}  {header}")
    for step in args.steps:
        path = args.ckpt_dir / args.pattern.format(step=step)
        if not path.exists():
            continue
        model, tm = load_model(path, args.device)
        index = {name: k for k, name in enumerate(tm)}
        cells = []
        for _, _, idx in groups:
            se, n, worst = 0.0, 0, 0.0
            for i in idx:
                f = frames[int(i)]
                at = f["atype"].reshape(-1)
                c = torch.tensor(
                    f["coord"].reshape(1, -1, 3),
                    dtype=torch.float32,
                    device=args.device,
                )
                b = torch.tensor(
                    f["box"].reshape(1, 3, 3), dtype=torch.float32, device=args.device
                )
                t = torch.tensor(
                    [[index[type_map[a]] for a in at]],
                    dtype=torch.long,
                    device=args.device,
                )
                pred = (
                    model(c, t, box=b)["force"]
                    .detach()
                    .reshape(-1, 3)
                    .double()
                    .cpu()
                    .numpy()
                )
                err = pred - f["force"].reshape(-1, 3)
                se += float((err**2).sum())
                n += len(at)
                worst = max(worst, float(np.linalg.norm(err, axis=-1).max()))
            cells.append(f"{np.sqrt(se / (3 * n)):.4f} / {worst:7.2f}")
        print(
            f"{step:7d}  "
            + "  ".join(
                f"{c:>{len(h)}}" for c, h in zip(cells, header.split("  "), strict=True)
            ),
            flush=True,
        )
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

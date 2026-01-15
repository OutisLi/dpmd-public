# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, T201, B905
"""
Batched force-error scan of a checkpoint over a whole LMDB subset.

Frames are grouped by atom count and evaluated in batches in float32, which is
fast enough to cover the 10% OMat24 validation subset in minutes.  The output
lists every frame's largest per-atom force error and its largest predicted and
labelled force, so that the frames responsible for a large RMSE can be
identified and examined individually afterwards.
"""

from __future__ import (
    annotations,
)

import argparse
import sys
from collections import (
    defaultdict,
)
from pathlib import (
    Path,
)

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "diagnose"))
from repro_spike import (
    load_model,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--lmdb", default="/data/Datasets/LMDB/OMat24/sub01val.lmdb")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument(
        "--atom-budget", type=int, default=150, help="upper bound on atoms per batch"
    )
    ap.add_argument(
        "--n-random",
        type=int,
        default=0,
        help="scan this many randomly chosen frames instead of all",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, type_map = load_model(args.ckpt, args.device)
    from deepmd.dpmodel.utils.lmdb_data import (
        LmdbDataReader,
    )

    reader = LmdbDataReader(args.lmdb, type_map, batch_size=1)
    nlocs = np.asarray(reader.frame_nlocs)
    chosen = np.arange(len(nlocs))
    if args.n_random > 0:
        chosen = np.sort(
            np.random.default_rng(args.seed).choice(
                len(nlocs), size=min(args.n_random, len(nlocs)), replace=False
            )
        )
    groups = defaultdict(list)
    for i in chosen:
        groups[int(nlocs[i])].append(int(i))
    keys = [
        "idx",
        "natoms",
        "e_err_atom",
        "f_rmse",
        "f_mae",
        "f_component_mae",
        "f_maxerr",
        "fmax_label",
        "fmax_pred",
        "argmax_atom",
    ]
    rec = {k: [] for k in keys}
    done = 0
    out = args.out or args.ckpt.with_suffix(".fastscan.npz")
    for n, idxs in sorted(groups.items()):
        bs = max(1, min(args.batch, args.atom_budget // n))
        for s in range(0, len(idxs), bs):
            chunk = idxs[s : s + bs]
            frames = [reader[i] for i in chunk]
            c = torch.tensor(
                np.stack([f["coord"].reshape(-1, 3) for f in frames]),
                dtype=torch.float32,
                device=args.device,
            )
            b = torch.tensor(
                np.stack([f["box"].reshape(3, 3) for f in frames]),
                dtype=torch.float32,
                device=args.device,
            )
            t = torch.tensor(
                np.stack([f["atype"].reshape(-1) for f in frames]),
                dtype=torch.long,
                device=args.device,
            )
            try:
                pred = model(c, t, box=b)
                e = pred["energy"].detach().reshape(-1).double().cpu().numpy()
                fp = (
                    pred["force"]
                    .detach()
                    .reshape(len(chunk), -1, 3)
                    .double()
                    .cpu()
                    .numpy()
                )
                del pred
            except torch.OutOfMemoryError:
                # Fall back to one frame at a time; a frame that still does not
                # fit is recorded as missing rather than aborting the scan.
                torch.cuda.empty_cache()
                e, fp = np.full(len(chunk), np.nan), np.full((len(chunk), n, 3), np.nan)
                for k in range(len(chunk)):
                    try:
                        pred = model(c[k : k + 1], t[k : k + 1], box=b[k : k + 1])
                        e[k] = float(pred["energy"].detach().reshape(-1)[0])
                        fp[k] = (
                            pred["force"].detach().reshape(-1, 3).double().cpu().numpy()
                        )
                        del pred
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        print(
                            f"  frame {chunk[k]} (N={n}) skipped: out of memory",
                            flush=True,
                        )
            for k, f in enumerate(frames):
                fl = f["force"].reshape(-1, 3)
                err = np.linalg.norm(fp[k] - fl, axis=-1)
                vals = [
                    chunk[k],
                    n,
                    abs(e[k] - float(f["energy"].reshape(-1)[0])) / n,
                    float(np.sqrt((err**2).mean())),
                    float(err.mean()),
                    float(np.abs(fp[k] - fl).mean()),
                    float(err.max()),
                    float(np.linalg.norm(fl, axis=-1).max()),
                    float(np.linalg.norm(fp[k], axis=-1).max()),
                    int(err.argmax()),
                ]
                for key, v in zip(keys, vals):
                    rec[key].append(v)
            done += len(chunk)
        w = int(np.argmax(rec["f_maxerr"]))
        print(
            f"  natoms={n:4d} frames={len(idxs):6d} done={done:7d} worst so far {rec['f_maxerr'][w]:10.2f} (idx {rec['idx'][w]}, N {rec['natoms'][w]}, Fmax_pred {rec['fmax_pred'][w]:.1f})",
            flush=True,
        )
        np.savez(out, **{k: np.asarray(v) for k, v in rec.items()})
    r = {k: np.asarray(v) for k, v in rec.items()}
    np.savez(out, **r)
    tot = r["natoms"].sum()
    print(
        f"\nframes {len(r['idx'])}, atoms {tot}; force RMSE {np.sqrt(np.sum(r['f_rmse'] ** 2 * r['natoms']) / tot):.4f} eV/A; frames maxerr>1/5/20/100: {(r['f_maxerr'] > 1).sum()}/{(r['f_maxerr'] > 5).sum()}/{(r['f_maxerr'] > 20).sum()}/{(r['f_maxerr'] > 100).sum()}"
    )
    o = np.argsort(-r["f_maxerr"])[:30]
    print("worst frames: idx N f_maxerr Fmax_label Fmax_pred e_err_atom")
    for i in o:
        print(
            f"  {r['idx'][i]:6d} {r['natoms'][i]:4d} {r['f_maxerr'][i]:10.2f} {r['fmax_label'][i]:8.1f} {r['fmax_pred'][i]:10.1f} {r['e_err_atom'][i]:8.4f}"
        )
    print(f"saved {out}")


if __name__ == "__main__":
    main()

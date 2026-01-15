# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201
"""
Shortest contact of every element pair in an LMDB training set.

A fixed random subset of frames is scanned; for every frame the minimum-image distance of every
pair of real atoms is computed, and for every ordered-free element pair (A, B) the smallest
distance found, the number of frames in which the pair occurs, and the 0.1th, 1st, 5th and 50th
percentiles of the per-frame closest A-B distance are recorded, together with the onset of the
pair's distance distribution: all A-B distances below 4 Å are histogrammed in 0.02 Å bins, and the
onset is the smallest distance whose bin holds at least a fraction (0.1 %, 1 %) of the fullest bin —
the shortest separation at which the pair occurs commonly, which bounds the pair's bonds from above,
while the rare compressed tail of the data lies below it. The table is the range source of the isolated-pair
inequality (ledger E59): inside a fraction of the shortest data contact of a pair the model's
isolated pair is required to be repulsive.

Usage: pair_contact_stats.py <lmdb> --input input.json --sample 200000 --out contacts.json
"""

from __future__ import (
    annotations,
)

import argparse
import json

import numpy as np
from ase.data import (
    atomic_numbers,
    covalent_radii,
)

from deepmd.dpmodel.utils.lmdb_data import (
    LmdbDataReader,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("lmdb")
    ap.add_argument(
        "--input", required=True, help="training input whose type_map is used"
    )
    ap.add_argument("--sample", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    type_map = json.load(open(args.input))["model"]["type_map"]
    nt = len(type_map)
    reader = LmdbDataReader(args.lmdb, type_map, "auto")
    rng = np.random.default_rng(args.seed)
    n = int(reader.nframes)
    picks = np.sort(rng.choice(n, size=min(args.sample, n), replace=False))
    print(f"frames in the set: {n}; scanned: {len(picks)}", flush=True)
    dmin = np.full((nt, nt), np.inf)
    edges = np.arange(0.0, 4.0 + 1e-9, 0.02)
    hist = np.zeros((nt, nt, len(edges) - 1), dtype=np.int64)
    nfr = np.zeros((nt, nt), dtype=np.int64)
    closest: dict[tuple[int, int], list[float]] = {}
    for count, idx in enumerate(picks, 1):
        fr = reader[int(idx)]
        atype = np.asarray(fr["atype"]).reshape(-1)
        real = atype >= 0
        coord = np.asarray(fr["coord"]).reshape(-1, 3)[real]
        atype = atype[real]
        cell = np.asarray(fr["box"]).reshape(3, 3)
        d = coord[:, None, :] - coord[None, :, :]
        frac = d @ np.linalg.inv(cell)
        frac -= np.round(frac)
        dist = np.linalg.norm(frac @ cell, axis=-1)
        np.fill_diagonal(dist, np.inf)
        # per frame: the closest distance of every element pair present
        ti = np.minimum(atype[:, None], atype[None, :])
        tj = np.maximum(atype[:, None], atype[None, :])
        key = ti * nt + tj
        flat_key, flat_d = key.reshape(-1), dist.reshape(-1)
        near = flat_d < 4.0
        if near.any():
            kk, bb = (
                flat_key[near],
                np.minimum((flat_d[near] / 0.02).astype(np.int64), len(edges) - 2),
            )
            np.add.at(hist, (kk // nt, kk % nt, bb), 1)
        order = np.argsort(flat_d)
        seen: set[int] = set()
        for o in order:
            k = int(flat_key[o])
            if k in seen:
                continue
            seen.add(k)
            dd = float(flat_d[o])
            if not np.isfinite(dd):
                break
            a, b = divmod(k, nt)
            nfr[a, b] += 1
            if dd < dmin[a, b]:
                dmin[a, b] = dd
            closest.setdefault((a, b), []).append(dd)
        if count % 20000 == 0:
            print(f"  {count} scanned", flush=True)
    table = {}
    for (a, b), vals in closest.items():
        arr = np.asarray(vals)
        contact = (
            covalent_radii[atomic_numbers[type_map[a]]]
            + covalent_radii[atomic_numbers[type_map[b]]]
        )
        h = hist[a, b]
        onset = {}
        for tag, frac in (("onset001", 0.001), ("onset01", 0.01)):
            ok = np.where(h >= max(frac * h.max(), 1.0))[0] if h.max() > 0 else []
            onset[tag] = float(edges[ok[0]]) if len(ok) else 0.0
        table[f"{type_map[a]}-{type_map[b]}"] = {
            "pairs_below_4A": int(h.sum() // 2 if a == b else h.sum()),
            **onset,
            "min": float(arr.min()),
            "p001": float(np.percentile(arr, 0.1)),
            "p1": float(np.percentile(arr, 1.0)),
            "p5": float(np.percentile(arr, 5.0)),
            "p50": float(np.percentile(arr, 50.0)),
            "frames": len(arr),
            "contact": float(contact),
        }
    json.dump(
        {"lmdb": args.lmdb, "scanned": len(picks), "pairs": table},
        open(args.out, "w"),
        indent=1,
    )
    print(f"{len(table)} element pairs recorded to {args.out}")
    for p in "H-H,O-O,N-N,C-C,H-O,H-Li,Li-O,Cl-Na,Mg-O,Al-Al,O-Si,Ca-O,O-Ti,Fe-O,Cu-Cu,S-Zn,As-Ga,Ag-Ag,Au-Au,H-Pb,Cr-Cr,Mo-Mo,F-K,Ni-Ni".split(
        ","
    ):
        a, b = p.split("-")
        k = f"{a}-{b}" if type_map.index(a) <= type_map.index(b) else f"{b}-{a}"
        if k in table:
            t = table[k]
            print(
                f"{k:6s} contact {t['contact']:.2f} A: shortest {t['min']:.3f} A ({t['min'] / t['contact']:.2f} contacts), onset 0.1%/1%: {t['onset001']:.2f} {t['onset01']:.2f} A, percentiles 0.1/1/5/50: {t['p001']:.3f} {t['p1']:.3f} {t['p5']:.3f} {t['p50']:.3f}, frames {t['frames']}, pairs<4A {t['pairs_below_4A']}"
            )
        else:
            print(f"{k:6s} absent")


if __name__ == "__main__":
    main()

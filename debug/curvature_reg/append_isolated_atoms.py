# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201
"""
Append isolated-atom frames to an LMDB training set.

Every element of the energy table that the LMDB's type map contains receives
``--copies`` frames holding that atom alone at the centre of a cubic cell of
side ``--box`` Å, labelled with the tabulated isolated-atom energy, a zero
force and a zero virial. The frames are written under new sequential keys in
the dataset's own msgpack layout, and the ``__metadata__`` record is rebuilt
with the extended ``frame_nlocs``, so the ordinary sampler and dataloader
serve them like any other frame; how often they are visited follows from
their share of the dataset, which ``--copies`` sets.
"""

from __future__ import (
    annotations,
)

import argparse
import json

import lmdb
import msgpack
import numpy as np


def encode(array: np.ndarray) -> dict:
    """Encode an array in the dataset's ``{type, shape, data}`` layout."""
    return {
        "type": str(array.dtype),
        "shape": list(array.shape),
        "data": array.tobytes(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("lmdb_dir")
    ap.add_argument("table", help="JSON {symbol: isolated-atom energy in eV}")
    ap.add_argument("--copies", type=int, required=True)
    ap.add_argument("--box", type=float, default=20.0)
    ap.add_argument("--map-size-gb", type=int, default=200)
    args = ap.parse_args()
    table = json.load(open(args.table))
    env = lmdb.open(args.lmdb_dir, map_size=args.map_size_gb * 1024**3, lock=True)
    with env.begin(write=True) as txn:
        meta = msgpack.unpackb(txn.get(b"__metadata__"), raw=False)
        type_map = list(meta["type_map"])
        fmt = meta["frame_idx_fmt"]
        nframes = int(meta["nframes"])
        elements = [
            (type_map.index(s), float(e)) for s, e in table.items() if s in type_map
        ]
        centre = np.full((1, 3), args.box / 2.0, dtype=np.float32)
        cell = (args.box * np.eye(3)).astype(np.float32)
        zero_force = np.zeros((1, 3), dtype=np.float32)
        zero_virial = np.zeros((3, 3), dtype=np.float32)
        written = 0
        for _ in range(args.copies):
            for t, e0 in elements:
                numbs = [0] * len(type_map)
                numbs[t] = 1
                frame = {
                    "atom_types": encode(np.array([t], dtype=np.int32)),
                    "coords": encode(centre),
                    "cells": encode(cell),
                    "energies": encode(np.array(e0, dtype=np.float32)),
                    "forces": encode(zero_force),
                    "virials": encode(zero_virial),
                    "atom_numbs": numbs,
                }
                txn.put(
                    format(nframes + written, fmt).encode(),
                    msgpack.packb(frame, use_bin_type=True),
                )
                written += 1
        meta["nframes"] = nframes + written
        meta["frame_nlocs"] = list(meta["frame_nlocs"]) + [1] * written
        txn.put(b"__metadata__", msgpack.packb(meta, use_bin_type=True))
    env.close()
    print(
        f"appended {written} isolated-atom frames ({len(elements)} elements x {args.copies} copies) "
        f"to {args.lmdb_dir}: {nframes} -> {nframes + written} frames, "
        f"share {written / (nframes + written):.4%}"
    )


if __name__ == "__main__":
    main()

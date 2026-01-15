# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201
"""
Write a random subset of an LMDB dataset as a new LMDB.

Records are copied verbatim (the msgpack payload is not decoded) under new
sequential keys, and the ``__metadata__`` record is rebuilt with the sliced
``frame_nlocs`` so the reader's fast initialization keeps working.  A sparse
training set carved from a densely sampled dataset is what reproduces the
"high capacity, sparse coverage" regime in which spurious surface features
appear.
"""

from __future__ import (
    annotations,
)

import argparse
import shutil
from pathlib import (
    Path,
)

import lmdb
import msgpack
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--max-atoms", type=int, default=10**9)
    ap.add_argument("--min-atoms", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = lmdb.open(args.src, readonly=True, lock=False, max_readers=1)
    with src.begin() as txn:
        meta = msgpack.unpackb(txn.get(b"__metadata__"), raw=False)
        nlocs = np.asarray(meta["frame_nlocs"])
        fmt = meta["frame_idx_fmt"]
        eligible = np.where((nlocs >= args.min_atoms) & (nlocs <= args.max_atoms))[0]
        rng = np.random.default_rng(args.seed)
        chosen = np.sort(
            rng.choice(eligible, size=min(args.n, len(eligible)), replace=False)
        )
        if Path(args.dst).exists():
            shutil.rmtree(args.dst)
        dst = lmdb.open(args.dst, map_size=64 * 1024**3)
        with dst.begin(write=True) as wtxn:
            for new_idx, old_idx in enumerate(chosen):
                raw = txn.get(format(int(old_idx), fmt).encode())
                wtxn.put(format(new_idx, fmt).encode(), raw)
            new_meta = dict(meta)
            new_meta["nframes"] = len(chosen)
            new_meta["frame_nlocs"] = [int(v) for v in nlocs[chosen]]
            wtxn.put(b"__metadata__", msgpack.packb(new_meta, use_bin_type=True))
        dst.close()
    print(
        f"wrote {len(chosen)} of {len(eligible)} eligible frames to {args.dst}; natoms quantiles {np.quantile(nlocs[chosen], [0, 0.5, 1])}"
    )


if __name__ == "__main__":
    main()

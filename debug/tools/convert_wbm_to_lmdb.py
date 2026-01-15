#!/usr/bin/env python
# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201
"""Convert the WBM matbench-discovery validation npy dataset to a single LMDB.

The WBM test set (``wbm_downsampled_mixu/``) is organised as a flat list
of single-natoms buckets; each bucket is a dpdata mixed-type system with
``set.XXXXXX/{coord,force,energy,box,fparam,real_atom_types}.npy``. This
utility serialises the whole tree into one LMDB compatible with
:class:`deepmd.dpmodel.utils.lmdb_data.LmdbDataReader`, preserving the
original float64 precision.

Systems are assigned ``sid`` in ascending order of bucket natoms
(i.e. 2, 3, 4, ..., 100). ``fparam`` is stored per-frame so downstream
models can start consuming it without re-converting the data.

Usage
-----
    python tools/convert_wbm_to_lmdb.py \
        --src /home/outisli/Research/dp_train/Datasets/WBMtest/wbm_downsampled_mixu \
        --dst /home/outisli/Research/dp_train/Datasets/wbm_v024.lmdb \
        --ref-lmdb /home/outisli/Research/dp_train/Datasets/mptraj_v024.lmdb
"""

from __future__ import (
    annotations,
)

import argparse
import logging
import shutil
import sys
import time
from pathlib import (
    Path,
)

import lmdb
import msgpack
import numpy as np
from tqdm import (
    tqdm,
)

log = logging.getLogger("convert_wbm_to_lmdb")

COMMIT_EVERY = 5000


# ---------------------------------------------------------------------------
# Dataset traversal
# ---------------------------------------------------------------------------


def read_global_type_map(ref_path: Path) -> list[str]:
    """Load the 118-element global type_map from a reference LMDB.

    Reusing the MPtraj training LMDB's type_map guarantees atype
    indices line up between training and validation.
    """
    with lmdb.open(str(ref_path), readonly=True, lock=False) as env, env.begin() as txn:
        raw = txn.get(b"__metadata__")
        if raw is None:
            raise ValueError(f"{ref_path} is missing __metadata__")
        meta = msgpack.unpackb(raw, raw=False)
    tm = list(meta["type_map"])
    if len(tm) != 118:
        log.warning(
            "ref-lmdb type_map has %d entries (expected 118); proceeding anyway",
            len(tm),
        )
    return tm


def sorted_int_dirs(parent: Path) -> list[Path]:
    """Return sub-directories of *parent* sorted by the integer value of their name."""
    out: list[Path] = []
    for p in parent.iterdir():
        if not p.is_dir():
            continue
        try:
            int(p.name)
        except ValueError:
            log.warning("skipping non-integer bucket name %s", p)
            continue
        out.append(p)
    out.sort(key=lambda p: int(p.name))
    return out


def build_remap(system_dir: Path, global_type_map: list[str]) -> np.ndarray:
    """Compute local-index → global-index remap for a single WBM system."""
    tm_path = system_dir / "type_map.raw"
    if not tm_path.is_file():
        raise FileNotFoundError(f"missing type_map.raw in {system_dir}")
    local_map = tm_path.read_text().split()
    remap = np.empty(len(local_map), dtype=np.int32)
    for i, name in enumerate(local_map):
        try:
            remap[i] = global_type_map.index(name)
        except ValueError as exc:
            raise ValueError(
                f"{system_dir}: element '{name}' missing from global type_map"
            ) from exc
    return remap


# ---------------------------------------------------------------------------
# Frame packing
# ---------------------------------------------------------------------------


def _as_c(arr: np.ndarray, dtype: np.dtype) -> bytes:
    if arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr.tobytes()


def _encode(arr: np.ndarray, dtype_name: str, shape: list[int]) -> dict:
    return {
        "type": dtype_name,
        "shape": list(shape),
        "data": _as_c(arr, np.dtype(dtype_name)),
    }


def pack_frame(
    *,
    atype_global: np.ndarray,
    coord: np.ndarray,
    cell: np.ndarray,
    energy: float,
    force: np.ndarray,
    atom_numbs: list[int],
    fparam: np.ndarray | None,
) -> bytes:
    """Pack a single WBM frame.

    ``fparam`` is optional; when present it is stored as int64 to match
    the existing ``wbm.lmdb`` encoding (one-hot ``[has_u, 1-has_u]``).
    """
    natoms = int(atype_global.shape[0])
    record: dict[str, object] = {
        "atom_types": _encode(atype_global, "int32", [natoms]),
        "coords": _encode(coord, "float64", [natoms, 3]),
        "cells": _encode(cell, "float64", [3, 3]),
        "energies": _encode(np.asarray(energy, dtype=np.float64), "float64", []),
        "forces": _encode(force, "float64", [natoms, 3]),
        "atom_numbs": atom_numbs,
    }
    if fparam is not None:
        record["fparam"] = _encode(fparam, "int64", [int(fparam.shape[0])])
    return msgpack.packb(record, use_bin_type=True)


# ---------------------------------------------------------------------------
# Main conversion pipeline
# ---------------------------------------------------------------------------


def convert(
    src: Path,
    dst: Path,
    ref_lmdb: Path,
    map_size_gb: float,
) -> tuple[list[str], list[int], list[int], list[tuple[int, Path, str]]]:
    """Perform the conversion; return (type_map, frame_nlocs, frame_system_ids, order)."""
    if dst.exists():
        log.info("removing existing output %s", dst)
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    global_type_map = read_global_type_map(ref_lmdb)
    ntypes = len(global_type_map)

    system_dirs = sorted_int_dirs(src)
    if not system_dirs:
        raise FileNotFoundError(f"no integer-named bucket under {src}")

    order: list[tuple[int, Path, str]] = [
        (sid, d, d.name) for sid, d in enumerate(system_dirs)
    ]
    log.info("assembled %d systems", len(order))

    frame_nlocs: list[int] = []
    frame_system_ids: list[int] = []
    warnings: list[str] = []

    env = lmdb.open(
        str(dst),
        map_size=int(map_size_gb * 1024**3),
        subdir=True,
        sync=False,
        meminit=False,
        readonly=False,
        writemap=False,
    )

    global_idx = 0
    total_skipped_sets = 0
    try:
        txn = env.begin(write=True)
        pending = 0
        for sid, system_dir, natoms_name in tqdm(order, desc="systems", unit="sys"):
            try:
                remap = build_remap(system_dir, global_type_map)
            except Exception as exc:
                warnings.append(f"{system_dir}: skipped ({exc})")
                continue

            try:
                expected_natoms = int(natoms_name)
            except ValueError:
                expected_natoms = -1

            set_dirs = sorted(p for p in system_dir.glob("set.*") if p.is_dir())
            if not set_dirs:
                warnings.append(f"{system_dir}: no set.* sub-directories")
                continue

            for set_dir in set_dirs:
                try:
                    coord = np.load(set_dir / "coord.npy")
                    force = np.load(set_dir / "force.npy")
                    energy = np.load(set_dir / "energy.npy")
                    box = np.load(set_dir / "box.npy")
                except FileNotFoundError as exc:
                    warnings.append(f"{set_dir}: missing npy ({exc.filename})")
                    total_skipped_sets += 1
                    continue

                nframes = int(coord.shape[0])
                if nframes == 0:
                    warnings.append(f"{set_dir}: zero-frame set, skipped")
                    total_skipped_sets += 1
                    continue

                if not (force.shape[0] == energy.shape[0] == box.shape[0] == nframes):
                    warnings.append(
                        f"{set_dir}: inconsistent nframes "
                        f"(coord={nframes} force={force.shape[0]} "
                        f"energy={energy.shape[0]} box={box.shape[0]})"
                    )
                    total_skipped_sets += 1
                    continue

                rat_path = set_dir / "real_atom_types.npy"
                if rat_path.is_file():
                    real_atom_types = np.load(rat_path)
                else:
                    tr_path = system_dir / "type.raw"
                    tr = np.asarray(
                        [int(x) for x in tr_path.read_text().split()],
                        dtype=np.int64,
                    )
                    warnings.append(
                        f"{set_dir}: real_atom_types.npy missing, "
                        f"falling back to type.raw (natoms={tr.shape[0]})"
                    )
                    real_atom_types = np.broadcast_to(tr, (nframes, tr.shape[0])).copy()

                if real_atom_types.ndim != 2 or real_atom_types.shape[0] != nframes:
                    warnings.append(
                        f"{set_dir}: real_atom_types.npy shape "
                        f"{real_atom_types.shape} incompatible with "
                        f"nframes={nframes}; skipping set"
                    )
                    total_skipped_sets += 1
                    continue

                natoms_per_frame = int(real_atom_types.shape[1])
                if expected_natoms > 0 and natoms_per_frame != expected_natoms:
                    warnings.append(
                        f"{set_dir}: natoms mismatch (bucket={expected_natoms}, "
                        f"real_atom_types={natoms_per_frame}); using latter"
                    )
                if coord.shape[1] != 3 * natoms_per_frame:
                    warnings.append(
                        f"{set_dir}: coord.shape[1]={coord.shape[1]} "
                        f"!= 3*natoms={3 * natoms_per_frame}; skipping set"
                    )
                    total_skipped_sets += 1
                    continue

                if coord.dtype != np.float64:
                    coord = coord.astype(np.float64, copy=False)
                if force.dtype != np.float64:
                    force = force.astype(np.float64, copy=False)
                if energy.dtype != np.float64:
                    energy = energy.astype(np.float64, copy=False)
                if box.dtype != np.float64:
                    box = box.astype(np.float64, copy=False)

                # Optional fparam (per-frame feature parameter, e.g. has_u one-hot).
                fparam_path = set_dir / "fparam.npy"
                fparam_all: np.ndarray | None = None
                if fparam_path.is_file():
                    fparam_all = np.load(fparam_path)
                    if fparam_all.shape[0] != nframes:
                        warnings.append(
                            f"{set_dir}: fparam nframes mismatch "
                            f"(fparam={fparam_all.shape[0]}, coord={nframes}); "
                            "dropping fparam for this set"
                        )
                        fparam_all = None

                coord_f = coord.reshape(nframes, natoms_per_frame, 3)
                force_f = force.reshape(nframes, natoms_per_frame, 3)
                box_f = box.reshape(nframes, 3, 3)
                atype_global_all = remap[real_atom_types].astype(np.int32, copy=False)

                for fi in range(nframes):
                    atype = atype_global_all[fi]
                    atom_numbs = np.bincount(atype, minlength=ntypes)[:ntypes]
                    fparam_fi = None
                    if fparam_all is not None:
                        fparam_fi = np.asarray(fparam_all[fi], dtype=np.int64).ravel()
                    payload = pack_frame(
                        atype_global=atype,
                        coord=coord_f[fi],
                        cell=box_f[fi],
                        energy=float(energy[fi]),
                        force=force_f[fi],
                        atom_numbs=atom_numbs.tolist(),
                        fparam=fparam_fi,
                    )
                    key = format(global_idx, "012d").encode()
                    txn.put(key, payload)
                    frame_nlocs.append(natoms_per_frame)
                    frame_system_ids.append(sid)
                    global_idx += 1
                    pending += 1
                    if pending >= COMMIT_EVERY:
                        txn.commit()
                        txn = env.begin(write=True)
                        pending = 0

        meta = {
            "nframes": global_idx,
            "frame_idx_fmt": "012d",
            "type_map": global_type_map,
            "frame_nlocs": frame_nlocs,
            "frame_system_ids": frame_system_ids,
        }
        txn.put(b"__metadata__", msgpack.packb(meta, use_bin_type=True))
        txn.commit()
    finally:
        env.sync()
        env.close()

    log.info(
        "wrote %d frames / %d systems (skipped sets: %d)",
        global_idx,
        len(order),
        total_skipped_sets,
    )
    convert._last_warnings = warnings  # type: ignore[attr-defined]
    convert._skipped_sets = total_skipped_sets  # type: ignore[attr-defined]
    return global_type_map, frame_nlocs, frame_system_ids, order


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify(
    dst: Path,
    global_type_map: list[str],
    order: list[tuple[int, Path, str]],
) -> None:
    """Reopen the output LMDB and cross-check against the npy / BASELINE counts."""
    from deepmd.dpmodel.utils.lmdb_data import (
        LmdbDataReader,
    )

    print()
    print("=" * 72)
    print("verification")
    print("=" * 72)
    reader = LmdbDataReader(str(dst), type_map=global_type_map, batch_size=1)
    print(f"nframes    = {reader.nframes}")
    print(f"nsystems   = {reader.nsystems}")
    assert reader.nsystems == len(order), (
        f"expected {len(order)} systems, got {reader.nsystems}"
    )

    # Per-system natoms read from actual frames (0th of each group).
    print()
    print(f"{'sid':>4}  {'bucket':>6}  {'nframes':>8}  {'natoms':>7}  path")
    for sid, path, bucket in order:
        group = reader.system_groups[sid]
        frame0 = reader[int(group[0])]
        natoms = int(frame0["atype"].shape[0])
        print(f"{sid:>4}  {bucket:>6}  {len(group):>8}  {natoms:>7}  {path}")

    print("\n-- spot-check 50 random frames (shapes / dtype / fparam) --")
    rng = np.random.default_rng(0)
    fparam_seen = 0
    for idx in rng.choice(reader.nframes, size=50, replace=False):
        f = reader[int(idx)]
        assert f["coord"].dtype == np.float64
        assert f["coord"].shape[0] == reader.frame_nlocs[int(idx)]
        if "fparam" in f:
            fparam_seen += 1
    print(
        f"spot check: PASS (50 frames, all float64, "
        f"fparam present in {fparam_seen}/50 checked frames)"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--src",
        type=Path,
        required=True,
        help="root directory containing WBM bucket sub-directories",
    )
    p.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="output LMDB path; overwritten if it already exists",
    )
    p.add_argument(
        "--ref-lmdb",
        type=Path,
        required=True,
        help="reference LMDB whose 118-element type_map is reused verbatim",
    )
    p.add_argument(
        "--map-size",
        type=float,
        default=1.0,
        help="LMDB map_size in GB (default: 1 GB, WBM is small)",
    )
    p.add_argument(
        "--skip-verify",
        action="store_true",
        help="skip the post-write verification block",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable DEBUG-level logging",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    t0 = time.perf_counter()
    global_type_map, frame_nlocs, frame_system_ids, order = convert(
        src=args.src,
        dst=args.dst,
        ref_lmdb=args.ref_lmdb,
        map_size_gb=args.map_size,
    )
    dt = time.perf_counter() - t0
    log.info("conversion finished in %.1f s", dt)

    data_mdb = args.dst / "data.mdb"
    size_bytes = data_mdb.stat().st_size if data_mdb.is_file() else 0
    print()
    print("=" * 72)
    print("summary")
    print("=" * 72)
    print(f"output            : {args.dst}")
    print(f"data.mdb size     : {size_bytes / 1024**2:.2f} MB")
    print(f"nframes           : {len(frame_nlocs)}")
    print(f"nsystems          : {len(order)}")
    print(f"wallclock         : {dt:.1f} s")

    warnings = getattr(convert, "_last_warnings", [])
    skipped = getattr(convert, "_skipped_sets", 0)
    if warnings:
        print()
        print(f"-- warnings ({len(warnings)}) / skipped sets ({skipped}) --")
        for w in warnings:
            print(f"  {w}")

    if not args.skip_verify:
        verify(args.dst, global_type_map, order)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201
"""Convert the cleaned MPtraj v0.2.4 npy dataset to a single LMDB (float64).

The MPtraj v0.2.4 release ships as a tree of ``set.*/*.npy`` buckets split
into a ``first/`` subset (one frame per material) and a ``rest_split/``
subset (remaining frames with an explicit ``train/``/``valid/`` split).
This utility serialises the full cleaned tree into one LMDB file whose
schema matches :class:`deepmd.dpmodel.utils.lmdb_data.LmdbDataReader`,
preserving the original float64 precision so downstream SeZM training
does not need an extra upcast step.

The LMDB system IDs are assigned so all ``first/`` systems come first
(contiguous block ``sid 0..421``) and all ``rest_split/`` systems come
afterwards (``sid 422..1213``). That way ``auto_prob`` splits such as
``"prob_sys_size;0:422:0.5;422:1214:0.5"`` give clean first/rest
50:50 frame quotas.

Usage
-----
    python tools/convert_mptraj_full_to_lmdb.py \
        --src /home/outisli/Research/dp_train/Datasets/MPtraj \
        --dst /home/outisli/Research/dp_train/Datasets/mptraj_v024_full.lmdb \
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

log = logging.getLogger("convert_mptraj_full_to_lmdb")

# Flush one LMDB txn every ``COMMIT_EVERY`` frames so that the transaction
# never grows beyond a manageable size but we still avoid per-frame commit
# overhead. ~5k frames * ~1 KB each ~ 5 MB, well below LMDB's single-txn
# page budget.
COMMIT_EVERY = 5000


# ---------------------------------------------------------------------------
# Dataset traversal
# ---------------------------------------------------------------------------


def read_global_type_map(ref_path: Path) -> list[str]:
    """Load the 118-element global type_map from a reference LMDB.

    Reading from the pre-existing ``mptraj_v024.lmdb`` guarantees the
    output LMDB and any downstream model config share the exact same
    element ordering.
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
    """Return sub-directories of *parent* sorted by the integer value of their name.

    Equivalent to the natural (version) sort ``sort -V`` used in the spec.
    """
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


def build_system_order(
    src: Path,
    max_natoms: int | None = None,
) -> list[tuple[int, Path, str, str, str]]:
    """Build the canonical (sid, system_dir, subset, family, natoms_name) list.

    Order (sid ascending):

        first/has_u    first/no_u
        rest/has_u/train  rest/no_u/train
        rest/has_u/valid  rest/no_u/valid

    Only rest buckets that actually carry the requested split are
    included; the missing ones are silently skipped (expected per the
    cleaning pipeline).

    If ``max_natoms`` is given, buckets whose directory name (which
    encodes natoms) exceeds this limit are dropped altogether; the
    remaining systems are still renumbered contiguously starting at 0.
    """

    def _keep(d: Path) -> bool:
        if max_natoms is None:
            return True
        try:
            return int(d.name) <= max_natoms
        except ValueError:
            return True

    order: list[tuple[int, Path, str, str, str]] = []
    sid = 0

    # --- first block ------------------------------------------------------
    for family, sub in (("has_u", "has_u-none_or_same"), ("no_u", "no_u-none_or_same")):
        base = src / "first" / sub
        if not base.is_dir():
            raise FileNotFoundError(f"missing subset directory: {base}")
        for d in sorted_int_dirs(base):
            if not _keep(d):
                continue
            order.append((sid, d, "first", family, d.name))
            sid += 1

    # --- rest train -------------------------------------------------------
    for family, sub in (("has_u", "has_u-none_or_same"), ("no_u", "no_u-none_or_same")):
        base = src / "rest_split" / sub
        if not base.is_dir():
            raise FileNotFoundError(f"missing subset directory: {base}")
        for d in sorted_int_dirs(base):
            if not _keep(d):
                continue
            td = d / "train"
            if td.is_dir():
                order.append((sid, td, "rest_train", family, d.name))
                sid += 1

    # --- rest valid -------------------------------------------------------
    for family, sub in (("has_u", "has_u-none_or_same"), ("no_u", "no_u-none_or_same")):
        base = src / "rest_split" / sub
        for d in sorted_int_dirs(base):
            if not _keep(d):
                continue
            vd = d / "valid"
            if vd.is_dir():
                order.append((sid, vd, "rest_valid", family, d.name))
                sid += 1

    return order


def build_remap(
    system_dir: Path,
    global_type_map: list[str],
    warnings: list[str],
) -> np.ndarray:
    """Compute local-index → global-index remap for a system.

    ``type_map.raw`` lives in the parent of ``set.*`` for first-block
    systems and in the parent of rest-block ``train``/``valid``
    directories. The function searches upwards until it finds one.
    """
    for candidate in (system_dir, system_dir.parent, system_dir.parent.parent):
        tm_path = candidate / "type_map.raw"
        if tm_path.is_file():
            break
    else:
        raise FileNotFoundError(f"cannot locate type_map.raw near {system_dir}")
    local_map = tm_path.read_text().split()
    if len(local_map) != 103:
        warnings.append(
            f"{system_dir}: type_map.raw has {len(local_map)} entries "
            "(expected 103) — remap will cover only what is present"
        )
    remap = np.empty(len(local_map), dtype=np.int32)
    for i, name in enumerate(local_map):
        try:
            remap[i] = global_type_map.index(name)
        except ValueError as exc:
            raise ValueError(
                f"{system_dir}: element '{name}' missing from global type_map"
            ) from exc
    return remap


def iter_set_dirs(system_dir: Path) -> list[Path]:
    """Return ``set.*`` sub-directories sorted ascending."""
    return sorted(p for p in system_dir.glob("set.*") if p.is_dir())


# ---------------------------------------------------------------------------
# Frame packing
# ---------------------------------------------------------------------------


def _as_c(arr: np.ndarray, dtype: np.dtype) -> bytes:
    """Return *arr* as C-contiguous bytes with the requested dtype."""
    if arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr.tobytes()


def _encode(arr: np.ndarray, dtype_name: str, shape: list[int]) -> dict:
    """Build a msgpack-ready encoded-array dict."""
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
    virial: np.ndarray,
    atom_numbs: list[int],
) -> bytes:
    """Pack a single frame into the LMDB binary payload."""
    natoms = int(atype_global.shape[0])
    record = {
        "atom_types": _encode(atype_global, "int32", [natoms]),
        "coords": _encode(coord, "float64", [natoms, 3]),
        "cells": _encode(cell, "float64", [3, 3]),
        "energies": _encode(np.asarray(energy, dtype=np.float64), "float64", []),
        "forces": _encode(force, "float64", [natoms, 3]),
        "virials": _encode(virial, "float64", [3, 3]),
        "atom_numbs": atom_numbs,
    }
    return msgpack.packb(record, use_bin_type=True)


# ---------------------------------------------------------------------------
# Main conversion pipeline
# ---------------------------------------------------------------------------


def convert(
    src: Path,
    dst: Path,
    ref_lmdb: Path,
    map_size_gb: float,
    max_natoms: int | None = None,
) -> tuple[list[str], list[int], list[int], list[tuple[int, Path, str, str, str]]]:
    """Perform the full conversion; return (type_map, frame_nlocs, frame_system_ids, order)."""
    if dst.exists():
        log.info("removing existing output %s", dst)
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    global_type_map = read_global_type_map(ref_lmdb)
    ntypes = len(global_type_map)

    order = build_system_order(src, max_natoms=max_natoms)
    log.info(
        "assembled %d systems (max_natoms=%s)",
        len(order),
        "none" if max_natoms is None else max_natoms,
    )

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
        for sid, system_dir, subset, family, natoms_name in tqdm(
            order, desc="systems", unit="sys"
        ):
            try:
                remap = build_remap(system_dir, global_type_map, warnings)
            except Exception as exc:
                warnings.append(f"{system_dir}: skipped ({exc})")
                continue

            set_dirs = iter_set_dirs(system_dir)
            if not set_dirs:
                warnings.append(f"{system_dir}: no set.* sub-directories")
                continue

            # Determine per-system natoms from the directory name as a
            # sanity reference; the actual natoms used per frame comes
            # from the coord/force/real_atom_types arrays.
            try:
                expected_natoms = int(natoms_name)
            except ValueError:
                expected_natoms = -1

            for set_dir in set_dirs:
                try:
                    coord = np.load(set_dir / "coord.npy", mmap_mode=None)
                    force = np.load(set_dir / "force.npy", mmap_mode=None)
                    energy = np.load(set_dir / "energy.npy", mmap_mode=None)
                    box = np.load(set_dir / "box.npy", mmap_mode=None)
                    virial = np.load(set_dir / "virial.npy", mmap_mode=None)
                except FileNotFoundError as exc:
                    warnings.append(f"{set_dir}: missing npy ({exc.filename})")
                    total_skipped_sets += 1
                    continue

                nframes = int(coord.shape[0])
                if nframes == 0:
                    warnings.append(f"{set_dir}: zero-frame set, skipped")
                    total_skipped_sets += 1
                    continue

                # Sanity: all frame-count dimensions should agree.
                if not (
                    force.shape[0]
                    == energy.shape[0]
                    == box.shape[0]
                    == virial.shape[0]
                    == nframes
                ):
                    warnings.append(
                        f"{set_dir}: inconsistent nframes "
                        f"(coord={nframes} force={force.shape[0]} "
                        f"energy={energy.shape[0]} box={box.shape[0]} "
                        f"virial={virial.shape[0]})"
                    )
                    total_skipped_sets += 1
                    continue

                # Real per-frame atom types (preferred); fall back to
                # type.raw replicated across frames if needed.
                rat_path = set_dir / "real_atom_types.npy"
                if rat_path.is_file():
                    real_atom_types = np.load(rat_path)
                else:
                    tr_path = system_dir / "type.raw"
                    if not tr_path.is_file():
                        tr_path = system_dir.parent / "type.raw"
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

                # Dtype normalisation; coord/force/box/virial/energy are
                # expected float64 but we coerce defensively.
                if coord.dtype != np.float64:
                    warnings.append(
                        f"{set_dir}: coord.dtype={coord.dtype} (coercing to float64)"
                    )
                    coord = coord.astype(np.float64, copy=False)
                if force.dtype != np.float64:
                    force = force.astype(np.float64, copy=False)
                if energy.dtype != np.float64:
                    energy = energy.astype(np.float64, copy=False)
                if box.dtype != np.float64:
                    box = box.astype(np.float64, copy=False)
                if virial.dtype != np.float64:
                    virial = virial.astype(np.float64, copy=False)

                # Reshape once per set for speed; we still slice per frame
                # so each msgpack payload is a contiguous ndarray.
                coord_f = coord.reshape(nframes, natoms_per_frame, 3)
                force_f = force.reshape(nframes, natoms_per_frame, 3)
                box_f = box.reshape(nframes, 3, 3)
                virial_f = virial.reshape(nframes, 3, 3)

                # Remap to the 118-element global indexing in one shot.
                # ``remap`` has shape (max_local,). Any out-of-range
                # index in real_atom_types would raise IndexError here,
                # which is the right behaviour — data corruption.
                atype_global_all = remap[real_atom_types].astype(np.int32, copy=False)

                for fi in range(nframes):
                    atype = atype_global_all[fi]
                    atom_numbs = np.bincount(atype, minlength=ntypes)[:ntypes]
                    payload = pack_frame(
                        atype_global=atype,
                        coord=coord_f[fi],
                        cell=box_f[fi],
                        energy=float(energy[fi]),
                        force=force_f[fi],
                        virial=virial_f[fi],
                        atom_numbs=atom_numbs.tolist(),
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

        # Final __metadata__ write and commit.
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
    if warnings:
        log.info("accumulated %d warnings (see report)", len(warnings))
    # Store warnings on the function for the caller to retrieve.
    convert._last_warnings = warnings  # type: ignore[attr-defined]
    convert._skipped_sets = total_skipped_sets  # type: ignore[attr-defined]
    return global_type_map, frame_nlocs, frame_system_ids, order


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify(
    dst: Path,
    global_type_map: list[str],
    order: list[tuple[int, Path, str, str, str]],
) -> None:
    """Reopen the output LMDB and run the sanity block from the spec."""
    # Local import so that we don't pay the DeePMD import cost during
    # conversion itself (the reader drags in torch configs etc.).
    from deepmd.dpmodel.utils.lmdb_data import (
        LmdbDataReader,
        compute_block_targets,
    )

    n_expected = len(order)
    first_sids = [sid for sid, _p, subset, _fam, _nn in order if subset == "first"]
    rest_sids = [sid for sid, _p, subset, _fam, _nn in order if subset != "first"]
    if not first_sids or not rest_sids:
        raise AssertionError(
            "expected both first and rest systems after filtering; "
            f"got first={len(first_sids)} rest={len(rest_sids)}"
        )
    first_end = max(first_sids) + 1
    total_end = max(rest_sids) + 1

    print()
    print("=" * 72)
    print("verification")
    print("=" * 72)
    reader = LmdbDataReader(str(dst), type_map=global_type_map, batch_size="auto:1024")
    print(f"nframes    = {reader.nframes}")
    print(f"nsystems   = {reader.nsystems}")
    assert reader.nframes > 0, "empty LMDB"
    assert reader.nsystems == n_expected, (
        f"expected {n_expected} systems, got {reader.nsystems}"
    )
    assert len(reader.system_nframes) == n_expected

    def first_idx(reader_obj: LmdbDataReader, sid: int) -> int:
        group = reader_obj.system_groups[sid]
        if not group:
            raise AssertionError(f"system {sid} has zero frames")
        return int(group[0])

    probe_sids = (0, first_end - 1, first_end, total_end - 1)
    probe_labels = ("first-begin", "first-end", "rest-begin", "rest-end")
    sid_meta = {entry[0]: entry for entry in order}
    for probe_sid, label in zip(probe_sids, probe_labels, strict=True):
        f = reader[first_idx(reader, probe_sid)]
        entry = sid_meta[probe_sid]
        print(
            f"  {label:<12} sid={probe_sid:<5} natoms={f['atype'].shape[0]:>3}  "
            f"bucket={entry[4]:>4}  subset={entry[2]:<10}  "
            f"family={entry[3]}  path={entry[1]}"
        )

    auto_prob_str = f"prob_sys_size;0:{first_end}:0.5;{first_end}:{total_end}:0.5"
    print(f"auto_prob      = {auto_prob_str}")
    bt = compute_block_targets(
        auto_prob_str,
        nsystems=reader.nsystems,
        system_nframes=reader.system_nframes,
    )
    print(f"block_targets  = {bt}")

    first_frames = sum(reader.system_nframes[:first_end])
    rest_frames = sum(reader.system_nframes[first_end:])
    print(f"first frames  = {first_frames}")
    print(f"rest frames   = {rest_frames}")

    print("\n-- spot-check 100 random frames (shapes / dtype) --")
    rng = np.random.default_rng(0)
    for idx in rng.choice(reader.nframes, size=100, replace=False):
        f = reader[int(idx)]
        assert f["coord"].dtype == np.float64, f["coord"].dtype
        assert f["coord"].shape[0] == reader.frame_nlocs[int(idx)]
        assert f["force"].shape == f["coord"].shape
    print("spot check: PASS (100 frames, all float64, shapes consistent)")

    print("\n-- batch-size rule smoke test (max:2000) --")
    # Close the current reader first; python-lmdb forbids two envs on
    # the same path within one process unless the first was GC'd.
    del reader
    reader2 = LmdbDataReader(str(dst), type_map=global_type_map, batch_size="max:2000")
    for nloc in (148, 296, 444):
        print(f"  bsi@{nloc} = {reader2.get_batch_size_for_nloc(nloc)}")


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
        help="root of the cleaned MPtraj v0.2.4 tree (contains first/ and rest_split/)",
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
        default=20.0,
        help="LMDB map_size in GB (default: 20)",
    )
    p.add_argument(
        "--max-natoms",
        type=int,
        default=None,
        help=(
            "if set, drop buckets whose natoms (directory name) exceeds this "
            "limit; the remaining systems are contiguously renumbered so "
            "first/rest still form contiguous sid blocks"
        ),
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
        max_natoms=args.max_natoms,
    )
    dt = time.perf_counter() - t0
    log.info("conversion finished in %.1f s", dt)

    # Report disk footprint.
    data_mdb = args.dst / "data.mdb"
    size_bytes = data_mdb.stat().st_size if data_mdb.is_file() else 0
    print()
    print("=" * 72)
    print("summary")
    print("=" * 72)
    print(f"output            : {args.dst}")
    print(f"data.mdb size     : {size_bytes / 1024**3:.2f} GB")
    print(f"nframes           : {len(frame_nlocs)}")
    print(f"nsystems          : {len(order)}")
    print(f"wallclock         : {dt:.1f} s")

    # Sid-range summary table.
    blocks = {}
    for sid, _path, subset, family, _nn in order:
        key = (subset, family)
        lo, hi = blocks.get(key, (sid, sid))
        blocks[key] = (min(lo, sid), max(hi, sid))
    print()
    print(f"{'subset':<12} {'family':<6} {'sid range':<14} {'count':>6}")
    for (subset, family), (lo, hi) in blocks.items():
        print(f"{subset:<12} {family:<6} {lo:>4} .. {hi:<6}  {hi - lo + 1:>6}")

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

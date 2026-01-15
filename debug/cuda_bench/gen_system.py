# SPDX-License-Identifier: LGPL-3.0-or-later
"""Diamond-carbon supercell writer shared by the LAMMPS and GPUMD scans.

A near-cubic supercell (8-atom conventional cell, a = 3.567 A) is tiled to the
requested atom count and written either as a LAMMPS data file or a GPUMD
extended-XYZ file from the *same* geometry, so the deepmd (LAMMPS) and NEP
(GPUMD) throughput curves are measured on identical systems. The spin variant
carries the identical coordinates and adds a per-atom magnetic moment, so a
spin-conditioned model is measured on the neighborhood of the energy runs. A small Gaussian
jitter breaks the perfect lattice so the neighbor list is representative of an
MD snapshot.
"""

from __future__ import (
    annotations,
)

import fcntl
import os
from typing import (
    TYPE_CHECKING,
)

import numpy as np

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

#: Magnitude of every benchmark moment, in Bohr magnetons.
SPIN_MOMENT = 2.0
#: Seed of the benchmark moment directions.
SPIN_SEED = 20240729

SMALL_TARGETS = (
    512,
    1_000,
    2_000,
    4_000,
    8_000,
    16_000,
    32_000,
    64_000,
    128_000,
    256_000,
    500_000,
)
SYSTEM_CACHE_VERSION = "diamond_v1"

A = 3.567  # diamond conventional lattice constant, angstrom
BASIS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.25, 0.25, 0.25],
        [0.25, 0.75, 0.75],
        [0.75, 0.25, 0.75],
        [0.75, 0.75, 0.25],
    ],
    dtype=np.float64,
)


def reps(target: int) -> tuple[int, int, int]:
    """Near-cubic ``(nx, ny, nz)`` whose ``8*nx*ny*nz`` best matches ``target``."""
    base = max(2, round((target / 8) ** (1.0 / 3.0)))
    best = None
    for nx in range(max(2, base - 2), base + 3):
        for ny in range(max(2, base - 2), base + 3):
            for nz in range(max(2, base - 2), base + 3):
                d = abs(8 * nx * ny * nz - target)
                if best is None or d < best[0]:
                    best = (d, nx, ny, nz)
    return best[1], best[2], best[3]


def atom_count(target: int) -> int:
    """Return the realized atom count for a requested system size.

    Parameters
    ----------
    target
        Requested atom count.

    Returns
    -------
    int
        Atom count of the nearest diamond supercell.
    """
    nx, ny, nz = reps(target)
    return 8 * nx * ny * nz


def build(target: int, jitter: float = 0.03, seed: int = 0):  # noqa: ANN201
    """Return ``(pos (N, 3), box (3,))`` for a diamond supercell near ``target``."""
    nx, ny, nz = reps(target)
    cells = np.stack(
        np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), -1
    ).reshape(-1, 3)
    rep = np.array([nx, ny, nz], dtype=np.float64)
    frac = (BASIS[None] + cells[:, None]).reshape(-1, 3) / rep
    box = A * rep
    pos = frac * box
    rng = np.random.default_rng(seed)
    pos = (pos + jitter * rng.standard_normal(pos.shape)) % box
    return pos, box


def write_data(path: str, pos: np.ndarray, box: np.ndarray) -> int:
    """Write a LAMMPS data file (single carbon atom type); return atom count."""
    n = pos.shape[0]
    lx, ly, lz = box
    with open(path, "w") as f:
        f.write(f"diamond C {n} atoms\n\n{n} atoms\n1 atom types\n\n")
        f.write(f"0.0 {lx:.6f} xlo xhi\n0.0 {ly:.6f} ylo yhi\n0.0 {lz:.6f} zlo zhi\n\n")
        f.write("Masses\n\n1 12.011\n\nAtoms\n\n")
        f.write(
            "".join(
                f"{i} 1 {x:.6f} {y:.6f} {z:.6f}\n" for i, (x, y, z) in enumerate(pos, 1)
            )
        )
    return n


def write_spin_data(path: str, pos: np.ndarray, box: np.ndarray) -> int:
    """Write an ``atom_style spin`` data file; return atom count.

    The Atoms section carries ``id type x y z spx spy spz sp``, where the three
    spin components are a direction that LAMMPS normalizes on read and ``sp``
    is the moment magnitude. Directions are drawn from a fixed seed so a cached
    system is reproducible, and they are isotropic rather than collinear so the
    bond-projected spin families of the descriptor stay populated.
    """
    n = pos.shape[0]
    lx, ly, lz = box
    directions = np.random.default_rng(SPIN_SEED).normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    with open(path, "w") as f:
        f.write(f"diamond C {n} atoms\n\n{n} atoms\n1 atom types\n\n")
        f.write(f"0.0 {lx:.6f} xlo xhi\n0.0 {ly:.6f} ylo yhi\n0.0 {lz:.6f} zlo zhi\n\n")
        f.write("Masses\n\n1 12.011\n\nAtoms\n\n")
        f.write(
            "".join(
                f"{i} 1 {x:.6f} {y:.6f} {z:.6f} "
                f"{sx:.6f} {sy:.6f} {sz:.6f} {SPIN_MOMENT:.6f}\n"
                for i, ((x, y, z), (sx, sy, sz)) in enumerate(
                    zip(pos, directions, strict=True), 1
                )
            )
        )
    return n


def write_xyz(path: str, pos: np.ndarray, box: np.ndarray) -> int:
    """Write a GPUMD extended-XYZ file (same geometry); return atom count."""
    n = pos.shape[0]
    lx, ly, lz = box
    lattice = f"{lx:.6f} 0 0 0 {ly:.6f} 0 0 0 {lz:.6f}"
    with open(path, "w") as f:
        f.write(f"{n}\n")
        f.write(f'pbc="T T T" Lattice="{lattice}" Properties=species:S:1:pos:R:3\n')
        f.write("".join(f"C {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in pos))
    return n


def _ensure_cached(
    target: int,
    directory: str,
    extension: str,
    writer: Callable[[str, np.ndarray, np.ndarray], int],
) -> tuple[str, int]:
    """Return a shared cached system file, building a missing one under a lock.

    A process lock serializes construction, and the completed file is published
    atomically, so concurrent benchmark workers never observe partial
    coordinates.

    Parameters
    ----------
    target
        Requested atom count before diamond-supercell rounding.
    directory
        Shared system-cache directory.
    extension
        File extension identifying the format, which also separates the caches.
    writer
        Format writer returning the realized atom count.

    Returns
    -------
    tuple[str, int]
        Absolute file path and realized atom count.
    """
    n_atoms = atom_count(target)
    os.makedirs(directory, exist_ok=True)
    path = os.path.abspath(
        os.path.join(directory, f"{SYSTEM_CACHE_VERSION}_{n_atoms}.{extension}")
    )
    lock_path = path + ".lock"
    with open(lock_path, "a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not os.path.exists(path):
            positions, box = build(target)
            temporary = f"{path}.{os.getpid()}.tmp"
            realized = writer(temporary, positions, box)
            if realized != n_atoms:
                raise RuntimeError(
                    "Diamond size changed during construction: "
                    f"expected {n_atoms}, got {realized}"
                )
            os.replace(temporary, path)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return path, n_atoms


def ensure_data(target: int, directory: str) -> tuple[str, int]:
    """Return a shared cached LAMMPS data file for one target size."""
    return _ensure_cached(target, directory, "data", write_data)


def ensure_spin_data(target: int, directory: str) -> tuple[str, int]:
    """Return a shared cached ``atom_style spin`` data file for one target size."""
    return _ensure_cached(target, directory, "spin.data", write_spin_data)


def ensure_xyz(target: int, directory: str) -> tuple[str, int]:
    """Return a shared cached extended-XYZ file for one target size."""
    return _ensure_cached(target, directory, "xyz", write_xyz)

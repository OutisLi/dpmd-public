# SPDX-License-Identifier: LGPL-3.0-or-later
"""Diamond-carbon supercell writer shared by the LAMMPS and GPUMD scans.

A near-cubic supercell (8-atom conventional cell, a = 3.567 A) is tiled to the
requested atom count and written either as a LAMMPS data file or a GPUMD
extended-XYZ file from the *same* geometry, so the deepmd (LAMMPS) and NEP
(GPUMD) throughput curves are measured on identical systems. A small Gaussian
jitter breaks the perfect lattice so the neighbor list is representative of an
MD snapshot.
"""

from __future__ import (
    annotations,
)

import numpy as np

# Shared size schedule for the LAMMPS and GPUMD scans; each stops at its first
# out-of-memory (memory is monotonic in system size). A ~2x ramp resolves the
# small-system launch-overhead regime, then a uniform 500k-atom stride locates
# each curve's out-of-memory ceiling finely enough to compare across models.
# ``build`` snaps every target to the nearest diamond supercell (8 n^3 atoms).
TARGETS = [
    512,
    1000,
    2000,
    4000,
    8000,
    16000,
    32000,
    64000,
    128000,
    256000,
    *range(500_000, 10_000_001, 500_000),
]

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

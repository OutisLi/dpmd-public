# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Compare a LAMMPS dpa4spin/kk single point with the eager native-spin model.

``in_spin_point.lammps`` dumps the configuration LAMMPS read together with the
force and the magnetic force ``fm`` of the patched Newtonian adapter, which
must be the raw ``-dE/dS``. This script rebuilds the same randomized model as
``freeze_spin_random.py`` (same preset, seed and scale) without compression,
evaluates it on the dumped configuration, and reports the energy, force and
magnetic-force differences together with the magnitude ranges of the LAMMPS
outputs.

Usage
-----
    BENCH_OP_LIB=.../libdeepmd_op_pt.so python run_with_oplib.py \
        check_spin_point.py --preset dpa4c-nano-v20260911 \
        --dump point_nano.dump --log log_nano.lammps \
        --data systems/diamond_v1_512.spin.data
"""

from __future__ import (
    annotations,
)

import argparse
import os
import re
import sys

import numpy as np
import torch  # noqa: TID253

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import freeze_dpa4c
from freeze_spin_random import (
    REDRAW_SCALE,
    REDRAW_SEED,
    redraw_zero_parameters,
)

from deepmd.pt_expt.model.get_model import (  # noqa: TID253
    get_model,
)


def read_dump(path: str) -> np.ndarray:
    """Return the per-atom rows of a single-frame custom dump, sorted by ID."""
    with open(path, encoding="utf-8") as stream:
        lines = stream.read().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("ITEM: ATOMS"))
    rows = np.array([[float(v) for v in line.split()] for line in lines[start + 1 :]])
    return rows[np.argsort(rows[:, 0])]


def read_box(path: str) -> np.ndarray:
    """Return the orthogonal box lengths of a LAMMPS data file."""
    lengths = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            match = re.match(r"\s*(\S+)\s+(\S+)\s+[xyz]lo\s+[xyz]hi", line)
            if match:
                lengths.append(float(match.group(2)) - float(match.group(1)))
            if len(lengths) == 3:
                break
    return np.array(lengths)


def read_pe(path: str) -> float:
    """Return the step-0 potential energy printed by ``thermo_style custom step pe``."""
    with open(path, encoding="utf-8") as stream:
        lines = stream.read().splitlines()
    header = next(
        i for i, line in enumerate(lines) if line.split()[:2] == ["Step", "PotEng"]
    )
    return float(lines[header + 1].split()[1])


def main() -> None:
    """Evaluate the eager model on the dumped configuration and compare."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--dump", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    # === Step 1. Rebuild the randomized model without compression ===
    config = freeze_dpa4c.model_config(
        channels=None,
        fitting_width=64,
        fitting_depth=3,
        lmax=2,
        radial_modes=0,
        spin=True,
        charge_state=False,
        default_charge_spin=[0.0, 1.0],
        bridging_window=None,
        preset=args.preset,
    )
    model = get_model(config).eval()
    redraw_zero_parameters(model, REDRAW_SEED, REDRAW_SCALE)

    # === Step 2. Evaluate on the configuration LAMMPS read ===
    # Columns: id x y z sp spx spy spz fx fy fz fmx fmy fmz.
    rows = read_dump(args.dump)
    n_atoms = rows.shape[0]
    spin = rows[:, 5:8] * rows[:, 4:5]
    coord = torch.tensor(rows[None, :, 1:4], dtype=torch.float64).reshape(1, -1)
    atype = torch.full((1, n_atoms), config["type_map"].index("C"), dtype=torch.int64)
    moment = torch.tensor(spin[None], dtype=torch.float64).reshape(1, -1)
    box = torch.tensor(np.diag(read_box(args.data)).reshape(1, 9), dtype=torch.float64)
    output = model(coord, atype, moment, box=box)
    energy = output["energy"].item()
    force = output["force"].detach().numpy().reshape(-1, 3)
    force_mag = output["force_mag"].detach().numpy().reshape(-1, 3)

    # === Step 3. Compare ===
    lmp_energy = read_pe(args.log)
    lmp_force = rows[:, 8:11]
    lmp_force_mag = rows[:, 11:14]
    finite = bool(np.isfinite(rows).all() and np.isfinite(lmp_energy))
    f_norm = np.linalg.norm(lmp_force, axis=1)
    fm_norm = np.linalg.norm(lmp_force_mag, axis=1)
    f_error = np.abs(lmp_force - force).max()
    fm_error = np.abs(lmp_force_mag - force_mag).max()
    print(  # noqa: T201
        f"{args.preset}: finite={finite} N={n_atoms} "
        f"E_lmp={lmp_energy:.6f} E_eager={energy:.6f} eV "
        f"dE/atom={(lmp_energy - energy) / n_atoms:.2e} | "
        f"|F| min/mean/max={f_norm.min():.4f}/{f_norm.mean():.4f}/{f_norm.max():.4f} eV/A "
        f"max|dF|={f_error:.2e} | "
        f"|fm| min/mean/max={fm_norm.min():.4f}/{fm_norm.mean():.4f}/{fm_norm.max():.4f} eV/muB "
        f"max|dfm|={fm_error:.2e} rel={fm_error / fm_norm.max():.2e}",
        flush=True,
    )


if __name__ == "__main__":
    main()

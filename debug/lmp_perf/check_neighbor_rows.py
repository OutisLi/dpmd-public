# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compare complete GPU neighbor rows from frozen owned and ghost records."""

from __future__ import (
    annotations,
)

import argparse
import json
import os
import subprocess
from pathlib import (
    Path,
)

CASES = (
    ("full17", "full", 17, 1.5, False),
    ("full99", "full", 99, 3.0, False),
    ("half_tri17", "half", 17, 1.5, True),
    ("half_tri99", "half", 99, 3.0, True),
    ("ghost17", "ghost", 17, 1.5, False),
    ("ghost99", "ghost", 99, 3.0, True),
    ("size17", "size", 17, 1.5, True),
    ("size99", "size", 99, 3.0, False),
)


def input_text(mode: str, binsize: float, triclinic: bool) -> str:
    """Return the two-type input with exclusions and unequal cutoffs."""
    sphere = mode == "size"
    atom_style = "sphere" if sphere else "atomic"
    geometry = (
        "region box prism 0 6 0 6 0 6 0.3 0.1 0.2"
        if triclinic
        else "region box block 0 6 0 6 0 6"
    )
    masses = (
        "set type 1 diameter 1.0 density 1.0\nset type 2 diameter 1.6 density 1.0"
        if sphere
        else "mass * 1.0"
    )
    return f"""units lj
atom_style {atom_style}
atom_modify sort 0 0.0
lattice fcc 0.8
{geometry}
create_box 2 box
create_atoms 1 box
set type 1 type/fraction 2 0.4 12345
{masses}
pair_style neighbor_probe 3.0
pair_coeff * *
pair_coeff 1 1 1.8
pair_coeff 1 2 2.5
pair_coeff 2 2 3.0
neighbor 0.2 bin
neigh_modify binsize {binsize} exclude type 1 2
run 0
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--gpu", default="1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=args.gpu,
        OMP_NUM_THREADS="1",
        LAMMPS_PLUGIN_PATH="",
        LD_LIBRARY_PATH=str(args.library.resolve())
        + ":"
        + env.get("LD_LIBRARY_PATH", ""),
    )
    results = []
    for name, mode, capacity, binsize, triclinic in CASES:
        work = args.output / name
        work.mkdir()
        (work / "input.lmp").write_text(input_text(mode, binsize, triclinic))
        package = ["-pk", "kokkos", "nbin/atoms/per/bin", str(capacity)]
        if mode in ("half", "size"):
            package += ["neigh", "half", "newton", "on"]
        snapshot = (
            args.reference / name / "atoms.bin"
            if args.reference
            else work / "atoms.bin"
        )
        command = [
            str(args.probe.resolve()),
            "input.lmp",
            "rows.txt",
            mode,
            str(snapshot.resolve()),
            "read" if args.reference else "write",
            *package,
        ]
        proc = subprocess.run(
            command, cwd=work, env=env, text=True, capture_output=True, timeout=180
        )
        (work / "screen.txt").write_text(proc.stdout + proc.stderr)
        if proc.returncode:
            raise RuntimeError(f"Neighbor probe failed; see {work / 'screen.txt'}")
        rows = (work / "rows.txt").read_bytes()
        lines = rows.decode().splitlines()
        result = {
            "case": name,
            "metadata": lines[0],
            "row_count": sum(line.startswith("row ") for line in lines),
        }
        if args.reference:
            result["exact"] = rows == (
                args.reference / name / "rows.txt"
            ).read_bytes()
        results.append(result)
        print(result, flush=True)  # noqa: T201
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    if args.reference and not all(result["exact"] for result in results):
        raise RuntimeError(f"Different neighbor rows; see {args.output}")


if __name__ == "__main__":
    main()

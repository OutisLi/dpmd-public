# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compare neighbor-check decisions and deterministic trajectories."""

from __future__ import (
    annotations,
)

import argparse
import json
import os
import re
import subprocess
from pathlib import (
    Path,
)

from check_integrator import (
    thermo_rows,
)


def input_text(case: str, ranks: int) -> str:
    """Exercise repeated skin crossings, host coordinates and box changes."""
    processors = "processors 2 1 1" if ranks == 2 else ""
    region = "1 2 1 3 1 3" if case == "empty_rank" else "2 4 1 3 1 3"
    extra = ""
    integrator = "fix integrate all nve"
    if case == "host":
        integrator = "suffix off\nfix integrate all nve\nsuffix on"
    if case == "include":
        extra = "group mobile id <= 32\natom_modify first mobile\nneigh_modify include mobile"
    if case == "deform":
        extra = "fix deform all deform 1 x erate 0.1 remap x"
    if case == "triclinic":
        extra = "change_box all triclinic\nfix deform all deform 1 xy erate 0.1 remap x"
    return f"""units lj
atom_style atomic
{processors}
lattice fcc 0.8
region box block 0 8 0 4 0 4
create_box 1 box
region populated block {region}
create_atoms 1 region populated
mass 1 1.0
pair_style lj/cut 0.5
pair_coeff * * 1.0 1.0
neighbor 0.01 bin
neigh_modify every 1 delay 0 check yes
velocity all set 0.9 -0.6 0.3
timestep 0.001
fix drive all addforce 0.1 -0.2 0.3
{integrator}
{extra}
thermo 1
thermo_style custom step temp pe ke etotal press vol
thermo_modify format float %.17g
dump state all custom 1 trajectory.dump id type x y z vx vy vz fx fy fz
dump_modify state sort id format float %.17g
run 80
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmp", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--cases", default="crossing,host,include,deform,triclinic")
    parser.add_argument("--ranks", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS="1", LAMMPS_PLUGIN_PATH=""
    )
    env["LD_LIBRARY_PATH"] = args.library + ":" + env.get("LD_LIBRARY_PATH", "")
    command = [args.lmp, "-k", "on", "g", "1", "-sf", "kk", "-in", "input.lmp"]
    if args.ranks > 1:
        env["HWLOC_COMPONENTS"] = "-cuda,-nvml,-gl,-opencl,-rsmi"
        command = [
            "/nas/outisli/Software/miniforge3/envs/dpmd/bin/mpiexec",
            "-n",
            str(args.ranks),
            *command,
        ]
    results = {}
    for case in args.cases.split(","):
        work = args.output / case
        work.mkdir()
        (work / "input.lmp").write_text(input_text(case, args.ranks))
        proc = subprocess.run(
            command, cwd=work, env=env, capture_output=True, text=True, timeout=180
        )
        (work / "screen.txt").write_text(proc.stdout + proc.stderr)
        if proc.returncode:
            raise RuntimeError(
                f"{case} exited with {proc.returncode}: {work / 'screen.txt'}"
            )
        thermo = thermo_rows(proc.stdout)
        if len(thermo) != 81:
            raise RuntimeError(f"{case}: expected 81 thermo rows, got {len(thermo)}")
        (work / "thermo.json").write_text(json.dumps(thermo, indent=2) + "\n")
        counts = {
            "builds": int(re.findall(r"Neighbor list builds = (\d+)", proc.stdout)[-1]),
            "dangerous": int(re.findall(r"Dangerous builds = (\d+)", proc.stdout)[-1]),
        }
        if counts["builds"] < 10:
            raise RuntimeError(
                f"{case}: too few neighbor rebuilds to test repeated crossings: {counts}"
            )
        if case == "empty_rank" and not re.search(r"Nlocal:.*\s0 min", proc.stdout):
            raise RuntimeError("Expected an MPI rank with zero local atoms")
        (work / "counts.json").write_text(json.dumps(counts, indent=2) + "\n")
        results[case] = {"counts": counts}
        if args.reference:
            results[case]["equal"] = {
                name: (work / name).read_bytes()
                == (args.reference / case / name).read_bytes()
                for name in ("thermo.json", "trajectory.dump", "counts.json")
            }
        print(case, results[case], flush=True)  # noqa: T201
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    if args.reference and not all(
        all(row["equal"].values()) for row in results.values()
    ):
        raise RuntimeError(
            f"Numerical differences found; see {args.output / 'results.json'}"
        )


if __name__ == "__main__":
    main()

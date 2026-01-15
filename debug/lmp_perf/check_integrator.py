# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compare LAMMPS integrator trajectories and thermostat observables."""

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

CASES = {
    "nvt": ("atomic", "all", "nvt temp 0.8 0.8 0.2", ""),
    "loops": ("atomic", "all", "nvt temp 0.8 1.2 0.2 tchain 5 tloop 3 drag 0.1", ""),
    "group": ("atomic", "mobile", "nvt temp 0.8 0.8 0.2", ""),
    "firstgroup": (
        "atomic",
        "mobile",
        "nvt temp 0.8 0.8 0.2",
        "atom_modify first mobile",
    ),
    "other_temp": (
        "atomic",
        "mobile",
        "nvt temp 0.8 0.8 0.2",
        "compute other all temp/kk\nfix_modify integrate temp other",
    ),
    "rmass": ("sphere", "all", "nvt temp 0.8 0.8 0.2 tloop 3", ""),
    "bias": (
        "atomic",
        "all",
        "nvt temp 0.8 0.8 0.2",
        "compute other all temp/partial 1 1 0\nfix_modify integrate temp other",
    ),
    "dynamic": (
        "atomic",
        "all",
        "nvt temp 0.8 0.8 0.2",
        "compute_modify integrate_temp dynamic/dof yes",
    ),
    "npt": ("atomic", "all", "npt temp 0.8 0.8 0.2 iso 0.5 0.5 1.0", ""),
    "aniso": ("atomic", "all", "npt temp 0.8 0.8 0.2 aniso 0.5 0.5 1.0", ""),
    "observed": (
        "atomic",
        "all",
        "nvt temp 0.8 0.8 0.2",
        "group void empty\ncompute empty void temp/kk\ncompute host all temp/kk/host",
    ),
    "restart": ("atomic", "all", "nvt temp 0.8 1.2 0.2 tloop 3", ""),
}


def observables(filename: str, extra: str = "") -> str:
    """Return commands that expose particle and thermostat state."""
    return f"""
thermo 1
thermo_style custom step temp pe ke etotal press vol f_integrate f_integrate[1] f_integrate[2] {extra}
thermo_modify format float %.17g
dump state all custom 1 {filename} id type x y z vx vy vz fx fy fz
dump_modify state sort id format float %.17g
"""


def input_text(case: str) -> str:
    """Isolate the integrator with deterministic nonzero external forces."""
    atom_style, group, integrator, extra = CASES[case]
    masses = "mass 1 1.0\nmass 2 2.0"
    if atom_style == "sphere":
        masses = (
            "set type 1 diameter 1.0 density 1.0\nset type 2 diameter 1.2 density 1.5"
        )
    outputs = ""
    if case == "observed":
        outputs = (
            "c_host c_host[1] c_host[2] c_host[3] c_host[4] c_host[5] "
            "c_host[6] c_empty c_empty[1]"
        )
    result = f"""units lj
atom_style {atom_style}
lattice fcc 0.8
region box block 0 4 0 4 0 4
create_box 2 box
create_atoms 1 box
set type 1 type/fraction 2 0.4 45123
{masses}
group mobile id <= 128
pair_style lj/cut 0.5
pair_coeff * * 1.0 1.0
neighbor 0.15 bin
neigh_modify every 1 delay 0 check yes
velocity all create 0.8 4928459 mom yes rot no dist gaussian
timestep 0.001
fix drive all addforce 0.1 -0.2 0.3
fix integrate {group} {integrator}
{extra}
{observables("trajectory.dump", outputs)}
run 20 start 0 stop 40
"""
    if case == "restart":
        result += f"""write_restart state.restart
clear
read_restart state.restart
neighbor 0.15 bin
neigh_modify every 1 delay 0 check yes
fix drive all addforce 0.1 -0.2 0.3
fix integrate {group} {integrator}
{observables("after_restart.dump")}
"""
    result += "run 20 start 0 stop 40\n"
    return result


def thermo_rows(output: str) -> list[str]:
    """Extract numerical thermodynamic rows without timing reports."""
    rows = []
    active = False
    for line in output.splitlines():
        if line.split()[:2] == ["Step", "Temp"]:
            active = True
        elif line.startswith("Loop time"):
            active = False
        elif active and line.split() and line.split()[0].isdigit():
            rows.append(" ".join(line.split()))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmp", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--cases", default=",".join(CASES))
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
        (work / "input.lmp").write_text(input_text(case))
        proc = subprocess.run(
            command, cwd=work, env=env, capture_output=True, text=True, timeout=180
        )
        (work / "screen.txt").write_text(proc.stdout + proc.stderr)
        if proc.returncode:
            raise RuntimeError(
                f"{case} exited with {proc.returncode}: {work / 'screen.txt'}"
            )
        thermo = thermo_rows(proc.stdout)
        if len(thermo) != 42:
            raise RuntimeError(f"{case}: expected 42 thermo rows, got {len(thermo)}")
        (work / "thermo.json").write_text(json.dumps(thermo, indent=2) + "\n")
        if args.reference:
            names = ["thermo.json", "trajectory.dump"]
            if case == "restart":
                names.append("after_restart.dump")
            results[case] = {
                name: (work / name).read_bytes()
                == (args.reference / case / name).read_bytes()
                for name in names
            }
        else:
            results[case] = {"thermo_rows": len(thermo)}
        print(case, results[case], flush=True)  # noqa: T201
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    if args.reference and not all(all(row.values()) for row in results.values()):
        raise RuntimeError(
            f"Numerical differences found; see {args.output / 'results.json'}"
        )


if __name__ == "__main__":
    main()

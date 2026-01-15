# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compare both geometry formats across two real MPI ranks and CUDA devices."""

from __future__ import (
    annotations,
)

import argparse
import json
import os
import subprocess
import sys
from pathlib import (
    Path,
)

BENCH = Path(__file__).resolve().parents[2] / "cuda_bench"
sys.path.insert(0, str(BENCH))

import gen_system
import machine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", required=True)
    args = parser.parse_args()
    machine.load()
    root = args.artifacts.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    systems = output / "systems"
    systems.mkdir()
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES=args.gpus,
        OMP_NUM_THREADS="1",
        DP_INTER_OP_PARALLELISM_THREADS="0",
        DP_INTRA_OP_PARALLELISM_THREADS="0",
        DP_BACKEND_PLUGIN_PATH=str(root / "integrated"),
        OMPI_MCA_pml="ob1",
        OMPI_MCA_btl="self,tcp",
        OMPI_MCA_opal_cuda_support="false",
    )
    environment["LD_LIBRARY_PATH"] = ":".join(
        [
            str(root / "lammps"),
            str(root / "integrated"),
            environment["BENCH_DP_LIB"],
            environment["BENCH_TORCH_LIB"],
            environment.get("LD_LIBRARY_PATH", ""),
        ]
    )
    results = []
    for flavor in ("energy", "spin"):
        writer = (
            gen_system.ensure_bcn_spin_data
            if flavor == "spin"
            else gen_system.ensure_bcn_data
        )
        data, atoms = writer(253952, str(systems))
        assert atoms == 253952
        spin = flavor == "spin"
        model = (
            root / "models_shape_restored" / ("spin_nano.pt2" if spin else "nano.pt2")
        )
        extra = (
            "compute moment all property/atom sp spx spy spz fmx fmy fmz"
            if spin
            else ""
        )
        fields = (
            "c_moment[1] c_moment[2] c_moment[3] c_moment[4] c_moment[5] c_moment[6] c_moment[7]"
            if spin
            else ""
        )
        inputs = f"""units metal
boundary p p p
atom_style {"spin" if spin else "atomic"}
atom_modify map yes sort 0 0.0
read_data ${{datafile}}
pair_style {"dpa4spin ${model} magnetic_force cartesian" if spin else "deepmd ${model}"}
pair_coeff * * B C N
neighbor 1 bin
neigh_modify every 1 delay 0 check yes
velocity all create 300 12345 mom yes rot yes dist gaussian
fix integrate all nvt temp 300 300 0.1
timestep 0.000002
compute pea all pe/atom
{extra}
dump state all custom 3 state.dump id type x y z c_pea fx fy fz {fields}
dump_modify state sort id format float %.17g
thermo 1
thermo_style custom step pe press pxx pyy pzz pxy pxz pyz
thermo_modify format float %.17g
run 0
run 3
compute av all centroid/stress/atom NULL pair
dump avout all custom 1 virial.dump id c_av[1] c_av[2] c_av[3] c_av[4] c_av[5] c_av[6] c_av[7] c_av[8] c_av[9]
dump_modify avout sort id format float %.17g
run 0
undump avout
uncompute av
run 3
"""
        reference = None
        for side in ("vector", "indexed"):
            work = output / f"{flavor}_{side}"
            work.mkdir()
            (work / "input.lmp").write_text(inputs)
            environment["LAMMPS_PLUGIN_PATH"] = str(
                root / ("vector_plugin" if side == "vector" else "integrated")
            )
            command = [
                environment["BENCH_MPIEXEC"],
                "--bind-to",
                "none",
                "-np",
                "2",
                str(root / "lammps/lmp"),
                "-k",
                "on",
                "g",
                "2",
                "-sf",
                "kk",
                "-pk",
                "kokkos",
                "comm",
                "host",
                "gpu/aware",
                "off",
                "-in",
                "input.lmp",
                "-var",
                "datafile",
                str(data),
                "-var",
                "model",
                str(model),
            ]
            process = subprocess.run(
                command,
                cwd=work,
                env=environment,
                text=True,
                capture_output=True,
                timeout=600,
            )
            (work / "screen.txt").write_text(process.stdout + process.stderr)
            (work / "command.json").write_text(json.dumps(command, indent=2) + "\n")
            process.check_returncode()
            rows = []
            active = False
            for line in process.stdout.splitlines():
                if line.split()[:1] == ["Step"]:
                    active = True
                elif line.startswith("Loop time"):
                    active = False
                elif active and line.split() and line.split()[0].isdigit():
                    rows.append(" ".join(line.split()))
            observed = (
                rows,
                (work / "state.dump").read_bytes(),
                (work / "virial.dump").read_bytes(),
            )
            assert rows and observed[1] and observed[2]
            if reference is None:
                reference = observed
            else:
                assert observed == reference, flavor
            record = {
                "flavor": flavor,
                "side": side,
                "atoms": atoms,
                "ranks": 2,
                "thermo_rows": len(rows),
                "state_bytes": len(observed[1]),
                "virial_bytes": len(observed[2]),
                "bit_exact": True,
            }
            results.append(record)
            (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(record, flush=True)  # noqa: T201


if __name__ == "__main__":
    main()

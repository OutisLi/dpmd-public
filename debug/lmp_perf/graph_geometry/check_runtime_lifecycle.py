# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check geometry selection across runs, virial requests and pair settings."""

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

from benchmark_matrix import (
    compare_trajectories,
    digest,
    input_text,
    numerical_difference,
    thermo_values,
    verify_trajectory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--grade", default="mini")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo / "debug/cuda_bench"))
    import machine

    machine.load()
    root = args.artifacts.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
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
        HWLOC_COMPONENTS=environment["BENCH_HWLOC_COMPONENTS"],
    )
    records = []
    for flavor in ("energy", "spin"):
        spin = flavor == "spin"
        runtime_name = "fork" if spin else "public"
        runtime = root / f"{runtime_name}_lammps"
        model = root / "models" / f"{args.grade}_{flavor}_zbl.pt2"
        data = root / "systems" / f"bcn_10296_zbl{'_spin' if spin else ''}.data"
        pair = (
            f'dpa4spin "{model}" magnetic_force cartesian'
            if spin
            else f'deepmd "{model}"'
        )
        fields = "id type x y z vx vy vz fx fy fz"
        if spin:
            fields += " " + " ".join(f"c_moments[{index}]" for index in range(1, 8))
        fields += " c_atom_energy"
        inputs = input_text(model, data, "bcn", spin, 4, 2, True)
        inputs += f"""compute av all centroid/stress/atom NULL pair
dump atomic_virial all custom 1 virial.dump id c_av[1] c_av[2] c_av[3] c_av[4] c_av[5] c_av[6] c_av[7] c_av[8] c_av[9]
dump_modify atomic_virial sort id format float %.17g
run 0
undump atomic_virial
uncompute av
run 3
pair_coeff * * B C N
run 0
pair_style {pair}
pair_coeff * * B C N
run 0
run 3
write_dump all custom final.dump {fields} modify sort id format float %.17g
"""
        environment["LD_LIBRARY_PATH"] = ":".join(
            [
                str(root / "integrated"),
                str(runtime),
                environment["BENCH_TORCH_LIB"],
                environment["BENCH_DP_LIB"],
                os.environ.get("LD_LIBRARY_PATH", ""),
            ]
        )
        reference = None
        for side in ("vector", "indexed", "auto"):
            work = output / f"{flavor}_{side}"
            work.mkdir()
            (work / "input.lmp").write_text(inputs)
            environment["LAMMPS_PLUGIN_PATH"] = str(
                root / f"{runtime_name}_{side}_plugin"
            )
            command = [
                str(runtime / "lmp"),
                "-k",
                "on",
                "g",
                str(args.ranks),
                "-sf",
                "kk",
            ]
            if args.ranks > 1:
                command = [
                    environment["BENCH_MPIEXEC"],
                    "--bind-to",
                    "none",
                    "-np",
                    str(args.ranks),
                    *command,
                ]
                command += ["-pk", "kokkos", "comm", "host", "gpu/aware", "off"]
            command += ["-in", "input.lmp", "-log", "none"]
            (work / "command.json").write_text(json.dumps(command, indent=2) + "\n")
            # The input file owns the command stream; MPI must not forward
            # the launcher's standard input to its ranks.
            process = subprocess.run(
                command,
                cwd=work,
                env=environment,
                stdin=subprocess.DEVNULL,
                text=True,
                capture_output=True,
                timeout=900,
            )
            (work / "screen.txt").write_text(process.stdout + process.stderr)
            process.check_returncode()
            trajectory = verify_trajectory(work, spin)
            compared = trajectory | {
                "final_sha256": digest(work / "final.dump"),
                "virial_sha256": digest(work / "virial.dump"),
                "thermo": thermo_values(process.stdout),
            }
            comparisons = {}
            if reference is None:
                reference = compared
                reference_work = work
            else:
                import numpy as np

                comparisons = compare_trajectories(
                    reference_work,
                    work,
                    spin,
                    trajectory["atoms"],
                    reference["thermo"],
                    compared["thermo"],
                    states=("before", "after", "final"),
                )
                left = np.loadtxt(reference_work / "virial.dump", skiprows=9)
                right = np.loadtxt(work / "virial.dump", skiprows=9)
                assert np.array_equal(left[:, 0], right[:, 0])
                # LAMMPS metal units convert eV to bar angstrom^3 by nktv2p.
                comparisons["atomic_virial_ev"] = numerical_difference(
                    left[:, 1:] / 1.6021765e6,
                    right[:, 1:] / 1.6021765e6,
                    8e-6,
                    1e-4,
                )
                assert all(value["passed"] for value in comparisons.values()), (
                    flavor,
                    side,
                    comparisons,
                )
            calibrations = process.stdout.count("DeePMD canonical geometry:")
            assert calibrations == (3 if side == "auto" else 0), (
                flavor,
                side,
                calibrations,
            )
            record = {
                "flavor": flavor,
                "side": side,
                "grade": args.grade,
                "ranks": args.ranks,
                "calibrations": calibrations,
                "bit_exact": compared == reference,
                "comparisons": comparisons,
                **compared,
            }
            records.append(record)
            (output / "results.json").write_text(json.dumps(records, indent=2) + "\n")
            print(  # noqa: T201
                {
                    key: record[key]
                    for key in ("flavor", "side", "ranks", "calibrations", "bit_exact")
                },
                flush=True,
            )


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: LGPL-3.0-or-later
"""Sample device memory during fixed and automatic geometry evaluation."""

from __future__ import (
    annotations,
)

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import (
    Path,
)

from artifact_snapshot import (
    capture_snapshot,
    verify_snapshot,
)
from benchmark_matrix import (
    calibration_values,
    input_text,
    thermo_values,
)


def memory_mib(gpu: str) -> int:
    value = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            gpu,
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    )
    return int(value.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo / "debug/cuda_bench"))
    import machine

    machine.load()
    root = args.artifacts.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = capture_snapshot(root)
    (output / "snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES=args.gpu,
        OMP_NUM_THREADS="1",
        DP_INTER_OP_PARALLELISM_THREADS="0",
        DP_INTRA_OP_PARALLELISM_THREADS="0",
        DP_BACKEND_PLUGIN_PATH=str(root / "integrated"),
    )
    records = []
    for grade, flavor in (("nano", "energy"), ("plus", "spin")):
        spin = flavor == "spin"
        runtime_name = "fork" if spin else "public"
        runtime = root / f"{runtime_name}_lammps"
        environment["LD_LIBRARY_PATH"] = ":".join(
            [
                str(root / "integrated"),
                str(runtime),
                environment["BENCH_TORCH_LIB"],
                environment["BENCH_DP_LIB"],
                os.environ.get("LD_LIBRARY_PATH", ""),
            ]
        )
        model = root / "models" / f"{grade}_{flavor}.pt2"
        data = root / "systems" / f"bcn_100800{'_spin' if spin else ''}.data"
        for side in ("vector", "indexed", "auto"):
            work = output / f"{grade}_{flavor}_{side}"
            work.mkdir()
            (work / "input.lmp").write_text(
                input_text(model, data, "bcn", spin, 200, 20, False)
            )
            environment["LAMMPS_PLUGIN_PATH"] = str(
                root / f"{runtime_name}_{side}_plugin"
            )
            command = [
                str(runtime / "lmp"),
                "-k",
                "on",
                "g",
                "1",
                "-sf",
                "kk",
                "-in",
                "input.lmp",
                "-log",
                "none",
            ]
            (work / "command.json").write_text(json.dumps(command, indent=2) + "\n")
            active = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    args.gpu,
                    "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=10,
            ).strip()
            assert not active, (args.gpu, active)
            baseline = memory_mib(args.gpu)
            start = time.monotonic()
            samples = []
            with (work / "screen.txt").open("w") as screen:
                process = subprocess.Popen(
                    command,
                    cwd=work,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=screen,
                    stderr=subprocess.STDOUT,
                )
                while process.poll() is None:
                    elapsed = time.monotonic() - start
                    if elapsed > 900:
                        process.kill()
                        process.wait()
                        raise TimeoutError(
                            f"Memory measurement exceeded 900 seconds: {work}"
                        )
                    samples.append(
                        {"elapsed_s": elapsed, "memory_mib": memory_mib(args.gpu)}
                    )
                    time.sleep(0.1)
            assert process.returncode == 0, (work, process.returncode)
            screen = (work / "screen.txt").read_text()
            thermo_values(screen)
            calibration = calibration_values(screen)
            assert bool(calibration) == (side == "auto"), work
            assert samples, work
            record = {
                "grade": grade,
                "flavor": flavor,
                "atoms": 100800,
                "elements": "bcn",
                "side": side,
                "baseline_mib": baseline,
                "sampled_peak_mib": max(sample["memory_mib"] for sample in samples),
                "sampled_increment_mib": max(sample["memory_mib"] for sample in samples)
                - baseline,
                "calibration": calibration,
                "sampling_pause_s": 0.1,
                "samples": samples,
                "snapshot_sha256": snapshot["sha256"],
            }
            records.append(record)
            (output / "results.json").write_text(json.dumps(records, indent=2) + "\n")
            print(  # noqa: T201
                {key: value for key, value in record.items() if key != "samples"},
                flush=True,
            )
    verify_snapshot(root, output, snapshot)


if __name__ == "__main__":
    main()

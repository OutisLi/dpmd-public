# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201
"""Run one recorded recipe and its checkpoint observer in one allocation."""

from __future__ import (
    annotations,
)

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from datetime import (
    datetime,
    timezone,
)
from pathlib import (
    Path,
)


def stop(process: subprocess.Popen) -> None:
    """Stop an owned process group and wait for its exit."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    run = directory / "runs" / args.run
    definition = json.loads((run / "experiment.json").read_text())
    if (run / "launch.log").exists():
        raise FileExistsError(f"Recipe already launched: {run}")
    visibility = os.environ["CUDA_VISIBLE_DEVICES"]
    if len(visibility.split(",")) != 1:
        raise ValueError(f"Expected a single allocated GPU, got {visibility!r}")
    subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        check=True,
    )
    sources = (
        "train_refresh.py",
        "radial_experiment.py",
        "message_envelope.py",
        "attention_normalization.py",
        "edge_type_neighbors.py",
        "seed_radial_gate.py",
        "outer_radial_gate.py",
        "pair_fade.py",
        "isolated_reference.py",
        "adam_route.py",
        "fitting_normalization.py",
        "plain_fitting.py",
        "reference_loss.py",
        "original_reference.py",
        "cluster_anchors.py",
        "pair_anchors.py",
        "replay_anchors.py",
        "watch_run.py",
        "dimer_survey.py",
        "diagnose/needle_geometry.py",
    )
    definition["source_sha256"] = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in sources
    }
    (run / "experiment.json").write_text(json.dumps(definition, indent=2) + "\n")
    command = [
        sys.executable,
        "-u",
        str(directory / "train_refresh.py"),
        "input.json",
        *definition["train_flags"],
    ]
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    with (run / "dp.log").open("x") as output:
        trainer = subprocess.Popen(
            command,
            cwd=run,
            stdout=output,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    (run / "launch.log").write_text(
        f"launched at {stamp} on {os.uname().nodename} GPU {visibility} "
        f"(Slurm job {os.environ.get('SLURM_JOB_ID', '-')}, one GPU): train_refresh.py "
        + " ".join(definition["train_flags"])
        + "\n"
    )
    with (run / "watch.log").open("x") as output:
        observer = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(directory / "watch_run.py"),
                str(run),
                "--trainer",
                str(trainer.pid),
                "--gpu",
                visibility,
            ],
            cwd=directory,
            stdout=output,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    (run / "runtime.json").write_text(
        json.dumps(
            {
                "host": os.uname().nodename,
                "gpu": visibility,
                "trainer_pid": trainer.pid,
                "watcher_pid": observer.pid,
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "started": stamp,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"STARTED {args.run}: trainer={trainer.pid}, observer={observer.pid}, CUDA_VISIBLE_DEVICES={visibility}",
        flush=True,
    )
    try:
        result = observer.wait()
        if result:
            raise RuntimeError(
                f"Observer failed for {args.run} with exit code {result}"
            )
        print(
            f"FINISHED {args.run}: scheduled checkpoint coverage verified", flush=True
        )
    finally:
        stop(observer)
        stop(trainer)


if __name__ == "__main__":
    main()

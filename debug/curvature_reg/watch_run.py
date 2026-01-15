# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201
"""Stage published checkpoints and survey every saved step of one live experiment."""

from __future__ import (
    annotations,
)

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import (
    Path,
)


def trainer_alive(pid: int, run: Path) -> bool:
    """Check the trainer's process identity and state on the current host."""
    process = Path("/proc") / str(pid)
    try:
        command = process.joinpath("cmdline").read_bytes().split(b"\0")
        state = process.joinpath("stat").read_text().rsplit(")", 1)[1].split()[0]
        directory = process.joinpath("cwd").resolve(strict=True)
    except FileNotFoundError:
        return False
    return (
        state != "Z"
        and directory == run
        and any(item.endswith(b"/train_refresh.py") for item in command)
    )


def stage_checkpoints(run: Path, horizon: int) -> list[int]:
    """Copy only checkpoints covered by the trainer's published EMA pointer."""
    pointer = run / "model_ema.ckpt.pt"
    if not pointer.exists():
        return []
    published = int(pointer.resolve().stem.rsplit("-", 1)[1])
    for source in sorted((run / "ckpt").glob("model_ema.ckpt-[0-9]*.pt")):
        step = int(source.stem.rsplit("-", 1)[1])
        target = run / f"ema_{step}.pt"
        if step > min(published, horizon) or target.exists():
            continue
        partial = target.with_suffix(".partial")
        shutil.copyfile(source, partial)
        with zipfile.ZipFile(partial) as archive:
            damaged = archive.testzip()
        if damaged is not None:
            raise ValueError(f"Checkpoint CRC failed for {source}: {damaged}")
        partial.replace(target)
    return sorted(
        int(path.stem.split("_")[1])
        for path in run.glob("ema_[0-9]*.pt")
        if int(path.stem.split("_")[1]) <= horizon
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--trainer", type=int, required=True)
    parser.add_argument(
        "--gpu", required=True, help="CUDA visibility mask inherited from the launcher"
    )
    args = parser.parse_args()
    run = args.run.resolve()
    definition = json.loads((run / "experiment.json").read_text())
    horizon = int(definition["horizon_step"])
    inputs = json.loads((run / "input.json").read_text())
    stride = int(inputs["training"]["save_freq"])
    expected = set(range(stride, horizon + 1, stride))
    directory = Path(__file__).resolve().parent
    if not trainer_alive(args.trainer, run):
        raise RuntimeError(f"No matching live trainer {args.trainer} in {run}")
    environment = dict(
        os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="4"
    )
    pairs = (
        ["H-H"]
        if run.name.startswith("h_") and not definition.get("survey_all_pairs", False)
        else []
    )
    evaluator: subprocess.Popen | None = None
    eval_log = None
    evaluating = None
    completed: set[int] = set()
    for record in run.glob("survey_full_[0-9]*.json"):
        step = int(record.stem.rsplit("_", 1)[1])
        if record.with_suffix(".npz").exists():
            data = json.loads(record.read_text())
            if len(data["pairs"]) == (1 if pairs else 24):
                completed.add(step)
    failed: set[int] = set()
    stopped_at_horizon = False
    print(
        f"STARTUP verified trainer={args.trainer}, run={run.name}, GPU={args.gpu}, "
        f"expected_checkpoints={len(expected)}, stride={stride}, horizon={horizon}",
        flush=True,
    )
    while True:
        alive = trainer_alive(args.trainer, run)
        available = stage_checkpoints(run, horizon)
        if horizon in available and alive and not stopped_at_horizon:
            if os.getpgid(args.trainer) != args.trainer:
                raise RuntimeError(
                    f"Trainer {args.trainer} is not its own process-group leader"
                )
            os.killpg(args.trainer, signal.SIGTERM)
            stopped_at_horizon = True
            print(
                f"HORIZON saved {horizon}; stopped trainer {args.trainer}", flush=True
            )
        if evaluator is not None and evaluator.poll() is not None:
            eval_log.close()
            if evaluator.returncode == 0:
                record = json.loads(
                    (run / f"survey_full_{evaluating}.json").read_text()
                )
                count = 1 if pairs else 24
                if len(record["pairs"]) != count:
                    raise ValueError(
                        f"Incomplete survey at {evaluating}: expected {count} pairs"
                    )
                completed.add(evaluating)
                print(
                    f"SURVEY completed step={evaluating}, pairs={count}, "
                    f"tail_failures={sum(not p['tail_pass'] for p in record['pairs'])}",
                    f"tail45_failures={sum(not p.get('tail45_pass', True) for p in record['pairs'])}",
                    flush=True,
                )
            else:
                failed.add(evaluating)
                print(
                    f"SURVEY failed step={evaluating}, returncode={evaluator.returncode}",
                    flush=True,
                )
            evaluator = None
        pending = [step for step in available if step not in completed | failed]
        if evaluator is None and pending:
            evaluating = pending[0]
            output = run / f"survey_full_{evaluating}.npz"
            eval_log = (run / f"survey_full_{evaluating}.log").open("w")
            command = [
                sys.executable,
                "-u",
                str(directory / "dimer_survey.py"),
                str(run / f"ema_{evaluating}.pt"),
                f"{run.name} {evaluating}",
                *pairs,
                "--out",
                str(output),
            ]
            evaluator = subprocess.Popen(
                command,
                cwd=run,
                env=environment,
                stdout=eval_log,
                stderr=subprocess.STDOUT,
            )
        status = {
            "host": os.uname().nodename,
            "trainer_pid": args.trainer,
            "trainer_alive": alive,
            "horizon": horizon,
            "horizon_reached": horizon in available,
            "available": available,
            "completed": sorted(completed),
            "failed": sorted(failed),
            "evaluating": evaluating if evaluator is not None else None,
            "missing": sorted(expected - set(available)),
            "updated": time.time(),
        }
        partial = run / "watch_state.partial"
        partial.write_text(json.dumps(status, indent=2) + "\n")
        partial.replace(run / "watch_state.json")
        if not alive and evaluator is None and not pending:
            break
        time.sleep(5)
    if failed or expected - completed:
        raise RuntimeError(
            f"Experiment ended with failed={sorted(failed)} and "
            f"unevaluated={sorted(expected - completed)}"
        )
    print(
        f"COMPLETE verified {len(completed)} checkpoint surveys through {horizon}",
        flush=True,
    )


if __name__ == "__main__":
    main()

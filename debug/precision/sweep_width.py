# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201
"""
Sweep descriptor width against AMP to locate the training bottleneck.

The question the sweep answers is whether the training step is limited by
arithmetic or by memory traffic and kernel launches. Reduced precision only
pays where arithmetic dominates, so the AMP speed-up as a function of channel
width tells directly how much headroom a lower format could ever reach.

Each configuration is a copy of the reference training script with
``descriptor.channels`` and ``descriptor.use_amp`` overridden; everything else,
including the dataset and the optimizer, is held fixed.
"""

from __future__ import (
    annotations,
)

import argparse
import json
import re
import subprocess
from pathlib import (
    Path,
)

BENCH = Path("debug/train_bench/bench.py")
STEP_RE = re.compile(r"median\s+([\d.]+)\s*ms")
VRAM_RE = re.compile(r"peak VRAM\s+([\d.]+)\s*GiB")


def write_variant(
    base: dict, channels: int, batch: int, use_amp: bool, out: Path
) -> None:
    """
    Write a training script variant with width, batch size and AMP overridden.

    Parameters
    ----------
    base : dict
        Parsed reference training script.
    channels : int
        Descriptor channel width to set.
    batch : int
        Training batch size in frames.
    use_amp : bool
        Whether the descriptor runs its interaction blocks under bf16 autocast.
    out : Path
        Destination path of the variant.
    """
    cfg = json.loads(json.dumps(base))
    cfg["model"]["descriptor"]["channels"] = channels
    cfg["model"]["descriptor"]["use_amp"] = use_amp
    cfg["training"]["training_data"]["batch_size"] = batch
    # Each variant needs its own statistics file: the descriptor width changes
    # the cached shapes.
    cfg["training"]["stat_file"] = f"./stat_c{channels}.hdf5"
    out.write_text(json.dumps(cfg, indent=2))


def run_bench(
    script: Path, repo: Path, cwd: Path, warmup: int, steps: int, device: str
) -> tuple[float, float]:
    """
    Run the training benchmark and return the steady-state cost.

    Parameters
    ----------
    script : Path
        Training script to benchmark.
    repo : Path
        Repository root, used to locate the benchmark harness.
    cwd : Path
        Working directory, which fixes the relative dataset paths.
    warmup : int
        Warm-up steps.
    steps : int
        Measured steps.
    device : str
        Value for ``CUDA_VISIBLE_DEVICES``.

    Returns
    -------
    tuple[float, float]
        Median wall time per training step in milliseconds and peak device
        memory in GiB; both are ``float("nan")`` when the run failed.
    """
    env = {
        "OMP_NUM_THREADS": "1",
        "DP_INTER_OP_PARALLELISM_THREADS": "0",
        "DP_INTRA_OP_PARALLELISM_THREADS": "0",
        "CUDA_VISIBLE_DEVICES": device,
        "PATH": "/usr/bin:/bin",
    }
    proc = subprocess.run(
        [
            "/nas/outisli/Software/miniforge3/envs/dpmd/bin/python",
            str(repo / BENCH),
            script.name,
            "--warmup",
            str(warmup),
            "--steps",
            str(steps),
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    hits = STEP_RE.findall(proc.stdout)
    if not hits:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        return float("nan"), float("nan")
    vram = VRAM_RE.findall(proc.stdout)
    return float(hits[0]), float(vram[0]) if vram else float("nan")


def main() -> None:
    """Run the width/AMP sweep and print the speed-up table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo", type=Path, default=Path("/nas/outisli/Software/deepmd-kit")
    )
    ap.add_argument(
        "--script",
        type=Path,
        default=Path("/nas/outisli/Software/deepmd-kit/examples/water/dpa4/input.json"),
    )
    ap.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--batch", type=int, nargs="+", default=[1])
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    base = json.loads(args.script.read_text())
    cwd = args.script.parent

    rows: list[str] = []
    header = (
        f"{'channels':>9} {'batch':>6} {'fp32 ms':>9} {'amp ms':>9} "
        f"{'speed-up':>9} {'fp32 GiB':>9} {'amp GiB':>9}"
    )
    for ch in args.channels:
        for bs in args.batch:
            res: dict[bool, tuple[float, float]] = {}
            for amp in (False, True):
                variant = cwd / f"_bench_c{ch}_b{bs}_{'amp' if amp else 'fp32'}.json"
                write_variant(base, ch, bs, amp, variant)
                res[amp] = run_bench(
                    variant, args.repo, cwd, args.warmup, args.steps, args.device
                )
                variant.unlink(missing_ok=True)
            t0, m0 = res[False]
            t1, m1 = res[True]
            rows.append(
                f"{ch:>9} {bs:>6} {t0:>9.2f} {t1:>9.2f} {t0 / t1:>9.2f} "
                f"{m0:>9.2f} {m1:>9.2f}"
            )
            print(f"[done] channels={ch} batch={bs}")

    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()

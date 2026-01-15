# SPDX-License-Identifier: LGPL-3.0-or-later
"""Scan the DPA1 and NEP reference curves of the capacity figure.

The DPA4C curves of that figure come from ``parameter_sweep.py``, which owns
every DPA4C configuration so that the shared corner is scanned once. This
script produces only the external references, and the figure is drawn by
``plot.py`` once both halves exist.
"""

from __future__ import (
    annotations,
)

import argparse
import concurrent.futures
import os
import subprocess
import threading
from dataclasses import (
    dataclass,
)

import benchmark_runtime

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = "/aisi-vepfs/outisli/miniforge3/envs/dpmd/bin/python"


@dataclass(frozen=True)
class Reference:
    """Describe one external reference curve."""

    tag: str
    kind: str
    dpa1_shape: str = ""


REFERENCES = (
    Reference("nep", "nep"),
    Reference("dpa1_s", "dpa1", dpa1_shape="S"),
    Reference("dpa1_m", "dpa1", dpa1_shape="M"),
    Reference("dpa1_l", "dpa1", dpa1_shape="L"),
)


def _model_path(reference: Reference) -> str:
    return os.path.join(HERE, "models", f"dpa1_{reference.dpa1_shape}_compress.pt2")


def _prepare(reference: Reference, gpu: str) -> None:
    if reference.kind != "dpa1":
        return
    env = dict(os.environ)
    env["DP_CUDA_INFER"] = "2"
    env["CUDA_VISIBLE_DEVICES"] = gpu
    subprocess.run(
        [PYTHON, os.path.join(HERE, "prep_models.py"), reference.dpa1_shape],
        cwd=HERE,
        env=env,
        check=True,
    )


def _scan(reference: Reference, gpu: str, fresh: bool) -> None:
    env = dict(os.environ)
    env["BENCH_GPU"] = gpu
    env["BENCH_FRESH"] = "1" if fresh else "0"
    if reference.kind == "nep":
        subprocess.run(
            [PYTHON, os.path.join(HERE, "nep_scan.py")],
            cwd=HERE,
            env=env,
            check=True,
        )
        return
    env.update(
        {
            "BENCH_MODEL": _model_path(reference),
            "BENCH_TAG": reference.tag,
            "BENCH_INCLUDE_SMALL": "1",
        }
    )
    subprocess.run(
        [PYTHON, os.path.join(HERE, "lmp_scan.py")],
        cwd=HERE,
        env=env,
        check=True,
    )


def _worker(gpu: str, assigned: list[Reference], fresh: bool) -> None:
    for reference in assigned:
        print(f"[gpu {gpu}] {reference.tag}", flush=True)  # noqa: T201
        if fresh or not (
            reference.kind == "nep" or os.path.exists(_model_path(reference))
        ):
            _prepare(reference, gpu)
        _scan(reference, gpu, fresh)


def main() -> None:
    """Parse GPU assignments and execute the reference curves."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="re-prepare models and overwrite every result CSV",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="assume the installed DeePMD-kit and LAMMPS binaries are current",
    )
    args = parser.parse_args()
    benchmark_runtime.preflight(args.gpus, require_nep=True)
    if not args.skip_build:
        benchmark_runtime.build_runtime()

    stop = threading.Event()

    def guarded(gpu: str, assigned: list[Reference]) -> None:
        for reference in assigned:
            if stop.is_set():
                return
            try:
                _worker(gpu, [reference], args.fresh)
            except (OSError, RuntimeError, subprocess.SubprocessError, ValueError):
                stop.set()
                raise

    assignments = [
        list(REFERENCES[index :: len(args.gpus)]) for index in range(len(args.gpus))
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [
            pool.submit(guarded, gpu, assigned)
            for gpu, assigned in zip(args.gpus, assignments, strict=True)
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()

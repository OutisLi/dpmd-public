# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201
"""
Per-row drift of the first radial layer across a run's EMA checkpoints.

The first radial layer reads the Gaussian channels; row c multiplies channel c.
For every checkpoint the script reports the norm of each row's displacement
since the previous checkpoint and since the hand-over checkpoint, grouped by the
channel's centre, so that the rows the data never constrain (the innermost
centres) can be compared with the rows they do. One line of JSON per checkpoint
is appended to the output file; a summary table is printed.
"""

from __future__ import (
    annotations,
)

import argparse
import json
import re
from pathlib import (
    Path,
)

import numpy as np

KEY = "model.Default.atomic_model.descriptor.radial_embedding.net.0.matrix"
CENTRES = "model.Default.atomic_model.descriptor.radial_basis.adam_freqs"


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-step", type=int, default=10000)
    args = parser.parse_args()
    files = sorted(
        args.run.glob("ema_*.pt"),
        key=lambda f: int(re.search(r"ema_(\d+)", f.name).group(1)),
    )
    rows: dict[int, np.ndarray] = {}
    centres = None
    for f in files:
        step = int(re.search(r"ema_(\d+)", f.name).group(1))
        state = torch.load(f, map_location="cpu", weights_only=False)["model"]
        rows[step] = state[KEY].double().numpy()
        centres = state[CENTRES].double().numpy().reshape(-1)
    steps = sorted(rows)
    ref = rows.get(args.reference_step)
    print(
        f"{'step':>6s} "
        + " ".join(f"{c:6.1f}" for c in centres[:8])
        + "  | mid(2-4A) | rel. drift since hand-over: "
        + " ".join(f"{c:4.1f}" for c in centres[:6])
        + " mid"
    )
    with args.out.open("a") as stream:
        for i, step in enumerate(steps):
            w = rows[step]
            prev = rows[steps[i - 1]] if i else None
            per_row = (
                np.linalg.norm(w - prev, axis=1)
                if prev is not None
                else np.zeros(len(centres))
            )
            mid = (centres >= 2.0) & (centres <= 4.0)
            since = (
                np.linalg.norm(w - ref, axis=1) / np.linalg.norm(ref, axis=1)
                if (ref is not None and step > args.reference_step)
                else None
            )
            record = {
                "run": args.run.name,
                "step": step,
                "centres_A": centres.tolist(),
                "row_step_norm": per_row.tolist(),
                "row_relative_drift_since_reference": None
                if since is None
                else since.tolist(),
            }
            stream.write(json.dumps(record) + "\n")
            line = (
                f"{step:6d} "
                + " ".join(f"{v:6.3f}" for v in per_row[:8])
                + f"  | {per_row[mid].mean():8.3f} |"
            )
            if since is not None:
                line += (
                    " "
                    + " ".join(f"{v:4.2f}" for v in since[:6])
                    + f" {since[mid].mean():4.2f}"
                )
            print(line)


if __name__ == "__main__":
    main()

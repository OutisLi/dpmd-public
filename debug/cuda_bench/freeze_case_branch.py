# SPDX-License-Identifier: LGPL-3.0-or-later
"""Freeze a trained DPA4C checkpoint as a branch of a shared fitting network.

The fitting network gains case-embedding columns whose first-layer rows are
zero, and the one-hot case embedding selects the second branch, so the branch
computes the function of the checkpoint and runs its dynamics. The export
folds the case embedding into the first fitting layer; ``--unfolded`` keeps
it, which is the archive that the export wrote before it folded a case
embedding. Against an archive of the checkpoint itself (``freeze_dpa4c.py
--checkpoint``) the folded branch gives bitwise identical LAMMPS dumps.

Usage
-----
    DP_CUDA_INFER=2 python freeze_case_branch.py \
        --checkpoint MINI.ckpt.pt --out models/case_branch/mini.pt2
"""

from __future__ import (
    annotations,
)

import argparse

import numpy as np
from freeze_dpa4c import (
    COMPRESS_STRIDE,
    load_checkpoint,
    show_route_measurements,
)

from deepmd.pt_expt.model.model import (  # noqa: TID253
    BaseModel,
)
from deepmd.pt_expt.utils import (  # noqa: TID253
    serialization,
)


def main() -> None:
    """Parse the options and write the compressed branch archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="trained checkpoint")
    parser.add_argument("--branches", type=int, default=2, help="case columns")
    parser.add_argument(
        "--unfolded",
        action="store_true",
        help="keep the case embedding, as the export did before folding it",
    )
    parser.add_argument("--out", required=True, help="output .pt2 path")
    args = parser.parse_args()
    show_route_measurements()

    data = load_checkpoint(args.checkpoint).serialize()
    fitting = data["fitting"]
    fitting["dim_case_embd"] = args.branches
    fitting["@variables"]["case_embd"] = np.eye(args.branches, dtype=np.float32)[1]
    for network in fitting["nets"]["networks"]:
        weight = network["layers"][0]["@variables"]["w"]
        network["layers"][0]["@variables"]["w"] = np.concatenate(
            [weight, np.zeros((args.branches, weight.shape[1]), weight.dtype)]
        )
        network["in_dim"] += args.branches
    model = BaseModel.deserialize(data).eval()
    model.get_descriptor().enable_compression(1.4, table_stride_1=COMPRESS_STRIDE)
    if args.unfolded:
        serialization._fold_case_embeddings = lambda model: None
    data = {"model": model.serialize()}
    lower_kind = serialization._resolve_lower_kind(args.out, data, "auto")
    serialization.deserialize_to_file(
        args.out, data, lower_kind=lower_kind, do_atomic_virial=True
    )
    print(f"frozen -> {args.out} ({lower_kind})", flush=True)  # noqa: T201


if __name__ == "__main__":
    main()

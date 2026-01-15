# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Freeze a trained DPA4C checkpoint, compressed, with or without a long-range module.

The short-range part is the trained checkpoint, so that molecular dynamics
stays physical; the long-range module takes the package defaults of its
kernel, a neutral constraint and a seeded charge network whose output head is
scaled to give charges of order 0.1 e. The archive is the compact canonical
lower of the device that freezes it, whose fitting routes are measured there;
the route tuner prints a record of each table it measures, that of the
energy fitting and that of the charge network.

Usage
-----
    python freeze_long_range.py --checkpoint Air.ckpt.pt --kernel les --periodic \
        --out air_les_p.pt2
"""

from __future__ import (
    annotations,
)

import argparse
import copy

import torch  # noqa: TID253
from freeze_dpa4c import (
    show_route_measurements,
)

from deepmd.pt_expt.model.get_model import (  # noqa: TID253
    get_model,
)
from deepmd.pt_expt.train.wrapper import (  # noqa: TID253
    ModelWrapper,
)
from deepmd.pt_expt.utils.serialization import (  # noqa: TID253
    deserialize_to_file,
)


def main() -> None:
    """Freeze one archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--kernel", choices=("none", "les", "sog"), required=True)
    parser.add_argument("--periodic", action="store_true")
    parser.add_argument("--head-scale", type=float, default=0.05)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    show_route_measurements()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    config = copy.deepcopy(state["_extra_state"]["model_params"])
    if args.kernel != "none":
        config["long_range"] = {
            "type": "latent_charge",
            "kernel": {"type": args.kernel},
            "charge_constraint": {"source": "neutral"},
            "periodic": args.periodic,
        }
    model = get_model(copy.deepcopy(config))
    missing, unexpected = ModelWrapper(model).load_state_dict(state, strict=False)
    if unexpected or not all(".long_range." in key for key in missing):
        raise RuntimeError(f"checkpoint mismatch: {missing=} {unexpected=}")
    model.eval()
    if args.kernel != "none":
        head = model.atomic_model.long_range.charge_net.nets[0].layers[-1]
        generator = torch.Generator().manual_seed(11)
        with torch.no_grad():
            head.w.copy_(
                args.head_scale * torch.randn(head.w.shape, generator=generator)
            )
    model.atomic_model.enable_compression(1.4, table_stride_1=0.01)
    deserialize_to_file(
        args.out,
        {"model": model.serialize(), "model_def_script": config},
        lower_kind="dpa4c_canonical",
        do_atomic_virial=True,
    )
    print(f"frozen -> {args.out}", flush=True)  # noqa: T201


if __name__ == "__main__":
    main()

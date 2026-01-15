# SPDX-License-Identifier: LGPL-3.0-or-later
"""Interpolation error of the compressed DPA4C radial table at several strides.

The compressed descriptor replaces the radial network ``g(r)`` (and the mode
profiles ``q(r)``) by a quintic Hermite table. This script measures, for the
trained released checkpoints, how far the table departs from the network it
replaces as a function of the knot spacing: the network is evaluated in fp64
on a dense grid together with its derivative, the table is evaluated from its
stored fp32 coefficients both in fp64 and in the fp32 Horner form the kernels
use, and the error is reported relative to the largest magnitude of each
channel, for the value and for the radial derivative that enters the force.

Usage
-----
    python table_accuracy.py --grades nano neo --strides 0.002 0.01 0.05
"""

from __future__ import (
    annotations,
)

import argparse
import copy
import os
from typing import (
    TYPE_CHECKING,
)

import machine
import torch  # noqa: TID253

from deepmd.pt_expt.kernels.dpa4c.graph_compress import (  # noqa: TID253
    build_radial_table,
)
from deepmd.pt_expt.model.get_model import (  # noqa: TID253
    get_model,
)
from deepmd.pt_expt.train.wrapper import (  # noqa: TID253
    ModelWrapper,
)

if TYPE_CHECKING:
    from deepmd.pt_expt.descriptor.dpa4c import (
        DescrptDPA4C,
    )

#: Trained EMA checkpoint of a released grade, ``{}`` standing for the
#: capitalized grade name, from the machine profile (machine.sh).
machine.load()
TRAINED = os.environ["DPA4C_CHECKPOINTS"]


def load_trained(grade: str, device: torch.device) -> torch.nn.Module:
    """Load the trained EMA checkpoint of one released grade."""
    state = torch.load(
        TRAINED.format(grade.capitalize()), map_location=device, weights_only=True
    )
    state = state.get("model", state)
    model = get_model(state["_extra_state"]["model_params"])
    ModelWrapper(model).load_state_dict(state)
    return model.to(device).eval()


def network(
    descriptor: DescrptDPA4C, distance: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the tabulated radial maps and their radial derivative in fp64."""
    basis = copy.deepcopy(descriptor.radial_basis).to(torch.float64)
    embedding = copy.deepcopy(descriptor.radial_embedding).to(torch.float64)
    head = (
        None
        if descriptor.radial_mode_head is None
        else copy.deepcopy(descriptor.radial_mode_head).to(torch.float64)
    )
    distance = distance.clone().requires_grad_(True)
    hidden = embedding.call_hidden(basis(distance[:, None]))
    values = embedding.call_output(hidden)
    if head is not None:
        values = torch.cat([values, head.call(hidden)], dim=-1)
    slopes = torch.stack(
        [
            torch.autograd.grad(values[:, c].sum(), distance, retain_graph=True)[0]
            for c in range(values.shape[1])
        ],
        dim=-1,
    )
    return values.detach(), slopes.detach()


def table_eval(
    table: torch.Tensor, stride: float, distance: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the split-layout quintic table and its derivative."""
    width = table.shape[1] // 6
    index = torch.clamp((distance / stride).floor().long(), max=table.shape[0] - 1)
    x = (distance - index.to(distance.dtype) * stride).to(dtype)[:, None]
    row = table[index].to(dtype)
    quartet = row[:, : 4 * width].reshape(-1, width, 4)
    pair = row[:, 4 * width :].reshape(-1, width, 2)
    c0, c1, c2, c3 = quartet.unbind(-1)
    c4, c5 = pair.unbind(-1)
    value = c0 + (c1 + (c2 + (c3 + (c4 + c5 * x) * x) * x) * x) * x
    slope = c1 + (2 * c2 + (3 * c3 + (4 * c4 + 5 * c5 * x) * x) * x) * x
    return value, slope


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grades", nargs="+", default=["nano", "mini", "neo", "air", "plus"]
    )
    parser.add_argument(
        "--strides", type=float, nargs="+", default=[0.002, 0.005, 0.01, 0.02, 0.05]
    )
    parser.add_argument("--rmin", type=float, default=0.5)
    args = parser.parse_args()
    device = torch.device("cuda")
    for grade in args.grades:
        model = load_trained(grade, device)
        descriptor = model.atomic_model.fused_decomposition()[0].descriptor
        rcut = float(descriptor.rcut)
        grid = torch.arange(args.rmin, rcut, 1.0e-4, dtype=torch.float64, device=device)
        values, slopes = network(descriptor, grid)
        value_scale = values.abs().amax(0).clamp_min(1e-30)
        slope_scale = slopes.abs().amax(0).clamp_min(1e-30)
        print(  # noqa: T201
            f"== {grade}: {values.shape[1]} channels, |g| up to "
            f"{float(values.abs().max()):.3g}, |g'| up to {float(slopes.abs().max()):.3g}",
            flush=True,
        )
        for stride in args.strides:
            table, _ = build_radial_table(descriptor, stride)
            parts = []
            for dtype in (torch.float64, torch.float32):
                value, slope = table_eval(table, stride, grid, dtype)
                value_error = (
                    (value.double() - values).abs().amax(0) / value_scale
                ).max()
                slope_error = (
                    (slope.double() - slopes).abs().amax(0) / slope_scale
                ).max()
                tag = "fp64" if dtype == torch.float64 else "fp32"
                parts.append(f"{tag}: value {value_error:.2e} slope {slope_error:.2e}")
            print(  # noqa: T201
                f"  h={stride:<6g} rows={table.shape[0]:5d} "
                f"{table.numel() * 4 / 1024:8.1f} KiB  " + " | ".join(parts),
                flush=True,
            )


if __name__ == "__main__":
    main()

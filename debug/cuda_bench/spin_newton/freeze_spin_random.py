# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Freeze native-spin DPA4C benchmark models with an open spin branch.

``freeze_dpa4c.py --preset X --spin`` freezes an untrained model whose spin
branch is closed: the scalar ``spin_gate`` that multiplies the calibrated spin
invariants is initialized to zero, so the magnetic force ``fm = -dE/dS`` is
identically zero and a spin trajectory driven by that model exercises nothing
of the spin coupling. This script builds the same model, redraws every
parameter that is exactly zero from a fixed-seed normal distribution, and then
compresses and freezes it exactly as ``freeze_dpa4c.py`` does.

The default standard deviation of 0.3 is the gate magnitude at which the
untrained grades produce mean magnetic forces of 0.25 to 0.7 eV/muB on the
benchmark diamond with isotropic 2 muB moments, the magnitude range of trained
magnetic models; the response is close to linear in the gate.

Usage
-----
    DP_CUDA_INFER=2 CUDA_VISIBLE_DEVICES=0 \
    BENCH_OP_LIB=$BENCH_DP_LIB/libdeepmd_op_pt.so \
    python run_with_oplib.py freeze_spin_random.py \
        --preset dpa4c-nano-v20260911 --out ../models/spin_random/nano.pt2
"""

from __future__ import (
    annotations,
)

import argparse
import os
import sys

import torch  # noqa: TID253

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import freeze_dpa4c

from deepmd.pt_expt.model.get_model import (  # noqa: TID253
    get_model,
)
from deepmd.pt_expt.utils.serialization import (  # noqa: TID253
    deserialize_to_file,
)

#: Seed of the redrawn parameters.
REDRAW_SEED = 20261015
#: Standard deviation of the redrawn parameters.
REDRAW_SCALE = 0.3


def redraw_zero_parameters(
    model: torch.nn.Module, seed: int, scale: float
) -> list[tuple[str, torch.Tensor]]:
    """Redraw every exactly-zero parameter from ``N(0, scale**2)``.

    Parameters are visited in registration order and drawn from one generator,
    so the redrawn values depend only on the seed and the model structure.

    Parameters
    ----------
    model
        Model before compression.
    seed
        Seed of the normal generator.
    scale
        Standard deviation of the normal distribution.

    Returns
    -------
    list[tuple[str, torch.Tensor]]
        Name and new value of every redrawn parameter.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    redrawn = []
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.numel() == 0 or bool((parameter != 0).any()):
                continue
            values = scale * torch.randn(
                parameter.shape, generator=generator, dtype=torch.float64
            )
            parameter.copy_(values.to(dtype=parameter.dtype, device=parameter.device))
            redrawn.append((name, parameter.detach().cpu().clone()))
    return redrawn


def main() -> None:
    """Build, randomize, compress and freeze one native-spin preset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset", required=True, help="DPA4C preset, e.g. dpa4c-nano-v20260911"
    )
    parser.add_argument("--seed", type=int, default=REDRAW_SEED)
    parser.add_argument("--scale", type=float, default=REDRAW_SCALE)
    parser.add_argument(
        "--stride",
        type=float,
        default=freeze_dpa4c.COMPRESS_STRIDE,
        help="radial table spacing in Å",
    )
    parser.add_argument("--out", required=True, help="output .pt2 path")
    args = parser.parse_args()
    freeze_dpa4c.show_route_measurements()

    # === Step 1. Build the model of ``freeze_dpa4c.py --preset X --spin`` ===
    # The width arguments are placeholders that the preset replaces.
    model = get_model(
        freeze_dpa4c.model_config(
            channels=None,
            fitting_width=64,
            fitting_depth=3,
            lmax=2,
            radial_modes=0,
            spin=True,
            charge_state=False,
            default_charge_spin=[0.0, 1.0],
            bridging_window=None,
            preset=args.preset,
        )
    ).eval()

    # === Step 2. Open the zero-initialized parameters ===
    redrawn = redraw_zero_parameters(model, args.seed, args.scale)
    if not redrawn:
        raise RuntimeError(f"{args.preset} has no zero parameter to redraw")
    for name, value in redrawn:
        print(  # noqa: T201
            f"redrawn {name} {tuple(value.shape)}: {value.flatten()[:4].tolist()}",
            flush=True,
        )

    # === Step 3. Compress and freeze as freeze_dpa4c.py does ===
    descriptor = model.atomic_model.fused_decomposition()[0].descriptor
    descriptor.enable_compression(1.4, table_stride_1=args.stride)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    deserialize_to_file(
        args.out,
        {"model": model.serialize()},
        lower_kind="dpa4c_canonical",
        do_atomic_virial=True,
    )
    print(f"frozen -> {args.out}", flush=True)  # noqa: T201


if __name__ == "__main__":
    main()

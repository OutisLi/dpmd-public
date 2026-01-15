# SPDX-License-Identifier: LGPL-3.0-or-later
"""Freeze a compressed DPA4C model for canonical CUDA benchmarks.

The descriptor and fitting widths are independent. The fused graph fitting
operator accepts any positive float4-aligned hidden width.

Usage
-----
    DP_CUDA_INFER=2 python freeze_dpa4c.py \
        --channels 64 --fitting-width 128 --out models/dpa4c_c64_f128.pt2

``--zbl`` freezes the same architecture as a zone-bridging composition with the
ZBL repulsion, which selects the bridged specialization of the fused kernels.
"""

from __future__ import (
    annotations,
)

import argparse
from typing import (
    Any,
)

from freeze import (
    TYPE_MAP,
)

from deepmd.pt_expt.model.get_model import (  # noqa: TID253
    get_model,
)
from deepmd.pt_expt.utils.serialization import (  # noqa: TID253
    deserialize_to_file,
)

CHANNEL_WIDTHS = (8, 16, 32, 64, 128)


def model_config(
    channels: int,
    fitting_width: int,
    fitting_depth: int,
    lmax: int,
    radial_modes: int,
    spin: bool,
    charge_state: bool,
    default_charge_spin: list[float],
    bridging_window: tuple[float, float] | None = None,
) -> dict:
    """Build the benchmark model configuration.

    Parameters
    ----------
    channels
        DPA4C degree-zero channel width.
    fitting_width
        Width of every fitting hidden layer.
    fitting_depth
        Number of fitting hidden layers.
    lmax
        Maximum angular degree.
    radial_modes
        Number of shared radial mode profiles mixed per ordered type pair.
    spin
        Whether every type carries a magnetic moment, which selects the
        spin-conditioned descriptor and the spin branch of the fused kernel.
    charge_state
        Whether the descriptor is conditioned on the frame charge state.
        Compression folds the condition into the frozen tables, so this costs
        nothing per step and leaves the compiled kernel untouched.
    default_charge_spin
        The ``[charge, multiplicity]`` compression bakes into those tables.
    bridging_window
        Inner and outer radius in Å of the ZBL zone bridging, or ``None`` for
        a model without it.

    Returns
    -------
    dict
        Complete energy-model configuration.
    """
    return {
        "type_map": TYPE_MAP,
        **(
            {"spin": {"scheme": "native", "use_spin": [True] * len(TYPE_MAP)}}
            if spin
            else {}
        ),
        "descriptor": {
            "type": "dpa4c",
            "rcut": 6.0,
            "channels": channels,
            "lmax": lmax,
            "basis_type": "bessel",
            "n_radial": 16,
            "radial_modes": radial_modes,
            "precision": "float32",
            "seed": 42,
            **(
                {
                    "add_chg_spin_ebd": True,
                    "default_chg_spin": default_charge_spin,
                }
                if charge_state
                else {}
            ),
        },
        "fitting_net": {
            "type": "ener",
            "neuron": [fitting_width] * fitting_depth,
            "resnet_dt": False,
            "activation_function": "silu",
            "precision": "float32",
            "seed": 42,
        },
        **(
            {
                "bridging_method": "zbl",
                "bridging_r_inner": bridging_window[0],
                "bridging_r_outer": bridging_window[1],
            }
            if bridging_window is not None
            else {}
        ),
    }


def _activate_charge_state(descriptor: Any) -> None:
    """Give the charge-state head a non-zero output projection.

    The projection is zero initialized so that an untrained descriptor is
    independent of the condition, which is the right default but makes a
    benchmark model measure and compare a mechanism that does nothing. Drawing
    the head from a fixed seed leaves the rest of the model untouched and makes
    the condition observable.

    Parameters
    ----------
    descriptor
        A charge-conditioned DPA4C descriptor before compression.
    """
    import torch

    head = descriptor.charge_spin_embedding.network.layers[-1]
    generator = torch.Generator(device=head.w.device).manual_seed(5)
    with torch.no_grad():
        head.w.copy_(
            0.5
            * torch.randn(
                head.w.shape,
                dtype=head.w.dtype,
                device=head.w.device,
                generator=generator,
            )
        )


def main() -> None:
    """Parse benchmark dimensions and write a canonical PT2 model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", type=int, choices=CHANNEL_WIDTHS, required=True)
    parser.add_argument(
        "--fitting-width",
        type=int,
        default=64,
        help="float4-aligned fitting hidden width",
    )
    parser.add_argument("--fitting-depth", type=int, default=3)
    parser.add_argument("--lmax", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument(
        "--radial-modes",
        type=int,
        choices=(0, 2, 4, 8),
        default=0,
        help="shared radial mode profiles mixed per ordered type pair",
    )
    parser.add_argument(
        "--spin",
        action="store_true",
        help="freeze the spin-conditioned descriptor and its fused kernel",
    )
    parser.add_argument(
        "--charge-state",
        action="store_true",
        help="condition the descriptor on the frame charge state",
    )
    parser.add_argument(
        "--default-charge-spin",
        type=float,
        nargs=2,
        default=(0.0, 1.0),
        metavar=("CHARGE", "MULTIPLICITY"),
        help="charge state compression bakes into the frozen tables",
    )
    parser.add_argument(
        "--zbl",
        action="store_true",
        help="bridge the model to the ZBL repulsion",
    )
    parser.add_argument(
        "--bridging-window",
        type=float,
        nargs=2,
        default=(0.5, 0.8),
        metavar=("R_INNER", "R_OUTER"),
        help="radii in Å between which a pair enters the descriptor",
    )
    parser.add_argument("--out", required=True, help="output .pt2 path")
    args = parser.parse_args()
    if args.fitting_depth <= 0:
        raise ValueError(f"`fitting_depth` must be positive, got {args.fitting_depth}")
    if args.fitting_width <= 0 or args.fitting_width % 4:
        raise ValueError(
            "`fitting_width` must be a positive multiple of four, "
            f"got {args.fitting_width}"
        )

    model = get_model(
        model_config(
            args.channels,
            args.fitting_width,
            args.fitting_depth,
            args.lmax,
            args.radial_modes,
            args.spin,
            args.charge_state,
            list(args.default_charge_spin),
            tuple(args.bridging_window) if args.zbl else None,
        )
    ).eval()
    # A bridged model is a composition; its learned part owns the descriptor.
    descriptor = model.atomic_model.fused_decomposition()[0].descriptor
    if args.charge_state:
        _activate_charge_state(descriptor)
    descriptor.enable_compression(1.4)
    deserialize_to_file(
        args.out,
        {"model": model.serialize()},
        lower_kind="dpa4c_canonical",
        do_atomic_virial=True,
    )
    print(f"frozen -> {args.out}", flush=True)  # noqa: T201


if __name__ == "__main__":
    main()

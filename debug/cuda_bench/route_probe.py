# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Time the fitting routes of a compressed DPA4C model as the route tuner does.

The script builds the model of ``freeze_dpa4c.py --checkpoint`` or of
``spin_newton/freeze_spin_random.py --preset`` and times every compared route
of the fused energy-force operator at the given run sizes with the bench of the
freeze-time tuner (``deepmd.pt_expt.kernels.fitting_routes._Bench``): replays
of a captured CUDA graph of the whole operator on a periodic bulk system of
exactly the run size. The node types are the tuner's (``tuner``: drawn among
the first three types of the model) or those of a benchmark system
(``carbon``, or ``bcn`` with boron, carbon and nitrogen at random).

By default the replays of the routes alternate within each capture, as in the
samples of the tuner. ``--block`` times each route as a train of its own
instead, and ``--heat`` replays a route back to back for the given seconds
before its train is timed; a board at its power limit clocks a route by the
power that the route draws, which only the trains show. Per run size the
script prints the median replay time of every route, its gain over the route
of the rules with the range over the captures and, with ``--winner``, the
decision of the tuner's ``winner`` on the same inputs.

``--charge-network`` times the charge network of an LES module attached to a
trained checkpoint in place of its energy fitting, as the tuner measures it.

Usage
-----
    BENCH_OP_LIB=/path/to/libdeepmd_op_pt.so DP_CUDA_INFER=2 python route_probe.py \
        --preset dpa4c-neo-v20260911 --sizes 23040,100800 --types tuner,carbon \
        --heat 10 --block --replays 30 --captures 3
"""

from __future__ import (
    annotations,
)

import argparse
import os
import statistics
import sys
import time

import torch  # noqa: TID253

# A development build of the operator library is loaded ahead of the package,
# whose loader then finds the operators registered and keeps the installed
# library out of the process.
if os.environ.get("BENCH_OP_LIB"):
    torch.ops.load_library(os.environ["BENCH_OP_LIB"])

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "spin_newton"))

import freeze_dpa4c
import freeze_spin_random

from deepmd.pt_expt.kernels import (  # noqa: TID253
    fitting_routes,
)
from deepmd.pt_expt.kernels.graph_fitting import (  # noqa: TID253
    fitting_operator_arguments,
)
from deepmd.pt_expt.model.get_model import (  # noqa: TID253
    get_model,
)


def build_model(
    checkpoint: str | None, preset: str | None, long_range: bool = False
) -> torch.nn.Module:
    """Return the compressed model that the freeze scripts of the harness export.

    With ``long_range``, the trained checkpoint takes the LES module of
    ``freeze_long_range.py``, whose charge network the probe can time.
    """
    if checkpoint is not None and long_range:
        import copy

        from deepmd.pt_expt.train.wrapper import (
            ModelWrapper,
        )

        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = state.get("model", state)
        config = copy.deepcopy(state["_extra_state"]["model_params"])
        config["long_range"] = {
            "type": "latent_charge",
            "kernel": {"type": "les"},
            "charge_constraint": {"source": "neutral"},
        }
        model = get_model(config)
        missing, unexpected = ModelWrapper(model).load_state_dict(state, strict=False)
        if unexpected or not all(".long_range." in key for key in missing):
            raise RuntimeError(f"checkpoint mismatch: {missing=} {unexpected=}")
        model.eval()
    elif checkpoint is not None:
        model = freeze_dpa4c.load_checkpoint(checkpoint)
    else:
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
                preset=preset,
            )
        ).eval()
        freeze_spin_random.redraw_zero_parameters(
            model, freeze_spin_random.REDRAW_SEED, freeze_spin_random.REDRAW_SCALE
        )
    descriptor = model.atomic_model.fused_decomposition()[0].descriptor
    descriptor.enable_compression(1.4, table_stride_1=freeze_dpa4c.COMPRESS_STRIDE)
    return model


def heat(graph: torch.cuda.CUDAGraph, seconds: float) -> None:
    """Replay ``graph`` back to back for ``seconds``, keeping the device under load."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        for _ in range(8):
            graph.replay()
        torch.cuda.synchronize()


def route_name(choice: int) -> str:
    return "per-layer" if choice == fitting_routes.PER_LAYER else f"{choice:#x}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="trained checkpoint, as freeze_dpa4c.py")
    source.add_argument("--preset", help="native-spin preset, as freeze_spin_random.py")
    parser.add_argument("--sizes", required=True, help="run sizes, comma-separated")
    parser.add_argument(
        "--types", default="tuner", help="tuner, carbon or bcn, comma-separated"
    )
    parser.add_argument("--captures", type=int, default=4)
    parser.add_argument("--replays", type=int, default=9)
    parser.add_argument(
        "--heat", type=float, default=0.0, help="seconds of load before each train"
    )
    parser.add_argument(
        "--block", action="store_true", help="time each route as a train of its own"
    )
    parser.add_argument(
        "--winner", action="store_true", help="also print the tuner's decision"
    )
    parser.add_argument(
        "--charge-network",
        action="store_true",
        help="time the charge network of an LES module of the checkpoint",
    )
    args = parser.parse_args()

    model = build_model(args.checkpoint, args.preset, args.charge_network)
    label = args.preset or os.path.basename(os.path.dirname(args.checkpoint))
    type_map = model.get_type_map()
    sizes = [int(size) for size in args.sizes.split(",")]
    device = torch.device("cuda", 0)
    with torch.cuda.device(device), torch.no_grad():
        networks = fitting_routes._measured_networks(model)
        if args.charge_network and len(networks) < 2:
            raise SystemExit("--charge-network needs --checkpoint")
        fitting, atom_bias = networks[-1] if args.charge_network else networks[0]
        bench = fitting_routes._Bench(model, device, max(sizes), fitting, atom_bias)
        tuner_types = bench.atype.clone()
        weights = list(fitting_operator_arguments(bench.fitting).weights)
        width = int(weights[0].shape[1])
        generator = torch.Generator(device=device).manual_seed(fitting_routes.SEED + 1)
        for mode in args.types.split(","):
            # === Step 1. Node types of the synthetic systems ===
            if mode == "tuner":
                bench.atype = tuner_types
            elif mode == "carbon":
                bench.atype = torch.full_like(tuner_types, type_map.index("C"))
            elif mode == "bcn":
                elements = torch.tensor(
                    [type_map.index(e) for e in ("B", "C", "N")], device=device
                )
                draw = torch.randint(
                    0, 3, tuner_types.shape, device=device, generator=generator
                )
                bench.atype = elements[draw]
            else:
                raise ValueError(
                    f"unknown type mode {mode!r}; expected tuner, carbon or bcn"
                )
            for node_count in sizes:
                # === Step 2. Time the routes of the run size ===
                rule = int(
                    torch.ops.deepmd.graph_fitting_route_rule(weights, node_count)
                )
                choices = fitting_routes._choices(device, width, node_count)
                if rule not in choices:
                    choices.append(rule)
                inputs = bench._inputs(node_count)
                medians: dict[int, list[float]] = {choice: [] for choice in choices}
                for capture in range(args.captures):
                    graphs = bench._capture(inputs, choices, capture)
                    times: dict[int, list[float]] = {choice: [] for choice in choices}
                    heat(graphs[rule], args.heat)
                    if args.block:
                        for choice in choices:
                            heat(graphs[choice], args.heat)
                            bench._replay({choice: graphs[choice]}, args.replays, times)
                    else:
                        bench._replay(graphs, args.replays, times)
                    del graphs
                    for choice in choices:
                        medians[choice].append(statistics.median(times[choice]))
                # === Step 3. Report every route against the route of the rules ===
                print(  # noqa: T201
                    f"[{label} {mode} {node_count}] rule route = {route_name(rule)}",
                    flush=True,
                )
                for choice in choices:
                    gains = [
                        r / m - 1.0
                        for r, m in zip(medians[rule], medians[choice], strict=True)
                    ]
                    print(  # noqa: T201
                        f"  {route_name(choice):>20s}: {statistics.median(medians[choice]):8.4f} ms"
                        f" (captures {' '.join(f'{m:.4f}' for m in medians[choice])})"
                        f"  gain over rule {100 * statistics.median(gains):+6.2f}%"
                        f" [{100 * min(gains):+.2f}, {100 * max(gains):+.2f}]",
                        flush=True,
                    )
                if args.winner:
                    decision = bench.winner(node_count, rule, choices)
                    print(f"  winner(): {route_name(decision)}", flush=True)  # noqa: T201


if __name__ == "__main__":
    main()

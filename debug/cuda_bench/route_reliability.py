# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Repeat the train check of the route tuner and print the gain of every capture.

The check of a candidate route times a train of the route of the rules and one
of the per-layer kernels in each of two captures, in opposite orders, each
train after a lead-in of ``LEAD_IN_MS`` of device time of its own replays
(``deepmd.pt_expt.kernels.fitting_routes._Bench._train_gain``). This script runs
that check ``--repeats`` times per run size, node composition and lead-in with
the tuner's own bench, and prints both per-capture gains of the per-layer
kernels over the rules, their mean, which for two captures is their median, and
the median train times; a run size at which the rules take the per-layer kernels
is skipped. The node types are the tuner's (``tuner``: drawn among
the first three types of the model) or elements drawn at random per node,
joined by ``+`` (``C``, ``B+C+N``, ``H+He+Li``).

Usage
-----
    BENCH_OP_LIB=/path/to/libdeepmd_op_pt.so DP_CUDA_INFER=2 python route_reliability.py \
        --preset dpa4c-air-v20260911 --sizes 100800,129536 --types tuner,C \
        --lead-ins 500,3000 --repeats 8
"""

from __future__ import (
    annotations,
)

import argparse
import os
import statistics
import sys

import torch  # noqa: TID253

# A development build of the operator library is loaded ahead of the package,
# whose loader then finds the operators registered and keeps the installed
# library out of the process.
if os.environ.get("BENCH_OP_LIB"):
    torch.ops.load_library(os.environ["BENCH_OP_LIB"])

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import route_probe

from deepmd.pt_expt.kernels import (  # noqa: TID253
    fitting_routes,
)
from deepmd.pt_expt.kernels.graph_fitting import (  # noqa: TID253
    fitting_operator_arguments,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="trained checkpoint, as freeze_dpa4c.py")
    source.add_argument("--preset", help="native-spin preset, as freeze_spin_random.py")
    parser.add_argument("--sizes", required=True, help="run sizes, comma-separated")
    parser.add_argument(
        "--types", default="tuner", help="tuner or '+'-joined elements, comma-separated"
    )
    parser.add_argument(
        "--lead-ins", default="500", help="LEAD_IN_MS values, comma-separated"
    )
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()

    model = route_probe.build_model(args.checkpoint, args.preset)
    type_map = model.get_type_map()
    sizes = [int(size) for size in args.sizes.split(",")]
    device = torch.device("cuda", 0)
    with torch.cuda.device(device), torch.no_grad():
        fitting, atom_bias = fitting_routes._measured_networks(model)[0]
        bench = fitting_routes._Bench(model, device, max(sizes), fitting, atom_bias)
        tuner_types = bench.atype.clone()
        weights = list(fitting_operator_arguments(bench.fitting).weights)
        generator = torch.Generator(device=device).manual_seed(fitting_routes.SEED + 1)
        for mode in args.types.split(","):
            # === Step 1. Node types of the synthetic systems ===
            if mode == "tuner":
                bench.atype = tuner_types
            else:
                elements = torch.tensor(
                    [type_map.index(element) for element in mode.split("+")],
                    device=device,
                )
                draw = torch.randint(
                    0,
                    len(elements),
                    tuner_types.shape,
                    device=device,
                    generator=generator,
                )
                bench.atype = elements[draw]
            for node_count in sizes:
                rule = int(
                    torch.ops.deepmd.graph_fitting_route_rule(weights, node_count)
                )
                if rule == fitting_routes.PER_LAYER:
                    print(  # noqa: T201
                        f"{mode} {node_count}: the rules take the per-layer kernels",
                        flush=True,
                    )
                    continue
                inputs = bench._inputs(node_count)
                for lead_in in (float(value) for value in args.lead_ins.split(",")):
                    # === Step 2. Repeat the check on trains ===
                    fitting_routes.LEAD_IN_MS = lead_in
                    means: list[float] = []
                    rule_ms: list[float] = []
                    layer_ms: list[float] = []
                    for repeat in range(args.repeats):
                        gains = []
                        for capture in range(fitting_routes.TRAIN_CAPTURES):
                            choices = [rule, fitting_routes.PER_LAYER]
                            graphs = bench._capture(inputs, choices, capture)
                            order = choices if capture % 2 == 0 else choices[::-1]
                            times = {c: bench._train_ms(graphs[c]) for c in order}
                            del graphs
                            rule_ms.append(times[rule])
                            layer_ms.append(times[fitting_routes.PER_LAYER])
                            gains.append(
                                times[rule] / times[fitting_routes.PER_LAYER] - 1.0
                            )
                        means.append(statistics.mean(gains))
                        print(  # noqa: T201
                            f"{mode} {node_count} lead-in {lead_in:.0f} ms repeat {repeat}:"
                            f" captures {' '.join(f'{100 * g:+.2f}%' for g in gains)},"
                            f" mean {100 * means[-1]:+.2f}%",
                            flush=True,
                        )
                    # === Step 3. Summary of the repeats ===
                    print(  # noqa: T201
                        f"SUMMARY {mode} {node_count} lead-in {lead_in:.0f} ms: mean"
                        f" {100 * statistics.mean(means):+.2f}%, range"
                        f" {100 * min(means):+.2f}% to {100 * max(means):+.2f}%,"
                        f" {sum(mean > 0 for mean in means)} of {len(means)} positive,"
                        f" {sum(mean >= -fitting_routes.TIE for mean in means)} at least"
                        f" -TIE; median train ms: rules {statistics.median(rule_ms):.4f},"
                        f" per-layer {statistics.median(layer_ms):.4f}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()

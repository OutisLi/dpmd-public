# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Device time of the fused fitting network on one network shape.

Times ``deepmd::graph_fitting_energy_gradient``, the energy and the input
cotangent of a DPA4C fitting network -- ``--layers`` hidden layers of one
width, three by default and two for the charge network of a long-range module,
every one after the first with an identity residual, and a linear energy head
-- on random weights and inputs. The operator chooses its fused shape or the
per-layer kernels as in a model evaluation; ``--per-layer`` times the separate
forward and backward operators instead, which always take the per-layer
kernels. The operator writes the input cotangent over its input, so every
evaluation restores the input first, and the time of those copies alone is
subtracted. The checksums of the energy and the cotangent identify the result
bit for bit across builds, shapes and the two paths.

Usage
-----
    BENCH_OP_LIB=/path/to/libdeepmd_op_pt.so python bench_fitting.py \
        --width 96 --din 70 --nodes 100800
"""

from __future__ import (
    annotations,
)

import argparse
import itertools
import os

import torch  # noqa: TID253

#: Hidden activation codes of the fused fitting operator.
ACTIVATIONS = {"tanh": 0, "silu": 1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--width", type=int, required=True, help="hidden width")
    parser.add_argument(
        "--din", type=int, required=True, help="network input, the descriptor width"
    )
    parser.add_argument("--nodes", type=int, required=True, help="nodes evaluated")
    parser.add_argument("--layers", type=int, default=3, help="hidden layers")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--activation", choices=sorted(ACTIVATIONS), default="silu")
    parser.add_argument(
        "--node-tile", type=int, default=131072, help="nodes per run of the operator"
    )
    parser.add_argument(
        "--per-layer",
        action="store_true",
        help="time the separate forward and backward operators",
    )
    args = parser.parse_args()

    torch.ops.load_library(os.environ["BENCH_OP_LIB"])
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(5)
    dims = [args.din] + [args.width] * args.layers
    weights = [
        torch.randn(a, b, generator=gen, device=device) / a**0.5
        for a, b in itertools.pairwise(dims)
    ]
    biases = [
        0.1 * torch.randn(args.width, generator=gen, device=device) for _ in weights
    ]
    residuals = [0] + [1] * (args.layers - 1)
    head_weight = (
        torch.randn(args.width, generator=gen, device=device) / args.width**0.5
    )
    head_bias = torch.zeros(1, device=device)
    bias_atom_e = torch.zeros(2, device=device, dtype=torch.float64)
    x0 = torch.randn(args.nodes, args.din, generator=gen, device=device)
    atype = torch.randint(0, 2, (args.nodes,), generator=gen, device=device)
    seed = torch.ones(args.nodes, device=device, dtype=torch.float64)
    activation = ACTIVATIONS[args.activation]
    x = x0.clone()

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        """Return the energy and the input cotangent of one evaluation."""
        x.copy_(x0)
        if args.per_layer:
            energy, saved = torch.ops.deepmd.graph_fitting(
                x,
                atype,
                weights,
                biases,
                residuals,
                head_weight,
                head_bias,
                bias_atom_e,
                activation,
            )
            cotangent = torch.ops.deepmd.graph_fitting_backward(
                seed[:, None],
                saved,
                weights,
                biases,
                residuals,
                head_weight,
                activation,
            )
            return energy, cotangent
        energy = torch.ops.deepmd.graph_fitting_energy_gradient(
            x,
            atype,
            weights,
            biases,
            residuals,
            head_weight,
            head_bias,
            bias_atom_e,
            activation,
            seed,
            args.node_tile,
        )
        return energy, x

    for _ in range(5):
        energy, cotangent = run()
    # === Time the restoring copies alone, then the evaluations with them ===
    torch.cuda.synchronize()
    copy_start = torch.cuda.Event(enable_timing=True)
    copy_stop = torch.cuda.Event(enable_timing=True)
    copy_start.record()
    for _ in range(args.repeats):
        x.copy_(x0)
    copy_stop.record()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.repeats):
        energy, cotangent = run()
    stop.record()
    torch.cuda.synchronize()
    ms = (start.elapsed_time(stop) - copy_start.elapsed_time(copy_stop)) / args.repeats
    # Multiply-adds of the forward and the input-gradient backward.
    fma = args.nodes * (
        2 * args.din * args.width
        + 2 * (args.layers - 1) * args.width**2
        + 2 * args.width
    )
    print(  # noqa: T201
        f"width {args.width} din {args.din} nodes {args.nodes}: {ms:.4f} ms, "
        f"{2 * fma / ms / 1e9:.1f} TFLOP/s, "
        f"energy sum {energy.double().sum().item():.10e}, "
        f"cotangent sum {cotangent.double().sum().item():.10e}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, T201, B905
"""
Operator-level micro-benchmarks for the SeZM / DPA4 training hot spots.

Every case is timed through the full training pattern used by the force loss:
a forward pass, a first backward that differentiates with respect to the input
(the force), and a second backward that differentiates the force loss with
respect to the parameters. Candidate implementations of a case are timed
against each other under identical shapes, dtypes and seeds.

Shapes default to the ``mix:400`` OMat24 configuration measured on one
RTX PRO 6000: 396 nodes, 13930 edges, ``lmax=5``, ``mmax=1``, 128 wide
channels split into 2 focus streams.
"""

from __future__ import (
    annotations,
)

import argparse
import statistics
from typing import (
    TYPE_CHECKING,
)

import torch

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16


def timed(fn: Callable[[], None], iters: int, warmup: int) -> float:
    """
    Return the median wall time of a callable in milliseconds.

    Parameters
    ----------
    fn : Callable[[], None]
        Work to time. Must be self-contained and side-effect free across calls.
    iters : int
        Number of measured repetitions.
    warmup : int
        Number of untimed repetitions run first.

    Returns
    -------
    float
        Median elapsed time in milliseconds.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(starts, ends))


def double_backward(
    apply: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    params: list[torch.Tensor],
) -> None:
    """
    Run the forward, force, and force-loss backward of one candidate.

    Parameters
    ----------
    apply : Callable[[torch.Tensor], torch.Tensor]
        Candidate implementation, a pure function of the differentiable input.
    x : torch.Tensor
        Differentiable input; stands in for the coordinate-dependent activation.
    params : list[torch.Tensor]
        Parameters whose gradients the second backward produces.
    """
    for p in params:
        p.grad = None
    x.grad = None
    out = apply(x)
    (force,) = torch.autograd.grad(out.square().sum(), x, create_graph=True)
    force.square().sum().backward()


def compare(
    name: str,
    candidates: dict[str, tuple[Callable[[torch.Tensor], torch.Tensor], list]],
    x: torch.Tensor,
    iters: int,
    warmup: int,
) -> None:
    """
    Time every candidate of one case and print the results relative to the first.

    Parameters
    ----------
    name : str
        Case name shown in the report.
    candidates : dict[str, tuple[Callable, list]]
        Mapping from candidate name to ``(apply, params)``. The first entry is
        the reference against which the others are reported.
    x : torch.Tensor
        Differentiable input shared by all candidates.
    iters : int
        Measured repetitions per candidate.
    warmup : int
        Untimed repetitions per candidate.
    """
    print(f"\n=== {name} ===")
    reference: float | None = None
    baseline_out: torch.Tensor | None = None
    for label, (apply, params) in candidates.items():
        with torch.no_grad():
            out = apply(x).float()
        if baseline_out is None:
            baseline_out = out
            mismatch = ""
        else:
            delta = (out - baseline_out).abs().max().item()
            scale = baseline_out.abs().max().item() + 1e-30
            mismatch = f"  rel-err {delta / scale:.2e}"
        elapsed = timed(lambda: double_backward(apply, x, params), iters, warmup)
        if reference is None:
            reference = elapsed
            speedup = ""
        else:
            speedup = f"  {reference / elapsed:5.2f}x"
        print(f"  {label:<34}{elapsed:8.3f} ms{speedup}{mismatch}")


def case_focus_linear(args: argparse.Namespace) -> None:
    """Per-focus channel projection ``(B, F, Cin) -> (B, F, Cout)``."""
    b, f, c_in, c_out = args.edges, args.focus, 64, 320
    x = torch.randn(b, f, c_in, device=DEVICE, dtype=DTYPE, requires_grad=True)
    weight = torch.randn(
        c_in, f * c_out, device=DEVICE, dtype=DTYPE, requires_grad=True
    )

    def einsum_bmm(inp: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bfi,ifo->bfo", inp, weight.view(c_in, f, c_out))

    def split_mm(inp: torch.Tensor) -> torch.Tensor:
        w = weight.view(c_in, f, c_out)
        return torch.stack(
            [inp[:, i] @ w[:, i] for i in range(f)],
            dim=1,
        )

    def split_mm_contig(inp: torch.Tensor) -> torch.Tensor:
        w = weight.view(c_in, f, c_out)
        parts = [inp[:, i].contiguous() @ w[:, i].contiguous() for i in range(f)]
        return torch.stack(parts, dim=1)

    compare(
        f"FocusLinear  B={b} F={f} Cin={c_in} Cout={c_out}",
        {
            "einsum bfi,ifo->bfo (bmm)": (einsum_bmm, [weight]),
            "per-focus mm + stack": (split_mm, [weight]),
            "per-focus mm (contiguous)": (split_mm_contig, [weight]),
        },
        x,
        args.iters,
        args.warmup,
    )


def case_so2_linear(args: argparse.Namespace) -> None:
    """Block-diagonal SO(2) mixing over ``|m|`` groups in focus-major layout."""
    f, e = args.focus, args.edges
    c_f = 64
    m0, m1 = 6 * c_f, 10 * c_f
    total = m0 + m1
    x = torch.randn(f, e, total, device=DEVICE, dtype=DTYPE, requires_grad=True)
    w0 = torch.randn(f, m0, m0, device=DEVICE, dtype=DTYPE, requires_grad=True)
    w1 = torch.randn(f, m1, m1, device=DEVICE, dtype=DTYPE, requires_grad=True)

    def block_bmm(inp: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [torch.bmm(inp[:, :, :m0], w0), torch.bmm(inp[:, :, m0:], w1)], dim=-1
        )

    def dense_bmm(inp: torch.Tensor) -> torch.Tensor:
        dense = torch.zeros(f, total, total, device=DEVICE, dtype=DTYPE)
        dense[:, :m0, :m0] = w0
        dense[:, m0:, m0:] = w1
        return torch.bmm(inp, dense)

    compare(
        f"SO2Linear  F={f} E={e} blocks=({m0},{m1})",
        {
            "two block bmm + cat": (block_bmm, [w0, w1]),
            "dense block-diagonal bmm": (dense_bmm, [w0, w1]),
        },
        x,
        args.iters,
        args.warmup,
    )


def case_aggregate(args: argparse.Namespace) -> None:
    """Destination reduction of the per-edge message onto nodes."""
    n, e, d, c = args.nodes, args.edges, 36, 128
    x = torch.randn(e, d, c, device=DEVICE, dtype=DTYPE, requires_grad=True)
    weight = torch.randn(c, c, device=DEVICE, dtype=DTYPE, requires_grad=True)
    dst = torch.randint(0, n, (e,), device=DEVICE)
    dst_expand = dst.reshape(e, 1, 1).expand(e, d, c)

    def index_add(inp: torch.Tensor) -> torch.Tensor:
        message = inp @ weight
        out = message.new_zeros(n, d, c)
        return out.index_add(0, dst, message)

    def scatter_add(inp: torch.Tensor) -> torch.Tensor:
        message = inp @ weight
        out = message.new_zeros(n, d, c)
        return out.scatter_add(0, dst_expand, message)

    compare(
        f"Aggregation  N={n} E={e} D={d} C={c}",
        {
            "index_add": (index_add, [weight]),
            "scatter_add": (scatter_add, [weight]),
        },
        x,
        args.iters,
        args.warmup,
    )


def case_rotate(args: argparse.Namespace) -> None:
    """Gather the source node feature and rotate it into the edge frame."""
    n, e, d, d_m, c = args.nodes, args.edges, 36, 16, 128
    x = torch.randn(n, d, c, device=DEVICE, dtype=DTYPE, requires_grad=True)
    rot = torch.randn(e, d_m, d, device=DEVICE, dtype=DTYPE)
    src = torch.randint(0, n, (e,), device=DEVICE)
    weight = torch.randn(c, c, device=DEVICE, dtype=DTYPE, requires_grad=True)

    def gather_bmm(inp: torch.Tensor) -> torch.Tensor:
        return torch.bmm(rot, (inp @ weight).index_select(0, src))

    def bmm_rot_first(inp: torch.Tensor) -> torch.Tensor:
        return torch.bmm(rot, inp.index_select(0, src)) @ weight

    compare(
        f"RotateToLocal  N={n} E={e} D={d}->{d_m} C={c}",
        {
            "index_select then bmm": (gather_bmm, [weight]),
            "bmm then channel mix": (bmm_rot_first, [weight]),
        },
        x,
        args.iters,
        args.warmup,
    )


def case_basis_grad(args: argparse.Namespace) -> None:
    """Channel-basis gradient of the radial degree mixer.

    Contracts ``sum_{e,o,i} K[e,o,i,r] x[e,i,c] g[e,o,c]`` into an ``(R, C)``
    parameter. The contraction keeps both ``r`` and ``c`` to the end, so some
    intermediate is unavoidable; what differs between the candidates is whether
    that intermediate is materialized in a layout the reduction can consume
    without a transposing copy.
    """
    e, num_l, channels, rank = args.edges, args.lmax + 1, 128, args.rank
    kernel = torch.randn(e, num_l, num_l, rank, device=DEVICE, dtype=DTYPE)
    x = torch.randn(e, num_l, channels, device=DEVICE, dtype=DTYPE)
    grad = torch.randn(e, num_l, channels, device=DEVICE, dtype=DTYPE)

    def two_step() -> torch.Tensor:
        inner = torch.einsum("eoir,eic->eocr", kernel, x)
        return torch.einsum("eocr,eoc->rc", inner, grad)

    def three_operand() -> torch.Tensor:
        return torch.einsum("eoir,eic,eoc->rc", kernel, x, grad)

    def degree_first() -> torch.Tensor:
        weighted = torch.einsum("eoir,eoc->reic", kernel, grad)
        return (weighted * x.unsqueeze(0)).sum(dim=(1, 2), dtype=torch.float32)

    def per_rank_bmm() -> torch.Tensor:
        # One bmm per rank contracts the degree axis, leaving an (E, i, C)
        # intermediate that is R times smaller than the fused layouts and is
        # already contiguous for the elementwise reduction.
        terms = [
            (torch.bmm(kernel[:, :, :, r].transpose(1, 2), grad) * x).sum(
                dim=(0, 1), dtype=torch.float32
            )
            for r in range(rank)
        ]
        return torch.stack(terms)

    reference = two_step().float()
    print(f"\n=== ChannelBasisGrad  E={e} L={num_l} C={channels} R={rank} ===")
    for label, fn in (
        ("two-step einsum", two_step),
        ("three-operand einsum", three_operand),
        ("degree-first, fp32 reduce", degree_first),
        ("per-rank bmm, fp32 reduce", per_rank_bmm),
    ):
        out = fn()
        error = (out.float() - reference.float()).abs().max().item()
        scale = reference.abs().max().item() + 1e-30
        elapsed = timed(fn, args.iters, args.warmup)
        print(f"  {label:<34}{elapsed:8.3f} ms  rel-err {error / scale:.2e}")


def case_focus_layout(args: argparse.Namespace) -> None:
    """Cost of reaching the SO(2) mixing layout from the rotation output.

    The rotation writes ``(E, D_m, C_wide)`` with the focus stream interleaved
    into the channel axis. The mixing stack needs the focus stream as the matmul
    batch axis, i.e. ``(F, E, D_m * Cf)``. Merging ``D_m`` with ``Cf`` after the
    permute requires ``stride[D_m] == Cf``, which holds only at ``F == 1``, so at
    ``F > 1`` the reshape materializes a copy of the whole edge tensor on every
    layer. The alternative is for the producer to write the focus-major layout
    directly, which this case models as an already-contiguous operand.
    """
    e, f, d_m, c_f = args.edges, args.focus, 3 * args.lmax + 1, 64
    c_wide = f * c_f
    layers = 4

    edge_major = torch.randn(e, d_m, c_wide, device=DEVICE, dtype=DTYPE)
    focus_major = torch.randn(f, e, d_m * c_f, device=DEVICE, dtype=DTYPE)
    weight = torch.randn(f, d_m * c_f, d_m * c_f, device=DEVICE, dtype=DTYPE)

    def via_permute() -> None:
        x = edge_major.reshape(e, d_m, f, c_f).permute(2, 0, 1, 3)
        for _ in range(layers):
            flat = x.reshape(f, e, d_m * c_f)
            out = torch.bmm(flat, weight)
            x = out.reshape(f, e, d_m, c_f)

    def already_focus_major() -> None:
        x = focus_major
        for _ in range(layers):
            x = torch.bmm(x, weight)

    print(f"\n=== FocusLayout  E={e} F={f} D_m={d_m} Cf={c_f} layers={layers} ===")
    for label, fn in (
        ("permute from edge-major", via_permute),
        ("producer writes focus-major", already_focus_major),
    ):
        elapsed = timed(fn, args.iters, args.warmup)
        print(f"  {label:<34}{elapsed:8.3f} ms")


def case_tall_skinny(args: argparse.Namespace) -> None:
    """Weight-gradient contraction over the edge axis.

    ``grad_W = x^T g`` reduces the whole edge axis into a small matrix. At
    ``(M, N) = (64, 320)`` the output covers barely one 128x128 tile, so a
    plain batched GEMM occupies a couple of the device's SMs and the ``K``
    reduction runs serially inside them: the measured rate is far below both the
    arithmetic and the bandwidth roof, which is the signature of a split-K
    candidate rather than of a badly shaped kernel.
    """
    e, f, m, n = args.edges, args.focus, 64, 320
    x = torch.randn(f, e, m, device=DEVICE, dtype=DTYPE)
    g = torch.randn(f, e, n, device=DEVICE, dtype=DTYPE)

    def batched() -> torch.Tensor:
        return torch.bmm(x.transpose(1, 2), g)

    def per_focus_mm() -> torch.Tensor:
        return torch.stack([x[i].T @ g[i] for i in range(f)])

    chunks = 16
    kept = (e // chunks) * chunks

    def split_k() -> torch.Tensor:
        # Partition the reduction so each partial covers its own tile, then sum
        # the partials; this is what a split-K GEMM does internally. The tail
        # rows are folded back in separately.
        xs = x[:, :kept].reshape(f, chunks, kept // chunks, m)
        gs = g[:, :kept].reshape(f, chunks, kept // chunks, n)
        partial = torch.einsum("fcem,fcen->cfmn", xs, gs).sum(0)
        if kept < e:
            partial = partial + torch.bmm(x[:, kept:].transpose(1, 2), g[:, kept:])
        return partial

    def fold_focus() -> torch.Tensor:
        # Fold the focus axis into the output so one larger GEMM replaces the
        # batch; the off-diagonal blocks are computed and discarded.
        flat_x = x.permute(1, 0, 2).reshape(e, f * m)
        flat_g = g.permute(1, 0, 2).reshape(e, f * n)
        full = flat_x.T @ flat_g
        return torch.stack(
            [full[i * m : (i + 1) * m, i * n : (i + 1) * n] for i in range(f)]
        )

    reference = batched().float()
    flops = 2 * f * e * m * n
    print(f"\n=== TallSkinnyWeightGrad  F={f} E={e} M={m} N={n} ===")
    for label, fn in (
        ("batched bmm", batched),
        ("per-focus mm", per_focus_mm),
        ("manual split-K (16)", split_k),
        ("folded focus, one GEMM", fold_focus),
    ):
        out = fn().float()
        error = (out - reference).abs().max().item()
        scale = reference.abs().max().item() + 1e-30
        elapsed = timed(fn, args.iters, args.warmup)
        rate = flops / (elapsed * 1e-3) / 1e12
        print(
            f"  {label:<26}{elapsed:8.3f} ms{rate:9.1f} TFLOP/s"
            f"  rel-err {error / scale:.2e}"
        )


CASES = {
    "tall_skinny": case_tall_skinny,
    "focus_layout": case_focus_layout,
    "focus_linear": case_focus_linear,
    "so2_linear": case_so2_linear,
    "aggregate": case_aggregate,
    "rotate": case_rotate,
    "basis_grad": case_basis_grad,
}


def main() -> None:
    """Parse the command line and run the selected cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", choices=list(CASES))
    parser.add_argument("--edges", type=int, default=13930)
    parser.add_argument("--nodes", type=int, default=396)
    parser.add_argument("--focus", type=int, default=2)
    parser.add_argument("--lmax", type=int, default=5)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"device: {torch.cuda.get_device_name(0)}  dtype: {DTYPE}")
    for name in args.cases or list(CASES):
        CASES[name](args)


if __name__ == "__main__":
    main()

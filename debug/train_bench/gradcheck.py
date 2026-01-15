#!/usr/bin/env python
# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, T201, B905
"""
Second-order gradient verification for the fused SeZM training operators.

Each case pits a fused operator against the eager reference that defines it and
compares three things: the forward value, the first-order gradients with respect
to every differentiable input, and the second-order gradients obtained by
differentiating a linear functional of the first-order gradients. The third
check is the one that matters for training, since the force loss differentiates
the backward pass itself.

The comparison is run in float64 against the reference where the kernels permit
it and in float32 otherwise; tolerances are set per dtype.

Examples
--------
Run every case::

    python debug/train_bench/gradcheck.py

Run one case verbosely::

    python debug/train_bench/gradcheck.py rotate_back_block_so2 -v
"""

from __future__ import (
    annotations,
)

import argparse
import traceback
from typing import (
    TYPE_CHECKING,
)

import torch
from torch import (
    Tensor,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

DEVICE = torch.device("cuda")

CASES: dict[str, Callable[[argparse.Namespace], None]] = {}


def case(name: str) -> Callable[[Callable], Callable]:
    """
    Register a verification case under a name.

    Parameters
    ----------
    name : str
        Case name used on the command line.

    Returns
    -------
    Callable
        Decorator that records the function and returns it unchanged.
    """

    def wrap(fn: Callable) -> Callable:
        CASES[name] = fn
        return fn

    return wrap


def _flat_outputs(value: object) -> list[Tensor]:
    """Return the tensor outputs of a call as a flat list."""
    if isinstance(value, torch.Tensor):
        return [value]
    return [v for v in value if isinstance(v, torch.Tensor)]


def degree_block_mask(lmax: int, dtype: torch.dtype) -> Tensor:
    r"""
    Return the degree-block sparsity pattern of a Wigner-D matrix.

    A Wigner-D matrix is block diagonal over the degree :math:`l`, with block
    :math:`l` spanning rows and columns :math:`l^2 \\ldots (l+1)^2`. The fused
    rotation kernels read and differentiate only those structural non-zeros,
    while a dense eager reference touches the whole matrix. Multiplying the
    differentiable leaf by this mask puts both paths on the same footing: the
    off-block entries are constant zero for the kernel and are annihilated for
    the reference, so their gradients agree everywhere.

    Parameters
    ----------
    lmax : int
        Maximum degree.
    dtype : torch.dtype
        Element type of the result.

    Returns
    -------
    Tensor
        Mask with shape ``((lmax+1)**2, (lmax+1)**2)``, one inside the blocks.
    """
    dim = (lmax + 1) ** 2
    mask = torch.zeros(dim, dim, device=DEVICE, dtype=dtype)
    for degree in range(lmax + 1):
        start, stop = degree**2, (degree + 1) ** 2
        mask[start:stop, start:stop] = 1.0
    return mask


def second_order(
    fn: Callable[..., object],
    inputs: list[torch.Tensor],
    cotangents: list[torch.Tensor],
    tangents: list[torch.Tensor],
    first_order_only: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
    r"""
    Evaluate the first-order gradients and one second-order projection.

    The projection differentiates :math:`\\sum_i \\langle t_i, g_i \\rangle`,
    where ``g`` are the first-order gradients driven by the fixed cotangents.
    Any second-order bug shows up here because the projection touches every
    component of the backward.

    Parameters
    ----------
    fn : Callable
        Operator under test; returns one tensor or a tuple of tensors.
    inputs : list[torch.Tensor]
        Differentiable inputs, each with ``requires_grad=True``.
    cotangents : list[torch.Tensor]
        Output cotangents, one per tensor output of ``fn``.
    tangents : list[torch.Tensor]
        Weights of the linear functional applied to the first-order gradients,
        one per entry of ``inputs``.
    first_order_only : bool, default=False
        Stop after the first-order gradients. Used while an operator's backward
        is complete but its own autograd formula is not yet in place.

    Returns
    -------
    tuple[list[torch.Tensor], list[torch.Tensor | None]]
        The first-order gradients and the second-order gradients.
    """
    outputs = _flat_outputs(fn())
    grads = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=cotangents[: len(outputs)],
        create_graph=first_order_only is False,
        allow_unused=True,
    )
    detached = [g.detach() if g is not None else None for g in grads]
    if first_order_only:
        return detached, [None] * len(inputs)
    projection = sum((g * t).sum() for g, t in zip(grads, tangents) if g is not None)
    second = torch.autograd.grad(projection, inputs, allow_unused=True)
    return detached, list(second)


# Tolerance of the autocast pass, set by ``main``; None disables it. Under
# bf16 the fused and the dense path accumulate rounding differently, so the
# comparison bounds implementation divergence rather than machine precision.
AMP_TOL: float | None = None


def _autocast_call(fn: Callable[..., object]) -> Callable[..., object]:
    """Run a forward under bf16 autocast, as the training step does."""

    def wrapped() -> object:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return fn()

    return wrapped


def report(
    name: str,
    fused: Callable[..., object],
    reference: Callable[..., object],
    inputs: list[torch.Tensor],
    labels: list[str],
    seed: int,
    tol: float,
    verbose: bool,
    first_order_only: bool = False,
) -> bool:
    """
    Compare a fused operator against its reference through second order.

    Every case runs twice: once in the ambient precision with ``tol``, and --
    when :data:`AMP_TOL` is set and the inputs are fp32 -- once more with both
    forwards under bf16 autocast, mirroring a production training step. The
    autocast pass compares the fused path against the dense path in the same
    regime, so it catches dtype-handling defects that a pure-fp32 check cannot
    reach.

    Parameters
    ----------
    name : str
        Case name shown in the report.
    fused : Callable
        Fused operator, called with no arguments (close over the inputs).
    reference : Callable
        Eager reference, called with no arguments.
    inputs : list[torch.Tensor]
        Differentiable inputs shared by both callables.
    labels : list[str]
        Human-readable name of each input, used in failure messages.
    seed : int
        Seed for the random cotangents and tangents.
    tol : float
        Maximum tolerated relative error.
    verbose : bool
        Print the per-input relative errors even when they pass.

    Returns
    -------
    bool
        Whether every comparison stayed within tolerance.
    """
    ok = _report_pass(
        name, fused, reference, inputs, labels, seed, tol, verbose, first_order_only
    )
    if AMP_TOL is not None and inputs[0].dtype is torch.float32:
        ok = (
            _report_pass(
                f"{name}[amp]",
                _autocast_call(fused),
                _autocast_call(reference),
                inputs,
                labels,
                seed,
                AMP_TOL,
                verbose,
                first_order_only,
            )
            and ok
        )
    return ok


def _report_pass(
    name: str,
    fused: Callable[..., object],
    reference: Callable[..., object],
    inputs: list[torch.Tensor],
    labels: list[str],
    seed: int,
    tol: float,
    verbose: bool,
    first_order_only: bool,
) -> bool:
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    with torch.no_grad():
        ref_out = _flat_outputs(reference())
        fused_out = _flat_outputs(fused())
    cotangents = [
        torch.randn(o.shape, device=o.device, dtype=o.dtype, generator=generator)
        for o in ref_out
    ]
    tangents = [
        torch.randn(i.shape, device=i.device, dtype=i.dtype, generator=generator)
        for i in inputs
    ]

    ref_first, ref_second = second_order(
        reference, inputs, cotangents, tangents, first_order_only
    )
    fused_first, fused_second = second_order(
        fused, inputs, cotangents, tangents, first_order_only
    )

    failures: list[str] = []

    def compare(tag: str, a: torch.Tensor | None, b: torch.Tensor | None) -> None:
        if a is None and b is None:
            return
        if a is None or b is None:
            failures.append(
                f"{tag}: one side is None (ref={a is None} fused={b is None})"
            )
            return
        scale = a.abs().max().item()
        error = (a.float() - b.float()).abs().max().item() / (scale + 1e-30)
        if verbose or error > tol:
            print(f"    {tag:<34} rel-err {error:.3e}")
        if error > tol:
            failures.append(f"{tag}: rel-err {error:.3e} > {tol:.1e}")

    for i, (a, b) in enumerate(zip(ref_out, fused_out)):
        compare(f"forward[{i}]", a, b)
    for label, a, b in zip(labels, ref_first, fused_first):
        compare(f"d/d{label}", a, b)
    for label, a, b in zip(labels, ref_second, fused_second):
        compare(f"d2/d{label}", a, b)

    status = "PASS" if not failures else "FAIL"
    print(f"  [{status}] {name}")
    for line in failures:
        print(f"    !! {line}")
    return not failures


# ======================================================================
# Cases
# ======================================================================
@case("rotate_back_block_so2")
def _case_rotate_back_block_so2(args: argparse.Namespace) -> None:
    """Block-diagonal local->global rotation reading the per-focus layout."""
    from deepmd.pt.model.descriptor.sezm_nn.indexing import (
        build_m_major_index,
    )
    from deepmd.pt_expt.kernels.triton.sezm.so2_rotation import (
        rotate_back_block_so2,
        rotate_back_reference,
    )

    lmax, n_focus, focus_dim = args.lmax, args.focus, 32
    n_edge = args.edges
    dim = (lmax + 1) ** 2
    reduced = (lmax + 1) + 2 * lmax
    dtype = torch.float64 if args.fp64 else torch.float32

    x = torch.randn(
        n_edge, n_focus, reduced, focus_dim, device=DEVICE, dtype=dtype
    ).requires_grad_()
    wigner_raw = torch.randn(
        n_edge, dim, dim, device=DEVICE, dtype=dtype
    ).requires_grad_()
    mask = degree_block_mask(lmax, dtype)
    coeff = build_m_major_index(lmax, 1, device=DEVICE)

    def eager() -> torch.Tensor:
        x_std = x.transpose(1, 2).reshape(n_edge, reduced, n_focus * focus_dim)
        return rotate_back_reference(x_std, wigner_raw * mask, coeff, dim)

    report(
        "rotate_back_block_so2",
        lambda: rotate_back_block_so2(x, wigner_raw * mask, lmax),
        eager,
        [x, wigner_raw],
        ["x_local", "wigner"],
        args.seed,
        args.tol_fp64 if args.fp64 else args.tol,
        args.verbose,
        args.first_order,
    )


@case("rotate_to_local_block")
def _case_rotate_to_local_block(args: argparse.Namespace) -> None:
    """Block-diagonal global->local rotation with the source gather fused in."""
    from deepmd.pt.model.descriptor.sezm_nn.indexing import (
        build_m_major_index,
    )
    from deepmd.pt_expt.kernels.triton.sezm.so2_rotation import (
        rotate_to_local_block,
        rotate_to_local_reference,
    )

    lmax, channels = args.lmax, 64
    n_node, n_edge = args.nodes, args.edges
    dim = (lmax + 1) ** 2
    dtype = torch.float64 if args.fp64 else torch.float32

    x = torch.randn(n_node, dim, channels, device=DEVICE, dtype=dtype).requires_grad_()
    wigner_raw = torch.randn(
        n_edge, dim, dim, device=DEVICE, dtype=dtype
    ).requires_grad_()
    mask = degree_block_mask(lmax, dtype)
    src = torch.randint(0, n_node, (n_edge,), device=DEVICE)
    coeff = build_m_major_index(lmax, 1, device=DEVICE)

    report(
        "rotate_to_local_block",
        lambda: rotate_to_local_block(x, src, wigner_raw * mask, lmax),
        lambda: rotate_to_local_reference(x, src, wigner_raw * mask, coeff, dim),
        [x, wigner_raw],
        ["x", "wigner"],
        args.seed,
        args.tol_fp64 if args.fp64 else args.tol,
        args.verbose,
        args.first_order,
    )


@case("block_diag_gemm")
def _case_block_diag_gemm(args: argparse.Namespace) -> None:
    """Block-diagonal SO(2) GEMM over the ``|m|`` groups."""
    from deepmd.pt_expt.kernels.triton.sezm.so2_block_gemm import (
        block_diag_gemm,
        block_diag_gemm_reference,
    )

    n_focus, n_edge = args.focus, args.edges
    lmax, channels = args.lmax, 64
    m0 = (lmax + 1) * channels
    m1 = 2 * lmax * channels
    slices = [(0, m0, 0, m0), (m0, m0 + m1, m0, m0 + m1)]
    dtype = torch.float64 if args.fp64 else torch.float32

    x = torch.randn(
        n_focus, n_edge, m0 + m1, device=DEVICE, dtype=dtype
    ).requires_grad_()
    weight = torch.randn(
        n_focus, m0 + m1, m0 + m1, device=DEVICE, dtype=dtype
    ).requires_grad_()

    report(
        "block_diag_gemm",
        lambda: block_diag_gemm(x, weight, slices),
        lambda: block_diag_gemm_reference(x, weight, slices),
        [x, weight],
        ["x_flat", "weight"],
        args.seed,
        args.tol_fp64 if args.fp64 else args.tol,
        args.verbose,
        args.first_order,
    )


@case("radial_mix")
def _case_radial_mix(args: argparse.Namespace) -> None:
    """Edge-conditioned low-rank degree mixer in the reduced SO(2) layout."""
    from deepmd.pt_expt.kernels.triton.sezm.radial_mix import (
        radial_mix_block,
        radial_mix_reference,
    )

    lmax, channels, rank = args.lmax, 64, args.rank
    n_edge = args.edges
    reduced = (lmax + 1) + 2 * lmax
    kernel_size = (lmax + 1) ** 2 + lmax**2
    dtype = torch.float64 if args.fp64 else torch.float32

    compact = torch.randn(
        n_edge, kernel_size, rank, device=DEVICE, dtype=dtype
    ).requires_grad_()
    x_local = torch.randn(
        n_edge, reduced, channels, device=DEVICE, dtype=dtype
    ).requires_grad_()
    basis = torch.randn(rank, channels, device=DEVICE, dtype=dtype).requires_grad_()

    report(
        "radial_mix",
        lambda: radial_mix_block(compact, x_local, basis, lmax),
        lambda: radial_mix_reference(compact, x_local, basis, lmax),
        [compact, x_local, basis],
        ["compact", "x_local", "channel_basis"],
        args.seed,
        args.tol_fp64 if args.fp64 else args.tol,
        args.verbose,
        args.first_order,
    )


@case("flash_atten")
def _case_flash_atten(args: argparse.Namespace) -> None:
    """Fused rotate-back, attention weighting and destination reduction."""
    from deepmd.pt_expt.kernels.triton.sezm.flash_atten import (
        flash_atten_aggregate,
        flash_atten_aggregate_reference,
    )

    lmax, n_focus, focus_dim = args.lmax, args.focus, 32
    n_node, n_edge, n_head = args.nodes, args.edges, 1
    dim = (lmax + 1) ** 2
    reduced = (lmax + 1) + 2 * lmax
    dtype = torch.float64 if args.fp64 else torch.float32

    x_local = torch.randn(
        n_edge, n_focus, reduced, focus_dim, device=DEVICE, dtype=dtype
    ).requires_grad_()
    wigner_raw = torch.randn(
        n_edge, dim, dim, device=DEVICE, dtype=dtype
    ).requires_grad_()
    mask = degree_block_mask(lmax, dtype)
    rescale = torch.rand(dim, device=DEVICE, dtype=dtype) + 0.5
    alpha = torch.randn(
        n_edge, n_focus, n_head, device=DEVICE, dtype=dtype
    ).requires_grad_()
    dst = torch.randint(0, n_node, (n_edge,), device=DEVICE)
    order = torch.argsort(dst)
    counts = torch.bincount(dst, minlength=n_node)
    row_ptr = torch.cat(
        [torch.zeros(1, device=DEVICE, dtype=torch.long), counts.cumsum(0)]
    )

    report(
        "flash_atten",
        lambda: flash_atten_aggregate(
            x_local,
            wigner_raw * mask,
            rescale,
            alpha,
            order,
            row_ptr,
            dst,
            lmax,
            n_head,
        ),
        lambda: flash_atten_aggregate_reference(
            x_local, wigner_raw * mask, rescale, alpha, dst, n_node, lmax, n_head
        ),
        [x_local, wigner_raw, alpha],
        ["x_local", "wigner", "alpha"],
        args.seed,
        args.tol_fp64 if args.fp64 else args.tol,
        args.verbose,
        args.first_order,
    )


@case("rotate_mix")
def _case_rotate_mix(args: argparse.Namespace) -> None:
    """Fused rotate-to-local and radial degree mixing of the SO(2) value path."""
    from deepmd.pt_expt.kernels.triton.sezm.so2_value_path import (
        _rotate_mix_op,
        _rotate_mix_reference,
    )

    lmax, n_focus, rank = args.lmax, args.focus, args.rank
    n_node, n_edge = args.nodes, args.edges
    c_wide = args.focus_dim * n_focus
    dim = (lmax + 1) ** 2
    kernel_size = (lmax + 1) ** 2 + lmax**2
    dtype = torch.float64 if args.fp64 else torch.float32

    x = torch.randn(n_node, dim, c_wide, device=DEVICE, dtype=dtype).requires_grad_()
    wigner_raw = torch.randn(
        n_edge, dim, dim, device=DEVICE, dtype=dtype
    ).requires_grad_()
    mask = degree_block_mask(lmax, dtype)
    # rank 0 is the mixer-free variant: kc carries per-degree radial features.
    kc_cols = kernel_size * rank if rank > 0 else (lmax + 1) * c_wide
    kc = torch.randn(n_edge, kc_cols, device=DEVICE, dtype=dtype).requires_grad_()
    cb = torch.randn(max(rank, 1), c_wide, device=DEVICE, dtype=dtype).requires_grad_()
    src = torch.randint(0, n_node, (n_edge,), device=DEVICE)
    order = torch.argsort(src)
    row_ptr = torch.cat(
        [
            torch.zeros(1, device=DEVICE, dtype=torch.long),
            torch.bincount(src, minlength=n_node).cumsum(0),
        ]
    )

    report(
        "rotate_mix",
        lambda: _rotate_mix_op(
            x, src, order, row_ptr, wigner_raw * mask, kc, cb, lmax, n_focus, rank
        ),
        lambda: _rotate_mix_reference(
            x, src, wigner_raw * mask, kc, cb, lmax, n_focus, rank
        ),
        [x, wigner_raw, kc, cb],
        ["x", "wigner", "kc", "channel_basis"],
        args.seed,
        args.tol_fp64 if args.fp64 else args.tol,
        args.verbose,
        args.first_order,
    )


@case("mixing_stack")
def _case_mixing_stack(args: argparse.Namespace) -> None:
    """Gated SO(2) mixing stack: the one operator that is not multilinear."""
    from deepmd.pt_expt.kernels.triton.sezm.so2_value_path import (
        _mixing_stack_op,
        _mixing_stack_reference,
    )

    lmax, n_focus, focus_dim = args.lmax, args.focus, args.focus_dim
    n_edge, n_gated = args.edges, args.layers
    m0 = (lmax + 1) * focus_dim
    m1 = 2 * lmax * focus_dim
    row = m0 + m1
    dtype = torch.float64 if args.fp64 else torch.float32
    scale = 0.1

    u0 = torch.randn(n_focus, n_edge, row, device=DEVICE, dtype=dtype).requires_grad_()
    alpha = (
        torch.rand(n_edge, n_focus, device=DEVICE, dtype=dtype) + 0.5
    ).requires_grad_()
    w0 = (
        torch.randn(n_gated + 1, n_focus, m0, m0, device=DEVICE, dtype=dtype) * scale
    ).requires_grad_()
    w1 = (
        torch.randn(n_gated + 1, n_focus, m1, m1, device=DEVICE, dtype=dtype) * scale
    ).requires_grad_()
    gw = (
        torch.randn(
            n_gated, n_focus, focus_dim, lmax * focus_dim, device=DEVICE, dtype=dtype
        )
        * scale
    ).requires_grad_()

    # Only the first output is a real output: the pre-activations and the final
    # activation exist so the backward can avoid storing them, and no consumer
    # ever sends a gradient into them.
    report(
        "mixing_stack",
        lambda: _mixing_stack_op(u0, alpha, w0, w1, gw, lmax, focus_dim, True)[0],
        lambda: _mixing_stack_reference(u0, alpha, w0, w1, gw, lmax, focus_dim, True)[
            0
        ],
        [u0, alpha, w0, w1, gw],
        ["u0", "alpha", "w0_all", "w1_all", "gw_all"],
        args.seed,
        args.tol_fp64 if args.fp64 else args.tol,
        args.verbose,
        args.first_order,
    )


def main() -> None:
    """Parse the command line and run the selected cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", choices=list(CASES))
    parser.add_argument("--lmax", type=int, default=3)
    parser.add_argument("--focus", type=int, default=1)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--focus-dim", type=int, default=64)
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--edges", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=2e-4)
    parser.add_argument("--tol-fp64", type=float, default=1e-10)
    parser.add_argument(
        "--amp-tol",
        type=float,
        default=5e-2,
        help="tolerance of the bf16 autocast pass (fused vs dense in bf16)",
    )
    parser.add_argument(
        "--skip-amp",
        action="store_true",
        help="run the ambient-precision pass only",
    )
    parser.add_argument("--fp64", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--first-order",
        action="store_true",
        help="stop at the first-order gradients",
    )
    args = parser.parse_args()

    global AMP_TOL
    AMP_TOL = None if args.skip_amp or args.fp64 else args.amp_tol
    torch.manual_seed(args.seed)
    selected = args.cases or list(CASES)
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"lmax={args.lmax} focus={args.focus} nodes={args.nodes} edges={args.edges}")
    failed = []
    for name in selected:
        try:
            CASES[name](args)
        except Exception:
            print(f"  [ERROR] {name}")
            traceback.print_exc()
            failed.append(name)
    if failed:
        raise SystemExit(f"cases raised: {', '.join(failed)}")


if __name__ == "__main__":
    main()

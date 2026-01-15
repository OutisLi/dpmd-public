#!/usr/bin/env python
# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201, ANN001, ANN201, ANN202, TID253, B905
"""Parity check for the fused CUDA SO(2) value path (training form).

Runs the mega-kernel operator chain (forward, first-order backward,
second-order projection) against the eager reference composition
(rotate-mix reference, competition head, mixing-stack reference) on
identical inputs, in ambient fp32 and under bf16 autocast. Wigner operands
are masked to the degree-block structure so the kernel (which reads only
structural non-zeros) and the dense reference agree by construction.

The verdict is the arbitration of the operator unit tests
(source/tests/pt_expt/kernels/conditioning.py): the fused error against the
fp64 ground truth is bounded by a multiple of the eager reference's own error
in the same working precision, and the bound is applied to the median over
independent operand draws. A single draw is no verdict: the extreme error of a
quantity with few elements, such as the two-entry competition-bias gradient,
swings by an order of magnitude between draws while both sides carry the same
rounding.

The fp64 pair separates logic errors from precision. Evaluated in float64,
the fused operator and the eager reference differ by reduction order only,
which every draw keeps below a fixed ceiling, while a structural error leaves
a disagreement many orders of magnitude above it. The pair needs the float64
kernels, which only a build with ``DEEPMD_ENABLE_DPA4_FP64=ON`` carries;
without them the fp32 and bf16 comparisons still run.
"""

import sys
from pathlib import (
    Path,
)

import torch

from deepmd.pt.cxx_op import ENABLE_CUSTOMIZED_OP  # noqa: F401
from deepmd.pt_expt.kernels.cuda.dpa4.so2_conv_train import (
    _value_train_op,
    float64_available,
    op_available,
)
from deepmd.pt_expt.kernels.triton.sezm.so2_value_path import (
    _mixing_stack_reference,
    _rotate_mix_reference,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from source.tests.pt_expt.kernels.conditioning import (
    deviations,
    median_deviations,
)

# ``(lmax, n_focus, focus_dim, mixing_layers, mixer_rank, focus_compete)``.
SHAPES = (
    (3, 2, 32, 3, 1, True),
    (5, 2, 64, 4, 2, True),
    (3, 1, 64, 3, 0, False),
    (6, 2, 96, 4, 1, True),
)
N_NODE = 700
N_EDGE = 3001
SOFTMAX_TAU = 1.0
LABEL_SMOOTHING = 0.02
NORM_EPS = 1e-7

# Independent operand draws per shape; the verdict is the median over them.
DRAWS = 5
# Multiple of the eager error attributed to reduction order, as in the unit
# tests, with the competition-bias curvature's distinct conditioning bound.
FACTOR = 4.0
FACTOR_OVERRIDES = {"g2_bias": 8.0}
# Largest float64 disagreement of the fused operator from the eager reference
# that reduction order explains, relative to the ground-truth scale. Double
# rounding through the deepest second-order chain here stays near 3e-11, while
# float32 arithmetic in any kernel of the chain leaves disagreements on the
# order of 1e-7.
FLOAT64_CEILING = 1e-9

LEAVES = ("x", "wig", "kc", "cb", "wfc", "bias", "w0", "w1", "gw")


def block_mask(lmax, device):
    dim = (lmax + 1) ** 2
    mask = torch.zeros(dim, dim, device=device, dtype=torch.float64)
    for l in range(lmax + 1):
        b = l * l
        w = 2 * l + 1
        mask[b : b + w, b : b + w] = 1.0
    return mask


def pack_wigner_rows(wigner, lmax):
    """Pack the m=0 and m=+-1 rows consumed by the reduced rotation."""
    m0, mm, mp = [], [], []
    for l in range(lmax + 1):
        start, end = l * l, (l + 1) ** 2
        row0 = start + l
        m0.append(wigner[:, row0, start:end])
        if l >= 1:
            mm.append(wigner[:, row0 - 1, start:end])
            mp.append(wigner[:, row0 + 1, start:end])
    return torch.cat(m0 + mm + mp, dim=1)


def relative_error(truth, value):
    """Max-norm distance relative to the ground-truth scale (floored at one)."""
    scale = truth.abs().max().clamp_min(1.0).item()
    return (value - truth).abs().max().item() / scale


def make_case(shape, seed, second):
    """Draw one operand set of a block shape.

    The operands are drawn once in float64 and cast per evaluation, so every
    evaluation of the draw sees the same numbers and differs only in the
    arithmetic that consumes them.

    Returns
    -------
    tuple
        The quantity names, the evaluator ``evaluate(fused, dtype, amp)`` and
        the projection onto the comparison domain.
    """
    lmax, focus, cf, layers, rank, apply_alpha = shape
    torch.manual_seed(seed)
    device = "cuda"
    double = {"device": device, "dtype": torch.float64}
    dim = (lmax + 1) ** 2
    c_wide = focus * cf
    m0 = (lmax + 1) * cf
    m1 = 2 * lmax * cf
    n_gated = layers - 1
    mask = block_mask(lmax, device)

    x0 = torch.randn(N_NODE, dim, c_wide, **double)
    src = torch.randint(0, N_NODE, (N_EDGE,), device=device, dtype=torch.long)
    wig0 = torch.randn(N_EDGE, dim, dim, **double) * mask.unsqueeze(0)
    if rank == 0:
        kc0 = torch.randn(N_EDGE, lmax + 1, c_wide, **double)
        cb0 = torch.zeros(1, **double)
    else:
        ksz = (lmax + 1) ** 2 + lmax * lmax
        kc0 = 0.3 * torch.randn(N_EDGE, ksz * rank, **double)
        cb0 = torch.randn(rank, c_wide, **double)
    wfc0 = 0.05 * torch.randn(cf, focus, **double)
    bias0 = 0.05 * torch.randn(focus, **double)
    w00 = 0.2 * torch.randn(n_gated + 1, focus, m0, m0, **double)
    w10 = 0.2 * torch.randn(n_gated + 1, focus, m1, m1, **double)
    gw0 = 0.3 * torch.randn(n_gated, focus, cf, lmax * cf, **double)
    cot = torch.randn(N_EDGE, focus, (3 * lmax + 1) * cf, **double)
    # A force loss also sends cotangents through the Wigner and degree-kernel
    # gradients, whose producers precede the operator on the coordinate
    # graph; the Wigner cotangent lives on the structural block diagonal.
    h_x = torch.randn_like(x0)
    h_wig = torch.randn_like(wig0) * mask
    h_kc = torch.randn_like(kc0)

    src_order = torch.argsort(src, dim=0, stable=True)
    counts = src.new_zeros(N_NODE).scatter_add(0, src, torch.ones_like(src))
    src_rowptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)])

    operands = (x0, wig0, kc0, cb0, wfc0, bias0, w00, w10, gw0)
    requires_grad = (
        True,
        True,
        True,
        rank > 0,
        apply_alpha,
        apply_alpha,
        True,
        True,
        True,
    )
    active = [name for name, req in zip(LEAVES, requires_grad) if req]
    names = ["fwd"] + [f"g_{name}" for name in active]
    if second:
        names += [f"g2_{name}" for name in active]

    def competition(u0, wfc, bias):
        """Label-smoothed cross-focus softmax over the scalar gate rows.

        The head runs in the operator's accumulator precision: float32 for
        the float32 and bfloat16 evaluations, float64 for the ground truth.
        """
        if not apply_alpha:
            return torch.ones(N_EDGE, focus, device=device, dtype=u0.dtype)
        acc = torch.float64 if u0.dtype == torch.float64 else torch.float32
        gate = u0[:, :, :cf].permute(1, 0, 2).to(acc)
        logits = torch.einsum("efi,if->ef", gate, wfc.to(acc)) + bias.to(acc)
        p = torch.softmax(logits / SOFTMAX_TAU, dim=1)
        return (p * (1.0 - LABEL_SMOOTHING) + LABEL_SMOOTHING / focus).to(u0.dtype)

    def evaluate(fused, dtype, amp):
        leaves = tuple(
            operand.to(dtype).clone().requires_grad_(req)
            for operand, req in zip(operands, requires_grad)
        )
        targets = [leaf for leaf in leaves if leaf.requires_grad]
        # The fused operator's autocast rule casts every floating-point input
        # to bfloat16; the eager reference is fed the same casts explicitly so
        # both sides run one numerical regime.
        inputs = tuple(leaf.to(torch.bfloat16) for leaf in leaves) if amp else leaves
        x, wig, kc, cb, wfc, bias, w0, w1, gw = inputs
        kc_flat = kc.reshape(N_EDGE, -1) if rank > 0 else kc
        cb_flat = cb.reshape(-1) if rank > 0 else cb
        ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if amp
            else torch.autocast("cuda", enabled=False)
        )
        with ctx:
            if fused:
                out, _, _, _ = _value_train_op(
                    x,
                    src,
                    src_order,
                    src_rowptr,
                    pack_wigner_rows(wig, lmax),
                    kc_flat,
                    cb_flat,
                    wfc if apply_alpha else None,
                    bias if apply_alpha else None,
                    None,
                    w0,
                    w1,
                    gw,
                    lmax,
                    focus,
                    rank,
                    apply_alpha,
                    SOFTMAX_TAU,
                    LABEL_SMOOTHING,
                    NORM_EPS,
                )
            else:
                u0 = _rotate_mix_reference(
                    x, src, wig, kc_flat, cb_flat, lmax, focus, rank
                )
                out, _, _ = _mixing_stack_reference(
                    u0, competition(u0, wfc, bias), w0, w1, gw, lmax, cf, apply_alpha
                )
        loss1 = (out.double() * cot).sum()
        g1 = torch.autograd.grad(loss1, targets, create_graph=second)
        if not second:
            return (out.double(), *[g.double() for g in g1])
        # Force regime: the second differentiation reaches the node-feature,
        # Wigner and degree-kernel gradients (their producers sit on the
        # coordinate graph); the parameter gradients feed the optimizer.
        loss2 = (
            (g1[0].double() * h_x).sum()
            + (g1[1].double() * h_wig).sum()
            + (g1[2].double() * h_kc.reshape_as(g1[2])).sum()
        )
        g2 = torch.autograd.grad(loss2, targets, allow_unused=True)
        g2 = [g if g is not None else torch.zeros_like(t) for g, t in zip(g2, targets)]
        return (out.double(), *[g.double() for g in g1], *[g.double() for g in g2])

    def domain(name, t):
        # The kernels contract the Wigner matrix over its structural block
        # diagonal only; the model's Wigner construction reads the same
        # entries, so the gradient comparison lives on that domain.
        return t * mask.unsqueeze(0) if name.endswith("_wig") else t

    return names, evaluate, domain


def judge_shape(shape, second, fused_float64):
    """Arbitrate one block shape over ``DRAWS`` operand draws."""
    lmax, focus, cf, layers, rank, apply_alpha = shape
    label = (
        f"L={lmax} F={focus} Cf={cf} NL={layers} rank={rank} alpha={int(apply_alpha)}"
    )
    base_seed = lmax * 1009 + focus * 101 + cf + layers + rank * 7
    tables = {False: [], True: []}
    pair_errors = []
    for draw in range(DRAWS):
        names, evaluate, domain = make_case(shape, base_seed + 7919 * draw, second)
        gold = evaluate(fused=False, dtype=torch.float64, amp=False)
        if fused_float64:
            fused64 = evaluate(fused=True, dtype=torch.float64, amp=False)
            pair_errors.append(
                [
                    relative_error(domain(name, truth), domain(name, got))
                    for name, truth, got in zip(names, gold, fused64)
                ]
            )
        for amp in (False, True):
            tables[amp].append(
                deviations(
                    names,
                    gold,
                    evaluate(fused=False, dtype=torch.float32, amp=amp),
                    evaluate(fused=True, dtype=torch.float32, amp=amp),
                    factor=FACTOR,
                    factor_overrides=FACTOR_OVERRIDES,
                    working_dtype=torch.bfloat16 if amp else torch.float32,
                    project=domain,
                )
            )

    ok = True
    for amp in (False, True):
        for index, entry in enumerate(median_deviations(tables[amp])):
            pair_ok, pair_text = True, ""
            if not amp:
                if pair_errors:
                    worst = max(errors[index] for errors in pair_errors)
                    pair_ok = worst <= FLOAT64_CEILING
                    pair_text = f"fp64-pair {worst:.3e}  "
                else:
                    pair_text = "fp64-pair n/a  "
            flag = "ok" if (entry.ok and pair_ok) else "FAIL"
            ok = ok and flag == "ok"
            print(
                f"  {label} amp={int(amp)} {entry.name:>8}: {pair_text}"
                f"fused-vs-fp64 {entry.fused:.3e}  eager-vs-fp64 {entry.eager:.3e}  "
                f"bound {entry.bound:.3e} [{flag}]"
            )
    return ok


def main():
    second = "--first-order-only" not in sys.argv
    print("cuda so2 value available:", op_available())
    assert op_available()
    fused_float64 = float64_available()
    print("float64 kernels compiled:", fused_float64)
    print(
        f"verdict: medians over {DRAWS} draws; fp64 pair: worst draw "
        f"<= {FLOAT64_CEILING:.0e}"
    )
    ok = True
    for shape in SHAPES:
        ok = judge_shape(shape, second, fused_float64) and ok
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

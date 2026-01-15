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
"""

import sys

import torch

from deepmd.pt.cxx_op import ENABLE_CUSTOMIZED_OP  # noqa: F401
from deepmd.pt_expt.kernels.cuda.dpa4.so2_conv_train import (
    _value_train_op,
    op_available,
)
from deepmd.pt_expt.kernels.triton.sezm.so2_value_path import (
    _mixing_stack_reference,
    _rotate_mix_reference,
)


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


def run_case(
    lmax, focus, cf, n_node, edges, layers, rank, apply_alpha, amp, second=True
):
    torch.manual_seed(lmax * 1009 + focus * 101 + cf + layers + rank * 7)
    device = "cuda"
    dim = (lmax + 1) ** 2
    c_wide = focus * cf
    m0 = (lmax + 1) * cf
    m1 = 2 * lmax * cf
    n_gated = layers - 1
    tau, ls = 1.0, 0.02

    x0 = torch.randn(n_node, dim, c_wide, device=device, dtype=torch.float64)
    src = torch.randint(0, n_node, (edges,), device=device, dtype=torch.long)
    wig0 = torch.randn(edges, dim, dim, device=device, dtype=torch.float64)
    wig0 = wig0 * block_mask(lmax, device).unsqueeze(0)
    if rank == 0:
        kc0 = torch.randn(edges, lmax + 1, c_wide, device=device, dtype=torch.float64)
        cb0 = torch.zeros(1, device=device, dtype=torch.float64)
    else:
        ksz = (lmax + 1) ** 2 + lmax * lmax
        kc0 = 0.3 * torch.randn(edges, ksz * rank, device=device, dtype=torch.float64)
        cb0 = torch.randn(rank, c_wide, device=device, dtype=torch.float64)
    wfc0 = 0.05 * torch.randn(cf, focus, device=device, dtype=torch.float64)
    bias0 = 0.05 * torch.randn(focus, device=device, dtype=torch.float64)
    w00 = 0.2 * torch.randn(
        n_gated + 1, focus, m0, m0, device=device, dtype=torch.float64
    )
    w10 = 0.2 * torch.randn(
        n_gated + 1, focus, m1, m1, device=device, dtype=torch.float64
    )
    gw0 = 0.3 * torch.randn(
        n_gated, focus, cf, lmax * cf, device=device, dtype=torch.float64
    )
    cot = torch.randn(
        edges, focus, (3 * lmax + 1) * cf, device=device, dtype=torch.float64
    )
    h_x = torch.randn_like(x0)
    # A force loss also sends cotangents through the Wigner and degree-kernel
    # gradients, whose producers precede the operator on the coordinate
    # graph; the Wigner cotangent lives on the structural block diagonal.
    h_wig = torch.randn_like(wig0) * block_mask(lmax, device)
    h_kc = torch.randn_like(kc0)

    src_order = torch.argsort(src, dim=0, stable=True)
    counts = src.new_zeros(n_node).scatter_add(0, src, torch.ones_like(src))
    src_rowptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)])

    leaves = ("x", "wig", "kc", "cb", "wfc", "bias", "w0", "w1", "gw")

    def path(fused, gold=False):
        dt = torch.float64 if gold else torch.float32
        x = x0.to(dt).clone().requires_grad_(True)
        wig = wig0.to(dt).clone().requires_grad_(True)
        kc = kc0.to(dt).clone().requires_grad_(True)
        cb = cb0.to(dt).clone().requires_grad_(rank > 0)
        wfc = wfc0.to(dt).clone().requires_grad_(apply_alpha)
        bias = bias0.to(dt).clone().requires_grad_(apply_alpha)
        w0 = w00.to(dt).clone().requires_grad_(True)
        w1 = w10.to(dt).clone().requires_grad_(True)
        gw = gw0.to(dt).clone().requires_grad_(True)
        params = (x, wig, kc, cb, wfc, bias, w0, w1, gw)
        grad_targets = [p for p, name in zip(params, leaves) if p.requires_grad]
        if amp and not gold:
            # The fused operator's autocast rule casts every floating-point
            # input to bfloat16; the eager reference is fed the same casts
            # explicitly so both sides run one numerical regime.
            x, wig, kc, cb, wfc, bias, w0, w1, gw = (
                t.to(torch.bfloat16) for t in params
            )
        ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if amp and not gold
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
                    kc.reshape(edges, -1) if rank > 0 else kc,
                    cb.reshape(-1) if rank > 0 else cb,
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
                    tau,
                    ls,
                    1e-7,
                )
            else:
                u0 = _rotate_mix_reference(
                    x,
                    src,
                    wig,
                    kc.reshape(edges, -1) if rank > 0 else kc,
                    cb.reshape(-1) if rank > 0 else cb,
                    lmax,
                    focus,
                    rank,
                )
                if apply_alpha:
                    gate = u0[:, :, :cf].permute(1, 0, 2)
                    logits = (
                        torch.einsum("efi,if->ef", gate.float(), wfc.float())
                        + bias.float()
                    )
                    p = torch.softmax(logits / tau, dim=1)
                    alpha = (p * (1.0 - ls) + ls / focus).to(u0.dtype)
                else:
                    alpha = torch.ones(edges, focus, device=device, dtype=u0.dtype)
                out, _, _ = _mixing_stack_reference(
                    u0, alpha, w0, w1, gw, lmax, cf, apply_alpha
                )
        loss1 = (out.double() * cot).sum()
        g1 = torch.autograd.grad(loss1, grad_targets, create_graph=second)
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
        g2 = torch.autograd.grad(loss2, grad_targets, allow_unused=True)
        g2 = [
            gi if gi is not None else torch.zeros_like(p)
            for gi, p in zip(g2, grad_targets)
        ]
        return (
            out.double(),
            *[g.double() for g in g1],
            *[g.double() for g in g2],
        )

    active = [
        n
        for n, p0, req in zip(
            leaves,
            (x0, wig0, kc0, cb0, wfc0, bias0, w00, w10, gw0),
            (True, True, True, rank > 0, apply_alpha, apply_alpha, True, True, True),
        )
        if req
    ]
    names = ["fwd"] + [f"g_{n}" for n in active]
    if second:
        names += [f"g2_{n}" for n in active]
    # The kernels contract the Wigner matrix over its structural block
    # diagonal only; the model's Wigner construction reads the same entries,
    # so the gradient comparison lives on that domain.
    wmask = block_mask(lmax, "cuda")

    def domain(name, t):
        return t * wmask.unsqueeze(0) if name.endswith("_wig") else t

    ok = True
    if not amp:
        # The fp64 pair separates logic errors from fp32 conditioning. The
        # kernels keep float accumulators internally (production runs fp32 or
        # AMP), so the double pair agrees to reduction-order rounding,
        # amplified by the depth of the second-order chain; logic errors sit
        # orders of magnitude higher. The fp32 pair is judged against the
        # fp64 ground truth so the deep unnormalized chain's error
        # amplification affects both sides equally.
        gold_ref = path(False, gold=True)
        gold_got = path(True, gold=True)
        ref = path(False)
        got = path(True)
        for name, gr, gg, r, g in zip(names, gold_ref, gold_got, ref, got):
            gr, gg = domain(name, gr), domain(name, gg)
            r, g = domain(name, r), domain(name, g)
            scale = gr.abs().max().clamp_min(1.0)
            err64 = (gr - gg).abs().max().item() / scale.item()
            err_ref = (r - gr).abs().max().item() / scale.item()
            err_got = (g - gr).abs().max().item() / scale.item()
            # The bound is a multiple of the eager reference's own distance
            # from the fp64 gold plus one rounding of the format, never an
            # operator-specific absolute figure: such a floor absorbs
            # precision regressions (it hid the bf16 competition-weight anchor
            # defect, dpa4_cuda.md 11.6).
            bound = max(4.0 * err_ref, float(torch.finfo(torch.float32).eps))
            flag = "ok" if (err64 <= 5e-6 and err_got <= bound) else "FAIL"
            ok = ok and flag == "ok"
            print(
                f"  L={lmax} F={focus} Cf={cf} NL={layers} rank={rank} "
                f"alpha={int(apply_alpha)} amp=0 {name:>8}: fp64-pair "
                f"{err64:.3e}  fused-vs-fp64 {err_got:.3e}  eager-vs-fp64 "
                f"{err_ref:.3e} [{flag}]"
            )
        return ok
    gold = path(False, gold=True)
    ref = path(False)
    got = path(True)
    for name, gd, r, g in zip(names, gold, ref, got):
        gd, r, g = domain(name, gd), domain(name, r), domain(name, g)
        scale = gd.abs().max().clamp_min(1.0)
        err_ref = (r - gd).abs().max().item() / scale.item()
        err_got = (g - gd).abs().max().item() / scale.item()
        # Same rule under bfloat16. Single-draw extremes of a small
        # gradient are noisy, so a ratio near the bound here is not a verdict:
        # the committed unit tests decide on a median over independent draws
        # (source/tests/pt_expt/kernels/test_so2_value_train.py).
        bound = max(4.0 * err_ref, float(torch.finfo(torch.bfloat16).eps))
        flag = "ok" if err_got <= bound else "FAIL"
        ok = ok and err_got <= bound
        print(
            f"  L={lmax} F={focus} Cf={cf} NL={layers} rank={rank} "
            f"alpha={int(apply_alpha)} amp=1 {name:>8}: fused-vs-fp64 "
            f"{err_got:.3e}  eager-vs-fp64 {err_ref:.3e} [{flag}]"
        )
    return ok


def main():
    second = "--first-order-only" not in sys.argv
    print("cuda so2 value available:", op_available())
    assert op_available()
    ok = True
    for lmax, focus, cf, layers, rank, apply_alpha in [
        (3, 2, 32, 3, 1, True),
        (5, 2, 64, 4, 2, True),
        (3, 1, 64, 3, 0, False),
        (6, 2, 96, 4, 1, True),
    ]:
        for amp in (False, True):
            ok = (
                run_case(
                    lmax,
                    focus,
                    cf,
                    700,
                    3001,
                    layers,
                    rank,
                    apply_alpha,
                    amp,
                    second=second,
                )
                and ok
            )
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

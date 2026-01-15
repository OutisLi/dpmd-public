# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201, ANN001, ANN201, TID253
"""Arbitrate second-order gradients of the fused value path by finite
differences.

Small fp64 case: compares eager-autograd and the fused CUDA operator chain
against a central finite difference of the first-order force-style scalar
L2 = <d L1 / d x, h_x>, for the parameter curvatures the eager pair cannot
arbitrate on its own (the competition head and the channel basis).
"""

import sys

import torch

sys.path.insert(0, "/nas/outisli/Software/deepmd-kit")

from deepmd.pt.cxx_op import ENABLE_CUSTOMIZED_OP  # noqa: F401
from deepmd.pt_expt.kernels.cuda.dpa4.so2_conv_train import (
    _value_train_op,
    ensure_registered,
    op_available,
)
from deepmd.pt_expt.kernels.triton.sezm.so2_value_path import (
    _mixing_stack_reference,
    _rotate_mix_reference,
)

assert op_available()
ensure_registered()
torch.manual_seed(7)
dev = "cuda"
DT = torch.float64

lmax, focus, cf, n_node, edges, layers, rank = 2, 2, 32, 40, 121, 2, 1
tau, ls = 1.0, 0.05
dim = (lmax + 1) ** 2
cw = focus * cf
m0 = (lmax + 1) * cf
m1 = 2 * lmax * cf

x0 = torch.randn(n_node, dim, cw, device=dev, dtype=DT)
src = torch.randint(0, n_node, (edges,), device=dev)
src_order = torch.argsort(src, dim=0, stable=True)
_counts = src.new_zeros(n_node).scatter_add(0, src, torch.ones_like(src))
src_rowptr = torch.cat([_counts.new_zeros(1), torch.cumsum(_counts, 0)])
_bm = torch.zeros(dim, dim, device=dev, dtype=DT)
for _l in range(lmax + 1):
    _b = _l * _l
    _bm[_b : _b + 2 * _l + 1, _b : _b + 2 * _l + 1] = 1.0
# The kernels read the Wigner matrix on the structural block diagonal only;
# masking keeps the eager reference on the same function.
wig0 = torch.randn(edges, dim, dim, device=dev, dtype=DT) * _bm
kc0 = torch.randn(edges, (lmax + 1) ** 2 + lmax * lmax, rank, device=dev, dtype=DT)
cb0 = torch.randn(rank, cw, device=dev, dtype=DT)
wfc0 = torch.randn(cf, focus, device=dev, dtype=DT) * 0.5
bias0 = torch.randn(focus, device=dev, dtype=DT) * 0.1
n_gated = layers - 1
w00 = torch.randn(layers, focus, m0, m0, device=dev, dtype=DT) * 0.2
w10 = torch.randn(layers, focus, m1, m1, device=dev, dtype=DT) * 0.2
gw0 = torch.randn(n_gated, focus, cf, lmax * cf, device=dev, dtype=DT) * 0.3
cot = torch.randn(edges, focus, (3 * lmax + 1) * cf, device=dev, dtype=DT)
h_x = torch.randn_like(x0)


def forward(x, wig, kc, cb, wfc, bias, w0, w1, gw, fused):
    if fused:
        return _value_train_op(
            x,
            src,
            src_order,
            src_rowptr,
            wig,
            kc.reshape(edges, -1),
            cb.reshape(-1),
            wfc,
            bias,
            w0,
            w1,
            gw,
            lmax,
            focus,
            rank,
            True,
            tau,
            ls,
        )[0]
    u0 = _rotate_mix_reference(
        x, src, wig, kc.reshape(edges, -1), cb.reshape(-1), lmax, focus, rank
    )
    fgate = u0[:, :, :cf].permute(1, 0, 2)
    logits_f = torch.einsum("efi,if->ef", fgate, wfc) + bias
    p = torch.softmax(logits_f / tau, dim=1)
    alpha_f = p * (1.0 - ls) + ls / focus
    out, _, _ = _mixing_stack_reference(u0, alpha_f, w0, w1, gw, lmax, cf, True)
    return out


PARAMS = [x0, wig0, kc0, cb0, wfc0, bias0, w00, w10, gw0]
NAMES = ["x", "wig", "kc", "cb", "wfc", "bias", "w0", "w1", "gw"]


def l2_value(params, fused):
    """L2 = <dL1/dx, h_x> with L1 = <out, cot>."""
    xr = params[0].detach().requires_grad_(True)
    out = forward(xr, *params[1:], fused)
    (gx,) = torch.autograd.grad((out * cot).sum(), (xr,))
    return (gx * h_x).sum()


def analytic(fused):
    leaves = [t.detach().requires_grad_(True) for t in PARAMS]
    out = forward(*leaves, fused)
    g1 = torch.autograd.grad((out * cot).sum(), leaves, create_graph=True)
    l2 = (g1[0] * h_x).sum()
    g2 = torch.autograd.grad(l2, leaves, allow_unused=True)
    return [
        g if g is not None else torch.zeros_like(p)
        for g, p in zip(g2, leaves, strict=True)
    ]


def fd(param_idx, eps=1e-5, max_probe=80):
    base = PARAMS[param_idx]
    g = torch.zeros_like(base)
    flat = g.view(-1)
    n = base.numel()
    probes = (
        list(range(n)) if n <= max_probe else torch.randperm(n)[:max_probe].tolist()
    )
    for i in probes:
        for sgn in (1.0, -1.0):
            pert = [p.clone() for p in PARAMS]
            pv = pert[param_idx].view(-1)
            pv[i] += sgn * eps
            flat[i] += sgn * l2_value(pert, False) / (2 * eps)
    return g, probes


ea = analytic(False)
fa = analytic(True)
for idx in (4, 5, 3):
    fd_ref, probes = fd(idx)
    mask = torch.zeros(fd_ref.numel(), dtype=torch.bool, device=dev)
    mask[torch.as_tensor(probes, device=dev)] = True
    fd_v = fd_ref.view(-1)[mask]
    scale = fd_v.abs().max().clamp_min(1e-10)
    e_err = (ea[idx].reshape(-1)[mask] - fd_v).abs().max() / scale
    f_err = (fa[idx].reshape(-1)[mask].to(DT) - fd_v).abs().max() / scale
    print(
        f"g2_{NAMES[idx]}: eager-vs-fd {e_err.item():.3e}  "
        f"fused-vs-fd {f_err.item():.3e}  scale {scale.item():.3e}"
    )

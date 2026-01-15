# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, T201, ANN001, ANN205
"""
Probe whether quantization-aware training absorbs fp8 trunk rounding.

An fp8 e4m3 GEMM carries a relative error near 3.7e-2, set by the three
mantissa bits and unreachable by any scaling strategy. In the
coefficient/geometry split the trunk output *is* the potential's parameter
vector, so that error transfers one-to-one into the potential-energy surface
unless training compensates for it.

The probe fits a fixed fp32 teacher map from a chemical summary to a
coefficient vector and compares four precision policies of the student trunk:

``bf16``
    Reference. Trunk in bf16 for both training and evaluation.
``bf16 -> fp8``
    Trained in bf16, evaluated in fp8. The mismatched case, which bounds what
    naive post-training quantization costs.
``fp8 QAT``
    Trained and evaluated in fp8, so the optimizer sees the forward that
    deployment runs.
``fp8 branches + bf16 edges``
    Residual-branch GEMMs in fp8, input projection and output head in bf16.
    The residual stream never accumulates a raw fp8 product, at the cost of
    leaving a small fraction of the FLOPs in bf16.

The reported metric is the relative error of the predicted coefficients,
which is exactly the relative error the potential-energy surface inherits.
"""

from __future__ import (
    annotations,
)

import argparse
import math

import torch
from torch import (
    Tensor,
    nn,
)

E4M3 = torch.float8_e4m3fn
E4M3_MAX = 448.0


def _row_quant(x: Tensor) -> tuple[Tensor, Tensor]:
    """
    Quantize a 2-D tensor to e4m3 with one scale per row.

    A row-wise scale depends only on its own row, so a per-atom row is
    independent of batch composition: a training batch and a single-structure
    evaluation quantize the same atom identically. Per-tensor scaling does not
    have this property and would make inference depend on batch content.

    Parameters
    ----------
    x : Tensor
        Input matrix with shape (M, K).

    Returns
    -------
    tuple[Tensor, Tensor]
        The e4m3 tensor with shape (M, K) and float32 scales with shape (M, 1).
    """
    amax = x.abs().amax(dim=-1, keepdim=True).float().clamp_min(1e-12)
    scale = amax / E4M3_MAX
    return (x.float() / scale).to(E4M3), scale


class _Fp8Matmul(torch.autograd.Function):
    """
    ``x @ w.T`` with the forward on fp8 tensor cores and a bf16 backward.

    An fp8 GEMM can only carry scales on its non-contracted axes, so the three
    GEMMs of a linear layer would each need their operands re-quantized along
    a different axis. The backward therefore stays in bf16: it is a small
    share of the step, carries no smoothness requirement, and gives the
    forward quantization a straight-through gradient, which is the signal
    that lets the trunk learn to compensate for its own rounding.
    """

    @staticmethod
    def forward(ctx, x: Tensor, w: Tensor) -> Tensor:
        xq, sx = _row_quant(x)
        wq, sw = _row_quant(w)
        ctx.save_for_backward(x.bfloat16(), w.bfloat16())
        ctx.in_dtype = x.dtype
        # ``wq`` is row-major (N, K), so ``wq.t()`` is the column-major (K, N)
        # operand the kernel requires. Row-wise scaling accepts only bf16 or
        # fp16 output, so the accumulator result carries bf16 rounding.
        out = torch._scaled_mm(
            xq.contiguous(),
            wq.t(),
            scale_a=sx,
            scale_b=sw.reshape(1, -1),
            out_dtype=torch.bfloat16,
        )
        return out.to(x.dtype)

    @staticmethod
    def backward(ctx, g: Tensor):
        x, w = ctx.saved_tensors
        gb = g.bfloat16()
        return (gb @ w).to(ctx.in_dtype), (gb.t() @ x).to(ctx.in_dtype)


def linear(x: Tensor, layer: nn.Linear, mode: str) -> Tensor:
    """
    Apply a linear layer at the requested arithmetic precision.

    Parameters
    ----------
    x : Tensor
        Input with shape (M, K).
    layer : nn.Linear
        Layer supplying weight and optional bias.
    mode : str
        One of ``"fp32"``, ``"bf16"``, ``"fp8"``.

    Returns
    -------
    Tensor
        Output with shape (M, N), in the dtype of ``x``.
    """
    if mode == "fp8":
        y = _Fp8Matmul.apply(x, layer.weight)
    elif mode == "bf16":
        y = (x.bfloat16() @ layer.weight.bfloat16().t()).to(x.dtype)
    else:
        y = x @ layer.weight.t()
    return y + layer.bias if layer.bias is not None else y


class Trunk(nn.Module):
    """
    Per-atom residual trunk mapping a chemical summary to coefficients.

    Parameters
    ----------
    d_in : int
        Input summary width.
    d_hidden : int
        Residual stream width.
    d_out : int
        Coefficient width.
    n_layers : int
        Number of residual layers.
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int, n_layers: int) -> None:
        super().__init__()
        self.proj_in = nn.Linear(d_in, d_hidden)
        self.norms = nn.ModuleList(nn.LayerNorm(d_hidden) for _ in range(n_layers))
        self.up = nn.ModuleList(
            nn.Linear(d_hidden, 2 * d_hidden, bias=False) for _ in range(n_layers)
        )
        self.down = nn.ModuleList(
            nn.Linear(d_hidden, d_hidden, bias=False) for _ in range(n_layers)
        )
        self.head = nn.Linear(d_hidden, d_out)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1.0 / math.sqrt(m.in_features))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, s: Tensor, branch: str, edge: str) -> Tensor:
        """
        Map a chemical summary to a coefficient vector.

        Parameters
        ----------
        s : Tensor
            Chemical summary with shape (N, d_in).
        branch : str
            Precision mode of the residual-branch GEMMs.
        edge : str
            Precision mode of the input projection and the output head.

        Returns
        -------
        Tensor
            Coefficients with shape (N, d_out).
        """
        h = linear(s, self.proj_in, edge)
        for norm, up, down in zip(self.norms, self.up, self.down, strict=True):
            a, b = linear(norm(h), up, branch).chunk(2, dim=-1)
            h = h + linear(a * torch.sigmoid(b), down, branch)
        return linear(h, self.head, edge)


def main() -> None:
    """Train the four precision policies and report coefficient accuracy."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d-in", type=int, default=64)
    ap.add_argument("--d-hidden", type=int, default=1024)
    ap.add_argument("--d-out", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=8192)
    ap.add_argument("--n-test", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    # Teacher: a fixed fp32 map, standing in for the coefficient field a
    # trained descriptor would have to reproduce.
    teacher = Trunk(args.d_in, args.d_hidden, args.d_out, args.n_layers).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    s_train = torch.randn(args.n_train, args.d_in, device=device)
    with torch.no_grad():
        y_train = teacher(s_train, "fp32", "fp32")
    # A random deep teacher over a 64-dimensional input is close to a random
    # function, so a finite sample cannot pin it down and held-out error is
    # dominated by generalization rather than by arithmetic. The comparison of
    # interest is in-distribution: what accuracy each precision policy can
    # reach on data it was fitted to, and what switching precision at
    # evaluation costs a fixed set of weights.
    s_eval = s_train[: args.n_test]
    y_eval = y_train[: args.n_test]
    denom = y_eval.abs().mean()

    # (label, train branch, train edge, eval branch, eval edge)
    policies = [
        ("bf16", "bf16", "bf16", "bf16", "bf16"),
        ("bf16 -> fp8", "bf16", "bf16", "fp8", "fp8"),
        ("fp8 QAT", "fp8", "fp8", "fp8", "fp8"),
        ("fp8 branches + bf16 edges", "fp8", "bf16", "fp8", "bf16"),
    ]

    print(
        f"trunk: d_in={args.d_in} d_hidden={args.d_hidden} "
        f"d_out={args.d_out} layers={args.n_layers}"
    )
    print(f"train: {args.steps} steps, batch {args.batch}, lr {args.lr}\n")
    print(
        f"  {'policy':26s} {'final loss':>12s} {'coef rel err':>13s} "
        f"{'switch cost':>12s}"
    )
    print("  " + "-" * 67)

    for label, tr_branch, tr_edge, ev_branch, ev_edge in policies:
        torch.manual_seed(args.seed + 1)
        student = Trunk(args.d_in, args.d_hidden, args.d_out, args.n_layers).to(device)
        opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

        for _ in range(args.steps):
            idx = torch.randint(0, args.n_train, (args.batch,), device=device)
            loss = (
                (student(s_train[idx], tr_branch, tr_edge) - y_train[idx]).pow(2).mean()
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            sched.step()

        student.eval()
        with torch.no_grad():
            pred = student(s_eval, ev_branch, ev_edge)
            ref = student(s_eval, "bf16", "bf16")
        rel = float((pred - y_eval).abs().mean() / denom)
        # Cost of running these fixed weights at the evaluation precision
        # instead of bf16, isolated from whatever accuracy training reached.
        switch = float((pred - ref).abs().mean() / denom)
        print(f"  {label:26s} {float(loss.detach()):12.4e} {rel:13.4e} {switch:12.4e}")


if __name__ == "__main__":
    main()

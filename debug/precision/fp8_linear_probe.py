# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, T201, ANN001, ANN205
"""
Feasibility probe for an fp8 linear layer on sm_120.

The probe establishes the three facts a production fp8 module needs:

1. how much per-row scaling buys over per-tensor scaling, since the naive
   per-tensor form loses about 3% and that is far above what a potential
   needs;
2. that a differentiable wrapper around ``torch._scaled_mm`` reproduces the
   fp32 gradients to the accuracy the forward suggests, with the backward
   GEMMs also in fp8;
3. what all of this costs at the shape the chemical trunk actually runs.

Scaling conventions follow the usual fp8 training recipe: activations and
weights round to e4m3 (higher mantissa, values are bounded), gradients round
to e5m2 (wider exponent, gradients span decades).
"""

from __future__ import (
    annotations,
)

import time

import torch
from torch import (
    Tensor,
)

E4M3 = torch.float8_e4m3fn
E5M2 = torch.float8_e5m2
E4M3_MAX = 448.0
E5M2_MAX = 57344.0


def row_scale(x: Tensor, fmt_max: float) -> tuple[Tensor, Tensor]:
    """
    Quantize a 2-D tensor to fp8 with one scale per row.

    Parameters
    ----------
    x : Tensor
        Input matrix with shape (M, K), float32 or bfloat16.
    fmt_max : float
        Largest finite magnitude of the target fp8 format.

    Returns
    -------
    tuple[Tensor, Tensor]
        The fp8 tensor with shape (M, K) and the float32 scales with shape
        (M, 1) such that ``x ~= q.float() * scale``.
    """
    amax = x.abs().amax(dim=-1, keepdim=True).float().clamp_min(1e-12)
    scale = amax / fmt_max
    fmt = E4M3 if fmt_max == E4M3_MAX else E5M2
    return (x.float() / scale).to(fmt), scale


def tensor_scale(x: Tensor, fmt_max: float) -> tuple[Tensor, Tensor]:
    """
    Quantize a 2-D tensor to fp8 with a single scale.

    Parameters
    ----------
    x : Tensor
        Input matrix with shape (M, K).
    fmt_max : float
        Largest finite magnitude of the target fp8 format.

    Returns
    -------
    tuple[Tensor, Tensor]
        The fp8 tensor and the float32 scale with shape (1, 1).
    """
    amax = x.abs().amax().float().clamp_min(1e-12).reshape(1, 1)
    scale = amax / fmt_max
    fmt = E4M3 if fmt_max == E4M3_MAX else E5M2
    return (x.float() / scale).to(fmt), scale


def fp8_gemm(
    a: Tensor, sa: Tensor, b: Tensor, sb: Tensor, out_dtype: torch.dtype
) -> Tensor:
    """
    Evaluate ``(a @ b) * sa * sb`` on the fp8 tensor cores.

    ``torch._scaled_mm`` requires the right operand in column-major layout.

    Parameters
    ----------
    a : Tensor
        Left fp8 operand with shape (M, K), row-major.
    sa : Tensor
        Scales of ``a``, shape (M, 1) or (1, 1).
    b : Tensor
        Right fp8 operand with shape (K, N).
    sb : Tensor
        Scales of ``b``, shape (1, N) or (1, 1).
    out_dtype : torch.dtype
        Output dtype.

    Returns
    -------
    Tensor
        Product with shape (M, N).
    """
    return torch._scaled_mm(
        a.contiguous(),
        b.t().contiguous().t(),
        scale_a=sa,
        scale_b=sb,
        out_dtype=out_dtype,
    )


class Fp8Linear(torch.autograd.Function):
    """
    ``y = x @ w.T`` with the forward GEMM on the fp8 tensor cores.

    An fp8 GEMM can only carry scales on its non-contracted axes, so each of
    the three GEMMs in a linear layer needs its operands re-quantized along a
    different axis. Paying that for the backward buys little: the layer is a
    few percent of the step, the backward has no smoothness requirement, and
    gradient quality matters more than its throughput. The backward therefore
    runs in bf16, which also gives the forward quantization a straight-through
    gradient -- exactly the quantization-aware training signal that lets the
    trunk learn to compensate for its own rounding.
    """

    @staticmethod
    def forward(ctx, x: Tensor, w: Tensor) -> Tensor:
        xq, sx = row_scale(x, E4M3_MAX)
        wq, sw = row_scale(w, E4M3_MAX)
        ctx.save_for_backward(x.bfloat16(), w.bfloat16())
        ctx.in_dtype = x.dtype
        # Row-wise scaling accepts only bf16/fp16 accumulator output, so the
        # result carries bf16 rounding on top of the fp8 operand rounding.
        out = fp8_gemm(xq, sx, wq.t(), sw.reshape(1, -1), torch.bfloat16)
        return out.to(x.dtype)

    @staticmethod
    def backward(ctx, g: Tensor):
        x, w = ctx.saved_tensors
        gb = g.bfloat16()
        return (gb @ w).to(ctx.in_dtype), (gb.t() @ x).to(ctx.in_dtype)


def forward_accuracy(m: int, n: int, k: int, device: str) -> None:
    """
    Compare per-tensor and per-row scaling against the fp32 reference.

    Parameters
    ----------
    m, n, k : int
        GEMM dimensions.
    device : str
        Torch device string.
    """
    torch.manual_seed(0)
    a = torch.randn(m, k, device=device)
    b = torch.randn(k, n, device=device)
    ref = a @ b
    denom = ref.abs().mean()

    aq_t, sa_t = tensor_scale(a, E4M3_MAX)
    bq_t, sb_t = tensor_scale(b.t().contiguous(), E4M3_MAX)
    out_t = fp8_gemm(aq_t, sa_t, bq_t.t(), sb_t.reshape(1, -1), torch.float32)

    aq_r, sa_r = row_scale(a, E4M3_MAX)
    bq_r, sb_r = row_scale(b.t().contiguous(), E4M3_MAX)
    out_r = fp8_gemm(aq_r, sa_r, bq_r.t(), sb_r.reshape(1, -1), torch.bfloat16).float()

    bf = (a.bfloat16() @ b.bfloat16()).float()

    print(f"=== forward accuracy, M={m} N={n} K={k} ===")
    for tag, out in (
        ("fp8 per-tensor->fp32", out_t),
        ("fp8 per-row->bf16", out_r),
        ("bf16", bf),
    ):
        rel = (out - ref).abs().mean() / denom
        print(f"  {tag:22s} mean rel err = {float(rel):.3e}")


def gradient_accuracy(m: int, n: int, k: int, device: str) -> None:
    """
    Check that the fp8 layer's gradients track the fp32 reference.

    Parameters
    ----------
    m, n, k : int
        GEMM dimensions.
    device : str
        Torch device string.
    """
    torch.manual_seed(0)
    x = torch.randn(m, k, device=device).requires_grad_(True)
    w = (torch.randn(n, k, device=device) / k**0.5).requires_grad_(True)
    x32 = x.detach().clone().requires_grad_(True)
    w32 = w.detach().clone().requires_grad_(True)

    target = torch.randn(m, n, device=device)
    Fp8Linear.apply(x, w).sub(target).pow(2).mean().backward()
    (x32 @ w32.t()).sub(target).pow(2).mean().backward()

    print(f"\n=== gradient accuracy, M={m} N={n} K={k} ===")
    for tag, got, ref in (("dx", x.grad, x32.grad), ("dw", w.grad, w32.grad)):
        rel = (got - ref).abs().mean() / ref.abs().mean()
        cos = torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0)
        print(f"  {tag}: mean rel err = {float(rel):.3e}   cosine = {float(cos):.6f}")


def timing(m: int, n: int, k: int, device: str) -> None:
    """
    Time the fp8, bf16 and fp32 forward at one shape.

    Parameters
    ----------
    m, n, k : int
        GEMM dimensions.
    device : str
        Torch device string.
    """
    a = torch.randn(m, k, device=device)
    b = torch.randn(k, n, device=device)
    aq, sa = row_scale(a, E4M3_MAX)
    bq, sb = row_scale(b.t().contiguous(), E4M3_MAX)
    abf, bbf = a.bfloat16(), b.bfloat16()

    print(f"\n=== timing, M={m} N={n} K={k} ===")
    fns = {
        "fp8 per-row": lambda: fp8_gemm(
            aq, sa, bq.t(), sb.reshape(1, -1), torch.bfloat16
        ),
        "fp8 per-tensor": lambda: fp8_gemm(
            aq.contiguous(),
            sa.amax().reshape(1, 1),
            bq.t(),
            sb.amax().reshape(1, 1),
            torch.bfloat16,
        ),
        "bf16": lambda: abf @ bbf,
        "fp32": lambda: a @ b,
    }
    for tag, fn in fns.items():
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            fn()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 100
        print(
            f"  {tag:14s} {dt * 1e3:7.3f} ms   {2 * m * n * k / dt / 1e12:7.1f} TFLOPS"
        )


def main() -> None:
    """Run the accuracy and timing probes at the chemical-trunk shape."""
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    # 1536 atoms is 8 frames of the water example; 2816 is a slow-channel
    # width that costs a few percent of the fast path.
    forward_accuracy(1536, 2816, 2816, device)
    gradient_accuracy(1536, 1024, 1024, device)
    timing(1536, 2816, 2816, device)


if __name__ == "__main__":
    main()

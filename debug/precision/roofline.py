# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, T201
"""
Where a per-atom trunk layer crosses the roofline knee.

The fast path of the descriptor is bandwidth bound: its dominant tensors are
per-edge, and its narrow SO(3) contractions reach a fraction of a percent of
peak. A per-atom trunk is the opposite regime -- the tensors are four orders
of magnitude smaller and the matrices are square -- but only above a width
where the GEMM work outgrows the elementwise traffic around it.

For a layer of the form ``x + W2 @ act(W1 @ norm(x))`` with ``W1: (2C, C)``
and ``W2: (C, 2C)`` over ``M`` rows, the arithmetic intensity is

    I = 8 M C^2 / (2 * (12 M C + 4 C^2))   FLOP/byte

and the machine's knee is ``peak_flops / peak_bandwidth``, which on this GPU
is 1e15 / 1.597e12 = 626 FLOP/byte. The probe measures where the realized
throughput actually saturates, for a training row count and for a
single-structure row count.
"""

from __future__ import (
    annotations,
)

import argparse
import time

import torch
from torch import (
    Tensor,
    nn,
)

PEAK_BF16_TFLOPS = 1000.0
PEAK_BW_GBS = 1597.0


class Layer(nn.Module):
    """
    One residual trunk layer, in the shape the production module would use.

    Parameters
    ----------
    width : int
        Residual stream width.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(width)
        self.up = nn.Linear(width, 2 * width, bias=False)
        self.down = nn.Linear(2 * width, width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the layer.

        Parameters
        ----------
        x : Tensor
            Residual stream with shape (M, width).

        Returns
        -------
        Tensor
            Updated stream with shape (M, width).
        """
        return x + self.down(torch.nn.functional.gelu(self.up(self.norm(x))))


def intensity(m: int, c: int) -> float:
    """
    Arithmetic intensity of one layer, in FLOP per byte.

    Parameters
    ----------
    m : int
        Number of rows.
    c : int
        Layer width.

    Returns
    -------
    float
        Ratio of layer FLOPs to layer bytes moved, assuming bf16 operands.
    """
    flops = 8.0 * m * c * c
    byts = 2.0 * (12.0 * m * c + 4.0 * c * c)
    return flops / byts


def measure(m: int, c: int, n_layers: int, device: str) -> tuple[float, float]:
    """
    Time a stack of trunk layers and return realized throughput.

    Parameters
    ----------
    m : int
        Number of rows.
    c : int
        Layer width.
    n_layers : int
        Number of stacked layers.
    device : str
        Torch device string.

    Returns
    -------
    tuple[float, float]
        Milliseconds per forward and realized TFLOPS.
    """
    stack = nn.Sequential(*[Layer(c) for _ in range(n_layers)]).to(
        device=device, dtype=torch.bfloat16
    )
    x = torch.randn(m, c, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        for _ in range(5):
            stack(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            stack(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 20
    flops = 8.0 * m * c * c * n_layers
    return dt * 1e3, flops / dt / 1e12


def main() -> None:
    """Sweep width at two row counts and print the roofline table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--widths", type=int, nargs="+", default=[512, 1024, 2048, 3072, 4096, 6144]
    )
    ap.add_argument("--rows", type=int, nargs="+", default=[192, 1536, 6144])
    ap.add_argument("--layers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    knee = PEAK_BF16_TFLOPS * 1e12 / (PEAK_BW_GBS * 1e9)
    print(
        f"roofline knee: {knee:.0f} FLOP/byte  (bf16 peak {PEAK_BF16_TFLOPS:.0f} TFLOPS, {PEAK_BW_GBS:.0f} GB/s)"
    )
    print(f"layers per measurement: {args.layers}\n")

    for m in args.rows:
        label = {192: "1 frame", 1536: "8 frames", 6144: "32 frames"}.get(m, "")
        print(f"  M = {m} rows  ({label})")
        print(
            f"    {'width':>6} {'intensity':>10} {'ms':>8} {'TFLOPS':>8} {'% peak':>7}"
        )
        for c in args.widths:
            ms, tflops = measure(m, c, args.layers, device)
            print(
                f"    {c:>6} {intensity(m, c):>10.0f} {ms:>8.3f} "
                f"{tflops:>8.1f} {100 * tflops / PEAK_BF16_TFLOPS:>6.1f}%"
            )
        print()


if __name__ == "__main__":
    main()

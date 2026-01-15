# SPDX-License-Identifier: LGPL-3.0-or-later
"""A Triton reduction that updates one of its inputs in place computes a wrong sum under a replicated layout.

The kernel follows the pattern of the reduction kernels that TorchInductor
generates for an ``in_out_ptr``: in a loop over the reduction axis it loads a
block of ``io``, adds the block of ``b``, accumulates ``(io + b) * w`` and
stores ``io + b`` back to the same addresses, then reduces the accumulator
along the inner axis. ``EXTRA`` further loads lengthen each iteration, as the
loads of the generated kernel do.

A 16 x 4 block (XBLOCK 16, RBLOCK 4) under ``num_warps`` 8 runs 256 threads
on 64 elements: the blocked layout gives warps 0 and 1 the block and warps 2
to 7 three copies of it. Every copy loads the block, the store is issued by
the first copy alone, and no barrier separates the loads of an iteration
from its store. A copy that runs behind the first one loads the stored
``io + b`` instead of ``io`` and accumulates ``(io + 2 b) * w``, and the
cross-warp epilogue of the reduction can take that copy's partial sum. The
stored block stays correct; the sum of some rows is wrong. With
``num_warps`` 2 the 64 threads hold the block once, and with the
``barrier`` argument a ``tl.debug_barrier()`` orders every load of an
iteration before its store; ``num_warps`` 4 holds two copies.

Wrong launches out of 3,000 (16 rows, nine columns, fp64) at num_warps
8 / 4 / 2 / 8 with the barrier:

- Tesla V100-SXM2-32GB (sm_70), torch 2.13.0+cu126, Triton 3.7.1, three
  passes: with EXTRA 6, 42 to 79 / 0 to 1 / 0 / 0; with EXTRA 0 at
  num_warps 8, 0.
- RTX PRO 6000 (sm_120), torch 2.13.0+cu130, Triton 3.7.1: with EXTRA 6,
  79 / 0 / 0 / 0; with EXTRA 0 at num_warps 8, 42.

Usage
-----
    python triton_inplace_race.py [TRIALS] [NUM_WARPS] [EXTRA] [barrier]
"""

import sys

import torch  # noqa: TID253
import triton
import triton.language as tl


@triton.jit
def inplace_reduce(  # noqa: ANN201
    io_ptr,  # noqa: ANN001
    b_ptr,  # noqa: ANN001
    w_ptr,  # noqa: ANN001
    out_ptr,  # noqa: ANN001
    xnumel,  # noqa: ANN001
    rnumel,  # noqa: ANN001
    EXTRA: tl.constexpr,
    BARRIER: tl.constexpr,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    x = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    xmask = x < xnumel
    acc = tl.full([XBLOCK, RBLOCK], 0, tl.float64)
    for r0 in tl.range(0, rnumel, RBLOCK):
        r = r0 + tl.arange(0, RBLOCK)[None, :]
        mask = (r < rnumel) & xmask
        a = tl.load(io_ptr + (r + rnumel * x), mask, other=0.0)
        b = tl.load(b_ptr + (r + rnumel * x), mask, other=0.0)
        w = tl.load(w_ptr + (r + rnumel * x), mask, other=0.0)
        for k in tl.static_range(EXTRA):
            w += 0.0 * tl.load(
                w_ptr + (r + rnumel * x) + (k + 1) * xnumel * rnumel, mask, other=0.0
            )
        t = a + b
        acc = tl.where(mask, acc + t * w, acc)
        if BARRIER:
            tl.debug_barrier()
        tl.store(io_ptr + (r + rnumel * x), t, mask)
    tl.store(out_ptr + x, tl.sum(acc, 1)[:, None], xmask)


def main() -> None:
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    num_warps = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    extra = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    barrier = len(sys.argv) > 4 and sys.argv[4] == "barrier"
    torch.manual_seed(0)
    rows, cols = 16, 9
    device = "cuda"
    flush = torch.empty(64 * 1024 * 1024 // 8, dtype=torch.float64, device=device)
    wrong_launches, wrong_rows, stored_wrong = 0, 0, 0
    for trial in range(trials):
        # A varying number of small live tensors moves the buffers between trials.
        spacers = [
            torch.empty(
                64 * (1 + (trial * 7 + k) % 5), dtype=torch.float64, device=device
            )
            for k in range(trial % 4)
        ]
        io = torch.randn(rows, cols, dtype=torch.float64, device=device)
        b = (
            io.clone()
            if trial % 2
            else torch.randn(rows, cols, dtype=torch.float64, device=device)
        )
        w = torch.randn(1 + extra, rows, cols, dtype=torch.float64, device=device)
        out = torch.empty(rows, dtype=torch.float64, device=device)
        expected = ((io + b) * w[0]).sum(1)
        expected_io = io + b
        flush.add_(1.0)  # evicts the operands from L2 before the launch
        inplace_reduce[(1,)](
            io,
            b,
            w,
            out,
            rows,
            cols,
            EXTRA=extra,
            BARRIER=barrier,
            XBLOCK=16,
            RBLOCK=4,
            num_warps=num_warps,
            num_stages=1,
        )
        torch.cuda.synchronize()
        bad = (out - expected).abs() > 1e-12 * expected.abs().max()
        stored_wrong += not torch.equal(io, expected_io)
        if bool(bad.any()):
            wrong_launches += 1
            wrong_rows += int(bad.sum())
            if wrong_launches <= 3:
                print(f"trial {trial}: rows {bad.nonzero().flatten().tolist()} wrong")  # noqa: T201
        del spacers
    print(  # noqa: T201
        f"torch {torch.__version__}, triton {triton.__version__}, {torch.cuda.get_device_name()}, "
        f"num_warps {num_warps}, EXTRA {extra}, barrier {barrier}: {wrong_launches} of {trials} launches "
        f"wrong ({wrong_rows} rows), stored block wrong in {stored_wrong}"
    )


if __name__ == "__main__":
    main()

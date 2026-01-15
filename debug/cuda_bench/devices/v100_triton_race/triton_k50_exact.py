# SPDX-License-Identifier: LGPL-3.0-or-later
"""Kernel _50 of the compiled vacuum-reference backward of DPA4C computes wrong sums under its cached configuration.

The Triton kernel below is the one TorchInductor (torch 2.13.0) generated
for the backward graph of the compiled training model of
``test_compiled_lower_accepts_a_compacted_node_axis[vacuum_ref=True]``,
copied from the cached output code unchanged but for the lint comment on its
signature (FX source nodes slice_170 ... bmm_139, six bmm reductions over nine
columns fused into one kernel over the 16 node rows of 14 atoms and the two
vacuum nodes). The kernel accumulates ``(in_out_ptr0 + in_ptr0) * in_ptr2``
into ``out_ptr1`` and ``(in_out_ptr1 + in_ptr1) * in_ptr2`` into ``out_ptr0``
while it stores the two sums back to ``in_out_ptr0`` and ``in_out_ptr1`` in
place; ``out_ptr2`` to ``out_ptr5`` read no buffer the kernel writes.

The script launches it at the configuration that the Inductor cache of the
failing run holds for it, XBLOCK 16, R0_BLOCK 4, num_warps 8, num_stages 1
(256 threads on a 16 x 4 block: four copies of the block in the blocked
layout, the in-place stores predicated to the first copy), or at another
num_warps, on fresh buffers with the shapes, strides and storage offsets of
the compiled graph: the arguments of a launch dumped from the compiled run
(DUMP.npz, a launch dumped by the investigation of section 12 of ``v100.md``), or values drawn with a
fixed seed. Every output is held against the same sums evaluated on the
host, and the stored in-place sums against ``a + b``. Triton specializes the
integer arguments of this launch, which the cached Inductor build did not.

Wrong launches out of 3,000 on a Tesla V100-SXM2-32GB (sm_70), torch
2.13.0+cu126, Triton 3.7.1, at num_warps 8 / 4 / 2, three passes:

- the dumped arguments of the failing launch: out_ptr1 0 to 1 / 2 to 8 / 0,
  out_ptr0 196 to 215 / 18 to 25 / 0;
- the drawn arguments: out_ptr1 0 to 1 / 2 to 11 / 0, out_ptr0 187 to 199 /
  19 to 37 / 0;
- out_ptr2 to out_ptr5, which read no buffer that the kernel writes, and the
  in-place stores: 0 in every case.

Usage
-----
    python triton_k50_exact.py [TRIALS] [NUM_WARPS] [DUMP.npz]
"""

import sys

import numpy as np
import torch  # noqa: TID253
import triton
import triton.language as tl

# Shape, stride and storage offset of every tensor argument in the compiled
# graph at 14 atoms (16 node rows), and the elements of the storage it views.
# in_ptr6 views the storage of in_ptr2 at the same offset.
LAYOUTS = {
    "in_out_ptr0": ((16, 9, 1), (9, 1, 144), 0, 144),
    "in_out_ptr1": ((16, 9, 1), (9, 1, 144), 0, 144),
    "in_ptr0": ((16, 9, 1), (9, 1, 1), 0, 144),
    "in_ptr1": ((16, 9, 1), (9, 1, 1), 0, 144),
    "in_ptr2": ((16, 9, 1), (56, 1, 1), 47, 896),
    "in_ptr3": ((16, 1, 9), (9, 9, 1), 0, 144),
    "in_ptr4": ((16, 56), (56, 1), 0, 896),
    "in_ptr5": ((16, 1, 9), (9, 9, 1), 0, 144),
    "in_ptr6": ((16, 1, 9), (56, 1, 1), 47, 896),
}
OUTPUTS = tuple(f"out_ptr{i}" for i in range(6))


# fmt: off
@triton.jit
def triton_red_fused_add_bmm_permute_slice_transpose_view_50(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):  # noqa: ANN001, ANN201
    r0_numel = 9
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :].to(tl.int64)
    rbase = r0_base
    x0 = xindex
    _tmp9 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float64)
    _tmp13 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float64)
    _tmp19 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float64)
    _tmp24 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float64)
    _tmp28 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float64)
    _tmp33 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float64)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 9*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r0_1 + 9*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_out_ptr1 + (r0_1 + 9*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r0_1 + 9*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (r0_1 + 56*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr3 + (r0_1 + 9*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr4 + (47 + r0_1 + 56*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr5 + (r0_1 + 9*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp30 = tl.load(in_ptr6 + (r0_1 + 56*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(r0_mask & xmask, tmp10, _tmp9)
        tmp11 = tmp2 * tmp6
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(r0_mask & xmask, tmp14, _tmp13)
        tmp17 = tmp15 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(r0_mask & xmask, tmp20, _tmp19)
        tmp22 = tmp21 * tmp16
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, R0_BLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(r0_mask & xmask, tmp25, _tmp24)
        tmp26 = tmp16 * tmp6
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(r0_mask & xmask, tmp29, _tmp28)
        tmp31 = tmp30 * tmp16
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
        tmp34 = _tmp33 + tmp32
        _tmp33 = tl.where(r0_mask & xmask, tmp34, _tmp33)
        tl.store(in_out_ptr0 + (r0_1 + 9*x0), tmp2, r0_mask & xmask)
        tl.store(in_out_ptr1 + (r0_1 + 9*x0), tmp5, r0_mask & xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tmp33 = tl.sum(_tmp33, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp9, xmask)
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tl.store(out_ptr2 + (x0), tmp19, xmask)
    tl.store(out_ptr3 + (x0), tmp24, xmask)
    tl.store(out_ptr4 + (x0), tmp28, xmask)
    tl.store(out_ptr5 + (x0), tmp33, xmask)

# fmt: on


def host_arguments(dump: str | None) -> dict[str, np.ndarray]:
    """Values of the tensor arguments: those of a dumped launch, or drawn with a fixed seed."""
    if dump is not None:
        with np.load(dump) as arrays:
            return {name: arrays[f"pre/{name}"].copy() for name in LAYOUTS}
    rng = np.random.default_rng(0)
    values = {name: rng.standard_normal(LAYOUTS[name][0]) for name in LAYOUTS}
    values["in_ptr6"] = values["in_ptr2"].reshape(16, 1, 9).copy()
    return values


def device_arguments(
    values: dict[str, np.ndarray], device: str
) -> dict[str, torch.Tensor]:
    """Fresh device buffers in the layouts of the compiled graph, holding ``values``."""
    views = {}
    for name, (shape, stride, offset, elements) in LAYOUTS.items():
        if name == "in_ptr6":
            continue
        storage = torch.zeros(elements, dtype=torch.float64, device=device)
        views[name] = storage.as_strided(shape, stride, offset)
        views[name].copy_(torch.from_numpy(values[name]))
    shape, stride, offset, _ = LAYOUTS["in_ptr6"]
    views["in_ptr6"] = views["in_ptr2"].as_strided(shape, stride, offset)
    return views


def expected_sums(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The six sums of the kernel over the nine columns, evaluated on the host."""
    a, b = values["in_out_ptr0"].reshape(16, 9), values["in_ptr0"].reshape(16, 9)
    a1, b1 = values["in_out_ptr1"].reshape(16, 9), values["in_ptr1"].reshape(16, 9)
    c = values["in_ptr2"].reshape(16, 9)
    d3, d5 = values["in_ptr3"].reshape(16, 9), values["in_ptr5"].reshape(16, 9)
    d4 = values["in_ptr4"].reshape(16, 56)[:, 47:56]
    d6 = values["in_ptr6"].reshape(16, 9)
    return {
        "out_ptr0": ((a1 + b1) * c).sum(1),
        "out_ptr1": ((a + b) * c).sum(1),
        "out_ptr2": (d3 * d4).sum(1),
        "out_ptr3": (d5 * d4).sum(1),
        "out_ptr4": (d4 * c).sum(1),
        "out_ptr5": (d6 * d4).sum(1),
    }


def main() -> None:
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    num_warps = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    dump = sys.argv[3] if len(sys.argv) > 3 else None
    device = "cuda"
    values = host_arguments(dump)
    expected = expected_sums(values)
    stored = {
        "in_out_ptr0": values["in_out_ptr0"] + values["in_ptr0"],
        "in_out_ptr1": values["in_out_ptr1"] + values["in_ptr1"],
    }
    flush = torch.empty(64 * 1024 * 1024 // 8, dtype=torch.float64, device=device)
    wrong = dict.fromkeys((*OUTPUTS, "in-place stores"), 0)
    for trial in range(trials):
        # A varying number of small live tensors moves the buffers between trials.
        spacers = [
            torch.empty(
                64 * (1 + (trial * 7 + k) % 5), dtype=torch.float64, device=device
            )
            for k in range(trial % 4)
        ]
        views = device_arguments(values, device)
        outs = [
            torch.empty_strided(
                (16, 1, 1), (1, 16, 16), dtype=torch.float64, device=device
            )
            for _ in OUTPUTS
        ]
        flush.add_(1.0)  # evicts the operands from L2 before the launch
        triton_red_fused_add_bmm_permute_slice_transpose_view_50[(1,)](
            views["in_out_ptr0"],
            views["in_out_ptr1"],
            views["in_ptr0"],
            views["in_ptr1"],
            views["in_ptr2"],
            views["in_ptr3"],
            views["in_ptr4"],
            views["in_ptr5"],
            views["in_ptr6"],
            *outs,
            16,
            9,
            XBLOCK=16,
            R0_BLOCK=4,
            num_warps=num_warps,
            num_stages=1,
        )
        torch.cuda.synchronize()
        for name, out in zip(OUTPUTS, outs, strict=True):
            want = expected[name]
            bad = (
                np.abs(out.reshape(16).cpu().numpy() - want)
                > 1e-12 * np.abs(want).max()
            )
            if bool(bad.any()):
                wrong[name] += 1
                if wrong[name] <= 2:
                    rows = np.nonzero(bad)[0].tolist()
                    print(f"trial {trial} {name}: rows {rows} wrong")  # noqa: T201
        wrong["in-place stores"] += any(
            not np.array_equal(views[name].cpu().numpy(), stored[name])
            for name in stored
        )
        del spacers
    print(  # noqa: T201
        f"torch {torch.__version__}, triton {triton.__version__}, {torch.cuda.get_device_name()}, "
        f"num_warps {num_warps}, {'dumped' if dump else 'drawn'} arguments: wrong launches out of {trials}: "
        + ", ".join(f"{name} {count}" for name, count in wrong.items())
    )


if __name__ == "__main__":
    main()

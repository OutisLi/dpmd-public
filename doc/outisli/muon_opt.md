# HybridMuon Optimizer Optimization

RTX PRO 6000 Blackwell, PyTorch 2.12 nightly, Pro production model (SeZM,
6 blocks, ~90 parameter tensors). All numbers are medians over 20 measured
steps with the `cuda+triton` training configuration active.

## 1. The step benchmark never saw the optimizer

`debug/train_bench/bench.py` times `zero_grad + forward + backward` -- the
optimization update and the gradient clip were outside every number the
kernel-optimization rounds produced. Phase timing on the real trainer
(`debug/train_bench/bench_opt.py`) showed the true step:

| phase     | ms/step | note                              |
| --------- | ------- | --------------------------------- |
| forward   | 44.9    |                                   |
| backward  | 68.9    | force pass + energy term          |
| clip      | 1.0     | `clip_grad_norm_`, negligible     |
| optimizer | 25.4    | HybridMuon, **18% of the step**   |
| total     | 140.2   | vs the 113.4 the step bench shows |

## 2. Diagnosis: host-bound

A profiler pass over `optimizer.step()` alone: **6.5 ms of GPU kernels
behind 25.4 ms of wall time**. The host self-time was ~42 ms of Python and
dispatch: ~630 `aten::mul`, ~500 `aten::add_`, ~280 `aten::copy_`, ~210
`aten::div`, ~170 `aten::baddbmm` calls per step, each launching a
microsecond-scale kernel. The Newton-Schulz iterations, the Magma scales,
the bucket assembly (cat/pad/reshape) and the per-parameter Adam
arithmetic all contribute launches; no single kernel is hot.

## 3. What was tried, in order

1. **Foreach-batching the Adam tails and the Muon delta application**
   (25.4 -> 23.7 ms). The bias-correction/update loops (6 kernels per
   parameter) and the per-entry `param.add_` collapsed into `_foreach_*`
   calls grouped by dtype. Correct, kept, but the host bulk was elsewhere:
   bucket assembly, Magma, the NS chains, the collection loops themselves.
1. **`torch.compile(mode="reduce-overhead")` on the NS orthogonalizers**
   (23.7 -> 18.6 ms). Inductor's CUDA-graph trees absorbed the NS launch
   chains per bucket shape. Superseded and removed: nested graphs are
   illegal inside the whole-step capture that replaced it, and the
   compiled wrappers themselves carried ~0.5 ms of guard evaluation per
   region per step.
1. **The whole step as one CUDA graph** (18.6 -> **5.9 ms**, the GPU
   floor). Two eager warmup steps build the routing, the state tensors and
   every lazy library handle (cuBLAS init inside a capture aborts it);
   the third step captures `_step_impl`; every later step is one
   `graph.replay()`. No new kernel was written anywhere -- the same
   cuBLAS/foreach/elementwise kernels run, submitted by the driver from
   the recorded graph instead of by ~3000 Python-dispatched launches.

Real step: 140.2 -> **119.4 ms** (-15%). The remaining 5.9 ms is the
kernel work itself (NS bmm chains at tensor-core rate, bandwidth-bound
foreach sweeps over ~30 M parameters), within ~2x of the pure-bandwidth
floor; a fused multi-tensor CUDA implementation could reclaim part of it
and was judged not worth the surface area.

## 4. What the graph demanded of the step

Everything the step reads or evolves must be a device tensor whose
address survives across replays:

- **Learning rate**: `group["lr_device"]`, a 0-dim tensor refreshed from
  the host value before every step (outside the capture); every use of
  the rate inside the step is tensor arithmetic (`_foreach_mul_` and
  friends broadcast 0-dim tensors).
- **Bias-correction powers**: per-group 0-dim tensors that advance by
  `mul_(beta)` *inside* the graph, so replays carry the recursion; they
  live in `param_groups` (serialized with the state dict) and older
  checkpoints' per-parameter float powers are adopted on load.
- **Gradients**: `zero_grad(set_to_none=True)` reallocates `p.grad` every
  step, so the graph reads static buffers filled by one
  `_foreach_copy_` per step.
- **Eager fallback**: the identical code path runs eagerly for
  non-plain-tensor parameters (FSDP2); there is one implementation, not
  two, and no configuration switch -- CUDA with plain tensors always
  captures. The equivalence tests flip the internal flag to obtain the
  eager reference trajectory.

## 5. The Magma EMA freeze -- the bug that vindicates the paranoia

The single-entry Magma path updated its EMA as

```python
magma_score = decay * magma_score + (1 - decay) * raw_score
state["magma_score"] = magma_score
```

Out-of-place: a fresh tensor rebound into the state dict. The rebinding
is a *host-side* assignment -- a captured graph executes it exactly once,
at capture time. On every replay the recorded kernels recompute
`decay * A + ...` against the **capture-time tensor A**, whose value is
never advanced again: the EMA recursion freezes, the damping scales
drift, and on the production model the drift compounded through the bf16
NS chain into NaN after a few hundred steps' worth of error in six.

The signature that found it: coupled-trajectory bisection was bitwise
clean at n = 3 (capture + first replay -- A still holds the latest value)
and diverged from n = 4 (the first replay whose input should have moved).
Single-step equivalence tests are structurally blind to this class of
bug. The fix is the in-place form the merged-bucket path already used
(`magma_score.mul_(decay).add_(raw, alpha=1-decay)`); after it, ten
coupled graph-vs-eager steps are bitwise identical.

**Rule extracted**: under graph capture, every state evolution must be an
in-place device operation on a persistent tensor. Rebinding a Python
reference is a silent freeze, not an error.

## 6. Verification matrix

- **Unit suite** (`source/tests/pt/test_hybrid_muon.py`,
  `TestHybridMuonCudaGraph`): a mixed-route model (square NS, rectangular
  Gram, name-routed AdamW, 1D Adam) over 8 coupled steps with a decaying
  learning-rate schedule -- graph vs eager asserted **bitwise**
  (`rtol=0, atol=0`; the replay re-executes the same kernels on the same
  operands, so any tolerance would hide a frozen scalar). Further cases:
  a frozen-rate canary (two schedules sharing the capture-time value must
  diverge), power evolution across replays, legacy checkpoint migration,
  save/load resume reproducing the uninterrupted run bitwise, the default
  engagement of the capture, and the eager reference path.
- **Production model**: ten coupled steps on Neo (force loss, clip,
  decaying rate); graph-vs-eager distance 1.2e-4 against an
  eager-vs-eager nondeterminism floor of 1.0e-4 (backward atomics), loss
  sequence identical to the printed digit, states finite.
- **Resume**: state-dict round trip mid-trajectory restores the powers
  exactly (0.9^n) and re-captures cleanly.

## 7. Boundaries

- The graph assumes training steady state: the parameter set and the
  presence of every gradient are fixed after warmup. A parameter losing
  its gradient after capture raises in the static-copy phase rather than
  updating silently wrong.
- DDP: each rank captures its own graph after the bucketed all-reduce;
  nothing crosses ranks inside the capture. FSDP2 falls back to the eager
  path automatically (DTensor parameters).
- The clip stays outside the graph: its total-norm feeds a host-side
  non-finite guard.
- `nonfinite_grad_guard` semantics are unchanged -- a skipped step simply
  does not replay.

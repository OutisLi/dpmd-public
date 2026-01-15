# The SeZM / DPA4 inference path on cuTile

This document describes the complete cuTile implementation of the SeZM (DPA4)
inference path: what each kernel computes, why it is shaped the way it is, what
the tile programming model costs and rewards, and which designs were measured
and rejected. It is written to be useful to someone about to build the same
pipeline again in another language, so every claim is a measurement and every
rejected design is recorded with the number that killed it.

The path is selected by `DP_CUTILE_INFER=1`. It is mutually exclusive with
`DP_TRITON_INFER` and `DP_CUTE_INFER`; enabling more than one is rejected at
construction. When it is enabled no Triton kernel executes, and a convolution
whose layout it does not support falls back to the dense reference rather than
to another accelerated backend.

______________________________________________________________________

## 1. Result

Measured on one NVIDIA RTX PRO 6000 Blackwell (`sm_120`) against the compiled
Triton path at its best setting (`DP_TRITON_INFER=3`), on an 8000-atom periodic
diamond cell (`nall = 216000`, `E = 1264000` edges) driven through
`forward_common_lower` with TF32 disabled:

| path                       | time per force step | peak memory |
| -------------------------- | ------------------- | ----------- |
| compiled + Triton, level 3 | 162.7 ms            | 68.0 GB     |
| compiled + cuTile          | **152.0 ms**        | **57.1 GB** |

That is 1.07x on time and 1.19x on memory. Section 9 explains why this is close
to the ceiling for kernel substitution on this model and what a materially
larger step would have to change.

Accuracy against the Triton path at the same setting: 5.9e-7 relative on the
per-atom energy and 8.0e-6 relative on the force. Both paths use
split-compensated fp16 tensor cores for the mixing stack, so this is the
difference between two independent realizations of the same approximation, not
the error of either against fp64. Per-kernel accuracy against the eager and fp64
references is in section 7.

______________________________________________________________________

## 2. The workload

One interaction block of the deployed DPA4-mini configuration: `lmax = 2`,
`mmax = 1`, focus width `Cf = 32`, one focus stream, three mixing layers, one
attention head, no SO(2) normalization and no layer scale. The model runs two
such blocks per step. Per block, the SO(2) message performs, for each of the
1.264 million edges:

1. **Wigner monomials** — a quaternion monomial basis feeding the degree-two
   rotation block; evaluated once per step rather than once per block.
1. **Rotate and mix** — gather the source node's `(D, C_wide)` features, project
   them onto the `3 * lmax + 1` reduced rows with the block-diagonal Wigner-D,
   then apply the edge-conditioned radial degree mixing.
1. **Mixing stack** — two gated SO(2) layers and one identity final layer, each
   a block-diagonal GEMM over the `m = 0` and `|m| = 1` groups with a sigmoid
   gate driven by the scalar rows.
1. **Attention aggregation** — rotate back to the global frame, rescale by
   degree, weight by the envelope-gated softmax, and reduce onto the destination
   node.
1. **Force assembly** — after the energy backward, two segmented reductions turn
   the per-edge energy gradient into the force and per-atom virial.

The arithmetic is concentrated in step 3. Per edge, the stack is 102 400
multiply-accumulates against 800 for the rotation and 1 600 for the mixing.
Everything else is bandwidth or latency.

### Graph shape, and why it decides kernel structure

Two facts about the SeZM neighbour graph shape almost every parallel decision in
this document, and getting either wrong silently invalidates a benchmark:

- **A destination is always one of the `nloc` local atoms.** Destination
  segments are long: 158 edges on this cell.
- **A source may be any of the `nall` extended atoms.** Source segments are
  short: about six edges, and there are twenty-seven times as many of them. The
  extended node feature table is 249 MB and does not fit in the last level
  cache, while the local one would.

A reduction over destinations and a reduction over sources are therefore
different problems, and a synthetic benchmark that draws both endpoints
uniformly measures neither. This mistake was made twice during development, once
for the rotate-and-mix backward and once for the force assembly, and both times
it selected a configuration that was wrong by a factor of two to four in
production.

### Measurement method

Kernels are compared in situ, from a CUDA profile of the compiled model on the
real checkpoint, because standalone harnesses proved easy to mis-specify. Where
standalone numbers appear they use a CUDA-event timer over 30 iterations after 8
warm-up iterations. End-to-end numbers come from `forward_common_lower` under
`torch.compile`, excluding warm-up, with peak memory read from
`torch.cuda.max_memory_allocated`. TF32 is disabled everywhere.

Two measurement traps cost real time before they were understood:

- **The compiler folds a loop-invariant `mma`.** A microbenchmark that issues
  the same `mma` in a loop reports up to 965 TFLOPS because the loop is
  optimized away. A loop-carried dependency on one operand is required for the
  number to mean anything.
- **`kernel.replace_hints()` returns an object with its own JIT cache.**
  Building a hinted variant inside a timing loop misses that cache on every
  iteration and inflates the measurement by fifteen to twenty times. Variants
  must be cached; see `kernel_variant`.

______________________________________________________________________

## 3. What cuTile does well, and what it does badly

cuTile is a tile-level DSL: a kernel is written over whole tiles, and the
compiler owns the mapping to warps, shared memory and tensor cores. On this
workload its behaviour splits cleanly.

**fp32 `mma` is not usable.** On `sm_120` the fp32 multiply-accumulate lowers to
separate `FMUL` and `FADD` instructions. A disassembled square GEMM contains no
`FFMA` at all. Measured on tall-skinny shapes representative of the stack:

| GEMM, fp32 | TFLOPS |
| ---------- | ------ |
| cuBLAS     | 74     |
| Triton     | 68     |
| cuTile     | 15     |

**fp16 tensor cores run at the hardware rate.** The same shapes in fp16 reach
440 TFLOPS through `ct.mma`. This single asymmetry determines the arithmetic of
the whole path: every contraction is evaluated in fp16 with split compensation,
which costs three tensor-core products per fp32 product and still leaves a large
margin over the fp32 path.

**Elementwise and reduction tile code is competitive.** The rotation, the degree
mixing, the attention weighting and the segment reductions are all plain tile
arithmetic and land within 10-20 % of the Triton kernels doing the same work,
sometimes ahead.

**Per-block fixed cost is high.** A block that owns a short segment spends more
time on the dependent scalar reads of its segment bounds than on the work they
gate. On the force assembly, the same total edge work took 3.5 ms spread over
216 000 six-edge segments against 0.78 ms over 8 000 long ones — a factor of 4.5.
Triton, whose programs are lighter, runs the same reduction in 0.72 ms.

The cost is a latency chain, not a launch cost, and the two have opposite
remedies. Batching several atoms into one block is worth 2.4x, because their
segment bounds then arrive in one coalesced read instead of one dependent pair
each. Going further and making the grid *persistent* -- a fixed block count,
each block striding over many atoms -- is 1.7x **slower** than the batched form
and improves monotonically as the persistent block count rises, which says the
kernel wants more concurrency to overlap those chains rather than fewer, fatter
blocks. **Shorten the dependent chain, and keep the block count high enough to
overlap what remains.**

**The compiler needs one hint and resents the others.** `occupancy` is load
bearing: left to itself the compiler spends the entire shared-memory budget on a
single block, and `occupancy=2` was worth 1.4x on the mixing-stack forward and
2.6x on the rotate-and-mix backward. `num_worker_warps` was never better than
the automatic choice and frequently much worse. Thread-block clusters
(`num_ctas`) were seven to sixty times slower on every kernel tried here and
should be considered unavailable.

______________________________________________________________________

## 4. Cross-cutting design

### 4.1 Split-compensated fp16 arithmetic

An fp32 product is recovered from fp16 tensor cores by splitting each operand
into a head and a tail,

$$
x = x_{\mathrm{hi}} + x_{\mathrm{lo}}, \qquad
x_{\mathrm{hi}} = \mathrm{fp16}(x), \qquad
x_{\mathrm{lo}} = x - x_{\mathrm{hi}},
$$

and evaluating three of the four cross products,

$$
A B \;\approx\; A_{\mathrm{hi}} B_{\mathrm{hi}}
\;+\; A_{\mathrm{hi}} B_{\mathrm{lo}}
\;+\; A_{\mathrm{lo}} B_{\mathrm{hi}},
$$

with fp32 accumulation. The dropped term is of order $2^{-22}$ relative.

Three details are not optional.

**The tail must be scaled.** An unscaled tail is $x \cdot 2^{-11}$, which is
subnormal in fp16 for any element below 0.125 and flushes to zero below
1.2e-4; the product then silently degrades to plain fp16 accuracy. The tail
carries a factor `TAIL_SCALE = 2048` and is divided out once per tile.

**The head and the tails need separate accumulators.** Folding them into one
accumulator requires scaling the head as well, which caps the admissible
activation magnitude at `65504 / TAIL_SCALE = 32`. That was tried and produced
`inf` on synthetic inputs. With only the tails scaled, the head represents the
operand unmodified and the representation is valid up to the fp16 maximum. The
cost is one extra accumulator tile and one merge per contraction.

**The split must be invisible to the compiler.** Expressed as tracer-visible
tensor operations, Inductor is free to keep the head in fp32 and elide the
narrowing round trip, which makes the tail identically zero. This is not
hypothetical: it produced a systematic 1e-4 error in the potential energy that
appeared only in the compiled path and vanished in eager. The split therefore
runs inside the opaque operator body, never in the traced graph.

### 4.2 Padding beats exactness

The SO(2) weight is block diagonal over the degree, which invites a contraction
split into `Cf`-wide blocks: exact, no padding, and roughly a quarter of the
multiply-adds of the padded form. Measured, that formulation is three to seven
times slower.

Below roughly 10^5 multiply-adds per `mma` call, the per-call weight load and
its latency dominate, and no arithmetic saving compensates for issuing many
small calls. The rule that follows shapes every kernel here:

> Pad the output axis to a power of two and issue one wide contraction. Skip the
> padded slabs on the *contraction* axis, where the weight rows are exact zeros
> and skipping is free.

Because the padded weight rows and the padded activation rows are both exactly
zero, the padded contraction is bit-identical to the exact one; padding buys
speed without an accuracy argument.

### 4.3 Code generation

The tile model has no list type and requires every tile extent to be a power of
two. A kernel that keeps one tile per spherical-harmonic degree can therefore
neither hold those tiles in an indexable container nor make the degree a tile
axis. The first fails at compile time with an unsupported-syntax error; the
second is impossible for the degree counts that occur (`lmax + 1` and
`2 * lmax`).

Every kernel whose structure depends on the degree layout is consequently
generated: one named tile per degree, one emitted statement per structural
non-zero of the rotation, exponents baked in for the monomials. The generator
lives next to the kernel it emits and shares one `Emitter` (see
`cutile/common.py`), which also provides the balanced `ct.cat` tree used to
assemble a padded group from its degree blocks — `ct.cat` takes exactly two
operands of equal shape, so a wide tile must be built as a binary tree.

Generated modules are written to a cache directory (`DP_CUTILE_CACHE_DIR`, else
a per-user directory under the system temporary directory) and imported as
files. This is a requirement, not a convenience: the tile compiler reads a
kernel from its Python *source*, so a kernel built by `exec` fails with
"could not get source code". The file name carries a digest of the source, which
makes the cache self-invalidating and shareable between processes.

### 4.4 Segmented reduction instead of scatter

Four kernels reduce per-edge quantities onto nodes. All four walk a CSR topology
— one block per node run, iterating its segments — rather than scattering with
atomics. At production neighbour counts a row atomic serializes on the order of
a hundred colliding edges per atom, and the segmented form additionally fixes
the summation order, which makes the result reproducible run to run.

The topology is built with `argsort` plus `searchsorted` (`topology.py`). Both
lower cleanly under `make_fx`, need no data-dependent control flow, and cost far
less than the traffic they save.

The segment key decides the tile, per the graph shape in section 2. The
attention aggregation reduces onto destinations and walks 32 edges per
iteration. The rotate-and-mix backward reduces onto sources and walks 8. The
force assembly reduces onto both and additionally gives each block two atoms,
because neither endpoint alone keeps a block busy.

Lanes that overrun a segment are handled without predicated stores: reads are
redirected to a duplicated in-range edge and masked out of the accumulation, and
writes are redirected to one scratch row appended to each per-edge output.

### 4.5 The launch-configuration table

The tile width and the occupancy hint are not shape-independent, and the penalty
for a wrong choice is large: on the attention aggregation, moving the backward
from a 32-edge to a 16-edge tile at `occupancy=2` is worth 21 %, while raising
the occupancy of that same 16-edge tile to four costs a factor of 1.9. Register
pressure scales with the degree count and the focus width, so the optimum moves
with `(lmax, focus_dim)`.

`launch_config.py` therefore resolves every launch through a table, in
decreasing precedence: process-local registrations, a built-in table keyed by
GPU name or by the longest model-name prefix ending at a space boundary, and a
family default. Defaults are correct on any device and merely untuned, so a new
GPU or a new block layout runs correctly on first contact.

The optima are not guessable. On this device the mixing-stack forward is fastest
with the occupancy left to the compiler while its backward wants two blocks per
multiprocessor; the aggregation forward wants four and its backward two. A sweep
of all seven families takes about forty-five minutes, most of it in the tile
compiler, and was worth 1.3 % of the whole step over hand-picked settings.

The tile width is a compile-time constant of the generated source, so each entry
produces its own cached kernel module; the occupancy is a compiler hint applied
through the cached variant mechanism. `sezm/sweep_launch_configs.py` measures
every candidate on a saturated edge list and registers the winner, and is the
tool that produced the built-in entries.

### 4.6 The operator boundary

Each kernel is exposed as a functional `custom_op` paired with an explicit
closed-form backward operator. The closed form is required rather than
stylistic: under the frozen inference graph the backward operator is dispatched
below autograd, where a nested `torch.autograd.grad` would not run.

A `custom_op` is opaque to Inductor — nothing inside it fuses with the
surrounding graph and its buffers are invisible to the memory planner — so the
signature is where the memory behaviour is decided. Only tensors that must cross
the boundary do. Weight padding, the fp16 split and the transposes the backward
needs are computed inside the operator and cached, not passed in as sixteen
small tensors.

**Cache the packed weights against the call site, never against the operand
storage.** The first implementation keyed the cache on `w0.data_ptr()`. In eager
that works; in a compiled graph the caller re-derives its stacked weights from
the live parameters on every call, so they land in a fresh temporary buffer whose
address is reused unpredictably between convolution instances. The result was
both spurious misses, which repack on the hot path, and collisions between two
different convolutions. The cache is now keyed on an integer identifying the call
site, allocated once per value-path entry.

______________________________________________________________________

## 5. The kernels

### 5.1 Wigner monomials — `wigner_monomials.py`

A Wigner-D block of degree `l` is a homogeneous polynomial of degree `2l` in the
four quaternion components, so the calculator evaluates a fixed monomial basis
and follows it with one coefficient matrix product. The kernel builds register
power ladders for the four components and emits one fully unrolled product per
monomial, with the exponent table baked into the generated source. The backward
is analytic — differentiating a monomial in one component leaves the same
monomial with that exponent reduced by one, times the original exponent — rather
than a replayed product tree.

### 5.2 Rotate and mix — `rotate_mix.py`

The forward is edge major. It gathers the source rows, projects them onto the
reduced rows over the structural non-zeros only, applies the degree mixing, and
writes the focus-major `(F, E, ROW)` activation the stack consumes. The rotated
pre-mix intermediate is never materialized. Neither stage is a matrix product
with a shared operand — the rotation coefficient and the mixing kernel are
per-edge scalars broadcast over channels — so both are elementwise tile
arithmetic and no tensor core is involved.

The backward is where the interesting decision is. The obvious structure mirrors
the forward: an edge-major kernel emitting a per-edge source gradient, followed
by a segmented reduction onto nodes. That was implemented first and measured
5.30 ms plus 1.74 ms standalone.

The structure that ships is **source major**: one block per source node walking
its CSR segment. Two things follow from every edge in a segment sharing one
source node:

- the node's features are read once instead of once per edge, which removes
  1.2 GB of gather traffic;
- the node gradient accumulates in registers and is written directly, which
  removes the per-edge intermediate entirely — 1.46 GB written and 1.46 GB read
  back.

In the model this stage went from 7.97 ms per step to 4.21 ms, and the separate
segmented-reduction kernel disappeared. Nsight Compute confirmed the diagnosis
before the rewrite: the edge-major backward ran at 71 % of DRAM throughput, so
it was already near its roof and only less traffic could help.

In both directions the focus streams are a compile-time loop inside the kernel
rather than a grid axis. As a grid axis they would race in the backward: the
rotation and degree-mixing gradients are shared across focus streams, so every
stream would write the same output element.

### 5.3 Mixing stack — `mixing_stack.py`

The whole stack runs inside one kernel, so neither the inter-layer activation
nor the gated-layer pre-activation reaches DRAM. The backward recovers the
pre-activation by replaying the stack from the operator's own input. This is the
memory result: the saved pre-activation is 2.11 GB per interaction block at
production edge counts, and it is most of what the 11 GB end-to-end reduction is
made of.

The gate needs one non-obvious transformation. Written directly, the sigmoid
gate multiplies a *column block* of the pre-activation, and the tile model offers
no scatter into a column block. Instead the gate projection is expanded into one
matrix per degree group whose column blocks carry the degree-to-gate mapping:
block 0 of the `m = 0` projection is the identity, so its sigmoid is the SiLU
gate of the scalar rows; block `r` is gate `r - 1`; and the `|m| = 1` projection
replicates gate `o mod lmax`. One contraction per group then produces a
full-width gate tile that multiplies the pre-activation elementwise.

The backward carries one further fusion. The gate-logit gradient re-enters
through the scalar rows only, and folding it into the operand of the contraction
that already reads that weight saves one traversal of the weight per gated layer.
That is worth 4 % of the backward, consistent with the picture that this kernel
spends its time moving weights.

### 5.4 Attention aggregation — `flash_atten.py`

The forward folds four stages into one destination-segmented pass: inverse
rotation, degree rescale, envelope-gated softmax weight, and the reduction onto
the destination node. Neither the rotated-back message nor the weighted value
reaches DRAM. The backward is edge major, because every gradient it produces is
per edge and it therefore needs no topology at all.

The degree rescale is baked into the generated source as a literal per full-basis
row. It is a configuration constant, and passing it as a tensor would cost a load
per row per edge for a value the compiler can fold.

### 5.5 Force assembly — `force_assembly.py`

Two segmented reductions, one per endpoint, over pre-built CSR topologies, with
float64 accumulation. The `(3, 3)` outer product is recomputed per edge in
registers and never materialized, which removes an `(E, 9)` intermediate. The
force and virial lanes are carried as four- and sixteen-lane tiles so both stay
on power-of-two extents; the unused lanes load as zeros and their stores fall
outside the output rows, where they are discarded.

Both endpoints are indexed over the extended atoms, so this kernel sees the worst
of both segment distributions at once: 8 000 long destination segments scattered
among 216 000 slots, and 216 000 short source segments. It is the kernel that
exposed cuTile's per-block fixed cost (section 3), and giving each block two
atoms rather than one is worth 2.4x. It remains the one stage where cuTile loses
to Triton, by a factor of six on a stage that is 3 % of the step.

______________________________________________________________________

## 6. What each kernel costs

From a CUDA profile of one compiled force step on the 8000-atom cell, summed per
stage. Both interaction blocks are included, so these are per-step figures.

| stage                                        | Triton level 3 | cuTile        | ratio |
| -------------------------------------------- | -------------- | ------------- | ----- |
| mixing stack, forward and backward           | 34.97 ms       | **28.80 ms**  | 1.21x |
| rotate and mix, including the node reduction | 10.14 ms       | **6.62 ms**   | 1.53x |
| attention aggregation, forward and backward  | 7.61 ms        | **7.29 ms**   | 1.04x |
| force and virial assembly                    | **0.72 ms**    | 4.62 ms       | 0.16x |
| SO(2) path total                             | 53.44 ms       | **47.33 ms**  | 1.13x |
| whole step, device time                      | 163.31 ms      | **152.53 ms** | 1.07x |

The step improves by more than the SO(2) path does, because the fused kernels
also remove zero-fill and elementwise traffic elsewhere in the graph: the
`FillFunctor` time alone halves.

The mixing-stack forward reaches roughly 48 TFLOPS of effective fp32 throughput,
which is 3.2x what cuTile's own fp32 `mma` can deliver and 71 % of what Triton's
fp32 path achieves on isolated GEMMs of this shape — while also fusing three
layers.

______________________________________________________________________

## 7. Accuracy

Per-kernel, at the shapes in `source/tests/pt/model/test_descriptor_sezm_cutile.py`,
as maximum relative error against the reference that defines each kernel:

| kernel                      | forward | backward |
| --------------------------- | ------- | -------- |
| rotate and mix              | 9.4e-8  | 2.2e-7   |
| mixing stack (against fp64) | 6.2e-7  | 9.4e-7   |
| attention aggregation       | 2.7e-7  | 1.7e-7   |
| Wigner monomials            | 1.1e-8  | 7.9e-8   |
| force and virial            | 3.5e-7  | —        |

The mixing stack is the only kernel using split-compensated fp16, and its error
is two to three orders of magnitude above the plain-fp32 kernels and roughly at
the level of an fp32 accumulation of the same length — which is the point of the
compensation.

The rotation gradient is compared on its structural block diagonal. Outside it
the dense reference differentiates coefficients that the Wigner construction
writes as constants, so its own backward discards them and the kernel does not
compute them.

End to end, against the Triton level-3 path on the 8000-atom cell: 5.9e-7
relative on the per-atom energy and 8.0e-6 relative on the force.

______________________________________________________________________

## 8. Designs that were measured and rejected

Recorded with the number that decided each one. Several are properties of the
tile model rather than of this workload and will re-appear in any cuTile kernel.

**fp32 `mma`.** 15 TFLOPS against 74 for cuBLAS. Disassembly shows separate
`FMUL` and `FADD` and zero `FFMA`. Every contraction moved to split-compensated
fp16.

**Exact block-diagonal contraction.** Splitting the stack GEMM into `Cf`-wide
blocks is exact and needs no padding, and is three to seven times slower than
one padded wide contraction. Below about 10^5 multiply-adds per call the weight
load dominates.

**Whole-edge-path mega fusion.** A single kernel covering rotation, mixing,
stack, inverse rotation, attention weighting and destination reduction was
implemented and validated. The forward was 1.36x faster than the separated
kernels. The backward, which must hold both the forward replay state and the
reverse sweep state, exhausted the register file and ran 3x slower than the
separated backward — DRAM bound on spills. Rejected: the forward gain is an
order of magnitude smaller than the backward loss.

**Edge-major backward plus a separate segmented reduction.** 7.04 ms against
2.74 ms standalone for the fused source-major backward, for the reasons in
section 5.2.

**One node per block in a short-segment reduction.** 4.5x slower than the same
work over long segments, and 2.4x slower than two nodes per block. See section 3.

**Single-accumulator fp16x3.** Overflows. See section 4.1.

**A coalesced padded row load for the rotation coefficients.** Loading a whole
padded Wigner row per reduced row and extracting the coefficients is a better
access pattern on paper and was 25 % *slower* in both directions than one scalar
load per structural non-zero. The rotation blocks are small enough to stay in
cache, so the wider tile buys nothing and costs registers.

**Element gather in a segmented reduction.** Replacing `ct.gather` over computed
element indices with sliced advanced indexing — sparse axis selects the edge,
dense axis stays a contiguous slice — was worth 1.8x on a DRAM-resident read. Use
the sliced form whenever the inner axis is contiguous.

**Thread-block clusters.** `num_ctas > 1` was seven to sixty times slower on
every kernel tried.

**`num_worker_warps`.** Honored from tile-compiler 13.3 onwards -- confirmed with
no ignore warning on `nvidia-cuda-tileiras` 13.3.36 -- and still not useful here:
the automatic choice equals `4` to within 3 % on the mixing stack, and `8` is
1.8x slower. This is the one knob the reference documentation points at for
register-bound kernels, so its absence of effect is what closes the occupancy
question on the stack backward rather than leaving it open.

**The `latency` load hint.** Raising the prefetch-depth hint on the mixing
stack's weight loads moved the forward non-monotonically between 3.59 and 4.41 ms
around a 3.76 ms baseline and left the backward flat within 1 %, with outputs
bit-identical throughout. The weight stream is already pipelined; the hint has
nothing left to buy. It remains worth trying on a kernel whose loads are not
loop-invariant.

**A persistent grid.** Replacing the force assembly's one-run-per-block grid with
a fixed block count striding over runs is 1.7x slower at its best setting, and it
improves monotonically as the persistent block count rises. Both facts say the
kernel is short of concurrency to overlap its dependent reads, not short of work
per block -- the opposite of what the earlier per-block-cost reading implied, and
the reason section 3 states the remedy as shortening the chain rather than
widening the block.

**`assume_divisible_by`.** No applicable site: every index in these kernels
derives from `ct.bid()` with power-of-two tile extents, which the compiler already
treats as aligned.

**`data_ptr`-keyed weight cache.** Correct in eager, wrong under compile. See
section 4.6.

**Tracer-visible fp16 splitting.** Silently elided by Inductor, producing a
1e-4 energy error only in the compiled path. See section 4.1.

**Uniformly random endpoints in a tuning benchmark.** Selected a tile that was
wrong by a factor of two to four for both endpoint reductions. See section 2.

______________________________________________________________________

## 9. What limits the path, and what a 2x step would require

The complete SO(2) value path is 53 ms of the 163 ms Triton step. Making all of
it free would yield 1.5x. Kernel substitution inside this path therefore cannot
reach 2x, and the 1.06x measured here is the honest consequence of the stack
already having been well optimized in Triton.

What is left, in order of size:

1. **The mixing-stack backward, 20.7 ms per step, is the single largest kernel in
   the step.** It is 2.6 times its own forward. Its profile is neither
   bandwidth nor issue limited -- 39 % of DRAM throughput, 30 % of SM throughput,
   64 % of L2 -- but occupancy limited: 255 registers per thread admit two blocks
   per multiprocessor, for 16.6 % achieved occupancy and eight active warps against
   a possible forty-eight, at 10.3 warp cycles per issued instruction. Against the
   fp16 tensor-core roof the kernel already runs at about 47 %, so the headroom is
   roughly 2x and it is gated by the register footprint of holding the replay and
   the reverse sweep at once. No compiler hint reaches it (section 8); only a
   design that carries less state would. It replays the forward and then sweeps back, so it traverses the
   weights roughly twice as many times as the forward. Its arithmetic is already
   at the tensor-core rate; the remaining term is weight traffic, and the only
   structural way to cut it is to hold more of the weight resident across layers,
   which is where the register file ran out in the mega-fusion attempt.
1. **The force assembly, 4.6 ms per step**, is six times Triton's and is limited
   by the dependent segment-bound reads of very short segments, not by traffic.
   The persistent-grid formulation that this diagnosis suggests was measured and
   is slower (section 8), so what remains is shortening the chain itself: a
   topology that stores segment lengths alongside the offsets would remove one of
   the two dependent reads per atom.
1. **Everything outside the SO(2) path** — the radial basis, the fitting network,
   the neighbour bookkeeping and the graph-lowering overhead — is the other 105
   ms and is untouched by this work.

A 2x step needs the third item, which means either an end-to-end fused operator
in the style of the DPA1 `graph_energy_force` path, or an analytic force that
removes the backward sweep altogether. Both are language-independent
observations; neither is a cuTile question.

______________________________________________________________________

## 10. Tile-model reference

Behaviours that cost time to discover and that a future kernel author should
assume from the start.

**Tile extents must be powers of two.** Every non-power-of-two structure must be
padded, and the padding must be provably zero if the result is to stay exact.
This includes `ct.arange`: a window of `NODES + 1` segment bounds must be read as
two power-of-two gathers.

**There is no list type in tile code.** A kernel that needs one tile per
structural element must be generated.

**The tile compiler reads the kernel from its source file.** Kernels built with
`exec` fail with "could not get source code". Generated kernels must be written
to disk and imported.

**`ct.cat` takes two operands of equal shape.** Wide tiles are assembled as
binary trees.

**Loop bounds must share one integer type.** A `range` whose bounds come from an
int64 array and whose step is an int32 literal fails verification. CSR offset
arrays are therefore int32.

**Selecting between operands of different integer widths fails verification.**
`ct.where(mask, int32_tile, int64_scalar)` emits a scalar narrowing that the
verifier rejects with a `trunci` shape error. Broadcast the scalar into a tile
with arithmetic first, then narrow explicitly:
`ct.where(mask, tile, (tile * 0 + scalar).astype(ct.int32))`.

**Index arity must match array rank exactly.** A rank-4 array needs four index
items even when three of the axes are singleton.

**`kernel.replace_hints()` has its own JIT cache.** Cache the variant.

**Runtime scalars enter through `ct.ScalarInt64`.** There is no `ScalarInt32`.

**Element offsets on edge-scaled arrays exceed 2^31 near 10^7 edges**, which is
inside the production range. Annotate those arrays with
`ct.ArrayAnnotation(index_dtype=ct.int64)`.

**Give a block enough work to hide its prologue.** Per-block fixed cost is high
relative to Triton; a kernel whose blocks each do a few hundred nanoseconds of
work will be dominated by it.

______________________________________________________________________

## 11. File inventory

The layout mirrors the Triton kernel package: one module per kernel family,
owning its generated kernels, its operator registration and one public entry
point, with the launch-configuration resolver separated from its data.

```
deepmd/pt_expt/kernels/cutile/
├── common.py              availability, fp16 split, hint cache, source generator
└── sezm/
    ├── indexing.py        reduced coefficient layout and its padded tile extents
    ├── wigner_monomials.py
    ├── so2_rotate_mix.py  rotation and degree mixing, source-major backward
    ├── so2_mixing_stack.py the complete gated stack, replay backward
    ├── so2_value_path.py  factory binding the value path into SO2Convolution
    ├── flash_atten.py     inverse rotation, weighting, destination reduction,
    │                      and the CSR row offsets the segmented kernels share
    ├── force_assembly.py  force and per-atom virial
    ├── tile_configs.py    launch-configuration resolver, families, defaults
    ├── tile_config_data.py built-in per-GPU tables
    └── sweep_tile_configs.py
```

`indexing.py` reuses the canonical reduced layout from
`deepmd/dpmodel/descriptor/dpa4_nn/indexing.py` rather than restating it, and
adds only what the tile model needs on top: the structural non-zeros of the
rotation and the power-of-two padded extents.

Gate: `deepmd/pt_expt/kernels/utils.py::use_cutile_infer`.
Binding sites: `SO2Convolution.__init__` and `so2_message`
(`deepmd/pt/model/descriptor/sezm_nn/so2.py`), the monomial basis
(`sezm_nn/wignerd.py`), and the force assembly
(`deepmd/pt/model/model/transform_output.py`).
Tests: `source/tests/pt/model/test_descriptor_sezm_cutile.py`.

### Supported configuration

The value-path factory validates the block layout and declines otherwise,
leaving the dense reference in charge. Support is narrower than the Triton path
in two respects, both forced by the tile model: the focus width must be a power
of two because it is a tile extent, and the radial degree mixer must be the
rank-one `degree_channel` form, whose per-edge kernel is a scalar per degree
pair and therefore elementwise. Cross-focus competition with more than one focus
stream is also excluded.

### Deployment status

This is a Python-inference path. `cuda.tile` kernels are JIT compiled at runtime
and do not bake into the AOTInductor `.pt2` artifact, so the path is not
available to the LAMMPS or GPUMD C++ inference route. Reaching those consumers
requires the kernels in a form that survives ahead-of-time export, which is what
a CUDA C++ implementation is for.

# SeZM / DPA4 Triton Kernels

This is the complete reference for the Triton path of the SeZM / DPA4 descriptor (PyTorch backend). It documents the operators, measured performance, launch-configuration and auto-tuning machinery, numerical invariants, and algorithm-selection constraints. The high-level model documentation (`doc/outisli/dpa4.md`) only summarizes this path and points here for the detail.

Every operator is registered as a functional `torch.library.triton_op` and launches its `@triton.jit` body through `torch.library.wrap_triton`, so it differentiates correctly under the traced (`make_fx`) force-autograd path and, unlike an opaque `custom_op`, is visible to Inductor and bakes into AOTInductor `.pt2` packages (see §5).

Inference and training are gated separately, by `DP_TRITON_INFER` and `DP_TRITON_TRAIN`. The distinction is a real one: an operator serves inference as soon as it reproduces the forward and the coordinate gradient with the parameters held fixed, whereas training additionally needs the gradient of every parameter it consumes *and* a second derivative of its own backward, because the force loss differentiates the backward pass again. §6 covers what that requires and which operators satisfy it.

## 1. The `DP_TRITON_INFER` gate

`DP_TRITON_INFER` is a cumulative numeric level read once at model construction (`deepmd.pt_expt.kernels.utils.triton_infer_level`); only the integers `0`–`3` are accepted. It takes effect during inference only.

| level | adds                                                                                                                                                                                                                                                                                                                                                                                         | numerics                   |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| `0`   | nothing (dense reference path)                                                                                                                                                                                                                                                                                                                                                               | —                          |
| `1`   | universal kernels needing no launch-config table: block-diagonal SO(2)/Wigner rotation (`so2_rotation`), dynamic radial degree mixing (`radial_mix`), the `SO2Linear` block GEMM (`so2_block_gemm`), Wigner-D monomial bases (`wigner_monomials`), the attention value-aggregation forward and its per-edge backward (`flash_atten`), and segmented force/virial assembly (`force_assembly`) | exact fp32                 |
| `2`   | the table-configured kernels: the fused SO(2) value path (`so2_value_path`: `so2_rotate_mix` + `so2_mixing_stack`) and the edge-block backward schedules of the flash aggregation and the rotate+mix. Unresolved table keys fall back to a level-1 kernel or a spill-safe configuration                                                                                                      | exact fp32                 |
| `3`   | the fp16x3 split-compensated mixing stack (`so2_stack_fp16x3`) on shapes whose configuration passed the fp64 validation sweep; unswept shapes keep the fp32 stack                                                                                                                                                                                                                            | ~fp32 (2⁻²² rounding step) |

The gate composes with `DP_COMPILE_INFER` and with `.pt2` freeze: the level chosen at construction is baked into the traced graph. `DP_CUTE_INFER` selects an experimental CuTe value-path backend instead and is mutually exclusive with `DP_TRITON_INFER` (both claim the fused SO(2) value path).

## 2. Measured inference performance

Training numbers are in §6.7.

### 2.1 RTX PRO 6000 Blackwell / DPA4-mini

The RTX PRO 6000 Blackwell table is keyed by the stable device-name prefix `NVIDIA RTX PRO 6000 Blackwell`, so the same measurements serve Server, Workstation, and other edition suffixes. The full supported family (`Cf ∈ {32, 64, 96, 128}`, `lmax ∈ [1, 6]`) was swept independently from H20. In particular, the table does not assume that Blackwell's higher fp32 peak makes the fp32 stack universally preferable: the final route compares the complete forward-plus-force-backward path and selects fp32 or fp16x3 per shape.

Production profile: `dpa4-mini.pt`, 4032-atom diamond, approximately 637k edges, TF32 off. Nsight Compute shows that the important kernels lie on or close to the memory roof rather than the fp32 roof:

| kernel / schedule                    | time     | DRAM BW    | DRAM peak | SM peak | registers/thread | achieved occupancy |
| ------------------------------------ | -------- | ---------- | --------- | ------- | ---------------- | ------------------ |
| separate sigmoid recompute           | 0.121 ms | 1.387 TB/s | 86.9%     | —       | 168              | 23.7%              |
| separate backward pointwise          | 1.221 ms | 1.496 TB/s | 93.7%     | —       | 40               | 98.9%              |
| fused recompute + backward pointwise | 1.112 ms | 1.494 TB/s | 93.6%     | 9.7%    | 254              | 16.5%              |
| fused fp32 `m=0` GEMM + forward gate | 0.669 ms | 1.26 TB/s  | 79.0%     | 36.1%   | 212              | 16.4%              |
| rotate+mix backward                  | 1.46 ms  | 1.26 TB/s  | 79%       | 67%     | 61               | 49.2%              |
| fp32 stack backward GEMM             | 0.756 ms | 1.43 TB/s  | 89.4%     | 62%     | 96               | 40.7%              |

The apparent H20/Pro 6000 parity therefore has a direct first-principles explanation: the advertised fp32 increase cannot accelerate kernels already transferring at 79–94% of the Pro 6000 DRAM roof. The profitable changes reduce bytes or repair a bad launch, rather than merely adding arithmetic throughput:

- Folding sigmoid recomputation into the backward pointwise kernel removes one `sig` store and reload, approximately 326 MB at the production edge count. The fused kernel gives up occupancy because its register count rises from 40 to 254, but the reduced traffic still wins by about 1.21× for the DPA4-mini shape in the NCU profile.
- Folding the fp32 `m=0` GEMM and forward gate keeps the degree-group outputs in registers and removes their round trip through `z`; the DPA4-mini microbenchmark improves by about 1.42×.
- The per-edge flash backward is pinned from a saturated-edge sweep because AOTInductor's tiny trace sample is not representative: four warps take 2.55 ms at production size, while one warp takes about 1.53 ms.

The tuned artifact reduces the 4032-atom Nsight Systems GPU total from 577.9 ms to 534.3 ms over the same nine-call profile window. The `debug/cuda_bench/lmp_scan.py` curve against the reference Pro 6000 package improves every one of the 18 finite sizes, with a 1.078× geometric-mean speedup. Unlike isolated kernel timings, these numbers include the entire MD step:

|  atoms | reference (atoms/ms) | tuned (atoms/ms) | speedup |
| -----: | -------------------: | ---------------: | ------: |
|  4,032 |              62.8416 |          67.7940 |  1.079× |
| 19,968 |              61.6178 |          66.5571 |  1.080× |
| 39,304 |              61.3855 |          66.2514 |  1.079× |
| 48,640 |              61.2839 |          66.1034 |  1.079× |

Both curves reach 48,640 atoms and OOM at 64,000. The capacity points use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; without it, allocator fragmentation can cause a false early OOM at 48,640 despite sufficient aggregate free space (see §5).

### 2.2 H20 Neo reference

Reference system: checkpoint `N_2-s_3-c_32x2-so3` (lmax 3, mmax 1, Cf 32, F 2, 2 interaction blocks × 3 mixing layers), 4096-atom diamond supercell (E ≈ 647k edges), H20, TF32 off, `DP_COMPILE_INFER=1`, full force evaluation, 20-iter mean (warmup excluded), `dF` against the level-2 output.

| level | time (ms) | peak alloc (GB) | dF_max (eV/Å) |
| ----- | --------- | --------------- | ------------- |
| 0     | 365.9     | 39.4            | 4.7e-6        |
| 1     | 244.2     | 31.9            | 3.4e-6        |
| 2     | 174.8     | 23.8            | reference     |
| 3     | 158.0     | 24.4            | 3.9e-6        |

Per-kernel CUDA self-time at level 3 (154.5 ms/step total; buckets fold the misattributed pointwise stack helpers back into `stack_gemm`):

| bucket                                              | ms/step | %   | note                                         |
| --------------------------------------------------- | ------- | --- | -------------------------------------------- |
| SO(2) mixing stack (fwd+bwd, gate/point/recompute)  | ~68.6   | 44  | FLOP core; at the fp16x3 tensor-core ceiling |
| cuBLAS GEMM (FFN / readout / attention projections) | 29.7    | 19  | at cuBLAS efficiency                         |
| Inductor pointwise                                  | 12.8    | 8   | already fused by Inductor                    |
| flash aggregation (fwd + edge-block bwd)            | 9.4     | 6   |                                              |
| rotate+mix (fwd + edge-block bwd)                   | 8.5     | 6   |                                              |
| scatter / index_add (aggregation + norm bwd)        | 8.0     | 5   | attention softmax proper is ~2.2 of this     |
| reductions (norm bwd, rsqrt chains)                 | 7.0     | 5   | already single Inductor fusions              |
| index / gather                                      | 6.6     | 4   |                                              |
| Wigner-D monomials                                  | 1.7     | 1   |                                              |

The single heaviest kernel is `_stack_fp16x3_bwd_kernel` at 28.8 ms/step (18.6 %) — the force gradient through the mixing stack. Roughly 63 % of the step (stack + cuBLAS) already runs at the hardware ceiling; the remainder is fragmented 0.5–2 ms Inductor fusions. Section 6 records the resulting algorithm-selection boundaries.

## 3. Operators

The operators are described in pipeline order. Each fuses a span of the reference `so2_message` / attention / force path and provides an exact hand-written backward that is itself differentiable under the force-autograd trace. All dispatch to an eager reference off the CUDA fp32 path (wrong dtype, CPU, or a zero-edge call).

### 3.1 Fused SO(2) value path — `so2_value_path.py` (level ≥ 2)

`SO2Convolution.__init__` binds `_triton_value_path = make_triton_value_path(self)` and `so2_message` routes through it in eval mode. The entry replaces Steps 1–5 of the reference message with two operators.

**`sezm_triton::so2_rotate_mix`** — one program per edge: gathers `x[src]`, applies the block-diagonal Wigner rotation over the structural non-zeros only (register-resident), applies the edge-conditioned radial degree mixing, and stores directly into the focus-major flat layout `(F, E, (3·lmax+1)·Cf)` the mixing stack consumes. A `RANK` constexpr covers the whole mixer family: `rank >= 1` forms the low-rank channel-factorized degree kernel in registers (`rank == 1` keeps the fast path where the basis factors out of the contraction) and `rank == 0` is the mixer-free variant (`radial_so2_mode: none`) that scales each reduced row by its degree's radial feature. The backward recomputes the rotation in registers (the forward saves no per-edge intermediate) and reduces the node gradient with `sezm_triton::segment_sum`, a contention-free CSR segment sum replacing the atomically serialized `index_add_`. Its forward `(warps, stages)` comes from the `rotate_mix_fwd` table; its backward dispatches to an edge-block schedule on `(C_wide, lmax)` keys present in the `rotate_mix_bwd_block` win list (requires `rank <= 1`).

**`sezm_triton::so2_mixing_stack`** — the whole mixing stack (`n_layers − 1` gated layers + identity final layer, cross-focus competition weight folded into the final store) as one operator. Keeping the inter-layer activations inside the op minimizes the compiled graph's activation footprint: only the stacked gated-layer pre-activations `z_all` and the result surface as outputs, and gate sigmoids are recomputed in the backward. Per gated layer: an `m = 0` block GEMM, a pointwise gate (sigmoid gates + SiLU + residual), and an `|m| = 1` GEMM with the gate/residual/alpha epilogue fused. On shapes in the `stack_m0_gate` win list, the `m = 0` GEMM also carries the gate epilogue and retains all degree-group outputs in registers, eliminating the intermediate read. The final identity layer streams straight into the edge-major `(E, F, ROW)` layout the flash aggregation consumes — no reassembly copy. The backward saves only `z_all` and the op's own `x_local` output; `grad_alpha = sum(g·x_local)/alpha` is exact from the identity, saving the two pre-scale activation copies.

The pointwise gate / recompute / backward-pointwise kernels hold `lmax` register tiles of width `next_power_of_2(Cf)`, so their winning `(BLOCK_M, warps, stages)` follows the register-pressure product `lmax · next_power_of_2(Cf)`; launch triples come from the `(Cf, lmax)`-keyed `gate` / `recompute` / `point` tables (unresolved keys fall back to the spill-safe `(16, 8, 2)`). The `point_recompute` win list folds sigmoid projection into the backward pointwise kernel only where eliminating the `sig` surface outweighs the extra registers. At `Cf >= GATE_BMM_MIN_FOCUS_DIM` (96) the per-group `CP × CP` gate dot spills regardless of tiling (a padded 96 behaves like 128), so an unfused path delegates the sigmoid projection and gate-logit contraction to cuBLAS batched matmuls.

The three IEEE-fp32 GEMM slots are independently configured by `stack_fp32`: forward `m = 0`, forward `|m| = 1`, and combined backward. Their matrix widths and epilogues differ enough that one shared launch tuple leaves performance on the table on RTX PRO 6000. Every selected tuple is checked through the complete stack against the conservative configuration because changing `BLOCK_K` regroups fp32 partial sums.

### 3.2 fp16x3 mixing stack — `so2_stack_fp16x3.py` (level 3)

The stack GEMMs are the FLOP core and the fp32 `tl.dot` tiling already sits at the H20 FFMA ceiling (~25 TFLOPS, matching cuBLAS/cutlass SIMT), so the only remaining GEMM lever is the compute unit. `sezm_triton::so2_mixing_stack_fp16x3` evaluates each fp32 GEMM as three fp16 tensor-core products with fp32 accumulation (a two-term Ootomo split, `A_hi·B_hi + A_hi·B_lo + A_lo·B_hi`; the `A_lo·B_lo` term at ~2⁻²² relative is dropped), reaching ~1.6× the FFMA GEMM throughput. An fp16 multiply feeding an fp32 accumulator is exact, so the only error is the two split truncations.

Three range/precision safeguards are structural, all exact powers of two:

- **Separate accumulators.** The head product and the two tail corrections accumulate in *separate* fp32 accumulators merged once per tile; chaining all three into one accumulator absorbs the small tail terms against the large head partial sums and doubles the error.
- **Tail scaling `2¹¹`.** Tails are stored pre-scaled by the fp16 mantissa width so an element below ~1.2e-4 does not fall out of the fp16 subnormal range and silently lose its correction; the epilogue scales back. `|x_lo · 2¹¹| <= |x|`, so the scaled tail never overflows where the head does not.
- **Activation prescale `2⁻⁴`.** The stack input rides the unnormalized residual stream (SeZM applies the equivariant norm *after* the SO(2) update), so the activation operand is prescaled before the split and rescaled in the epilogue. Measured layer inputs peak near 13 with ~6× per-block growth; the prescale keeps four orders of magnitude of head-overflow margin. Weights stay unscaled; a checkpoint whose stack weights or activations exceed the fp16 head range surfaces loudly as NaN on the first evaluation.

Accuracy contract: per-GEMM error against fp64 matches the fp32 reference (~5e-7); through a full force evaluation the deviation from the fp32 stack is ~4e-6 eV/Å maximum on a 4096-atom system (~1e-7 eV/atom on the energy). The 2⁻²² rounding step is three orders finer than TF32, so the PES smoothness matches fp32. The factory validates the supported family (`mmax == 1`, `lmax` 1–6, `Cf ∈ {32, 64, 96, 128}`, gated stack with identity final layer, mixer absent or `degree_channel` rank 1–4, fp32 weights, no `so2_norm`/`layer_scale`/`attn_res`/grid product) and returns `None` otherwise. Launch configurations come exclusively from the validated `stack_fp16x3` tables (see §4, §5).

### 3.3 Fused attention aggregation — `flash_atten.py` (level 1 fwd, level 2 edge-block bwd)

Under a force loss the aggregation's second order is served by two dedicated
kernels rather than the multilinear substitution form: the operator is
trilinear in `(x_local, Dt, alpha)`, so substituting each cotangent
separately costs three forward and three backward re-entries per
convolution, while `_flash_2nd_gather_kernel` (one CSR pass emitting the
output-cotangent term) and `_flash_2nd_edge_kernel` (one pass over the
structural non-zeros emitting the `x_local` / `Dt` / `alpha` terms)
carry the same mathematics in two launches. The substitution form remains
the fallback when only a subset of the cotangents is present. The Wigner
gradient buffer starts from zeros because the kernels write only the three
structural columns of each degree block.

`sezm_triton::flash_atten_aggregate` folds the block-diagonal `rotate_back` (transposed Wigner-D), the inverse-rotation degree rescale, the per-edge envelope-gated softmax weight `alpha`, and the destination reduction into one edge→node pass, so neither the rotate-back message `x_message` nor the weighted value is written to DRAM. The softmax `alpha` (scalar `(E, F, H)` logits) and the output-side head gate (node-level `(N, F, H)`) carry no bandwidth cost and stay outside; the gate is applied as `out = pre_gate · gate` afterwards.

The forward is one program per destination node performing an indirect CSR segment reduction: the operator builds a destination-sorted topology internally (`argsort` + `searchsorted`, integer ops, no gradient) and accumulates each edge segment in a `DIM`-row register tuple. This is deterministic and several times faster than a per-edge atomic scatter, which serializes on the ~10² colliding edges per atom at production neighbor counts. The internal sort is mandatory, not defensive: the traced graph keeps masked padding edges in arbitrary destination order.

The backward has two schedules. The per-edge kernel (level 1) emits all gradients from one program per edge — one cross-lane reduction per structural Wigner non-zero, serialized warp-shuffle chains that dominate on narrow hidden widths. `flash_bwd_edge` pins its `(warps, stages)` from a saturated-edge measurement; an uncovered key retains Triton's autotuner. On `(C_wide, lmax)` keys in the `flash_bwd_block` win list an edge-block kernel takes over: `BLOCK_E` edges per program with channels as the vector axis, turning each `grad_Dt` entry into one batched reduction (up to 1.84× at `C_wide = 64`). Wide widths lose on register-tile pressure (down to 0.36× at `C_wide = 192`), so a missing key keeps the per-edge kernel and no configuration can regress.

### 3.4 Segmented force assembly — `force_assembly.py` (level 1)

`edge_energy_deriv` (in `transform_output.py`) assembles force and per-atom virial from the per-edge energy gradient. Under the gate (inference only, `create_graph=False`), the four `index_add` scatters and the materialized `(E, 9)` outer product are replaced by `sezm_triton::edge_force_assembly`: two CSR segment-reduction passes (destination then source order) that recompute the outer product on the fly and accumulate the 12 output scalars per extended atom in float64 registers, removing the atomic serialization and tightening the summation error relative to fp32 atomics.

### 3.5 Wigner-D monomial evaluation — `wigner_monomials.py` (level 1)

`WignerDCalculator` builds the `l >= 2` blocks from fixed quaternion monomial bases followed by one coefficient matmul. On the CUDA inference path the basis evaluation routes through `sezm_triton::wigner_monomials`, whose exponent table is baked in at compile time (register power ladders, fully unrolled products, analytic leave-one-out backward). The CUDA training value path uses the same operator to build the packed block-diagonal runs consumed by every convolution block, followed by one coefficient matmul; the edge cache shares that result across blocks, so neither the dense Wigner pair nor a duplicate packed run is materialized. The `l = 2` block additionally collapses its 4⁴ rank-4 contraction onto the 35 unique degree-4 monomials at construction. Exponent tables are flattened to Python constants in `__init__` — a trace-time `.tolist()` would create unbacked symbols under `make_fx` and abort export.

### 3.6 Standalone rotation and block GEMM — `so2_rotation.py`, `so2_block_gemm.py` (level 1)

For convolutions outside the fused-value-path family (and at level 1 for all convolutions), the eager rotation `bmm` is replaced by the block-diagonal rotation kernels (`rotate_to_local` / `rotate_back` and their block/SO(2) variants), and the `SO2Linear` mixer by `sezm_triton::so2_block_diag_gemm` — a `BN = 64` block-diagonal GEMM that skips the structural off-`|m|` zeros. These need no launch-config table (single shape-independent or runtime-autotuned configs) and serve as the correct fallback whenever `make_triton_value_path` declines a convolution.

### 3.7 Standalone rotate-mix — `make_triton_rotate_mix` (level 1, wide blocks)

The level-1 companion of the fused value path: only the rotation and the
radial degree mixing fuse into the `so2_rotate_mix` operator (§3.1), writing
the focus-major mixing input directly while the mixing stack stays with the
compiler. The degree-expanded local intermediate and its relayout never
reach the traced graph. The operator is quadrilinear, so a force loss
re-enters its forward and backward several times for the second order; that
fixed cost is repaid only where the removed materialization is large. The
binding follows the measured crossover at `C_wide = 128`: the 128-wide Pro
shape gains ~2 ms per step while 64-wide shapes lose ~10%, so narrower
blocks keep the separate rotation and radial-mix kernels, whose backwards
are bilinear and trilinear.

### 3.8 Segmented attention softmax — `segment_softmax.py` (level 1)

`sezm_triton::segment_softmax` replaces the scatter/gather chain of
`segment_envelope_gated_softmax`: one
CSR-segmented kernel per direction (forward, first-order backward, second
order) walks the destination-sorted edge list the flash aggregation already
maintains, holding the per-channel vector (`C = F * H`, a handful) in
registers. The forward saves the normalized weights and the per-node null
weight, so neither backward re-reduces the segment maxima; the second order
is the hand-derived closed form of the softmax backward (four segment
scalars, two passes; see the module docstring for the derivation). The
envelope enters through `env**2` inside the operator, so its gradient and
curvature terms stay in-kernel as well. Under the force loss this removes
roughly a dozen materialized surfaces and the serialized `scatter_add` per
convolution block. A bridged model runs the same operator, because its source
gate is folded into the envelope the operator reads.

### 3.9 Standalone gated activation — `sezm_triton::gated_act` (level 1)

`GatedActivation` binds the fused operator at construction for its focus-major
(`fndc`) self-gated configuration — `mmax = 1`, SiLU scalar activation, no
gate bias — which is exactly the SO(2) mixing stack's nonlinearity. The
operator takes the pre-activation `(F, E, ROW)` together with the gate
projection and its transpose, evaluates the per-degree sigmoid gates through a
`CP × CP` register dot, and scales the three value rows sharing each gate.
Forward, first-order backward (the shared `_stack_point_bwd_kernel` with the
input-recovery path disabled) and the second order
(`gated_activation.py`) each run as one kernel per focus stream, so a
force-loss training step traverses the activation without expanding it into
the per-operation elementwise kernels the dense expression lowers to. The
gate weight's gradient contracts the whole edge axis and stays in ATen
(cuBLAS / tuned template). Launch triples reuse the `gate` / `point_train` /
`gated_second_order` tables with their spill-safe fallbacks, so no sweep is
required for correctness or basic performance.

The kernels hold `lmax` register tiles of width `next_power_of_2(Cf)` and
contract the gate projection as register dots, so the binding is limited to
the measured profitability region: all degrees at `Cf <= 32`, and
`lmax <= 3` at `Cf = 64`.

Beyond that region the operator is numerically complete through its
batched-matmul form — the projection and both logit contractions run as
cuBLAS matmuls around the Triton `SIG_IN` elementwise variants. A
hand-written CUDA form of those bodies was measured faster at the kernel
level yet lost end to end, and its dispatch was removed (the record lives in
`dpa4_cuda.md` §11.6): the operator boundary forces its saved tensors and
gradient surfaces to materialize, while the scheduler shares the dense
expression's intermediates with the surrounding graph. Kernel speed does
not transfer across an operator boundary unless it also beats that
boundary's materialization cost, so the dense path stays in place on the
wide shapes and the operator form remains available for callers outside the
compiled graph.

### 3.10 Fused grid pair product, training form — `grid_pair.py` (level 1)

Every grid operator of the model evaluates
`out = from_grid(to_grid(left) * to_grid(right))` on coefficient operands
(`GridProduct` directly, `GridBranch` at a single branch through a softmax
over one element, identically one with a structurally zero router
gradient). Unfused, the training graph materializes the grid field --
several times larger than its coefficient operand -- for the forward, the
backward and the force-loss second order, and surrounds each einsum with
full-size layout copies; on Pro this section measured ~14.5 ms and six
kinds of thin-tall GEMMs.

The composition is a GEMM-pointwise-GEMM sandwich, structurally the flash
attention pattern with the grid axis in the sequence role:
`sezm_triton::grid_pair_train` owns one `(pair, channel-block)` output tile
per program, walks the grid in blocks, and per block evaluates the two
projection `tl.dot` products, the pointwise product, and the
back-projection outer `tl.dot` into a resident fp32 accumulator -- the grid
field never reaches device memory on any differentiation order. The first
order is one kernel (five dots per block); the second order is one further
kernel: the backward is trilinear in `(grad_out, left, right)`, so each
curvature term is a traversal with one operand replaced by its cotangent
and all three outputs share the walk (eight dots). The projectors are
fixed quadrature matrices, so no parameter gradient exists. Under autocast
the operators run the bf16-with-fp32-accumulation regime through their own
autocast rule, exactly as the dense einsum composition lowers; fp32 runs
IEEE dots (TF32 off).

Three structural constraints shaped the kernels. The slot axis is covered
by the largest power of two below the count plus a low remainder segment
(`tl.arange` and `tl.dot` demand power-of-two tiles), so 147 slots (degree
six) pad to 128 + 32 instead of 256 -- the split took the Max per-step
kernel time from 24.6 to 15.5 ms. The channel block is capped so the fp32
accumulator tiles stay register-resident (a spilled accumulator dominated
the three-accumulator second-order kernel: capping took it from 0.90 to
0.31 ms on Pro). And the launch configuration is chosen by descending
trial-compilation with a per-shape cache, because the exact shared-memory
footprint depends on Triton's internal dot staging; shapes no candidate
fits (the wide-slot fp32 validation regime) fall back to an eager einsum
composition inside the operator.

The launch ABI separates fixed model geometry from workload topology. The
coefficient, focus, frame, channel, and grid dimensions (`D/F/K/C/G`) carry
explicit guards because they determine tile geometry and launch-table lookup;
the node dimension `N` and every physical tensor stride remain runtime scalar
arguments. Multi-focus producers expose a degree-major view whose coefficient
stride is proportional to `N`. Treating that stride as a compile-time constant
would specialize a complete forward/backward graph for every node count, while
canonicalizing the view would add full-size copies. Runtime strides preserve
the native zero-copy layout and let dynamic batches reuse one compiled graph
through the force-loss second order.

The binding (`BaseGridNet.__init__`, threshold 75 slots) follows the
measured crossover: Pro −5.0 ms / −2.0 GiB, Max −14.0 ms / −5.7 GiB,
Plus −4.7 ms / −0.9 GiB end to end, while below 75 slots the dense section
is small and the operator's dispatch chain costs more than its kernels
save on the host-bound configurations (Neo lost 4 ms when bound), so the
narrow grids stay with the compiler. A register-resident CUDA C++ form
(the inference operator's walk extended with the training differentiations)
was built and verified first: its FFMA-rate contractions reached only
~33 TFLOPS against the tensor cores' ~300 on these GEMM-shaped walks and
lost end to end on every production shape; the tensor-core sandwich form
replaced it outright. The tool follows the compute structure -- scalar
traversals and resident activation chains go to CUDA, GEMM sandwiches to
`tl.dot`.

## 4. Launch-config tables and freeze auto-tuning

Kernels whose optimal launch parameters depend on shape are configured from tables in `tile_configs.py`, resolved through two layers:

1. **Built-in tables** (`tile_config_data.py`) keyed at the top level by either an exact `torch.cuda.get_device_name()` string or a stable model-name prefix, then by kernel family and shape key. Resolution prefers an exact name and then the longest prefix ending at a space boundary, so `NVIDIA RTX PRO 6000 Blackwell` covers Server/Workstation Edition suffixes without allowing `NVIDIA H20` to match `NVIDIA H200`. The built-in models are H20 and RTX PRO 6000 Blackwell. A GPU without a matching table resolves every key to the family fallback — correct everywhere, merely untuned — so schedules are never blindly applied across architectures.
1. **Runtime registrations** (`register_tile_configs`) taking precedence in the current process. The freeze auto-tuner registers freshly swept winners here.

Two shape-key conventions: `(focus_dim, lmax)` for kernels whose register pressure is per focus stream (the pointwise, stack-GEMM, fused-gate, and fp16x3 families; valid for any focus count), and `(C_wide, lmax)` with `C_wide = n_focus · focus_dim` for kernels vectorizing over the full hidden width (rotate+mix and flash backward; swept at `F = 2`).

**Entry semantics** are three-valued, and `has_tile_config` distinguishes the last two:

- a configuration tuple — the swept winner;
- an explicit `None` — the sweep measured the family default as optimal (win-list families keep the per-edge kernel; `rotate_mix_fwd` keeps `(2, 2)`; `stack_fp16x3` keeps the fp32 stack);
- an absent key — never swept on that GPU; the only case the freeze auto-tuner treats as work.

**Fallbacks** by family: `gate`/`recompute`/`point` → spill-safe `(16, 8, 2)`; `rotate_mix_fwd` → upstream `(2, 2)`; `flash_bwd_edge` → the upstream autotuner; the edge-block and `point_recompute`/`stack_m0_gate` win lists → the separate schedule; `stack_fp32` → the shared conservative IEEE tuple; `stack_fp16x3` → the fp32 stack (deliberately no fallback configuration — an unvalidated one may be miscompiled into silent NaN, see §5).

**Freeze auto-tuning.** At `DP_TRITON_INFER >= 2` with a CUDA target, `freeze_pt2._tune_triton_configs` collects the `(Cf, lmax, F, H)` shapes of every fused-value-path convolution (`collect_model_shape_keys`), sweeps only the genuinely uncovered keys on the local GPU (`tune_missing_configs`), registers the winners, and rebinds the value-path entries so the construction-time fp32/fp16x3 selection sees the new coverage. Because Triton launch parameters are host-side constants resolved during Inductor lowering, the subsequent `aoti_compile_and_package` bakes the tuned launches into the `.pt2` — the correct persistence scope, since AOTInductor artifacts are not portable across GPU models. Freezing to a secondary GPU pins the current device to the target so lookup and sweep see the right hardware. Keys concluding "keep the default" register as `None` and are not re-swept.

**Cost and memory.** A full cold tune of the Neo shape takes ~8.3 minutes, dominated by the fp16x3 validation. Wide channels at high `lmax` take tens of minutes (each edge-block candidate is a fresh Triton compilation whose unrolled body grows with `C_wide · lmax`). `_saturating_edges` scales the synthetic edge count inversely with channel width and caps it to ~⅓ of device memory (cost model ~160 KB/edge at width 64, deliberately ~3× conservative so the tuner fits beside a resident model). The memory budget always wins over tuning quality: winners are stable down to ~2e5 edges and drift only into neighboring near-optimal candidates below that (a few percent of kernel time, never correctness — fp16x3 validity comes from the fp64 check, which is edge-count independent). The hard floor is 2e4 edges; sub-1e5 sweeps log a notice. Small-memory drills (`set_per_process_memory_fraction` as a hard limit, mocked uncovered GPU name, full level-3 tune of the Neo shape): a simulated 16 GB device sweeps at 3.6e4 edges with a 2.06 GB peak (~3.5 min); a simulated 8 GB device sweeps at the 2e4 floor with a 1.24 GB peak; both produce fully fp64-validated entries.

**Regeneration.** `python -m deepmd.pt_expt.kernels.triton.sezm.sweep_tile_configs --model ckpt.pt [--level 3]` tunes a checkpoint's uncovered keys and prints a merge-ready `tile_config_data.py` fragment for adding a new GPU. `--cf CF --lmax LMAX --kernels pointwise,point_recompute,rotate_fwd,rotate_bwd,flash_bwd,fp32,m0_gate,fp16x3` sweeps one explicit key. The fusion and edge-block families are win lists (a ≥ 3 % win records a configuration, otherwise `None`); fp16x3 must win the complete fp32 force path by at least 1%. The fp16x3 sweep validates every candidate against an fp64 reference for accuracy *and* for finiteness across an intermediate spread of edge counts (`_FP16X3_FINITE_EDGES`) plus the main count, because the pipeliner NaN is edge-count-dependent (see §5) — a config finite at one count can NaN at another.

## 5. Implementation invariants

The following constraints are part of the runtime and validation contract.

- **Functional `triton_op` registration.** LAMMPS loads `.pt2` through the C++ `AOTIModelPackageLoader` with no Python registration. `torch.library.triton_op` with `wrap_triton` packages the Triton cubin and keeps the operator visible to Inductor; an opaque `custom_op` would leave an unresolved dispatcher dependency.
- **Triton-owned fp16x3 split.** The split tail is defined by an `fp32 → fp16 → fp32` rounding round-trip. Expressing the split in aten allows Inductor to retain the intermediate in fp32 and elide the rounding. `_split_fp16_kernel` makes the conversion an explicit compiled boundary, enforced by `test_inductor_compiled_matches_eager`.
- **No edge-scaled scalar kernel arguments.** Triton may specialize a scalar kernel argument to int32, so a host-side `n_edge · ROW` stride can overflow beyond `2³¹ / ROW` edges. Mixing-stack kernels derive output and gradient strides from constexpr layout flags on int64 offsets. `test_dynamic_compile_survives_int32_stride_overflow_edge_counts` enforces this for both stack operators.
- **Multi-count fp16x3 validation.** Some `(num_warps, num_stages >= 2)` combinations of the three-`tl.dot` loop produce edge-count-dependent NaNs. Every table entry passes fp64 accuracy validation and finiteness checks across `_FP16X3_FINITE_EDGES` plus the sweep count. `num_stages == 1` remains in the candidate set as the structurally pipeliner-free schedule. Any kernel-body change invalidates the fp16x3 table.
- **Integer layer selection.** Per-layer weights use one `(n_layers, F, M, M)` tensor plus an integer kernel argument. Python-side `select` views are incompatible with Inductor's repeated functional-wrapper trace.
- **Exact shape keys.** Pointwise kernels run 20–50× slower when a narrow-width tile spills at a wider `Cf`. Exact `(Cf, lmax)` and `(C_wide, lmax)` keys therefore define routing; no cross-width heuristic is permitted.
- **Production-sized flash tuning.** AOT trace tensors are too small to select the saturated-edge flash backward launch. `flash_bwd_edge` stores the production-sized winner; uncovered devices retain the autotuner until freeze-time tuning supplies the key.
- **Package-level performance validation.** AOTInductor-generated pointwise and reduction kernels may choose different `XBLOCK` values across packages, causing about 0.9% throughput variation even when hand-written Triton timings agree within 0.4%. Performance claims therefore apply to the complete package rather than an isolated source schedule.
- **Expandable allocator segments at capacity.** At 48,640 atoms the caching allocator can hold about 10.9 GiB reserved but unallocated while rejecting a 3.62 GiB request. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` keeps both reference and tuned packages within the 48,640-atom capacity point; 64,000 atoms exceeds device capacity.
- **`reorder_for_peak_memory=False` for inference.** `make_fx` edge-count symbols carry no size hint, so the memory-ordering pass costs dynamic buffers as zero and hoists allocations (Neo: 33 GB instead of 22 GB). Training uses Dynamo hints and retains the upstream default. `build_inductor_compile_options(inference=...)` scopes the policy.
- **One-dimensional launch grids.** `triton.max_tiles=1` keeps data-dependent edge and node axes on the x grid (limit 2³¹−1) rather than y/z (limit 65535). Both training and inference require this bound.
- **CSR segment reduction.** At production density, indirect CSR reduction including topology sorting costs 0.94 ms versus 7.3 ms for `index_add_` on `(E, 16, 64) → (4096, 16, 64)`. Aggregation therefore uses the contention-free segmented form.
- **Explicit monomial multiply chains.** Dense `cumprod` lowers to expensive scan pairs. Compile-time-unrolled multiply chains fuse into the surrounding kernels.
- **Fixed GEMM autotuning policy.** `max_autotune_gemm=False` avoids an additional ~40 s compile cost; the selected block-GEMM tiles already operate at the FFMA ceiling.
- **IEEE dot products.** Every `tl.dot` uses `input_precision="ieee"`. fp16x3 at level 3 is the only deliberate departure from exact fp32.

## 6. Training and the second order

### 6.1 The `DP_TRITON_TRAIN` gate

`DP_TRITON_TRAIN` is a numeric level read once at model construction (`deepmd.pt_expt.kernels.utils.triton_train_level`); only the integers `0`–`1` are accepted. It takes effect during training only and is independent of `DP_TRITON_INFER`, so a deployed inference path stays frozen while training opts in separately. `active_triton_level` in `sezm_nn/so2.py` resolves which of the two applies; both are captured at construction, so the branch it drives is a trace-time constant.

| level | adds                                                                                                                                                            |
| ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `0`   | nothing (dense reference path)                                                                                                                                  |
| `1`   | the second-order-complete universal kernels: block-diagonal rotation, radial degree mixing, the `SO2Linear` block GEMM, the gated activation, flash aggregation |

The Triton training path is an *operator composition*: the compiler owns the graph and the fused operators replace individual segments. Within the value stream, its per-stage rotation and mixing operators are mutually exclusive with the fused CUDA value path (`DP_CUDA_TRAIN=1`, documented in `dpa4_cuda.md`), and the dispatch in `so2_message` gives the CUDA operator priority. The rest of the composition remains active, however: the production setting enables both gates so CUDA owns the value stream while Triton owns the attention and force-assembly segments. The fused Triton SO(2) value path (`so2_value_path.py`) remains an inference operator (`DP_TRITON_INFER >= 2`); training never composes that operator — its training form was measured slower than the level-1 composition and its dispatch was removed.

### 6.2 How the two backends reach the operators

Both gates serve the `pt` and `pt_expt` backends from one set of operators under `deepmd/pt_expt/kernels`, but the two backends reach them differently, and the difference is structural rather than stylistic. The `pt` backend owns its modules outright, so its dispatch branches sit inline in `sezm_nn/so2.py`. The `pt_expt` modules subclass the array-API `dpmodel` implementation and inject PyTorch behaviour by overriding it, so every point where an accelerated path can enter must exist in `dpmodel` as a seam: `_rotate_mix` (the rotation, radial mixing and focus-major cast of `so2_message` steps 1–3), `_rotate_to_local` / `_rotate_back`, `_attention_softmax`, `_mix_rank_compact`, `_block_diagonal_matmul`, and the `_cuda_value_train` / `_grid_pair_train_fn` / `_flash_atten_trains` hooks the dispatch reads. The array-API reference leaves every hook unbound and every seam at its dense body, so `dpmodel` keeps working unchanged; `deepmd/pt_expt/descriptor/dpa4_nn/` overrides the seams and binds the hooks with the same gate arithmetic the `pt` backend uses (`_active_triton_level`, mirroring `active_triton_level`). Adding a training path therefore means adding a seam to `dpmodel` and an override to `pt_expt`, not duplicating the operator.

The composed CUDA-and-Triton path also changes descriptor-level Wigner ownership. When every block binds the CUDA value operator and training-capable Triton flash aggregation, both backends set `_packed_wigner_train`: `_build_full_wigner()` then suppresses the dense `(E, D, D)` pair, and `_shared_wigner_runs()` takes the zonal coupling from the CUDA operator's packed run cache. The common decision lives in `dpmodel`, while `pt_expt` binds the PyTorch operators and mirrors the `pt` predicate exactly. This avoids a backend-specific dense transpose in the compiled graph and makes the degree-six Ultra layouts use the same packed path as `pt`.

Both `pt_expt` lower entry points pass their symbolic `make_fx` graph through `relax_views_to_reshapes()` before Inductor compilation. Fake-tensor strides can otherwise specialize a semantically copy-capable reshape of a transposed Wigner tensor to `aten.view`; a different runtime edge count then reaches a non-view-compatible stride and aborts compilation. The normalization leaves compatible views free and materializes a copy only for the dense fallback that actually needs one, so both the all-off baseline and the packed fast path remain valid.

What does *not* need mirroring is everything outside the module tree:
`build_inductor_compile_options` and `deepmd/pt_expt/kernels/utils.py` are one
module each, imported by both backends, so the gates of §6.6 and their option
set are shared by construction. Only the trainer needed a twin, for the
distributed precompile step of §6.6.

One asymmetry to watch when comparing the two backends: the compile switch is
spelled `model.use_compile` on `pt` and `training.enable_compile` on
`pt_expt`. An uncompiled `pt_expt` step of the nano configuration takes 111 ms
against 21.8 ms compiled, so a measurement that misses the difference is
measuring the wrong thing entirely (`bench.py --backend` carries the `pt`
spelling over).

### 6.3 What training demands beyond inference

The inference force differentiates the energy with respect to the coordinates while the parameters are constants, so an operator serving inference needs only the activation gradients. Training a force loss needs two things more:

1. **Parameter gradients.** `so2_block_diag_gemm` produced no weight gradient, and `radial_mix` / `so2_rotate_mix` produced no `channel_basis` gradient, because inference never asks for them.
1. **A differentiable backward.** The force is `-d(energy)/d(coord)` and the force loss is differentiated again, so autograd traverses the backward pass itself. A backward registered as an opaque `triton_op` has no autograd formula and stops that traversal.

### 6.4 Multilinearity supplies the second order

Every fused operator here except the mixing stack is *multilinear*: its forward `y = F(a_1, ..., a_n)` is linear in each argument separately. This is what makes the second order expressible with the operators that already exist, with no new kernel. Writing the first-order backward as `B(y_bar, a_1, ..., a_n) = (g_1, ..., g_n)` with `g_j` independent of `a_j`, the adjoint identity `<h_i, B_i(y_bar, .)> = <y_bar, F(..., h_i, ...)>` gives the whole formula:

```
grad(y_bar) = sum_i F(a_1, ..., h_i, ..., a_n)
grad(a_j)   = sum_{i != j} B_j(y_bar, {a_k}_{k != j, k != i}, h_i)
```

For `n = 2` the second line collapses to a single backward call, `B(y_bar, h_1, h_2)`: the `a_1` component of `B` depends only on `a_2` and vice versa, so substituting both cotangents at once cannot mix them. For `n >= 3` that shortcut would introduce cross terms, so each call substitutes one cotangent and keeps the components that see it. The derivation and the shared helpers live in `second_order.py`.

Per operator: rotations are bilinear in `(feature, Wigner-D)`; the block GEMM in `(activation, weight)`; `radial_mix` trilinear in `(kernel, activation, channel_basis)`; `flash_atten` trilinear in `(x_local, Dt, alpha)`; `so2_rotate_mix` quadrilinear in `(x, Wigner-D, kernel, channel_basis)`. `wigner_monomials` is a polynomial rather than a multilinear form, so its second order is a Hessian contraction taken from the eager closed form — exact, and negligible beside the rotations it feeds since its operand is only `(E, 4)`.

Two shape subtleties recur. Where a backward emits a *per-edge* gradient of a per-node input, the second-order term inherits the pending gather adjoint, and where such a cotangent re-enters the forward the gather is made the identity so each edge stands in for its own node. `flash_atten`'s backward carries the CSR view it does not itself read, because `setup_context` only sees the arguments of the operator it belongs to and the second order re-enters the segmented forward.

### 6.5 Parameter gradients and AMP

Parameter gradients that reduce the whole edge axis into a small tensor (`block_diag_weight_grad`, `channel_basis_grad`, `rotate_mix_basis_grad`) are expressed in ATen rather than Triton: cuBLAS already handles a tall-skinny reduction well, and an ATen expression is differentiable, which the second-order path needs. Their layout matters — contracting the output degree first leaves a rank-major intermediate the reduction consumes elementwise, whereas contracting the input degree forces a transposing copy of the whole edge tensor (worth 10.4 ms/step on the Pro shape).

Under AMP the activations arrive in bfloat16 while parameters and the Wigner-D buffer are float32, a mix the kernels cannot consume. Each forward operator therefore registers an autocast rule that aligns every floating-point input to the training dtype exactly as the built-in matmuls do; because the cast is recorded by autograd the parameters still accumulate float32 gradients. The rules are inert outside an autocast region.

### 6.6 Inductor options and distributed compilation

`DP_TUNE_TRAIN` grades the compile-time investment of the training graphs
(cumulative levels, default `0` = fast compilation; inference graphs ignore
it, the AOTI export forces its own C++ wrapper):

- **`DP_TUNE_TRAIN=1` — `cpp_wrapper`** replaces
  the generated Python wrapper that launches the compiled graph's kernels
  with a compiled C++ wrapper. The step launches thousands of kernels, and
  the Python dispatch overhead leaves the GPU idle most of the step on the
  small configurations (77% idle on Nano; measured 8-15% of the step on
  Nano/Neo/Air, ~1% on Plus/Pro). Its compile cost is minutes
  (`cpp_wrapper_build_separate` compiles the launch sequence at O1 in its
  own translation unit, 18 -> 11 minutes on the two-layer Pro graph).
- **`DP_TUNE_TRAIN=2` — additionally `max_autotune_gemm`** benchmarks Triton
  matmul templates against the cuBLAS call for every GEMM in the compiled
  graph. Its gain comes from batched weight-gradient contractions with a huge
  reduction axis and tiny output tiles. Current cuBLAS wheels serve most of
  those through native split-K kernels, and the CUDA value-path operator has
  its own cublasLt split-K path, so the residual is workload dependent: at an
  approximately 3,200-atom budget, the combined level-2 gain over untuned CUDA
  plus Triton is 0.3--3.1% except for the launch-bound `pt` Nano graph at 12.3%
  (§6.7). The benchmarking still dominates the first compilation and can take
  tens of minutes. Each GEMM dumps its candidate table to stderr
  (`AUTOTUNE mm(...)` plus `Autotune Choices Stats`); those two streams
  are independent and both default on. `apply_global_compile_patches`
  forces `autotune_num_choices_displayed = 0` and
  `max_autotune_report_choices_stats = False` so a `DP_TUNE_TRAIN=2` run
  does not flood `dp.log`. Export the matching `TORCHINDUCTOR_*`
  variables before launch to restore the dumps.

A defective autotune candidate must not abort compilation:
`patch_inductor_autotune_benchmark_tolerance` (applied with the other global
patches) scores a candidate whose benchmark harness raises `TypeError` as
infinitely slow instead of failing the step. On PyTorch 2.13 with
`cpp_wrapper` enabled, the in-process benchmark of some Triton matmul
templates assembles one more positional argument than the generated launcher
accepts; the candidate is unusable either way, and the cuBLAS fallback wins
that shape.

Fusion-widening options (`epilogue_fusion`, `aggressive_fusion`,
`combo_kernels`) stay disabled: the dynamic second-order graph is exactly where
widened fusion has historically produced defective kernels. `max_fusion_size`
is controlled by the shared positive-integer `DP_FUSION_SIZE` environment
variable for both training and inference and defaults to `8`; a larger value is
an opt-in workload-specific setting because measured gains are small and peak
memory can increase.

On a distributed job the autotune benchmarking makes compile time long and
non-deterministic across ranks (each rank benchmarks on its own device, and
the results do not share a cache key), and the desynchronization has aborted
multi-node training through NCCL collective timeouts: the first optimization
step both compiles the model and joins the first gradient all-reduce, so a
rank still compiling past the watchdog kills every peer already waiting in
that collective. Serializing compilation behind a per-node leader was tried
and made it worse -- the peers do not hit the leader's autotune cache, so the
node's total compile time doubles. The trainer therefore compiles *outside*
the collective window: before the optimization loop, every rank runs one
forward and backward per task under `DDP.no_sync`
(`_precompile_outside_collectives`, present in both backends' `training.py`),
which triggers exactly the training-graph compilations while issuing no
collective at all -- ranks compile in parallel, each taking however long it
takes -- and a rendezvous store counter (which has no watchdog) aligns all
ranks before the first real step.

### 6.7 Measured training performance

The final sweep measures one bf16 AMP training step with `torch.compile` on one
RTX PRO 6000 Blackwell. Each result uses an isolated Inductor and Triton cache,
30 warmup steps, and the median of 60 timed steps. Compilation is outside the
timed interval, and `torch.cuda.max_memory_allocated()` is reset after warmup so
the reported peak contains the timed training steps but not compilation. The
default `DP_FUSION_SIZE=8` is used throughout.

The variants are `base` (`DP_TRITON_TRAIN=0 DP_CUDA_TRAIN=0`), `tri`
(`DP_TRITON_TRAIN=1 DP_CUDA_TRAIN=0`), and `cuda-tri`
(`DP_TRITON_TRAIN=1 DP_CUDA_TRAIN=1`), all with `DP_TUNE_TRAIN=0`.
`tune2-cuda-tri` adds `DP_TUNE_TRAIN=2`. Untuned jobs may occupy independent
GPUs concurrently; tuned jobs run serially because concurrent host activity can
change the kernels selected by `max_autotune_gemm`.

The measured configurations are:

| size  | `lmax` | blocks | `channels` | `n_focus` | radial rank |
| ----- | ------ | ------ | ---------- | --------- | ----------- |
| Nano  | 1      | 2      | 32         | 1         | none        |
| Mini  | 2      | 2      | 32         | 1         | 1           |
| Neo   | 3      | 2      | 32         | 2         | 1           |
| Air   | 3      | 3      | 64         | 1         | 1           |
| Plus  | 4      | 4      | 64         | 1         | 2           |
| Pro   | 5      | 6      | 64         | 2         | 2           |
| Max   | 6      | 8      | 96         | 2         | 2           |
| Ultra | 6      | 10     | 96         | 4         | 4           |

The two backends intentionally exercise different batching layouts. `pt` pads
frames to a common atom count; `pt_expt` concatenates ragged frames behind an
`n_node` vector. Workload is therefore reported as atoms/frames, and only rows
within one backend and one workload are compared. Raw time across the two
backends is not a backend benchmark.

For the default `mix:400` atom budget, the serial small-model decision sweep is:

| size | backend   | workload | base (ms/GiB)  | tri (ms/GiB)   | cuda-tri (ms/GiB) | tune2-cuda-tri (ms/GiB) |
| ---- | --------- | -------- | -------------- | -------------- | ----------------- | ----------------------- |
| Nano | `pt`      | 396a/22f | 20.655 / 0.433 | 20.404 / 0.463 | --                | 16.055 / 0.384          |
| Nano | `pt_expt` | 376a/9f  | 23.927 / 0.491 | 21.756 / 0.507 | --                | 18.627 / 0.401          |
| Mini | `pt`      | 396a/22f | 23.352 / 0.837 | 21.262 / 0.838 | --                | 17.358 / 0.599          |
| Mini | `pt_expt` | 376a/9f  | 22.810 / 0.992 | 22.891 / 0.899 | --                | 19.304 / 0.672          |
| Neo  | `pt`      | 396a/22f | 28.923 / 1.920 | 25.348 / 1.968 | --                | 22.664 / 1.205          |
| Neo  | `pt_expt` | 376a/9f  | 24.166 / 2.230 | 24.343 / 2.260 | --                | 21.840 / 1.283          |
| Air  | `pt`      | 396a/22f | 29.355 / 3.207 | 31.615 / 3.566 | 34.767 / 2.147    | 25.396 / 2.314          |
| Air  | `pt_expt` | 376a/9f  | 32.735 / 3.659 | 30.516 / 3.787 | 30.366 / 2.258    | 26.148 / 2.349          |

The tuned composition is the fastest default-budget setting for every small
model: 1.16--1.35x over `base` on `pt` and 1.11--1.28x on `pt_expt`, with
peak memory at 0.58--0.89x of `base`. Triton alone can remove enough device work
to help an untuned small step, but it retains the dense value-stream surfaces;
its memory is therefore close to, and sometimes above, `base`.

The untuned large-model sweep at the same configured atom budget is:

| size  | backend   | workload | base (ms/GiB)    | cuda-tri (ms/GiB) | speedup | memory factor |
| ----- | --------- | -------- | ---------------- | ----------------- | ------- | ------------- |
| Plus  | `pt`      | 396a/22f | 47.434 / 6.275   | 36.846 / 3.134    | 1.29x   | 0.50x         |
| Plus  | `pt_expt` | 376a/9f  | 51.392 / 7.021   | 38.135 / 3.191    | 1.35x   | 0.45x         |
| Pro   | `pt`      | 396a/22f | 156.471 / 20.594 | 92.954 / 8.453    | 1.68x   | 0.41x         |
| Pro   | `pt_expt` | 376a/9f  | 160.254 / 21.908 | 95.015 / 8.482    | 1.69x   | 0.39x         |
| Max   | `pt`      | 396a/22f | 444.225 / 51.660 | 274.764 / 20.392  | 1.62x   | 0.39x         |
| Max   | `pt_expt` | 376a/9f  | 445.428 / 54.493 | 274.222 / 20.346  | 1.62x   | 0.37x         |
| Ultra | `pt`      | 192a/16f | 477.485 / 55.500 | 301.198 / 19.603  | 1.59x   | 0.35x         |
| Ultra | `pt_expt` | 198a/9f  | 410.531 / 54.698 | 255.935 / 16.990  | 1.60x   | 0.31x         |

No compile tuning is needed on these device-bound shapes. The composed path is
1.29--1.69x faster while using 0.31--0.50x of the baseline peak memory.

An eightfold atom-budget sweep (`mix:3200`) tests whether the small-model
decision survives a realistic change in batch size:

| size | backend   | workload   | base (ms/GiB)    | tri (ms/GiB)     | cuda-tri (ms/GiB) | tune2-cuda-tri (ms/GiB) |
| ---- | --------- | ---------- | ---------------- | ---------------- | ----------------- | ----------------------- |
| Nano | `pt`      | 3192a/57f  | 26.746 / 3.481   | 22.795 / 3.774   | 22.875 / 2.313    | 20.363 / 2.624          |
| Nano | `pt_expt` | 3171a/187f | 28.521 / 4.015   | 26.214 / 4.168   | 23.268 / 2.693    | 23.203 / 2.939          |
| Mini | `pt`      | 3192a/57f  | 51.825 / 7.320   | 46.555 / 7.034   | 36.273 / 4.181    | 35.199 / 4.661          |
| Mini | `pt_expt` | 3171a/187f | 57.094 / 8.710   | 49.428 / 7.591   | 38.131 / 4.806    | 37.603 / 5.240          |
| Neo  | `pt`      | 3192a/57f  | 128.673 / 17.437 | 109.638 / 17.515 | 89.937 / 9.780    | 89.363 / 10.230         |
| Neo  | `pt_expt` | 3171a/187f | 139.119 / 20.162 | 117.015 / 20.423 | 91.509 / 10.509   | 90.490 / 10.606         |
| Air  | `pt`      | 3192a/57f  | 224.563 / 28.529 | 207.910 / 31.307 | 148.981 / 18.054  | 147.148 / 18.900        |
| Air  | `pt_expt` | 3171a/187f | 235.391 / 32.671 | 213.300 / 33.806 | 148.013 / 19.172  | 147.371 / 19.510        |

At the larger batch, untuned CUDA plus Triton is already 1.17--1.59x faster
than `base` and uses 0.52--0.67x of its memory. Triton alone improves time by
1.08--1.19x but leaves memory at 0.87--1.10x of `base`. It is only nominally
faster than untuned CUDA on `pt` Nano (22.795 versus 22.875 ms), while consuming
63% more memory than that CUDA path.

The incremental `DP_TUNE_TRAIN=2` gain over untuned CUDA shrinks to
12.3/3.1/0.6/1.2% on `pt` Nano/Mini/Neo/Air and
0.3/1.4/1.1/0.4% on their `pt_expt` counterparts. Every tuned row also raises
peak memory relative to untuned CUDA. Tuning is therefore justified for the
small default-budget workloads, and remains material for the `pt` Nano large
batch, but untuned CUDA plus Triton is the practical speed-memory setting for
the other large-batch cases.

Level 1 remains the recommended Triton training level. By itself it preserves
compiler visibility around the fused segments; composed with
`DP_CUDA_TRAIN=1`, it owns the attention and force assembly while the CUDA
operator owns the value stream. A former level 2 dispatched the composed SO(2)
value path during training; it was numerically correct but expanded its
second-order backward into thousands of launches, so training now reaches that
value path only through the single CUDA operator chain documented in
`dpa4_cuda.md`.

Where the time actually goes (Pro, one step, level 1): the device is busy
97% of the step. Generated pointwise kernels carry ~44% of the device time,
of which roughly four fifths is layout-dominated data movement materializing
the backward graph's intermediate surfaces; GEMMs (cuBLAS plus tuned Triton
templates) carry ~30%; the hand-written kernels ~14%. The theoretical floor
for the mixing GEMMs alone is ~27 ms at peak bf16 throughput, so the GEMM
side is within a factor of ~1.5 of its bound while the data-movement side is
amplified several-fold by materialization -- the remaining reserve sits in
the backward graph's surface count, not in any single kernel.

### 6.8 Why the benefit tracks model size

The fused kernels reduce device work on every supported shape; whether that
reduction reaches the end-to-end step depends on the host cost of reaching
them. A default-budget Nano/Mini/Neo step lasts only 20--29 ms, so wrapper and
operator dispatch are a material fraction of the wall time. At an approximately
3,200-atom budget, the same models become device dominated enough for untuned
CUDA plus Triton to beat the baseline by 1.17--1.59x.

The cause is structural. A Triton operator called from the model body is
lowered into the compiled graph and launched directly, but an operator called
from a custom operator's backward is reached through a Python autograd formula
that Inductor does not inline. Every such call pays `torch.library` dispatch;
the force-loss second order multiplies that cost. `cpp_wrapper` removes the
generated Python graph-launch loop but cannot erase an opaque custom-operator
boundary, which is why tuning helps the shortest graphs most and the wider
models least.

Increasing either descriptor width or batch size raises the useful device work
behind each dispatch. Pro through Ultra therefore need no compile tuning, while
Nano can remain launch bound even at the larger atom budget. Model size alone
is not the decision variable; the measured atom/frame workload must accompany
it.

### 6.9 The mixing-stack backward, and its second order

The training backward of the mixing stack was built in five successive forms, each correct:

1. *Composable traversal in the autograd formula* (per-layer fused pointwise operator plus ATen glue). A force loss differentiates the formula again, and the engine expands the glue into roughly 13,000 eager elementwise launches per step.
1. *Monolithic backward as an inlined `triton_op` with a hand-derived second order.* Fast on one shape, but the compiler inlines the traversal into the traced graph, and under dynamic edge counts with several differently shaped convolutions in one graph it mis-sizes the reused inter-kernel buffers -- an illegal memory access on the Pro shape that no per-operator test can reproduce. Inlining also bakes launch configurations into compiled artifacts, so table updates silently stop applying until the Inductor cache is purged.
1. *Monolithic backward as an atomic `custom_op`.* Correct on every shape, but the per-layer kernel launches and batched matmuls inside the operator now dispatch eagerly at tens of microseconds each, costing several milliseconds per step against the inlined form.
1. *Whole-stack traversal as a single kernel* (one launch walks every layer, the running head and the recovered inputs flowing through L2 between the phases; weight gradients contract afterwards through layer-batched cuBLAS calls). This removes the host loop entirely and matches the multi-kernel numerics to 1e-7, but its inter-phase surfaces are re-read with several-fold amplification, so it only wins while one focus-major surface fits comfortably in the L2 cache.
1. *Whole traversal in hand-written CUDA.* Documented in `dpa4_cuda.md` §11; it is the interior of the fused CUDA value path (`DP_CUDA_TRAIN=1`), not a Python-visible operator. It replaces the Triton stages inside the value stream, while the Triton attention and force-assembly stages remain composable around it. The standalone Triton value composition remains the numerically verified reference the CUDA path is checked against.

#### The form that shipped

A fusion spanning several layers is profitable only if the backward *and* its own second order avoid autograd expansion; an intermediate form that fuses the forward and composes the backward from ATen is strictly worse than not fusing at all, because a second differentiation expands every ATen operation of the traversal into further eager kernels (roughly 13,000 launches per step on Pro).

The training path therefore runs the whole stack backward as one operator, `so2_mixing_stack_train_bwd`. It performs the same traversal as the inference backward, additionally contracting the per-layer weight gradients against layer inputs recovered inside the pointwise kernel, and accepting the gradients a differentiation of the whole graph sends back through the pre-activation and final-activation outputs (a tracer materializes those as zeros even when unused, so the dispatch must route on their presence, not treat them as an exceptional case).

Its autograd formula is the hand-derived adjoint of the traversal. The first order is linear in the incoming gradient and walks layers last to first, so its adjoint walks first to last:

```
h_gz(l)  = h(l) W_l + u_l h_W(l)          cotangent of the pre-activation gradient
h_gq(l)  = s_l h_G(l)                     cotangent of the gate-logit gradient
h(l+1)   = h(l) + P_l^T(h_gz, h_gq)       adjoint of the residual recursion
h_u(l)   = h_u(l-1) + gz_l h_W(l)^T       cotangent of the recovered inputs
```

where `P^T` and the gradients with respect to `z_l` and `G_l` come from the fused second order of the gated activation (`gated_activation.py`), and the recovery `u_l = u_{l+1} - act(z_l)` routes `h_u` back onto the pre-activation through the activation's own vector-Jacobian product -- which is the first-order pointwise kernel again. The head of the adjoint unwinds the final identity layer and the competition scale exactly as the first order applied them. Being the highest order, nothing differentiates this body in turn, so it composes fused kernels and cuBLAS contractions directly; the linearization points are recovered by replaying the first-order traversal, which is cheaper end to end than holding four stacked edge-size buffers per convolution alive across the force graph.

**Unused output cotangents must stay `None`.** Both wrapping operators set
`ctx.set_materialize_grads(False)`: the pre-activation and final-activation
outputs usually have no consumer, and in the force-loss regime the
parameter-gradient outputs never do -- their cotangents feed the optimizer,
not the force. Materialized zero cotangents would both drag the adjoint
through zero contributions (a full extra traversal per convolution per step)
and hide the force regime from the second-order dispatch, which routes on
`None`-ness because a traced graph cannot inspect tensor contents. The
one place a genuinely absent cotangent reaches dense arithmetic -- the
second differentiation re-enters the first-order backward through the saved
pre-activations alone, leaving the result surface without a gradient -- is
restored to zeros explicitly at that entry.

**The bottom layer's input is exact.** Layer inputs above the bottom exist in the backward only as values recovered from the forward output, and the recovery `u_l = u_{l+1} - act(z_l)` inherits the forward's accumulated store rounding: under bf16 autocast the recovered stack input carries the sum of every layer's rounding, which grows with depth along the unnormalized residual stream and dominated the weight-gradient error of the bottom layer (3.5x the eager reference on the Max shape, all other layers at parity). The stack input `u0` is an operand the autograd context already holds, so the operators accept it directly: the bottom layer's weight gradients contract against the exact value, and in the second order its input cotangent leaves through the `u0` slot instead of entering the recovery chain. This restores the bottom layer to parity at zero memory and zero traffic cost; a full fp32 recovery chain was measured against it and rejected -- it repairs none of the forward-store rounding (which precedes the backward) and costs ~13% of the replay's traffic.

### 6.10 Lessons

#### A dtype gate can silently disable everything

Two defects made every early measurement of the composed value-path training
form meaningless, and both were invisible to an fp32-only gradient check.

First, the value-path operators' `_use_triton` accepted only fp32 while every operator registers a bf16 autocast rule, so under AMP -- the production regime -- each operator quietly served its eager reference instead of its kernels. Second, one of those references reached `aten.repeat_interleave.Tensor` inside a `triton_op` body; Inductor has no lowering for it, and because the exception fires while compiling the *graph*, the whole model fell back to eager -- forward, backward, everything, a 4-6x step regression that no per-operator test can see. The segment adjoint is now a `custom_op` (atomic to the compiler) and the dtype gate admits what the kernels actually handle, with every transcendental and mixed-dtype `tl.dot` widened to fp32 explicitly.

The lesson is procedural as much as technical: benchmark logs must be read for compiler exceptions before their timings are believed, and the gradient check runs every case twice -- ambient fp32 against a tight tolerance, then both paths under bf16 autocast against a loose one -- so a dtype-handling defect fails the check instead of surfacing as a silent fallback months later.

#### Training kernels carry their own tile tables

The launch tables swept for inference do not transfer to training. The backward pointwise kernel runs there in bf16 with the layer-input recovery and the gate-logit store enabled, a heavier register profile whose winning tile sits far from the inference entry -- up to six times on the Air shape (`Cf` 64, `lmax` 3: the inference entry runs at 1.07 ms where the swept training entry runs at 0.18 ms). The gated activation's second order obeys the same register-pressure law as the other pointwise kernels, with an order of magnitude between a tile on the wrong side of the spill point and the swept one.

Both therefore have their own families, `point_train` and `gated_second_order`, resolved through the same two-layer lookup as the inference tables and swept by the same script (`sweep_tile_configs.py --kernels point_train,gated_second_order`). Unresolved keys fall back to the inference entry and the spill-safe pointwise default respectively.

#### Layout is part of a kernel's contract

The pointwise operator addresses its rows by a flat stride, so a caller must hand it a materialized tensor, not a view. The stack backward transposes an edge-major cotangent into focus-major, and

```python
grad_final = scaled.permute(1, 0, 2)
```

produces a view whose strides survive the subsequent addition. For a single focus stream the permuted strides coincide with the contiguous ones and the kernel reads correctly by accident; from two streams on it reads the wrong rows and every gradient downstream is wrong. Shape coverage in the gradient check has to include `n_focus > 1` for this class of defect to be visible at all -- the single-stream case cannot expose it.

### 6.11 Verifying a second-order path

Committed tests cover two of the three levels:

- `source/tests/pt_expt/kernels/` arbitrates the training operators against the eager references that define them — `test_grid_pair_train.py` for the coefficient-grid pair product, `test_segment_softmax.py` for the destination-segmented attention normalization (plus its two exact structural invariants: a muted edge carries no weight, and every segment keeps a strictly positive null mass), `test_so2_value_train.py` for the fused CUDA value path. Each covers forward, first order and the force-regime second order, in ambient fp32 and under bf16 autocast. The judgment rule (`conditioning.py`) bounds the fused error by a multiple of the eager reference's own distance from the fp64 truth and never by an operator-specific absolute tolerance; see `dpa4_cuda.md` §11.5 for why that distinction is load-bearing.
- `source/tests/pt/model/test_descriptor_sezm_train_paths.py` and its `pt_expt` mirror `source/tests/pt_expt/descriptor/test_dpa4_train_paths.py` assert the backend contract: each gate binds exactly the stages it owns (including the profitability bounds — the rotate-mix hidden-width crossover, the grid-pair slot count, the fused activation's register footprint, the block GEMM's tile alignment), the Triton and CUDA layers stay independent, and a training step whose objective is second order in the descriptor output reproduces the dense path's coordinate gradient.

Two further levels live under `debug/train_bench/`, on real models and shapes too large for a test suite:

- `gradcheck.py` compares each operator against the eager reference that defines it, on the forward, on every first-order gradient, and on a second-order projection of the first-order gradients. Random Wigner-D operands are multiplied by the degree-block mask so the kernel (which reads only structural non-zeros) and the dense reference agree by construction. Degree, focus count, focus width and mixer rank are all arguments, and a path is not considered verified until it has been checked with more than one focus stream.
- `verify_train.py` runs a whole training step of a real model twice on one set of weights and one batch, switching only the training dispatch (`--path triton` for this composition, `--path cuda` for the fused CUDA chain), and compares the loss and every parameter gradient. Because the reductions use atomics, the dense path is additionally run twice to measure its own non-determinism; that figure is the floor below which a fused-versus-dense difference carries no information. Observed fused-versus-dense differences sit at 1.2–2.6x that floor on both the Air and Pro shapes, and a 200-step training run reproduces the dense loss curve digit for digit.

## 7. Algorithm selection

- **Complex `|m| = 1` block.** The implementation uses the four-product block GEMM. Gauss three-multiplication reduces MACs by 25%, but its three simultaneous accumulator tiles reduce occupancy by 4–5× in a fused kernel, while a three-GEMM cuBLAS form reaches only 0.68–0.95× because of extra reads and launches.
- **Attention softmax.** `segment_envelope_gated_softmax` remains in the Inductor graph. A separate CSR softmax reaches 0.083 ms only when it shares sorted topology with flash aggregation; building its own topology yields 0.268 ms and approximately 1% end-to-end benefit, insufficient to justify a second segment-coupled backward interface.
- **Roofline boundary.** RTX PRO 6000 custom kernels operate at 79–94% of the DRAM roof, while the H20 mixing stack and cuBLAS pool are compute-bound. Additional speedup requires less traffic or less mathematical work rather than another wrapper-level fusion.

## 8. Regeneration and tests

- **Tables:** `deepmd/pt_expt/kernels/triton/sezm/sweep_tile_configs.py` (see §4). Built-in data in the adjacent `tile_config_data.py`.
- **Tests** (`source/tests/pt/model/test_descriptor_sezm_triton.py`): rotation / radial-mix / value-path / flash / monomial / force-assembly forward+backward parity against the eager reference and `make_fx` symbolic composability; `TestSeZMStackFP16x3` for fp16x3 accuracy against fp64, dynamic-range robustness, the Inductor-compiled-vs-eager split guard, and the int32-overflow guard; `TestTileConfigLookup` (CPU-safe) for the two-layer lookup / `None`-vs-absent semantics; `TestTileConfigLayering` for built-in dispatch, shape collection, and the freeze auto-tuner's skip logic; `TestTritonInferLevel` for gate parsing.
- **Training-path tests** (see §6.11): `source/tests/pt_expt/kernels/` for the operators, and the mirrored `test_descriptor_sezm_train_paths.py` / `test_dpa4_train_paths.py` for the two backends' gate bindings and training-step parity.
- **Second-order harnesses** (`debug/train_bench/`, see §6.11): `gradcheck.py` for per-operator first- and second-order parity, `verify_train.py` for whole-model gradient equivalence against the reduction-order noise floor, `bench.py` for steady-state step timing and operator breakdown, `micro.py` for candidate-implementation comparisons.

## 9. Generality across the design family

The kernels are parametric over the whole `mmax = 1` design family rather
than tied to one checkpoint. `lmax` (1--6), `Cf` (per-focus width in
`{32, 64, 96, 128}`, with non-power-of-two 96 padded and masked), `F` (focus
count), the layer count, and mixer `RANK` (0 mixer-free, 1--4
`degree_channel`) are compile-time or launch parameters. The committed CUDA
value-path arbitration additionally covers the two degree-six 384-channel
Ultra layouts, `F=4, Cf=96` and `F=3, Cf=128`, through forward, first order,
and force-regime second order in fp32 and bf16 autocast.

GPU-specific entries select only launch schedules. A built-in table is used
only when the CUDA device name and the complete shape key both match. Missing
devices and shapes use the family-specific conservative schedule, the
upstream Triton autotuner, or the spill-safe compile search in `grid_pair.py`;
an exact entry rejected by a later Triton compiler falls through to that same
search. These fallbacks preserve the same mathematics and layouts, so tuning
for RTX PRO 6000 Blackwell cannot make an uncovered device execute a
Blackwell-only launch. `make_triton_value_path` likewise returns `None` for an
unsupported convolution, leaving the reference composition in charge.

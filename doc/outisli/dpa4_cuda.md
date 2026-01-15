# The SeZM / DPA4 inference path on hand-written CUDA

This is the working record of the hand-written CUDA inference path for the
uncompressed DPA4 / SeZM descriptor. It is written for whoever continues the
same pipeline, so every claim is a measurement and every rejected design is
recorded with the number that killed it.

The companion documents are `dpa4_triton.md` (the Triton path this one is
measured against), `dpa4_cutile.md`, and `dpa1&4c_cuda.md` (the coding and
launch-policy reference for the compressed paths).

______________________________________________________________________

## 1. Scope and result

Sections 1--10 describe inference. Sections 11--12 describe the independent
training value-path operator selected by `DP_CUDA_TRAIN=1`. The numerical
contract is IEEE fp32 with TF32 disabled; bf16 autocast keeps all reductions
and library contractions in fp32 accumulation.

Measured across the model zoo on periodic diamond, RTX PRO 6000 Blackwell,
warmup discarded. The baseline is `DP_TRITON_INFER=2` with `DP_COMPILE_INFER=1`
-- level 2 rather than 3, because the fp16x3 tables of level 3 only cover swept
shape keys and the baseline must be float32 against float32 on every checkpoint
(section 6). Both paths are built in one process, so the force comparison is
element-wise.

| checkpoint      | atoms | baseline | `DP_CUDA_INFER=1` | `DP_CUDA_INFER=2` |    peak, level 2 |
| --------------- | ----: | -------: | ----------------: | ----------------: | ---------------: |
| `nano` l1 c32   | 8,000 |  56.0 ms |      50.8 (1.10x) |  **34.4 (1.64x)** |  6.2 GiB (1.23x) |
| `mini` l2 c32   | 8,000 | 116.9 ms |      91.0 (1.29x) |  **66.2 (1.77x)** | 10.1 GiB (1.53x) |
| `neo` l3 c32 F2 | 4,096 | 152.8 ms |     137.1 (1.12x) |     137.3 (1.11x) | 16.5 GiB (1.53x) |
| `air` l3 c64    | 4,096 | 313.9 ms |     272.6 (1.15x) |     272.8 (1.15x) | 27.6 GiB (1.17x) |
| `plus` l4 c64   | 1,728 | 275.8 ms |     247.0 (1.12x) |     246.9 (1.12x) | 20.5 GiB (1.33x) |
| `pro` l5 c64 F2 | 1,000 | 586.0 ms |     564.3 (1.04x) |     563.7 (1.04x) | 46.8 GiB (1.05x) |

Level 1 holds the operators whose profit is memory traffic and therefore holds
on every checkpoint: the grid pair product, the geometric initial embedding,
the dense Wigner build, the envelope and radial basis. Level 2 adds the fused
convolution, whose float32 SIMT arithmetic is profitable only while the
per-edge arithmetic stays small; a routing gate (section 3.7) takes it for the
narrow checkpoints and declines the wide ones, which is why level 2 equals
level 1 from `neo` upward and **raising the level never costs time**. Section
10 derives the same crossover when the part changes instead of the checkpoint
(H20).

One caveat on parity rather than speed: `air` at this cell size sits on a
pathological region of its own force Jacobian (amplification about 1e6, see
section 7.1), so its force comparison reflects amplified rounding noise
rather than implementation deviation; every other row's parity is 1e-5 eV/Å
or better.

The 2x gate on the lower graph is **not** met. Section 8 gives the measured
roadmap; the short version is that the fused convolution carries half of the
`mini` step at 30 % of the fp32 FMA roof and is latency bound rather than
issue bound, so what is left there costs a redesign, not a tile.

Deviation from the Triton path on the same input: energy 2.8e-9 relative, force
1.2e-5 eV/Å maximum. The energy agreement improved by two orders of magnitude
when the envelope moved behind an operator boundary (section 3.6): the compiler
had been inlining that chain into every consumer and re-evaluating it there, so
the same quantity reached different consumers with different rounding.

Correctness gates, all green at every zoo shape (`nano`, `mini`, `neo`, `air`,
`plus`, `pro`; degrees 1--6, focus widths 32/64, one and two focus streams,
mixer ranks 0--2):

- forward and attention weights against a dense reference: worst 1.2e-6;
- all nine convolution cotangents (`x`, `quat`, `kc`, `q`, `k`, `env`, `rad0`,
  `head_gate`, `fscale`) against reference autograd: worst 1.2e-6;
- the initial embedding and the radial chain against their module
  compositions, both directions, every degree: worst 5e-7 on values and 1e-5 on
  the Bessel cotangent, whose two terms cancel to leading order;
- bitwise reproducibility of the destination reduction;
- `make_fx(tracing_mode="symbolic")` traces the operators.

______________________________________________________________________

## 2. Why fuse, and where the budget is

Kernel substitution cannot reach 2x, and this was established before any kernel
was written. cuTile rewrote the same SO(2) kernels in another DSL and delivered
1.07x. The reason is visible in the arithmetic intensity: one mixing-stack
multiply is `(E, 96) @ (96, 96)`, which is 24 FLOP/byte when the activation
round-trips through device memory, against a ridge point near 78 FLOP/byte on
this GPU. Nsight Compute confirmed the Triton stack sitting at 88--94 % of the
DRAM roof. Those kernels are not badly written; the *graph granularity* is
wrong.

Two facts about this model make fusion unusually profitable.

- **The node feature is tiny.** `edge_schema_from_extended` maps every source
  endpoint back to its owning local atom, so the node count is `nloc` (8000),
  not `nall` (216000). The node feature is 9.2 MB and stays resident in L2.
  Every edge-to-node gather and node-to-edge scatter is therefore nearly free of
  DRAM traffic, and *all* of the DRAM pressure comes from per-edge
  intermediates.
- **The per-edge span is long.** Between the node feature going in and the node
  aggregate coming out lie the attention logits, the softmax, the rotation, the
  degree mixer, a three-layer gated stack, the inverse rotation and the
  weighted reduction. Unfused, each stage writes an `(E, ...)` tensor.

The fused-operator budget must also respect the hard FLOP floor. Per edge and
focus stream, the stack costs

$$
\;3\times\bigl(M_0^2 + M_1^2 + C_f\cdot\mathrm{GATE}\bigr)
   = 3\times(96^2 + 128^2 + 32\cdot 64) = 8.29\times 10^4\ \mathrm{FMA},
$$

which over 1.264e6 edges, two interaction blocks, and a backward that costs
about the same again, is 1.26 TFLOP per step. Against the 125 TFLOP/s fp32 FMA
peak of this GPU the convolution alone can never go below **10 ms/step**, and a
well-tuned SIMT kernel realistically lands at 30--40 % of that peak. Reaching
3x end to end (39 ms) is therefore not possible in fp32 without tensor cores or
an algorithmic change; 2x (58.7 ms) requires the convolution near 30 ms *and*
most of the non-convolution graph gone. Both numbers are stated here so nobody
re-derives them the hard way.

______________________________________________________________________

## 3. Operators

### 3.1 Fused convolution -- `dpa4_so2_conv`

One operator pair spans the complete per-edge path of one `SO2Convolution`:

```text
eff[e]  = q[dst] . k[src] / sqrt(Ch) + rad0[e] @ W_logit + 2 log env[e]
alpha   = segment_softmax(eff; null_logit, dst)          # online, flash style
x_local = Wigner_e @ x[src[e]]                           # (RED, Cf)
u0      = degree_mix(x_local; kc[e], cb)                 # (RED, Cf)
u       = gated_stack(u0; W0, W1, Gw)                    # n_layers
out[n]  = gate[n] * rescale * sum_{dst[e]=n} alpha[e] * Wigner_e^T @ u / D[n]
```

It replaces the attention-weight build (logits, envelope-gated segment
softmax), `so2_rotate_mix`, `so2_mixing_stack`, `flash_atten_aggregate` and the
output head gate, and none of the per-edge intermediates of that composition
reach device memory. The finished attention weights are emitted as an output
because the backward of the softmax stays outside the kernel (section 3.2).

**Layout.** The m-major reduced layout carries `RED = 3 * lmax + 1` rows
ordered `m = 0` degrees, then `m = -1`, then `m = +1`, so the flat row is
`u[r * Cf + c]` of width `ROW = RED * Cf`. The first `M0 = (lmax + 1) * Cf`
columns are the `m = 0` block and the remaining `M1 = 2 * lmax * Cf` are the
two `|m| = 1` blocks. The SO(2) weights are block diagonal over that split,
which is why the stack runs as two independent multiplies plus a shared gate.
In registers, activation column `slot * 32 + lane` lives in lane `lane` at slot
`slot`, and every phase -- the block multiplies, the rotations, the gate, the
attention -- agrees on that mapping.

**Rotation contract.** A production Wigner-D matrix is block diagonal in the
degree, so the kernels contract only the `2l + 1` entries of the degree block
of each selected row, staged per edge as a packed `NW = 3 (lmax + 1)^2 - 2` run
(25 floats at `lmax = 2`, against 63 for the full selected rows). The runs
arrive as a plain `(E, NW)` tensor built by the companion operator of section
3.3; the dense per-edge matrices never exist on the CUDA path. Note `NW > 32`
from degree three on, which is what broke the first Wigner-gradient
implementation (section 6).

**Decomposition.** The forward gives one block one destination node, the
backward one source node, so both reductions are owned by a single block: no
atomics, and a summation order fixed by the CSR. The CSR views of both
endpoints are built once per step in Python (stable argsort + preallocated
`scatter_add` counts + cumsum), cached on the edge cache, shared by both
interaction blocks, and enter
the operator as plain tensor inputs -- inside the operator they used to cost
four eager sorts per step and a bubble in the compiled stream. Each block walks
its segment in chunks of `BE = WARPS * TM` edges; the activation lives in
registers across the whole stack and the node accumulator in registers across
chunks.

The preallocated count tensor is required by symbolic graph export:
`torch.bincount(..., minlength=n_node)` is valid in eager execution, but a
symbolic `n_node` cannot define its output size in the exported CSR graph.
`scatter_add` writes the same counts into an already shape-defined tensor and
therefore preserves the dynamic-node contract.

**The attention weights are an online softmax.** The reference
`segment_envelope_gated_softmax` seeds every segment maximum with the null-mass
logit `log(softplus(z_bias) + eps)` and adds the null mass to the denominator.
The kernel reproduces this exactly in one pass: the running maximum starts at
the null logit, each chunk folds its logits in flash-attention style (the node
accumulator is rescaled by `exp(m_old - m_new)`), the raw logits are stashed in
the weight output, and a per-focus epilogue normalizes them once the segment
maximum is final. An edge outside the cutoff contributes `-1e30` and therefore
no mass. The optional cross-focus competition multiplies the *finished* weight
outside the softmax, so it rides through as a per-`(edge, focus)` multiplier
that scales the staged weight but not the denominator.

**The backward never replays the forward multiplies.** The forward saves one
slot per layer: the gated layers keep their pre-activation `z`, and the
identity layer keeps the final activation. Because a gated layer is
`u_{k+1} = u_k + inc(z_k)`, the reverse sweep peels the residual off with a
point-wise subtraction, and starting from the saved final activation removes
the rotation replay, the reconstruction and the identity multiply. One
activation buffer serves the whole sweep.

**Cotangent split.** The kernel differentiates the value path: `g_x` (stored,
source-major exclusive), `g_wigner` (block-diagonal entries only), `g_kc`, and
`g_alpha` with respect to the weight it applied. The Python autograd wrapper
then assembles the softmax Jacobian and the logit, query, key, radial-bias and
envelope cotangents from plain tensor operations that Inductor fuses; the null
mass changes no formula because it depends on no logit. When the competition
scale is present the saved weight is `alpha * fscale`, and the raw softmax
weight must be recovered before the Jacobian -- feeding the scaled weight into
it was a real bug that only the two-focus shape caught.

### 3.2 The Wigner build -- `dpa4_wigner_runs`

Every entry of the packed run of degree `l` is a homogeneous polynomial of
degree `2l` in the unit edge quaternion; multiplying by powers of
`|q|^2 = 1` lifts every entry onto the single degree `2 lmax` monomial
basis. The run is therefore one matrix product,

```text
runs = monomials(quat) @ C_run^T        # (E, M) @ (M, NW)
```

with `C_run` fitted once per degree in fp64 against the reference calculator
(residual below 5e-12 up to degree six) and cached per process. The backward is
the same construction on the derivative basis: exact exponent manipulation of
the fitted coefficients gives four tables of degree `2 lmax - 1`, and

```text
g_quat = einsum('et,tcm,em->ec', g_runs, C_der, monomials'(quat))
```

folds the accumulated run cotangent of all consumers onto the quaternions in
one product and one reduction. The extension of the fitted polynomial off the
unit sphere is immaterial: the quaternion normalization upstream projects the
radial gradient component out, so the operator's Euclidean gradient may carry
any radial part.

The operator exists so the runs are built **once per step**: both interaction
blocks consume the same tensor through the edge-cache store, their run
cotangents accumulate in autograd, and one contraction serves the whole step.
Implementation is one coalesced monomial kernel plus cuBLAS products; the two
hand-rolled alternatives, and why they lost, are in section 6.

What this removed from the graph: the dense `(E, D, D)` build (monomial
kernels, per-degree matmuls, the assembly copies and the transpose for
`Dt_full`), the 400 MB zero fill and dense scatter of its gradient, and the
dense backward contraction. The initial embedding takes its zonal rows from the
same polynomial tables through the calculator's existing `forward_zonal`
fallback, and the descriptor skips the dense build whenever every block runs
the fused path (`_wigner_free_conv`); cross-focus competition still reads dense
rows, so those configurations keep it.

### 3.3 Fused dense Wigner build -- `dpa4_wigner_dense`

Where the runs cannot replace the dense pair -- every configuration in which
some block still consumes `D_full`/`Dt_full`, which is all of level 1 and the
routed-out blocks of level 2 -- the same polynomial identity fuses the dense
build itself. Each block-diagonal element `(l, r, c)` is fitted on its own
degree `2l` basis and pruned to its structural non-zeros, which is what makes
this table affordable where the run table lifts everything onto one basis:

| degree | elements | entries | avg/element |
| -----: | -------: | ------: | ----------: |
|      2 |       35 |     197 |         5.6 |
|      3 |       84 |     909 |        10.8 |
|      4 |      165 |   3,103 |        18.8 |
|      6 |      455 |  20,033 |        44.0 |

The whole table is a few hundred kilobytes at most and is read identically by
every block, so it lives in L2; a warp walks the `(edge, element)` task space
edge-major, reading each entry once per 32 edges (one broadcast) against a
per-edge power table in shared memory, assembles the block-diagonal frame in
shared memory, and streams `D_full` and `Dt_full` out linearly. Traffic drops
to the quaternion read plus the two output writes -- the lower bound -- where
the module composition pays five full-size passes (monomial basis, GEMM, zero
fill, block scatter, transposed copy). The backward is the same walk with the
exponents differentiated in place, contracting `g_D + g_Dt^T` staged in shared
memory onto four per-edge accumulators.

Measured at the production edge count (1.26 M, RTX PRO 6000) against the
module path, forward: 4.89 to 0.59 ms at degree 2 (8.3x, 78 % of the DRAM
write roof), 14.1 to 2.1 ms at degree 3, 103.9 to 39.5 ms at degree 6 (2.6x;
the wide blocks turn shared-memory-read bound, which is acceptable since no
zoo checkpoint consumes a dense degree-6 pair). The polynomial fit runs at
construction -- always an eager context -- so tracing never re-enters it; a
lazy fit dies inside `make_fx` on a data-dependent guard, which is how the
first integration failed.

### 3.4 Fused grid pair product -- `dpa4_grid_pair`

Every grid operator of the model evaluates
`out = from_grid(to_grid(left) * to_grid(right))` on coefficient operands;
`GridProduct` directly, `GridMLP` inside channel projections, `GridBranch` at a
single branch through a softmax over one element, which is identically one. One
operator serves all five grid call sites.

The grid field is the reason to fuse: at the production SO(3) shape (`P = 27`,
`G = 104`, `C` up to 192) it is 39 times larger than the coefficient operand
that produces it -- 639 MB for 8000 nodes -- and the unfused einsums surround
each multiply with full-size layout copies. A warp owns one
`(node, 32-channel)` pair, holds both operands and the output accumulator in
registers, and walks the grid points, so the grid field never leaves registers.
The projector matrices are staged once per block into shared memory with rows
padded to the 16-byte vector width, and the inner loop reads them as `LDS.128`
broadcasts: from global memory those reads were two scalar loads per three
FMAs and cost forty percent of the issue bandwidth (section 5). The result is
bitwise identical to the projector composition because the grid sum runs in the
natural order of `g`.

**Training form.** The same expression differentiated inside the force
graph is served by the Triton tensor-core operator
(`sezm_triton::grid_pair_train`, `dpa4_triton.md` section 3.10), not by an
extension of this kernel. The register-resident walk above was extended
with the training differentiations first -- the frame-packed layout taken
apart in the load phase, the trilinear second order as one further kernel
with eight `P`-sized register arrays -- and verified to the same standard,
but its contractions run at FFMA rate (~33 TFLOPS measured) while the
training shapes are GEMM-shaped (`P` up to 147, `G` up to 460, the FLOP an
order of magnitude above the inference S2 grids), and it lost end to end
to the dense composition on every production size. The extension was
removed outright. The division of labor is structural: scalar traversals
and resident activation chains belong to hand-written CUDA, GEMM
sandwiches to the tensor cores.

### 3.5 Fused initial embedding -- `dpa4_zonal_scatter`

The geometric initial embedding broadcasts one radial feature per packed
non-scalar row, scales it by the zonal coupling, and reduces over the incident
edges of every destination node:

$$
\mathrm{out}[n, r, c] = \sum_{\mathrm{dst}(e) = n} \mathrm{zonal}[e, r]\;
   \mathrm{radial}[e, \mathrm{slot}(r), c].
$$

Written with tensor operations the per-edge message is materialized as an
`(E, R, C)` tensor, which is **1.3 GB** at the production shape and is written
once and read once by the scatter. Fused, a warp owns one node, walks its
incidence list through the destination CSR the convolution already builds, and
accumulates into a register tile, so the traffic is the two operands plus the
node result -- 533 MB, and no atomics. Because a lane owns one channel of a
32-wide block, wider features sweep the edge list once per block and read only
their own channels, so the traffic does not grow.

The operator also emits the packed layout and the smooth degree normalization,
which saves the caller a concatenation and a second full-size pass over the node
tile. Standalone against the composition it replaces, 8000 nodes at degree 158:

| degree | rows | reference |    fused | speedup |
| -----: | ---: | --------: | -------: | ------: |
|      1 |    3 |   1.99 ms | 0.153 ms |   13.0x |
|      2 |    8 |   5.39 ms | 0.282 ms |   19.1x |
|      3 |   15 |  10.05 ms | 0.480 ms |   21.0x |
|      6 |   48 |  27.11 ms | 1.127 ms |   24.1x |

A second consumer of this operator's inputs falls out of it. The packed Wigner
run of section 3.2 carries the `m = 0` row of degree `l` at entries `l^2` to
`(l + 1)^2`, and that *is* the zonal coupling: the descriptor slices it out of
the shared runs instead of rebuilding it from the quaternion, which is what
`_wigner_free_conv` had forced it to do. Verified elementwise against
`forward_zonal` at degrees 1 to 4, worst 1.1e-6.

### 3.6 Fused envelope and radial basis -- `dpa4_edge_radial`

Both quantities the edge cache derives from the pair distance are functions of
that distance alone,

$$
\mathrm{env}[e] = \mathrm{keep}[e]\, E_{p_1}(r), \qquad
\mathrm{rbf}[e, n] = \mathrm{keep}[e]\, \phi_n(r)\, E_{p_2}(r),
$$

with the C3 envelope in its cancellation-free factorization
$E_p(r) = u^4 S_p(x)$, $u = \mathrm{clamp}((r_c - r)/r_c, 0, 1)$, $x = 1 - u$,
and the basis either Bessel, $\phi_n = \sin(r f_n)/r$, or Gaussian,
$\phi_n = \exp(k (r - c_n)^2)$.

This chain is arithmetic on a 96 MB working set and looks free, which is exactly
why it was not: the compiler inlined it into every consumer of `env` and `rbf`
and re-evaluated it there, so the pass was paid several times over and each
consumer saw a slightly different rounding of the same quantity. Behind an
operator boundary it runs once. The speed effect is small (section 6), but the
**energy agreement with the Triton path improved by a factor of 72**, from
2.0e-7 to 2.8e-9 relative, because there is now one value rather than several.

### 3.7 The routing gate of the fused convolution

`make_cuda_so2_conv` declines a block the operator does not serve, and also one
it serves but would slow down. The convolution trades device traffic for
float32 SIMT arithmetic, so its profit falls as the arithmetic per edge grows;
measured as the end-to-end difference between taking the convolution over and
leaving it on the Triton path, on this part:

| checkpoint | per-layer FMA per edge (x F) | conv takeover |
| ---------- | ---------------------------: | ------------: |
| `mini`     |                       27,648 |         +43 % |
| `neo`      |                      112,640 |          -4 % |
| `air`      |                      225,280 |          -9 % |

The gate is `(M0^2 + M1^2 + Cf * GATE) * n_focus <= 65536`, sitting between the
measured signs. The focus-stream factor is the reason a plain width threshold
misjudges `neo`: its GEMMs are `mini`-narrow but its second stream doubles the
arithmetic. A declined block keeps the Triton path -- exactly what level 1
runs -- so `DP_CUDA_INFER=2` is safe to set unconditionally on this class of
part, and the level is a capability, not a promise.

______________________________________________________________________

## 4. Measurements

### 4.1 End-to-end progression

Compiled lower graph, 8000-atom cell, 30 timed iterations, warmup discarded.
Every row was measured against the same 117.4--117.5 ms Triton baseline in the
same process, with element-wise force parity checked each time.

| stage                                                    |         time |      peak |   speedup | note                                                          |
| -------------------------------------------------------- | -----------: | --------: | --------: | ------------------------------------------------------------- |
| Triton L3 baseline                                       |    117.47 ms |     15.37 |     1.00x |                                                               |
| fused convolution + fused grid pair, correct and general |     96.17 ms |     12.32 |     1.22x | session baseline after the correctness fixes of section 6     |
| + shared-memory broadcast in the block multiply          |     89.44 ms |     12.32 |     1.31x |                                                               |
| + `cp.async` panels, `TM = 4`, occupancy target          |     81.98 ms |     12.32 |     1.43x | includes the staging-race fix                                 |
| + staged mixer-gradient reduction                        |     78.72 ms |     12.32 |     1.49x |                                                               |
| + vectorized grid-pair projectors                        |     75.25 ms |     12.32 |     1.56x |                                                               |
| + CSR views built once per step in the graph             |     73.75 ms |     12.32 |     1.59x |                                                               |
| + fused attention weights                                |     73.92 ms |     12.32 |     1.59x | perf neutral; kept for the interface (section 5)              |
| + polynomial Wigner runs, dense matrices gone            |     73.97 ms |     11.21 |     1.59x | speed neutral by design iteration (section 6); the memory win |
| + packed reduction, column groups, `BUDGET = 2048`       |     67.67 ms |     11.26 |     1.73x | convolution 38.4 to 33.0 ms (section 5)                       |
| + fused initial embedding                                |     65.58 ms |     10.06 |     1.79x | the 1.3 GB per-edge message is gone                           |
| + fused envelope and radial basis                        | **65.62 ms** | **10.35** | **1.78x** | speed neutral, energy accuracy 72x better (section 3.6)       |

Peak memory is 10.35 GiB against 15.37 GiB (1.49x).

### 4.2 Fused convolution, standalone production shape

`E = 1,264,000`, `N = 8000`, uniform degree 158, per call. The backward figure
is kernel plus the Python cotangent assembly, driven through autograd.

| shape                    |   forward |  backward |     peak |
| ------------------------ | --------: | --------: | -------: |
| nano (`l1 c32 F1 L3 r0`) |   3.95 ms |   7.37 ms |  3.2 GiB |
| mini (`l2 c32 F1 L3 r1`) |   8.69 ms |  13.49 ms |  4.7 GiB |
| neo (`l3 c32 F2 L3 r1`)  |  32.00 ms |  46.77 ms | 12.9 GiB |
| air (`l3 c64 F1 L4 r1`)  | 120.40 ms | 181.41 ms | 15.9 GiB |
| plus (`l4 c64 F1 L4 r2`) | 258.30 ms | 310.39 ms | 23.6 GiB |

Over this session the mini pair went from 31.2 ms (12.7 + 19.4, weights-only
backward) to about 22 ms with strictly more work fused in; air went from
165 + 198 to about 120 + 190. The rows above predate the run-operator split of
section 3.2, which moved the polynomial build out of these timings and into
one shared per-step operator.

### 4.3 Where the 65.6 ms sits

Per-kernel self time of the final step (`torch.profiler`, 5 iterations, 471
launches, 67.5 ms of GPU time against 65.6 ms of wall time -- the graph is
launch-gap free):

| item                                                 |   ms | share |
| ---------------------------------------------------- | ---: | ----: |
| `so2_conv_bwd_kernel` (2 calls)                      | 17.0 |  25 % |
| `so2_conv_fwd_kernel` (2 calls)                      | 15.6 |  23 % |
| dense linear algebra (magma + cutlass, ~65 launches) | 11.9 |  18 % |
| Inductor fragments                                   | 11.8 |  18 % |
| `grid_pair` forward + backward (10 calls)            |  3.7 |   6 % |
| ATen kernels around the new operator boundaries      |  2.6 |   4 % |
| `zonal_scatter` and `edge_radial`                    |  2.3 |   3 % |
| monomials, force segment, fills, sorts, misc         |  2.2 |   3 % |

The convolution pair is at 32.6 ms against its 10 ms fp32 floor, i.e. 30 % MFU
(the Triton composition it replaced ran the same math at 20 % and materialized
every intermediate).

The dense linear algebra is now the second largest item and is *not* kernel
work that can be tuned: it is 65 launches of `bmm` shaped `(9, N, 32) @ (9, 32, 32)` and similar, i.e. the equivariant per-degree linear layers. At
`N = 8000` these carry 4 FLOP/byte and are purely bandwidth bound, so their
17 % efficiency is a granularity problem in the module code, not in cuBLAS.

______________________________________________________________________

### 4.4 Size dependence and parity

Both paths built in one process, identical inputs, so the force comparison is
element-wise:

| atoms |     edges | Triton L3 |     CUDA | speedup | peak (Triton / CUDA) | dE relative |      dF max |
| ----: | --------: | --------: | -------: | ------: | -------------------: | ----------: | ----------: |
| 4,096 |   647,168 |  59.06 ms | 38.33 ms |   1.54x |      7.90 / 5.77 GiB |      2.2e-7 | 1.1e-5 eV/Å |
| 8,000 | 1,264,000 | 117.46 ms | 73.97 ms |   1.59x |    15.37 / 11.21 GiB |      2.2e-7 | 1.1e-5 eV/Å |

The speedup rises slightly with size because the fused operators are
compute-bound while the path they replace was DRAM-bound.

______________________________________________________________________

## 5. Optimization study: what worked

Each entry gives the measured effect at the production mini shape, in the order
it landed. The through line: **this kernel was never FLOP-bound; every win came
from removing a hidden serialization or a quarter-rate instruction.**

1. **Panel staging with a register prefetch (1.74x forward).** The weight
   matrices are L2 resident, but a load consumed by the barrier right after it
   exposes hundreds of cycles per panel. Fetching the next panel into registers
   immediately after the barrier that publishes the current one hides all of
   it. Later replaced by `cp.async` (same effect, no registers).

1. **Keeping the activation tile out of local memory.** `ptxas` reported a
   224-byte stack frame: a runtime slot index (`areg[i][jslot]`) had demoted
   the whole activation array to local memory, an L1 round trip on every
   access. The first fix selected the slot with a predicated compare chain,
   which keeps the tile in registers with rolled loops and small code. Fully
   unrolling the panel loop to make the index compile-time constant works at
   narrow shapes but emits up to twelve panel bodies at wide ones and loses a
   third of the issue slots to instruction fetch (measured on `air`: 165 to
   237 ms). Superseded by the column-group split below, which gets the
   compile-time index without any of that.

1. **Splitting the loop nest at the column group (mini 23.2 to 21.3 ms,
   `air` 234 to 219 ms).** The predicated chain above costs `TM * AB`
   instructions every column group, which at the production shape is 1344 per
   chunk against 2592 products -- half again the work it enables. Hoisting the
   column-group loop out of the panel loop and unrolling only that level makes
   the slot a compile-time index, so the tile is read as `areg[i][JBEG + cg]`
   directly, while the panel loop inside stays rolled and the instruction
   footprint stays at three or four panel bodies rather than twelve. Dynamic
   instruction count fell 10.9 %, of which control flow halved; FFMA went from
   53.1 % to 59.6 % of all instructions.

1. **Vectorizing the weight side along the reduction (`air` forward 120 to 104
   ms, backward 181 to 131 ms; `plus` 258/310 to 185/222 ms).** The inner step
   read one weight per output column with a scalar `LDS`, one load per product
   row. Repacking the weight on the host to `(KK/4, NN, 4)` -- four consecutive
   reduction steps of one output column made contiguous -- turns those into one
   `LDS.128` per column group per four steps. The packing is independent of the
   panel depth, because a panel is always a whole number of step groups, so the
   staging is untouched, and it is conflict free: eight lanes of a load phase
   start four banks apart and together cover all thirty-two. Wide shapes gain
   28 %; the production `mini` shape is neutral, because at `TN = 3` it is
   latency bound rather than issue bound (section 6).

1. **Panel depth as a per-shape constant (nano 12.2 to 11.1 ms, mini 23.2 to
   21.5 ms, air 219 to 200 ms).** Every panel costs two barriers, and barrier
   stalls were 10 % of all issue slots. The optimum depth is a measured
   property of the shape -- 32/16/8/8/4/4 across the zoo, staged floats between
   1536 and 3072 throughout -- and both ends were measured: halving `mini`'s
   depth costs 8 % in barriers, doubling `air`'s past its optimum costs 49 % in
   residency. `panel_depth_of` encodes the thresholds.

1. **The wide tile at reduced residency (`air` conv 200 to 175 ms, -13 %).**
   The original policy dropped to `TM = 2` past `RB = 14` to protect the
   register file, halving the products each staged weight feeds. Keeping
   `TM = 4` and paying for it with a residency target of two instead of three
   is the better trade up to `RB = 26`; the widest row (`pro`, 32) cannot hold
   the wide tile at any residency and keeps the narrow one. Swept per shape on
   one GPU per configuration; the same sweep showed `TM = 4` at the default
   residency losing on `plus`, so neither constant generalizes without the
   other.

1. **Shared-memory broadcast instead of `__shfl_sync` (forward 11.8 to 10.0
   ms, backward 19.4 to 16.5 ms standalone; 96.2 to 89.4 ms end to end).** The
   reduction broadcast one activation value per shuffle, at one shuffle per
   3--4 FMAs -- and a shuffle is quarter rate, so the broadcast cost as many
   issue slots as the products it fed. Publishing the current column group into
   a 1 KB per-warp stage and reading it back as a uniform (full-rate) load
   removed that tax. The stage is step-major, so one `LDS.128` broadcasts a
   whole four-edge tile.

1. **`cp.async` double-buffered panels.** Same latency hiding as the register
   prefetch without the 32 prefetch registers and without the LDG-to-STS round
   trip. Performance neutral on mini at the old occupancy; what it bought was
   the register headroom the next item spends. It also introduced the staging
   race of section 6.

1. **`TM = 4` with small panels and an occupancy target (forward 9.9 to 8.6
   ms, backward 16.2 to 13.2 ms).** At `TM = 8` the kernel sat at 254
   registers, two resident blocks, eight warps per SM -- nothing covered the
   barriers (Nsight: 17 % issue-slot utilization, estimated 83 % local speedup
   from occupancy). Halving the edge tile halves the activation registers and
   `__launch_bounds__(NT, 3..4)` lets three to four blocks reside, which the
   assembler meets without spilling. The occupancy target is shape dependent:
   forcing it on the narrow shapes at the *old* register footprint made the
   in-model backward slower (96.2 to 102.4 ms end to end) even though the
   uniform-degree standalone improved -- variable-degree tails pay for the
   tighter schedule. Measure occupancy hints in the model graph, not the
   harness.

1. **Staged mixer-gradient reduction (backward 12.9 to 11.6 ms; neo 53.0 to
   44.5 ms).** The rank >= 1 degree-kernel gradient did one 5-shuffle warp
   reduction per `(slot, rank)` pair -- the same quarter-rate tax as item 3.
   The per-slot products now stage through the existing outer-product rows and
   one lane sums the 32 channels of its slot against the channel basis, read as
   a broadcast.

1. **Vectorized grid-pair projectors (6.8 to 3.3 ms).** Section 3.2. Two
   scalar broadcast loads per three FMAs became one `LDS.128` per four
   coefficients.

1. **CSR out of the operator (73.75 vs 75.25 ms).** Four eager radix sorts per
   step became two traced ones shared by both blocks, and the sort left the
   hot stream.

1. **Fused attention weights (perf neutral, kept).** The forward softmax and
   logit fragments left the graph, but the kernel gained the key gather, the
   logit reductions and the epilogue (+0.5 ms/call forward), and the softmax
   backward stayed graph-side by design. Net zero today; it removes an E-sized
   round trip, makes the operator self-contained for the LAMMPS/kokkos path,
   and is the prerequisite for ever moving the softmax backward in-kernel.

1. **Polynomial Wigner runs, shared per step (speed neutral, 1.1 GiB and the
   dense matrices gone).** Three designs were measured to get here. Inlining
   the polynomial evaluation into the megakernels cost 2.4 ms end to end: the
   monomial stage either displaced shared memory (an occupancy cliff) or, per
   focus stream and per chunk, repriced work the dense path did once. Splitting
   it into hand-rolled warp-per-edge kernels cost 14 ms: four active lanes and
   500-element serial reductions. The winning shape is one coalesced monomial
   kernel plus cuBLAS products, built once per step behind its own operator
   with its own autograd, shared by both blocks through the edge cache. The
   lesson generalizes: for E-sized dense contractions, a library GEMM on a
   well-shaped matrix beats a bespoke kernel unless the fusion removes real
   traffic.

Items that predate this session and still hold: the destination/source-major
exclusive reductions (no atomics, bitwise reproducible), the saved-activation
backward, the padded-lane liveness guards, and the `NCG = 32` column mapping
that keeps shared-memory wavefronts full.

______________________________________________________________________

## 6. Rejected, ineffective, or bugs worth remembering

- **Reading weights straight from L2/L1 instead of staging (1.85x slower).**
  Same load count, no barriers -- and still much slower: at eight warps per SM
  nothing covers the dependent-load latency inside the reduction. Staging plus
  prefetch is not optional at low occupancy.

- **The activation tile in shared memory (10.0 to 20.3 ms forward).** A
  29 KB per-block tile pushed dynamic shared memory to 50 KB; the driver's
  64 KB carveout then holds one block per SM. Shared memory is the scarcer
  resource for this kernel; the register file is the right home for the
  activation, local memory the wrong one (item 2 above), and the 1 KB
  column-group stage the right compromise.

- **Compile-time `n_layers` / `rank` / `n_head` / `n_focus` (2 % forward, 7 %
  backward on mini).** Pinning all four runtime parameters recovered almost
  nothing -- the loops they control sit outside the hot reduction. Not worth
  multiplying the 24 translation units.

- **`cp.async.bulk` (TMA) panel staging (mini +4 %, `neo` +9 %, `air` +13 %).**
  The obvious next step after the address arithmetic showed up as 13 % of all
  instructions: one thread issues a single bulk transfer per panel instead of
  128 threads issuing four 16-byte copies each. It works and it is slower
  everywhere except `nano`. The reason is the completion mechanism, not the
  transfer: a transaction barrier needs *every* thread of the block to arrive
  on it, which is a block-wide atomic per panel, and the copy itself is now
  issued serially by one thread. TMA pays off on tiles large enough to amortize
  that -- a 128x128 GEMM tile -- not on an 8 KB panel. Implemented against the
  `cuda::barrier` API rather than raw PTX after a hand-written
  `mbarrier.try_wait.parity` loop deadlocked; if this is revisited, note that
  the arrival count must be the block width and the phase must come from the
  arrival token, not from a hand-computed parity.

- **Uniform occupancy targets.** The narrow shapes want no occupancy cap at 254
  registers and want `OCC = 4` after the register diet; forcing `OCC = 5` or 6
  makes the assembler trade scheduling freedom for residency and costs 4 %.
  The panel budget, by contrast, *is* uniform at 2048 (section 5).

- **An operator boundary is not free, and it is the reason the fragment
  fusions returned so little.** Replacing the envelope chain with an operator
  cut the Inductor fragment total from 16.5 to 11.8 ms, but 2.6 ms came back as
  ATen kernels: the consumers that used to have the envelope inlined into them
  now read `env` and `rbf` from memory instead, and the compiler can no longer
  fuse across the boundary. Net was under 2 ms on a 4.4 ms target. The rule
  that survives: **fuse a chain out of the graph when it owns a large
  intermediate (`zonal_scatter`, 1.3 GB, 2.1 ms net) rather than when it merely
  looks expensive in the profile (`edge_radial`, arithmetic on 96 MB, neutral).**
  The envelope fusion was still kept, for the accuracy reason in section 3.6.

- **The `cp.async` staging race (forces off by 21 eV/Å, standalone tests
  green).** The first asynchronous copy was issued at function entry, before
  any barrier -- while other warps still read the same scratch region: the
  previous multiply's last panel (buffer parity makes it the same buffer when
  the panel count is odd) or the outer-product stage. The standalone checker
  passed on timing luck; the model graph exposed it immediately. Two rules
  survive: *a shared scratch region needs a barrier before the first async
  write into it*, and *force parity in the real graph is part of every
  measurement, not a final check*.

- **Four stale object files that survived every rebuild (`pro` launching with
  `invalid argument`).** The `(5, 64)` and `(6, 64)` translation units stopped
  recompiling after one interrupted parallel build: their objects sat at a
  stale timestamp while every header rewrite -- the weight repacking, the
  column-group split -- rebuilt the other twenty. The library still linked,
  so it carried kernels compiled against the old ABI next to a host compiled
  against the new one, and the mismatch surfaced as a launch failure on the
  one checkpoint (`pro`) that was also missing from the shape tables of every
  checker. Diagnosis was the absence of evidence: an error report added *at*
  the launch did not fire, which proved the executing launch function predated
  the report. Three rules: **a shape no test exercises is a shape that does
  not work** (`pro` is in `ZOO_SHAPES` now), launch-status checks belong at
  the launch where they can name their shape (they stay in), and when a fix
  visibly does not land, check the object timestamps before the code.

- **A baseline that was not what it claimed, and a wrong explanation built on
  top of it.** The end-to-end numbers were taken against `DP_TRITON_INFER=3`,
  and when level 2 turned out to lose on the wider checkpoints (1.05x on `neo`,
  0.74x on `air`, 0.52x on `plus`) the obvious story was the one section 10
  already tells: float32 SIMT against fp16 tensor cores, with the crossover
  moving from the part to the checkpoint as the arithmetic per edge grows. That
  story was wrong. Timing every Triton level at every CUDA level showed level 2
  and level 3 **identical to three digits** on this part (58.80 vs 58.81 ms at
  `DP_CUDA_INFER=0`, 34.72 vs 34.73 at level 2): the fp16x3 launch tables only
  cover swept shape keys, and on this part they cover none of them, so the
  "level 3" baseline had been level 2 all along and no tensor core was ever
  involved. The real cause is the launch tile, below. Two rules: **pin the
  baseline at the level whose kernels you can name**, which is why the zoo
  benchmark now runs at `DP_TRITON_INFER=2`, and a plausible explanation that
  matches an earlier finding deserves *more* suspicion, not less.

- **A uniform panel budget starves the wide shapes.** Section 5 sets
  `BUDGET = 2048` from a sweep on `mini`, checked against the wider shapes only
  for "does not get worse" -- and it did not, because they were already there.
  What that hid is that the budget divides by the panel width, so the wide
  shapes sit at the floor:

  | shape  | `NMAX` | `PK` | `TM` |
  | ------ | -----: | ---: | ---: |
  | `nano` |     64 |   32 |    4 |
  | `mini` |    128 |   16 |    4 |
  | `air`  |    384 |    4 |    2 |
  | `plus` |    512 |    4 |    2 |
  | `pro`  |    640 |    4 |    2 |

  A panel costs two barriers, so `PK = 4` gives `air` four times the barrier
  count of `mini` per unit of reduction, on top of the `TM = 2` tile that halves
  the products each weight load feeds. An earlier revision of this document
  recorded "`air` prefers the 4096-float budget"; unifying the constant deleted
  that without re-measuring it. **A constant validated on one shape is a
  constant for that shape.**

- **A dropped gradient on an input that looks like a constant (forces off by
  3e-2 eV/Å on `neo`).** The initial embedding ends in
  `out.mul_(inv_sqrt_deg)`, and folding that scaling into the operator saves a
  full pass over the node tile. But `inv_sqrt_deg` is not a constant: it is
  `rsqrt` of a sum over the cutoff envelope, so it carries a gradient back to
  the geometry, and the first version of the operator returned `None` for it.
  Everything else stayed green -- the forward matched to 1e-6, the operator's
  own backward matched its reference to 1e-7, `mini` and `nano` were within
  1e-5 end to end -- because the missing term is small wherever the degree is
  nearly uniform. The fix returns it, reconstructing the unscaled reduction
  from the saved output, which is exact because the degree floor keeps the
  scale strictly positive. Two rules: **when fusing a multiply, enumerate which
  of its operands are differentiable rather than which of them look constant**,
  and a unit test that only feeds the operator detached tensors cannot see this
  class of bug -- `test_backward_reaches_the_degree_normalization` now drives
  the scale through autograd on both paths.

- **A disagreement blamed on the wrong path.** On `air` at 8000 atoms the
  fused path and the Triton baseline differed by 1.3 eV/Å on a few atoms while
  the energies matched to 1e-10 relative, which looks exactly like a backward
  bug. Every candidate cleared: `grid_pair` was checked at all fourteen call
  sites the real model issues (worst 1.4e-6), the force assembly is shared with
  `mini` and `neo`, which were clean, and the same model at 2000 atoms was
  clean. Arbitrating all paths against the *dense* reference instead of against
  each other settled it: `DP_TRITON_INFER=3` alone deviates by 3.7e-1 on 1155
  atoms, and the fused path by 3.5e-1 on 472 -- both are off, the fp16x3
  tensor-core GEMMs are the source, and the fused path is the *more* accurate
  of the two. The 1.3 eV/Å was two independently perturbed paths being compared
  to each other. Two rules: **never arbitrate between two accelerated paths**,
  and `air` is not a model to run with `DP_TRITON_INFER=3` under molecular
  dynamics.

- **Padding contamination (session-opening bug).** Tail-chunk slots alias edge
  zero so their addresses stay in range; one unguarded accumulation then leaks
  spurious contributions into `g_x` and edge zero's `g_kc` / `g_wigner`. Fixed
  with a single warp-uniform liveness `continue`; the guard placement is load
  bearing, not cosmetic.

- **Wigner gradients past degree two (silently wrong for `lmax >= 3`).** The
  packed run `NW = 3(lmax+1)^2 - 2` exceeds a warp from degree three on, and a
  one-slot-per-lane reduction simply dropped slots 32 and up. Lanes now stride
  over the run. Any per-lane mapping must be checked against the largest
  instantiated shape, not the development one.

- **The scaled-weight softmax Jacobian (neo gradients off by 0.4).** With the
  competition scale folded into the saved weight, the Jacobian must run on the
  recovered raw softmax weight. Only the two-focus shape exercises this; a
  zoo-wide gradient test is what caught it.

- **The inline and warp-per-edge Wigner builds (2.4 ms and 14 ms end-to-end
  regressions).** Recorded in item 10 of section 5; both passed every
  correctness gate and lost on shape alone. The 11.6 ms `quat_grad` kernel is
  the canonical example of a pathological decomposition: correct, tiny FLOPs,
  four active lanes per warp walking 2 KB of tables serially.

- **The default-device sentinel is part of the contract.** The pt test suite
  pins the default device to a non-existent ordinal to catch library code that
  constructs tensors without an explicit device; the table-fitting helpers
  tripped it. Constructors in operator support code carry explicit devices.

- **fp16x3 / TF32 tensor-core paths.** Excluded by the accuracy contract
  (non-smooth potential surfaces under TF32; fp16x3 measured no win on this
  part in the Triton path). The fp32 FMA peak is the binding roof.

______________________________________________________________________

## 7. General lessons

1. **Profile the production shape.** A degree-7 harness made instruction-fetch
   stalls look dominant (every loop body ran once, cold); the production
   degree-158 profile disagreed. One misread profile cost a day of loop-shape
   churn.
1. **Check `ptxas -v` before trusting a register count story.** "255
   registers, 0 spills" hid a 224-byte stack frame -- the array was *demoted*,
   not spilled, and no profiler counter says so directly.
1. **Quarter-rate instructions are invisible FLOPs.** Shuffles and warp
   reductions issued per element cost four times what they look like in the
   source. Both big wins of this session (items 3 and 6) were the same fix in
   different clothes: replace per-element warp communication with a staged
   full-rate load.
1. **On-chip residency is a budget with three currencies** -- registers,
   shared memory, occupancy -- and the exchange rates are shape dependent. The
   activation tile visited all three homes (registers, local memory, shared
   memory); only measurement, not reasoning, picked the winner per shape.
1. **Custom operators are opaque to the compiler in both directions.** Eager
   work inside the op (the CSR sorts) never fuses and bubbles the stream;
   moving it into the graph as plain aten ops let Inductor dedupe it across
   both interaction blocks. Conversely, cotangent math left in Python fuses
   fine -- the softmax backward costs the same graph-side as it would
   in-kernel, so fuse forwards, not backwards, unless the memory says
   otherwise.
1. **Count how often a fused quantity is rebuilt.** The dense Wigner matrices
   were built once per step; the first fused designs rebuilt the equivalent
   runs four times (two forwards, two backwards) and gave the savings straight
   back. Sharing through the edge cache with autograd accumulating the
   cotangents restored the single build. The CSR views followed the same
   pattern earlier in the session.
1. **The end-to-end number is the only number.** Three separate times this
   session a standalone improvement (occupancy hints, the shared-memory
   activation, the attention fusion) was neutral or negative in the model
   graph -- variable degree, stream context and compile glue all shift the
   optimum.

### 7.1 Case study: the `air` force drift was the model, not the code

The benchmark's force parity flagged `air` at 4,096 atoms: up to 0.95 eV/Å
between two paths whose siblings agreed to 1e-5, present in old logs, so
predating every operator of this document. The investigation is recorded
because its *method* transfers and because every intuitive suspect was
innocent.

Symptom, sharpened by rerunning one configuration on fixed inputs and
differencing its own forces: the compiled forces drift run-to-run,
bimodally -- half the runs land near the 2e-6 atomic-order floor, half jump
to 0.35 eV/Å with a *fixed* set of atoms (4 above 0.1, 28 above 0.01) and a
highly reproducible magnitude. Energies stay at the 1e-5 eV floor throughout, and are bitwise
stable under deterministic algorithms, so the fault lives in the backward
graph. Only `air` and only from 16^3 diamond cells upward: `neo` on the same
cell (same rcut/sel, hence the same edge list) is clean, `air` on a 14^3 cell
is clean.

What did **not** move it, four to twelve runs each: every custom CUDA and
Triton operator disabled (pure Inductor + ATen still drifts); ten Inductor
codegen switches (buffer reuse, mix-order, split, persistent and cooperative
reductions, loop ordering and reindexing, online softmax, graph partition);
the cuBLAS workspace; `torch.use_deterministic_algorithms`. A one-hour
`compute-sanitizer initcheck` run bought nothing. The flakiness itself
poisoned the switch scan: a three-run "success" of one switch dissolved at
twelve runs, and no generated kernel even used the helper that switch
controls.

What worked was a launcher-level checksum probe: patching
`CachingAutotuner.run` plus the extern `mm/bmm` wrappers checksums every
tensor argument of all ~740 launches with device-side accumulation (a host
sync per launch retimes the schedule enough to halve the hit rate -- itself
evidence of timing sensitivity). Two runs give two sequences, and the first
divergence names its kernel. The result
reframed the problem: the drift enters at the 1e-7 relative floor in the
*first* backward reductions -- ordinary atomic-order noise -- and then
**grows along the chain**, reaching 1e-3 mid-graph and O(1) at the force
assembly, with the first-large-divergence site moving between hits. No
kernel computes wrong; the chain amplifies.

The verdict experiment took the compiler out entirely: perturb `edge_vec` by
1e-7 Å and compare forces. `air` at 4,096 atoms amplifies by **3.3e6 to
7.2e6** -- the same 0.35
eV/Å on the same atoms as the drift -- against 20 for `neo` and 30 for `air`
at 2,744 atoms, and a 0.02 Å jitter off the ideal lattice does not soften
it. A parallel agent independently pinned the noise injection to a backward
scatter whose summand cancels analytically (a `-Q + Q` pair, the projected
radial gradient of the quaternion normalization), so its float32 relative
error is O(1); that fixed noise direction lands on the Jacobian's large
singular vector, which fixes the affected atoms and the magnitude.

Conclusion: the checkpoint's force Jacobian is pathological (curvature about
1e6 eV/Å^2 against a physical 10--100) on large periodic diamond cells --
a property of the trained weights, reachable by any backend at any precision
policy, and a real warning for production trajectories entering such
configurations. For this document's purpose: `air`'s force-parity column at
4,096 atoms compares noise amplified by 1e6 and is meaningless; its speed
numbers and every other checkpoint's parity stand. The lessons: repetition
against oneself separates nondeterminism from implementation deviation
before any cross-path comparison is trusted; a flaky failure needs ten runs
before a switch is credited; and when every implementation is "wrong the
same way", stop debugging code and measure the conditioning of the
mathematics.

______________________________________________________________________

## 8. What remains

Measured budget at 65.6 ms: convolution 32.6, dense GEMMs 11.9, Inductor
fragments 11.8, grid 3.7, boundary ATen 2.6, new operators 2.3, misc 2.2. The
2x gate needs 58.5, so 7.1 ms are missing. The items below are ordered by
measured headroom; none is cheap.

1. **The convolution is latency bound, not issue bound (est. 5--10 ms, and a
   redesign).** This is the finding that reframes the rest. At the production
   shape the kernel runs at IPC 0.43 against a peak of 1.0, with 33 %
   occupancy, and the stall profile is *flat*: wait 13.5 %, short scoreboard
   12.7 %, not selected 10.8 %, long scoreboard 10.3 %, barrier 10.2 %, MIO
   throttle 6.8 %. No single stall dominates, which is the signature of too few
   warps to cover latency rather than of a bad instruction mix. Occupancy is
   set entirely by registers -- `65536 / (128 * 32) = 16` warps -- so the only
   lever is the register budget, and sweeping the occupancy target directly
   (`OCC = 5`, 6) loses: the assembler spills more than the residency returns.
   Reaching a higher occupancy therefore needs the *tile* to shrink, which
   means a different work decomposition, not a different constant. The FFMA
   count is already exactly the arithmetic the model requires (1.081e11 against
   a 1.048e11 floor), so there is no waste left to remove -- only latency to
   hide.
1. **The equivariant per-degree linear layers (est. 4--6 ms).** 65 launches of
   `bmm` shaped `(9, N, 32) @ (9, 32, 32)` and wider, carrying 4 FLOP/byte and
   running at about 17 % of the achievable bandwidth. One edge- or node-parallel
   kernel over the degree axis would collapse the launch count and the layout
   copies around it. This is now the largest item after the convolution and is
   ordinary work, not a redesign.
1. **The remaining Inductor fragments (est. 2--3 ms).** What is left is glue
   around the scatter and the readout heads. Section 6 is the caution: only
   fuse the ones that own a large intermediate.
1. **Backward odds and ends (est. 1--1.5 ms).** Saving the gate sigmoid in the
   forward (skips one multiply per gated layer in the reverse sweep at 0.7 GB
   of extra traffic), and merging one of the two attention barriers per chunk.

Sum of the estimates: 12--20 ms, so the 2x gate is reachable, but the reliable
part of it is item 2 rather than item 1. 3x remains out of reach in fp32
by the arithmetic of section 2.

______________________________________________________________________

## 9. Files, tests, reproduction

Layout (`source/op/pt/dpa4/`):

- `so2_conv.cuh` -- layout algebra, device primitives, `ConvArgs`, `ConvTile`;
- `so2_conv_kernel.cuh` -- shared-memory plan, block multiply, chunk loaders,
  attention, forward and backward kernels;
- `so2_conv_launch.h` -- per-shape tiles (`TM`, `WARPS`, panel budget,
  occupancy target) and the instantiation list, degrees 1--6 at widths 32/64;
- `so2_conv_instantiate.cuh` + `so2_conv_{fwd,bwd}_c{32,64}_l{1..6}.cu` -- one
  translation unit per (direction, width, degree), 24 units, wall-clock build
  about 3.5 minutes at `-j 180` with `nvcc --threads`;
- `so2_conv.cu` -- validation, argument packing, shape dispatch, the reduction
  repack, the monomial kernel behind `dpa4_wigner_runs`, the device ridge
  query behind the routing gate, torch bindings;
- `wigner_dense.cu` -- the fused dense Wigner build, degrees 1--10;
- `grid_pair.cu` -- the fused grid pair product;
- `zonal_scatter.cu` -- the fused geometric initial embedding, degrees 1--6;
- `edge_radial.cu` -- the fused cutoff envelope and radial basis.

Python bindings in `deepmd/pt_expt/kernels/cuda/dpa4/`: `so2_conv.py` (the
run-table fit and its cache, the shared run build, autograd with the softmax
cotangents, the support and routing gates), `wigner_dense.py` (the per-element
fit and its cache), `grid_pair.py`, `zonal_scatter.py`, `edge_radial.py`.
Dispatch: the convolution in `deepmd/pt/model/descriptor/sezm_nn/so2.py`
behind `DP_CUDA_INFER >= 2` and the routing gate of section 3.7; the initial
embedding in `sezm_nn/embedding.py`, the radial chain and the dense Wigner
build (`fused_radial`, `fused_wigner`) through `sezm_nn/edge_cache.py`, all
behind `DP_CUDA_INFER >= 1`; the dense-build skip (`_wigner_free_conv`) plus
the zonal slice of the shared runs in `deepmd/pt/model/descriptor/sezm.py`.

The CSR views live on `EdgeFeatureCache` and are built by one function,
`edge_cache.cached_edge_csr`: the CUDA convolution and initial embedding, the
Triton flash aggregation and the rotate-mix backward all consume the same
sorted view as an explicit operator input rather than rebuilding it inside
their ops. The measured effect of that convergence was small (five sorts per
step instead of seven, under 0.2 ms); its value is the contract, one
construction site, and segment reductions that agree bitwise across backends.

Tests: `source/tests/pt/model/test_descriptor_sezm_cuda.py` -- dense-reference
forward and attention weights over the full zoo, tail chunks, autograd-driven
gradients including the competition scale, bitwise reproducibility, `make_fx`
tracing, the factory gates, and both directions of the initial embedding and the
radial chain at every degree and several channel widths.

Standalone tools kept in `debug/cuda_bench/`: `check_conv.py` (dense-reference
check and standalone timing of the fused convolution; `sweep_tile.sh` builds
one library per tile candidate and times through it), `compare_paths.py` with
`sezm_harness.py` (both paths in one process, element-wise force parity), and
`benchmark_zoo.sh` (end to end over the zoo, one GPU per model in parallel).
The per-operator checkers that accompanied development were folded into
`source/tests/pt/model/test_descriptor_sezm_cuda.py` and removed, as were the
one-off diagnostics of the section 7.1 investigation; the section records
their methods, which take minutes to rebuild when needed.

Reproduction:

```bash
# build, then publish the library the package actually loads
cd build/py37-none-linux_x86_64 && ninja deepmd_op_pt
cp op/pt/libdeepmd_op_pt.so $ENV/lib/python3.13/site-packages/deepmd/lib/

# correctness
pytest source/tests/pt/model/test_descriptor_sezm_cuda.py -q

# end to end, both paths in one process, element-wise force parity
DP_TRITON_INFER=3 DP_COMPILE_INFER=1 python debug/cuda_bench/compare_paths.py --atoms 8000
```

The copy is not optional: the editable install resolves
`deepmd.lib.libdeepmd_op_pt` from site-packages, so a kernel change measured
without it is measuring the previous build. Confirm with
`cuobjdump -elf <lib> | grep ConvTile` that the tile in the loaded library is
the one just built -- an entire round of "no end-to-end effect" in this session
was this mistake.

## 10. The second part: what the H20 changed

The whole study above was measured on one part. Repeating it on an H20 turned
one of its conclusions inside out, which is worth recording precisely because
nothing about the code was wrong.

### 10.1 The two parts are opposites

|                                    | RTX PRO 6000 Blackwell |        H20 |
| ---------------------------------- | ---------------------: | ---------: |
| multiprocessors                    |                    188 |         78 |
| shared memory per multiprocessor   |                 100 KB |     228 KB |
| registers per multiprocessor       |                   64 K |       64 K |
| float32 peak (`SMs * 256 * clock`) |            117 TFLOP/s | 40 TFLOP/s |
| float32 GEMM, measured             |           ~110 TFLOP/s | 32 TFLOP/s |
| memory bandwidth, measured         |               1.6 TB/s |  2.05 TB/s |
| float32 FLOP per byte              |                    ~69 |        ~16 |

A fused operator buys arithmetic intensity: it trades memory traffic for
keeping intermediates on chip. That trade is priced by the ratio in the last
row, and the two parts sit a factor of four apart. Everything below follows.

### 10.2 The convolution loses on the H20, and the arithmetic says so first

The prediction was made before the measurement. The convolution's floor is

```text
1.26 TFLOP / step / 32 TFLOP/s = 39 ms      (H20, against 10 ms on Blackwell)
```

and at the 27--34 % of peak the kernel reaches, that is 120--145 ms. Measured:
144.1 ms, against a 114.2 ms Triton baseline, i.e. **0.79x**. The prediction and
the measurement agree to a few percent.

The reason the baseline is so much better placed is in its kernel names:
`DP_TRITON_INFER=3` routes the mixing stack through `_stack_fp16x3_*`, which is
fp16 tensor-core arithmetic with split compensation. Its effective throughput is
roughly `148 / 3 = 49 TFLOP/s` against the 32 TFLOP/s of float32 SIMT, and
tensor cores reach a higher fraction of their peak. On the Blackwell part the
same comparison runs the other way: 117 TFLOP/s of float32 is more than the fp16
path can deliver after its threefold compensation, which is why the sweep that
selected the Triton configuration there chose the plain float32 stack.

So the break-even is a property of the part, and `float32 peak * 0.35` against
the ~28 TFLOP/s the Triton stack delivers puts it near **80 float32 TFLOP/s**.
That is the number the user documentation quotes.

### 10.3 What the split looks like, and the level design that follows

Splitting both profiles by span:

| span                                            | Triton on H20 | fused CUDA on H20 |
| ----------------------------------------------- | ------------: | ----------------: |
| the convolution (stack, rotations, aggregation) |        ~45 ms |             92 ms |
| everything else (grid, Wigner, glue fragments)  |        ~70 ms |            ~52 ms |

The convolution is the only span that loses, and it loses for a reason that
depends on the part; everything else is a memory-traffic win that holds
everywhere. `DP_CUDA_INFER` therefore splits along exactly that line: level 1
carries the always-profitable operators and level 2 adds the convolution.

The part dependence itself is resolved by the routing gate rather than left
to the user: the break-even arithmetic scales with the part's
fp32-peak-to-bandwidth ridge, which four attribute queries compute at
construction time with no architecture list and no micro-benchmark
(section 3.7). The H20 measures a ridge of 9.83 FLOP/byte against the 73.2 of
the calibration part, which shrinks the admission budget below every zoo
checkpoint -- exactly the measured sign of the 0.79x takeover -- so
`DP_CUDA_INFER=2` degenerates to level 1 there by itself.

The transparent Triton fallback carries only rolling state when no second
derivative is requested. `_stack_backward_traversal` uses one pre-activation
gradient scratch, one gate-logit scratch and two recurrence buffers; stacked
per-layer surfaces are reserved for the second-order path. This lifetime is
part of the compiled representation: mutating slices of a stacked inference
surface makes functionalization emit `select_scatter` clones of the whole
surface. On the RTX PRO 6000 profile those clones occupied 17.0 ms per Mini
step and 56.8 ms per Pro step. The rolling form removes them, taking Mini at
1,264,000 edges from 106.56 to 88.90 ms and Pro at 80,896 edges from 344.18
to 287.99 ms. The kernels remain `torch.library.triton_op` so Inductor can
still fuse graph-side pointwise work around them.

Level 1 across the zoo on the H20 after the dense Wigner fusion, one GPU per
model, `DP_TRITON_INFER=2` baseline:

| checkpoint | atoms | baseline | `DP_CUDA_INFER=1` |             peak |
| ---------- | ----: | -------: | ----------------: | ---------------: |
| `nano`     | 8,000 |  60.3 ms |   52.5 ms (1.15x) |  7.3 GiB (1.07x) |
| `mini`     | 8,000 | 129.5 ms |  113.9 ms (1.14x) | 12.2 GiB (1.26x) |
| `neo`      | 4,096 | 175.5 ms |  165.2 ms (1.06x) | 16.6 GiB (1.51x) |
| `air`      | 4,096 | 464.8 ms |  440.2 ms (1.06x) | 27.7 GiB (1.16x) |
| `plus`     | 1,728 | 443.5 ms |  429.1 ms (1.03x) | 20.6 GiB (1.32x) |
| `pro`      | 1,000 |  1009 ms |  979.3 ms (1.03x) | 46.9 GiB (1.03x) |

Force parity is 1e-5 eV/Å or better on every row, `air` included: a single
pair comparison on a flaky amplifier (section 7.1) can land in the benign
state, which this one did.

### 10.4 The tile policy needs no second table

The launch tile was swept again on the H20 across the space that the shared
memory of the part opens up (three times the panel budget, eight resident
blocks, edge tiles from two to eight):

| TM / budget / occupancy | forward + backward |
| ----------------------- | -----------------: |
| 4 / 1024 / 4 (shipping) |            54.3 ms |
| 4 / 1024 / 8            |            54.1 ms |
| 4 / 1024 / 6            |            55.3 ms |
| 8 / 4096 / 2            |            61.7 ms |
| 8 / 1024 / 4            |            63.1 ms |
| 4 / 4096 / 2            |            65.5 ms |
| 4 / 16384 / 2           |            65.9 ms |
| 2 / 1024 / 6            |            70.7 ms |

The shipping point is within half a percent of the best, and the ordering is the
same as on Blackwell: what the kernel is sensitive to is the resident-block
count, not the shared-memory footprint, so the extra 128 KB per multiprocessor
buys nothing. One policy serves both parts and no per-part table is needed.
`DPA4_TILE_TM`, `DPA4_TILE_WARPS`, `DPA4_TILE_BUDGET` and `DPA4_TILE_OCC` stay
available on the build line for the next part.

### 10.5 Lessons

1. **A fused operator has a break-even that belongs to the part, not to the
   code.** The convolution is the same instructions on both parts; only the
   ratio of float32 throughput to what the alternative path can reach decides
   whether fusing pays. Before porting a fusion, price it against the peak of
   the target, not against the previous measurement.
1. **Read the baseline's kernel names on every new part.** The 0.79x was fully
   explained by four kernels whose names contain `fp16x3`. The same
   `DP_TRITON_INFER=3` selects different arithmetic on different parts, so
   "the same baseline" is not the same baseline.
1. **Split operators by what their profit depends on.** Traffic-bound fusions
   and arithmetic-bound fusions belong at different levels; bundling them forces
   a part that would gain from the first to pay for the second.

______________________________________________________________________

## 11. The training path: the fused value-path operator

Everything above this section serves inference. Training runs a different
regime -- parameter gradients, autograd-visible backward, and a force loss
that differentiates the backward again -- and is served by one operator
triple in `source/op/pt/dpa4/so2_conv_train.cu`, bound per convolution
block by `deepmd/pt_expt/kernels/cuda/dpa4/so2_conv_train.py`. The operator
spans the value stream of one `SO2Convolution` up to the attention
aggregation: rotation, radial degree mixing, competition weight, gated
mixing stack, final identity layer. The attention span (query-key logits
with the radial bias, the envelope-gated destination softmax with its null
mass, the weighted rotate-back aggregation, the output-side head gate)
runs as the Triton operator composition inside the traced graph, together
with the node-level projections -- a fused CUDA form of that span was
built, verified, measured slower at equal memory, and removed (section
12). The production operating point is therefore `DP_CUDA_TRAIN=1 DP_TRITON_TRAIN=1`: the CUDA operator owns the value stream, the Triton
level-1 operators own the attention and force-assembly segments.

### 11.1 Scope and gating

`DP_CUDA_TRAIN=1` (read once at model construction,
`deepmd.pt_expt.kernels.utils.cuda_train_enabled`) binds the fused value
path on every `SO2Convolution` matching the supported configuration:
`mmax` 1, degree 1--6, gated stack with an identity final layer, radial
mixer absent or `degree_channel` with rank at most 4, at most 256 wide
channels for degrees 1--5 or 384 wide channels at degree 6, and at most 4
focus streams. Unsupported blocks keep the narrower paths. The degree-six
width covers both Ultra layouts, `n_focus=4, channels=96` and
`n_focus=3, channels=128`.

A real competition norm (the focus entry of `edge_norm`) runs inside the
operator like the rest of the head: its learnable per-focus scales enter as
the `norm_scale` input and follow the same input/gradient/second-gradient
pattern as the projection weight. The forward folds the scales into the
head projection and rescales the logit by the inverse RMS of the gate row,
accumulated in the same lane-strided pass; the backward pushes the logit
gradient through the norm's Jacobian
`J_j = r s_j w_j - (r^3 S / cf) g_j` in the same per-edge kernel; and the
second order adds the closed-form curvature — the logit cotangent
`s = <hgg, J>`, the scale and weight curvature through a shared bracket
vector, and the Jacobian's own gate dependence, whose Hessian term is
mapped onto the rotation operands by one extra rotation backward. The
identity norm takes the original code paths unchanged (the dev checker's
output is byte-identical across the extension).

The value-stream dispatch in `so2_message` is mutually exclusive by
construction: a step runs either this operator or the Triton value-path
composition (`DP_TRITON_TRAIN=1` alone, `dpa4_triton.md`), never a mixture
within the value stream. The Triton composition remains the numerically
verified reference this operator is checked against; the dense
compiler-scheduled path is the base reference under both.

The operator serves both backends. On `pt` the binding sits inline in
`sezm_nn/so2.py`; on `pt_expt` the module subclasses the array-API `dpmodel`
implementation, so the binding target is a hook `dpmodel` declares
(`_cuda_value_train`) and the dispatch is the `training`-mode branch of
`dpmodel`'s own `so2_message`. The array-API reference leaves the hook
unbound and takes the dense branch, so nothing about `dpmodel` changes when
the operator is absent. The factory recognizes either spelling of the
identity competition norm — `nn.Identity` on `pt`, an unbound `None` hook
on `dpmodel` — and the shared `ScalarRMSNorm`, whose `adam_scale` parameter
feeds the operator's `norm_scale` input; any other norm module has no
closed form in the operator and keeps the block on the narrower paths.

When all blocks bind this value operator and the Triton training attention,
the descriptor sets `_packed_wigner_train`. The common
`dpmodel._build_full_wigner()` decision then omits the dense `(E, D, D)`
Wigner pair, and both `pt` and `pt_expt` obtain GIE zonal coupling through
`_shared_wigner_runs()` from the operator's packed edge-run cache. This is a
descriptor-level part of the training integration rather than a kernel
special case: unsupported blocks leave the predicate false and retain the
dense reference construction.

### 11.2 The operator triple

Three operators cover the three differentiation depths of a force loss:

- **`sezm_so2_value_fwd`** -- one resident tile kernel: the gather into
  the edge frame over the structural block-diagonal non-zeros of the
  Wigner-D matrix, the edge-conditioned radial degree mixing, the
  cross-focus competition weight from the `l = 0` scalars, every gated
  mixing layer, and the final identity layer with its edge-major store.
  The rotated input and all inter-layer activations live in shared memory
  for the lifetime of a block; the only global surfaces are the local
  activation and the backward anchors (the stacked pre-activations
  `z_all`, the final gated activation `u_final`, and the competition
  weight). Accumulators are typed `acc_type<scalar_t>` (float for
  float/bf16/half inputs, double for double), so the fp64 validation runs
  carry full double precision through the same kernels. Shapes whose tile
  would drop below eight edges bypass the resident kernel onto a C++
  composition with identical anchor layouts (section 11.7, item 3).
- **`sezm_so2_value_bwd`** -- the first order as one operator call: the
  rotated input is recomputed by the fused rotate-mix forward (never
  stored), the mixing traversal runs (`dpa4_sezm::mixing_bwd`) with its
  weight contractions gated by `with_weights` (section 12, item 2), the
  competition head is differentiated in closed form from the stored
  weight, and the rotation gradients flow through the fused rotation
  backward with the contention-free CSR segment reduction. The
  channel-basis gradient is not a separate pass: the rotation backward
  kernel accumulates the per-edge basis partials alongside its Phase-2
  register traversal (every operand is already resident) and the host
  reduces them in one sum. Under ambient grad mode (`keep_state`) the
  traversal retains its per-layer surfaces and the total input gradient as
  trailing outputs; the second order replays nothing.
- **`sezm_so2_value_bwd2`** -- the analytic second order of the force-loss
  regime, one operator call (section 11.4).

The constituent traversals live in named namespaces (`dpa4_sezm::mixing_*`
in `mixing_train.cu`, `dpa4_sezm::rotate_mix_*` and `segment_sum_csr` in
`rotate_mix_train.cu`, shared header `sezm_train_ops.cuh`) and are composed
as plain C++ calls; nothing below the three value-path operators is
exposed to the dispatcher. In the mixing traversals the host loops live in
C++, and cuBLASLt represents each block output with `ROW` as its physical
leading dimension. The GEMMs therefore write the `m0` and `m1` slices of the
`(F, E, ROW)` buffers directly; routing the same slices through ATen would
materialize contiguous outputs and copy them back. The elementwise bodies are
CUDA kernels at the DRAM roof (the backward gate kernel measures 93% DRAM
utilization). The measured second-order mixing chain on the Pro shape is
10.4 ms per convolution block against 18.1 ms for the same mathematics
composed through per-operator autograd formulas with an eager replay.

The traversals are not registered as Python-visible operators: the training
path exposes exactly the three operators above, and the historical per-piece
dispatch (`sezm_mixing_*`, `sezm_rotate_mix_*`, `sezm_gated_act_*` as
`torch.ops` entries with Python-side routing) was removed outright -- the
piecewise form pays one dispatcher round-trip and one autograd node per
piece, and its second order re-enters Python between every pair.

### 11.3 Weight gradients through cublasLt

The contraction `W_bar = u^T z_bar` reduces over the edge count, which dwarfs
the output tile. `lt_weight_grad` queries the cublasLt heuristic once per
shape on a dedicated handle with a shared 32 MB workspace, times the top
eight candidates once on the live operands, caches only the fastest heuristic
result, and contracts directly on the strided column blocks. Matrix layouts
and operation descriptors are local to one call and are destroyed after the
launch, so a run with changing edge counts does not retain one CUDA descriptor
plan per encountered batch. The heuristic's candidates in this regime are
split-K algorithms close to the traffic bound, where the framework's own
entry -- its handle coupled to the framework workspace budget -- degrades to
single-K kernels at ~3x the time. The `nvidia-cublas`
wheel must be at least the 13.6 series: the 13.1 wheels return no split-K
candidate for these shapes on `sm_120`. Double inputs bypass the path onto
`at::bmm` -- the Lt descriptors are fp32-compute, and reinterpreting double
memory as fp32 produced NaN weight gradients in the fp64 validation runs
until the guard was added.

### 11.4 The analytic second order

The force-loss regime sends cotangents into the backward's coordinate-graph
outputs: the node-feature gradient, the Wigner gradient and the
degree-kernel gradient (their producers precede the operator on the
coordinate graph, so a force loss reaches all three). The parameter
gradients feed the optimizer and carry none; the operator raises on any
other cotangent rather than silently mis-differentiating.

The second-order operator evaluates the curvature against those cotangents,
linearized around the kept first-order surfaces (nothing replays), and
additionally returns the cotangents of the backward's *anchor inputs* --
the local activation (the head's `ga` route), the pre-activations (the
nonlinearity derivatives are evaluated at `z`), and the competition weight
(the scale and the `ga` divisor); the final activation carries none, since
the force-regime first order never reads it (its weight contractions are
skipped and the alpha gradient contracts against the stored output).
Autograd routes those anchor cotangents back through the forward's own
output slots, which re-enters the first-order operator once more with the
`h_z`/`h_alpha` upstream slots populated -- the anchor re-entry that closes
the stored surfaces' dependence on the inputs. The mathematics inside:

1. **Rotation multilinearity.** The rotation backward is multilinear in
   `(grad_u0, x, wigner, kernel, basis)`, so its second order re-enters the
   existing forward and backward kernels with each cotangent placed in the
   slot of the operand it differentiates: the upstream cotangent of `grad_u0`
   collects one forward re-entry per differentiated output (`h_e` in the
   feature slot, `h_gwig` in the Wigner slot, `h_gkc` in the kernel slot),
   and each backward re-entry contributes the curvature of the two operands
   it does not occupy plus the channel basis.
1. **Mixing traversal.** `dpa4_sezm::mixing_bwd2` walks the hand-derived
   adjoint of the whole-stack backward in one call, exactly as in the Triton
   document's fifth form.
1. **Competition head, closed form.** The first order reads the softmax off
   the stored weight, `p = (alpha - ls/F) / (1 - ls)`, takes the traversal's
   alpha gradient `ga = <grad_out, x_local> / alpha`, and emits
   `gl = p (ga - <ga, p>) / tau` with the gate-slice gradient
   `g_gate = gl w_fc^T`. The second order is that map's exact VJP: onto
   `w_fc` directly, onto `(grad_out, x_local, alpha)` through `ga`, and onto
   the `alpha` anchor again through `p`. The head chain divides by `alpha`
   (as small as `ls/F`), so it runs in double accumulators -- the tensors
   are `(E, F)` scalars and one `(E, F, Cf)` slice, and the cost is
   unmeasurable.

**The second order differentiates the backward's computation graph, not an
equivalent formula.** This is the sharpest lesson of the chain. Two
mathematically equivalent forms of the same quantity distribute curvature
over *different* input slots of the operator, and autograd composes slots,
not mathematics. The first order reads `p` off the stored `alpha`; a first
draft of the second order linearized a softmax recomputed from `u0` instead,
which moved the head's curvature from the `alpha` slot (where autograd routes
it back through the forward's own head, reaching `u0`, `w_fc` and the bias
along the true graph) into the operator's `x` and bias slots -- every
individual formula verified, the total wrong by a factor of two. The same
applies to the traversal's alpha gradient: the kernel computes
`<grad_out, x_local> / alpha`, so the curvature lands on `(grad_out, x_local, alpha)`; an equivalent `<grad_out, u_final + u_final W>` form lands
it on `(grad_out, u_final, W)` -- correct only if the first order actually
computed it that way, which it does not. The same lesson fixes the anchor
cotangents' shape: the alpha gradient contracts `<grad_out, x_local> / alpha`, so the second order lands curvature on `(grad_out, x_local, alpha)` and the `u_final` slot stays empty -- an equivalent
`<grad_out, u_final + u_final W>` form would move it, and would be wrong
for *this* backward. The finite-difference contract of section 11.5 pins
exactly this: every output slot of `bwd2` must equal the finite difference
of `bwd` in that input slot, with anchors held fixed.

### 11.5 Verification

Four layers, each catching what the previous cannot:

0. **Unit tests** (`source/tests/pt_expt/kernels/test_so2_value_train.py`):
   the same operator-against-reference arbitration as the private harness
   below, but as a committed test over the four deployed block shapes, in
   ambient fp32, under bf16 autocast, and in fp64. The judgment rule lives in
   `conditioning.py` and is worth stating, because getting it wrong hid a
   real defect for a while: the bound on the fused error is *only* a multiple
   of the eager reference's own distance from the fp64 truth. An operator-
   specific absolute tolerance -- the private harness had `5e-2` under bf16 --
   records one machine's rounding and silently absorbs a precision
   regression; the competition-weight anchor defect above sat underneath that
   floor until it was removed. The one admitted absolute term is a single
   rounding of the working format, for quantities where the eager side comes
   out exact and a multiple of zero would reject any reduction order at all.
   The verdict is taken on the median over three independent operand draws:
   both errors are extremes over a tensor, and for a two-entry per-focus
   gradient a single draw's ratio swings between 0.17 and 8.4 with no trend
   in the operand count -- small-sample noise that a single draw would report
   as a failure.

1. **Operator parity** (`debug/train_bench/check_so2_value_cuda.py`): the
   fused operator against the eager reference composition (rotate-mix,
   competition head, mixing stack) on identical inputs -- forward, all
   nine first-order gradients, and the force-regime second order with
   cotangents on every coordinate-graph gradient (node features, Wigner,
   degree kernel). Every case runs as an fp64 pair (both implementations
   in double -- separates logic errors from conditioning; the pair is
   judged at the reduction-order ceiling `5e-6`, which the depth of the
   second-order chain sets), an fp32 pair against the fp64 ground truth,
   and a bf16-autocast pair. The Wigner comparisons are masked to the
   structural block diagonal the kernels read.

1. **Finite-difference arbitration** (`debug/train_bench/arb_wfc2.py`): the
   force-style scalar `L2 = <dL1/dx, h_x>` differentiated by central
   differences against the parameter curvatures the eager pair cannot
   arbitrate on its own -- the competition head and the channel basis --
   compared for both the eager graph and the fused operator. The fp64 case
   pins the fused curvatures to the finite difference at `~1e-7`.

1. **End-to-end** (`debug/train_bench/verify_train.py --path cuda`): one real
   training step (force loss) on a production configuration, dense and fused
   runs sharing one model instance, one weight set and one batch; every
   parameter gradient must sit within the dense path's own run-to-run noise
   floor (atomics reduction order), measured by running the dense step twice.
   Passes on Neo in eager fp32 (loss bit-identical to the dense step) and
   under bf16 AMP with `torch.compile`.

### 11.6 Dead ends and defects worth remembering

- **`keep_surfaces`, removed and then reinstated as `keep_state`.** The
  first-order mixing traversal can retain its per-layer adjoint heads,
  pre-activation gradients and gate-logit gradients for the second order to
  linearize around, sparing the replay. The piecewise-dispatch version was
  removed together with that dispatch, on the judgment that three stacked
  edge-size surfaces held across the force graph cost more memory than the
  replay costs compute. Once the chain was otherwise optimized the replay
  was exactly the remaining gap to the Triton composition (one full mixing
  traversal per second-order call), and the trade re-decided itself the
  other way: under ambient grad mode the first-order operator now runs the
  traversal with `keep_state`, returns the surfaces and the total input
  gradient as trailing outputs, and the second order replays nothing. The
  memory price is a fraction of what the fused chain saves elsewhere
  (Pro: +0.65 GiB against a 10 GiB saving; the surfaces are bf16). The
  lesson is that a compute-versus-memory trade is not a fixed decision but
  a function of everything else in the budget, and is worth re-measuring
  after each structural change.
- **Standalone CUDA gated-activation bodies (removed).** Hand-written
  elementwise bodies for the batched-matmul regime of the gated activation
  (one thread per site, register footprint independent of the channel
  width) beat every Triton tiling at the kernel level (first order 163 us
  against a swept 252 us on the Pro shape, second order 289 against 421 us)
  yet lost a few percent end to end: the operator boundary materializes
  saved tensors and gradient surfaces that the compiler-fused dense
  expression shares with the surrounding graph. Kernel speed does not
  transfer across an operator boundary unless it also beats that boundary's
  materialization cost. The operator was removed outright once its
  pointwise bodies had been absorbed into the mixing traversal's kernels
  (`mixing_train.cu`).
- **Double reinterpreted as fp32 in cublasLt** (section 11.3): silent NaN in
  the fp64 gold runs; dtype guards on every raw-pointer library call.
- **Slot-distribution bug in the second order** (section 11.4): every
  constituent formula correct, the composition wrong; only the
  finite-difference contract caught it.
- **The saturated-softmax edge.** At `alpha = ls/F` exactly (a focus stream
  the softmax has fully suppressed), the reconstruction
  `p = (alpha - ls/F)/(1 - ls)` sits on the `clamp` kink and the analytic
  subgradient differs from a finite difference straddling it. The
  disagreement is confined to `p = 0` points where `gl = 0` anyway; it is a
  property of the reconstruction, not an error, and the checks exclude the
  measure-zero boundary.
- **The head's conditioning under fp32.** The `1/alpha` chain amplifies the
  kernels' fp32 rounding by roughly a decade against the eager graph on the
  bias curvature (2e-4 relative against 1.4e-5); the fp64 pair is bitwise
  clean, so this is conditioning, not logic. The head now runs in double
  (11.4); the residual amplification through the fp32 anchors themselves is
  accepted and reflected in the check tolerances.
- **The competition weight was stored in working precision.** The head's
  whole backward hangs off one anchor: it reconstructs the softmax as
  `p = (alpha - ls/F)/(1 - ls)` and divides the traversal's weight gradient
  by `alpha`. Promoting that chain to double (above) does not recover bits
  the anchor never carried, and under bf16 AMP the anchor carried three
  decimal digits: replaying the closed-form chain on a bf16 round trip of an
  exact `alpha` costs `2.8e-3` relative, against `2.0e-7` for an fp32 round
  trip. The anchor and its gradient are now carried in accumulator precision
  (`dpa4_sezm::alpha_dtype`, mirrored by `_alpha_dtype` on the binding side)
  independent of the surfaces around them -- they are `(E, F)` scalars, so
  the storage is nothing next to the `(E, F, ROW)` surfaces, while every
  `(E, F, ROW)` product against them stays in working precision. Measured
  effect on the bias gradient under bf16: `8.1e-2` to `5.7e-2` relative,
  the remainder being the shared bf16 surfaces. The general lesson is that a
  reduced-precision anchor is not equivalent to a reduced-precision surface:
  an anchor a closed form divides by and reconstructs from deserves the
  accumulator width regardless of the ambient AMP regime.
- **The competition norm was first supported through an external seam
  (removed).** The norm support initially kept part of the head in the
  caller's autograd graph to avoid deriving the norm's closed form: first
  the whole head, passing the finished weight into the operator, then --
  after Inductor fused that softmax island's forward, backward and norm
  backward into one generated kernel that Triton's
  `tritongpu-remove-layout-conversions` pass miscompiles on `sm_120`
  ("operand does not dominate this use") -- only the gate, norm and linear
  layer, passing pre-temperature logits. The logits seam compiled, but the
  seam itself cost five milliseconds of host time per Neo step against the
  film-only baseline at equal device time: a larger traced graph, more
  kernel launches, and eager ATen chains where the internal head runs one
  kernel. Internalizing the norm removed the seam outright -- the scales
  are one more head parameter with the same input/gradient/second-gradient
  pattern as the projection weight, the derivation is a page of closed
  form, and the step time returned to the film-only level. The lesson is
  that an external-evaluation seam is not a free alternative to a
  derivation: it trades a bounded, verifiable page of calculus for a
  standing tax in graph surface, launches and host time, and for exposure
  to whatever the compiler does with the exported subgraph.

### 11.7 Kernel engineering of the training chain

The first profile of the chain (Neo, one RTX PRO 6000) put the forward mega
kernel at 15.6% of the device time and the rotation backward at 9.5% across
ten calls; the sections below record what was done about it, in order, with
the measured outcome of each step. Durations are per step on the Neo shape
(13,930 edges) unless stated.

1. **Edge tiling of the forward kernel (2.80 -> 1.94 ms).** The per-edge
   work is a chain of vector-matrix products against shared weights; one
   block per edge re-reads every weight column from L2 once per FMA (half a
   FLOP per byte), which Nsight Compute showed as L2 at 71% with FP32 at 6%
   of peak. Tiling eight edges per block (`TE` template parameter, chosen at
   launch from the shared-memory budget) amortizes each weight read over
   eight register accumulators: L2 dropped to 13.5%. The remaining gap to
   the FP32 roof is latency at two resident blocks per multiprocessor; the
   shared-memory footprint, not the register count, caps the residency.
1. **Bank-offset padding and partial unrolling (1.94 -> 1.78 ms).** One
   padding word per tile slot moves the eight-slot inner reads off a single
   bank (the row width is a multiple of the bank count), and `#pragma unroll 4` on the contraction loops lets the compiler batch the weight
   loads. Both were minor: the warp-level reads are broadcasts, so the
   conflict hypothesis was largely wrong, and the measured gain came from
   the load batching.
1. **Wide shapes bypass the resident kernel entirely.** On Pro
   (`row_w` 1024, F 2) the activation footprint forces the tile width to
   four and the residency to one block per multiprocessor, and the
   plain-FMA interior cannot approach the tensor-core GEMMs: 17.6 ms per
   call against a 1.4 ms cuBLAS-composition equivalent (105.9 ms of a
   276 ms step). Shapes whose tile would drop below eight edges now run the
   identical value stream as a C++ composition -- rotation kernel,
   closed-form competition head, cuBLAS-backed mixing traversal -- with
   identical anchor layouts, so the backward is oblivious to the choice:
   Pro fell to 178 ms. Double inputs stay on the resident kernel, whose
   accumulators follow the input precision (the parity gold standard).
1. **The second order shares the replay (178 -> 172 ms on Pro).**
   `mixing_bwd2` already replays the first-order traversal for its
   linearization points; the replayed input gradient now rides out as a
   trailing output, and the operator-level second order consumes it instead
   of running its own weightless `mixing_bwd`. One mixing traversal per
   second-order call disappeared.
1. **The rotation re-entries merged (172 -> 165.5 ms on Pro).** The second
   order used to launch four forward re-entries (u0, node-, Wigner- and
   kernel-cotangent routes) and three backward re-entries per convolution.
   Multilinearity makes every term landing on the same output linear in the
   rotated lanes, so the lanes sum *before* the contractions: one paired
   forward kernel now produces u0 and the full upstream cotangent (the node
   cotangent gathered onto edges in place, sparing the materialized
   `index_select`), and one curvature kernel evaluates all three backward
   re-entries against the shared upstream.
1. **Batched block reductions (Pro value-path total -12.8 ms).** The
   rotation backwards reduced every kernel-gradient and Wigner-gradient
   scalar with a two-barrier block reduction -- over a hundred
   `__syncthreads` per edge at high degree. Each output slot now takes a
   warp-level partial into a per-(slot, warp) scratch, and a single barrier
   precedes a parallel write-out phase; the kernel-gradient slots map
   linearly onto the compact kernel layout, and the Wigner slots follow the
   block-diagonal enumeration inverted in closed form at write-out.
   `rotate_mix_bwd` fell from 1.08 to 0.58 ms per call on Pro.
1. **Transposed weights stay views (neutral).** The backward operators
   materialized `W^T` per call (54 copies per step on Pro); the consumers
   are batched matmuls, which take the strided transpose views directly.
   The copies disappeared but cuBLAS serves the transposed operand with a
   slower kernel selection, and the net effect was within noise -- kept for
   the allocation traffic it removes, not for speed.
1. **Gate projections stay as logits until their pointwise consumers.** The
   forward, backward and second-order gate kernels evaluate sigmoid while
   loading each logit, removing 72 gate-sized launches per Pro step. On the
   396-atom, 13,930-edge, 22-frame workload this takes the median from 111.525
   to 108.160 ms without increasing any of the three consumer times.
1. **Upstream cotangents join the traversal in their consumers.** The reverse
   gate merges `grad_u_up` into the retained recurrence head and `grad_z_up`
   into its pre-activation gradient; the second-order gate writes the
   `grad_gz_up = hgz` route while consuming `hgz`. The redundant full-head
   clone is replaced by ownership transfer. Eighteen launches disappear and
   the same workload measures 106.824 ms.
1. **Block GEMMs write their strided destinations through cuBLASLt.** An ATen
   `bmm_out` whose destination is an `m0` or `m1` slice computes into a
   contiguous temporary and copies it into the `ROW`-strided view. The Pro
   profile contained 48 copies of `(2, E, 384)` and 48 of `(2, E, 640)` per
   step. Explicit row-major cuBLASLt layouts remove all 96 copies. The strided
   GEMMs cost about 0.8 ms more than their contiguous forms, while the removed
   copies cost about 2.1 ms; the median reaches 105.518 ms, 5.39% below the
   111.525 ms starting point, with peak memory moving from 9.13 to 9.15 GiB.

The maintained training templates live under `rotate_mix_train/` and
`so2_conv_train/`, separate from their thin operator hosts. CMake generates one
translation unit per `(lmax, radial rank, dtype)` for the rotation grid and one
per `(lmax, dtype)` for the resident value-path forward. Ninja can therefore
compile independent grid points concurrently without duplicating maintained
kernel source. The rotation grid instantiates float32, float64 and bfloat16;
unused float16 rotation templates are deliberately absent.

The final operating point is the composition
`DP_CUDA_TRAIN=1 DP_TRITON_TRAIN=1`: CUDA owns the value stream, while the
Triton level-1 operators remain visible to Inductor around the attention and
force-assembly segments. On one RTX PRO 6000 Blackwell with bf16 AMP,
`torch.compile`, the default `DP_FUSION_SIZE=8`, 30 warmup steps and 60 timed
steps, the untuned (`DP_TUNE_TRAIN=0`) large-model results are:

| shape | backend   | workload | base (ms/GiB)    | CUDA + Triton (ms/GiB) | speedup | memory factor |
| ----- | --------- | -------- | ---------------- | ---------------------- | ------- | ------------- |
| Plus  | `pt`      | 396a/22f | 47.434 / 6.275   | 36.846 / 3.134         | 1.29x   | 0.50x         |
| Plus  | `pt_expt` | 376a/9f  | 51.392 / 7.021   | 38.135 / 3.191         | 1.35x   | 0.45x         |
| Pro   | `pt`      | 396a/22f | 156.471 / 20.594 | 92.954 / 8.453         | 1.68x   | 0.41x         |
| Pro   | `pt_expt` | 376a/9f  | 160.254 / 21.908 | 95.015 / 8.482         | 1.69x   | 0.39x         |
| Max   | `pt`      | 396a/22f | 444.225 / 51.660 | 274.764 / 20.392       | 1.62x   | 0.39x         |
| Max   | `pt_expt` | 376a/9f  | 445.428 / 54.493 | 274.222 / 20.346       | 1.62x   | 0.37x         |
| Ultra | `pt`      | 192a/16f | 477.485 / 55.500 | 301.198 / 19.603       | 1.59x   | 0.35x         |
| Ultra | `pt_expt` | 198a/9f  | 410.531 / 54.698 | 255.935 / 16.990       | 1.60x   | 0.31x         |

The result is 1.29--1.69x faster at 0.31--0.50x of baseline peak memory. The
memory saving comes from two sides: the CUDA chain keeps the value stream's
inter-layer activations out of global memory, and the Triton attention path
avoids retaining the dense fp32 softmax surfaces. The largest remaining block
above the mixing GEMM floor is outside this value-path operator -- attention
backward, the radial network, fitting and loss.

Small models require a workload-qualified decision. With `mix:3200`, the same
matrix produces:

| shape | backend   | workload   | base (ms/GiB)    | CUDA + Triton (ms/GiB) | tuned CUDA + Triton (ms/GiB) |
| ----- | --------- | ---------- | ---------------- | ---------------------- | ---------------------------- |
| Nano  | `pt`      | 3192a/57f  | 26.746 / 3.481   | 22.875 / 2.313         | 20.363 / 2.624               |
| Nano  | `pt_expt` | 3171a/187f | 28.521 / 4.015   | 23.268 / 2.693         | 23.203 / 2.939               |
| Mini  | `pt`      | 3192a/57f  | 51.825 / 7.320   | 36.273 / 4.181         | 35.199 / 4.661               |
| Mini  | `pt_expt` | 3171a/187f | 57.094 / 8.710   | 38.131 / 4.806         | 37.603 / 5.240               |
| Neo   | `pt`      | 3192a/57f  | 128.673 / 17.437 | 89.937 / 9.780         | 89.363 / 10.230              |
| Neo   | `pt_expt` | 3171a/187f | 139.119 / 20.162 | 91.509 / 10.509        | 90.490 / 10.606              |
| Air   | `pt`      | 3192a/57f  | 224.563 / 28.529 | 148.981 / 18.054       | 147.148 / 18.900             |
| Air   | `pt_expt` | 3171a/187f | 235.391 / 32.671 | 148.013 / 19.172       | 147.371 / 19.510             |

Untuned CUDA plus Triton is already 1.17--1.59x faster and uses
0.52--0.67x of baseline memory at this atom budget. `DP_TUNE_TRAIN=2` adds
only 0.3--3.1% on every row except `pt` Nano (12.3%), raises peak memory in
every row, and imposes a long first compilation. It is therefore a
workload-specific option rather than part of the CUDA path's default. The full
small-budget and Triton-only comparison is recorded in `dpa4_triton.md` §6.7.

## 12. The whole-convolution experiment: optimization log and removal

The attention span was once folded into the training operator, making one
operator triple span the whole convolution (value stream and attention,
with analytic backward and second order -- verified to the same three-layer
standard as section 11.5, including finite-difference arbitration of the
logit-weight, null-logit and head-gate curvatures). It started measurably
*slower* than the composition it replaced -- Neo 26.5 against 21.7 ms per
step, Pro 144 against 121 -- and this section records the analysis, every
optimization step taken from there, and the terminal decision: the value
chain kept every optimization, the fused attention span was removed, and
the attention returned to the Triton composition (section 12.5). Per-call
numbers are from an operator micro-benchmark on the production shapes --
one batch of the shared test box, N = 396 nodes and E = 13930 edges, so
the per-node CSR segments average 35 edges -- timing the forward, the
first-order backward and the second order in isolation. This iterates in
seconds against minutes for an end-to-end step, and the isolation
separates kernel time from graph effects.

### 12.1 Where the time actually was

The production shapes per model size (F = 2 focus streams, H = 1 head):

| size | L   | layers | rank | Cf  | blocks | fwd  | bwd    | bwd2 (ms/call) |
| ---- | --- | ------ | ---- | --- | ------ | ---- | ------ | -------------- |
| Neo  | 3   | 3      | 1    | 32  | 2      | 0.92 | 3.77\* | 1.95\*         |
| Pro  | 5   | 4      | 2    | 64  | 6      | 1.37 | 3.77   | 8.85           |
| Max  | 6   | 4      | 2    | 96  | 8      | 3.06 | ~8     | ~20            |

(\*starting points where measured; the backward runs twice per step, once
for the force pass and once for the energy term.)

Three early hypotheses failed in order. Host launch overhead: the fused
attention span did add ~480 launches per step of eager ATen glue, and
folding the gate broadcast, the head reductions and the softmax
accumulation into the kernels recovered ~3 ms on Neo -- but Pro did not
move, because Pro's per-call wall time equals its CUDA time: the operator
is GPU-bound. Register-array pressure in the attention backward kernels
(the (DIM,) upstream arrays): streaming them per degree block from L2
freed the residency but moved nothing, because the attention span is ~9%
of the backward pair. Segment-scan serialization in the softmax kernels:
parallelizing the scans across the block was correct hygiene but the
segments average 35 edges and the scans were never hot.

What the kernel-level breakdown of one Pro (bwd + bwd2) pair (12.8 ms)
actually showed: the split-K weight-gradient GEMMs 1.74 ms, the rotation
backward 1.33 ms *because it ran twice* (once in the backward, once
replayed by the second order's anchor re-entry), the mixing gate kernels
1.32 ms (six calls -- the re-entry walks the layers again), the dX GEMM
chain ~3 ms, and 0.68 ms of elementwise glue additions. The attention
span: ~1.1 ms. The roofline standing of the pieces: the GEMMs run at
26-40% of the bf16 tensor-core roof (the shapes are K = 384/640 slabs at
E = 14k, no amount of scheduling turns them into large tiles), the gate
kernels at 72% of DRAM, and the rotation backward at 67% SM / 17% DRAM
with 16% occupancy -- a compute-latency problem, not a bandwidth one.

### 12.2 What was done, in order, with measured effect

1. **Second upstream in the rotation curvature kernel** (Pro bwd2
   8.85 -> 7.87 ms; removed with the fused span). With the anchor
   re-entry internal to one operator, it used to run a full
   `rotate_mix_bwd` traversal of its own plus a rotated-input recompute
   and a second CSR node reduction. The rotation backward's local gradient
   through the stored kernel projects through the stored Wigner exactly
   like the kernel-cotangent route already resident in
   `rotate_mix_bwd2_kernel`, so a second upstream folded into that route's
   accumulators at marginal register cost: one traversal served the
   curvature and the anchors' plain backward. The fold is only reachable
   when the re-entry lives inside the operator; under the value-only form
   the re-entry is an autograd-driven second `bwd` call whose upstream is
   produced outside the operator, so the second upstream was removed with
   the span rather than left as a dead lane in the kernel's instruction
   stream (the `gb` FMAs execute whether or not the pointer is null).
1. **Parameter gradients on demand** (Pro bwd 3.77 -> 2.98 ms for the
   force pass; Neo 0.84 -> 0.59; Max 8.3 -> 6.6). The force pass
   (`autograd.grad(E, coord)`) differentiates the coordinate chain alone,
   yet the backward unconditionally ran every weight contraction -- eleven
   split-K GEMMs and the attention parameter contractions per call, all
   discarded. The backward now takes `with_weights` from
   `ctx.needs_input_grad`; the energy-term backward still runs them. The
   competition-weight gradient stays unconditional (its gate-slice term
   enters the input gradient), which also retired a historical trap where
   the weightless traversal returned a zero-shaped competition gradient.
1. **Launch bounds on the rotation backward** (Pro force-bwd
   2.98 -> 2.82 ms). Unconstrained the kernel sits at ~165 registers and
   one resident block; `__launch_bounds__(256, 2)` trades a modest spill
   (absorbed by the 17%-utilized DRAM path) for doubled latency cover.
   The same bound on the second-order rotation and attention kernels moved
   nothing (their spill cancels the residency), and on the paired forward
   it is kept for uniformity.
1. **Retention snapshots written in place** (Max force-bwd −0.4 ms). The
   backward used to copy each layer head into its retention slot before
   updating it; the out-of-place residual contraction
   (`baddbmm_out(upstream[l-1], g_l, gz, W^T)`) now produces the next head
   directly in the next slot, so the three device-to-device copies per
   call vanish.
1. **Anchor curvature seeded into the kernels' stores** (Max bwd2
   18.2 -> 17.4 ms with the empty-branch fixes). The head curvature on the
   upstream gradient seeds the second-order final kernel's store
   (`ggout_init` in `mixing_bwd2`) instead of a separate addition pass;
   the unused `grad_u_final`/`grad_u0` zero surfaces of the head-free
   traversal became zero-shaped (each was a 97 MB fill plus a 97 MB blind
   addition on Max).
1. **cublasLt candidate timing** (~0.4 ms per Max step). The weight
   gradients' heuristic top-1 is not reliable across shapes; the top eight
   candidates are timed once on the live operands and the fastest cached.
   The measured gain is small -- the heuristic was already near-optimal on
   these shapes -- and the mechanism stays as insurance for future ones.

Step-equivalent effect on the operator's GPU time (fwd + 2 bwd + bwd2 per
block):

| size | before   | after            | end-to-end step    |
| ---- | -------- | ---------------- | ------------------ |
| Pro  | 106.5 ms | 92.1 ms (−13.5%) | 143.2 -> ~134.8 ms |
| Max  | ~305 ms  | 280.8 ms (−8%)   | 431.4 -> ~404 ms   |

### 12.3 Evaluated and rejected

- **Packing the reduction shuffles.** Two independent partials per 64-bit
  shuffle would halve the reduction instruction count on paper; SASS-level
  SHFL is 32-bit and the compiler emits two instructions for the 64-bit
  form. No net win, not implemented.
- **Fusing the anchor re-entry into the second-order mixing traversal.**
  The traversals walk the layers in opposite directions (the re-entry
  recovers inputs backward, the curvature recurses its head forward), so a
  single fused loop cannot serve both; the only shared work is the
  sigmoid recompute, worth ~0.15 ms against ~107 MB of retention.
  Rejected.
- **Concatenating the weight-gradient GEMMs across stages.** The pairs
  contract different operands into the same output; a K-dimension concat
  requires materializing adjacent copies (the copy costs what the fusion
  saves), and beta-accumulation only removes (M, N)-sized additions worth
  ~45 us. Not pursued.
- **Sharing the sigmoid surfaces between the two backward calls of a
  step.** The two calls differ only in the upstream, but carrying the
  (F, E, lg) fp32 surfaces across operator calls costs ~107 MB for
  ~0.2 ms. Rejected.
- **Moving the Wigner-gradient reduction onto the tensor cores**
  (implemented, measured, rolled back). The gradient is a per-edge outer
  product g_loc x^T over the channels -- structurally one batched matmul
  on materialized operand surfaces plus a structural scatter, which
  removes the warp-reduction chain that dominates the rotation backward's
  instruction stream. Measured end to end it *lost*: staging the gathered
  feature rows costs a (E, DIM, C) write (262 MB on Max) plus the matmul's
  re-read, and the added DRAM traffic exceeded the removed reduction time
  (Pro backward 2.93 -> 3.08 ms, Max step-equivalent +6.6 ms). The
  bandwidth-for-instructions trade runs the wrong way at these shapes;
  the reduction stays in the kernel.
- **Suspecting the cuBLAS SM80 kernel selection on Max.** The Max shapes
  (m0 = 672, not a multiple of 64) route to `cutlass_80`-generation
  kernels rather than the Blackwell-native `nvjet` family, which looked
  like the smoking gun; isolated measurement showed those kernels running
  at 200-320 TFLOPS -- 50-80% of the card's dense bf16 roof and consistent
  with the in-operator times. The selection is fine; nothing to fix.

### 12.4 Standing against the hardware

Where the chain stands, per Max step (8 blocks): the GEMM work is
~22 TFLOP running at 200-320 TFLOPS measured (50-80% of the card's dense
bf16 roof, consistent between isolated and in-operator measurement) for
~90 ms; the bandwidth-bound elementwise kernels (the gate pair measures
93-100% of DRAM) account for ~35 ms at their roofs; the instruction-bound
rotation traversals carry a further ~2-4x over their bandwidth floors that
both the register-pressure treatments and the tensor-core offload failed
to convert. About 2x remains in principle, essentially all of it inside
the instruction-bound traversals and the second order's two
direction-conflicting mixing walks.

The structural remainder that no kernel work removes: the backward runs
twice per step (the force pass and the energy term arrive as separate
autograd calls -- the force pass must precede the loss, and the energy
term's upstream additionally carries the force graph's second-order
feedback through the downstream layers, so the two upstreams are not
proportional and cannot be merged), and the second order's anchor
re-entry walks the mixing layers backward while its curvature walks them
forward. Reusing the force pass's traversal surfaces across the two
backward calls buys ~0.2 ms per call against ~100 MB of retention and was
rejected.

### 12.5 The terminal decision: the attention span moved back to Triton

After the optimization pass the whole-convolution operator stood at Pro
134.3 ms / 10.21 GiB and Max 403.9 ms / 25.76 GiB -- better than its own
starting point (144.4 / 431.4 ms) but still behind the hybrid it replaced
(Pro 121.2, Max 356.2 ms) at essentially equal memory. The deficit is
structural, not a kernel defect: the Triton attention span lives inside
the traced graph, where Inductor fuses the logits, the segmented softmax
and the aggregation epilogues with their graph neighbours (the projections,
the envelope, the head gate) and schedules them alongside independent
work; the CUDA span runs the same mathematics behind an operator boundary
that materializes every interface surface and serializes against the
graph. The attention span is also the part with the least to gain from
manual fusion -- its kernels were already at their roofs in both
implementations -- while the value stream (resident tile kernel, no
inter-layer materialization) is where the fusion pays.

The fused attention span was therefore removed outright (`atten_train.cu`,
`atten_train_kernels.cuh`, the whole-convolution operator schemas, and the
second upstream of the rotation curvature kernel that only the internal
anchor re-entry could exploit); the value chain kept every optimization of
section 12.2 -- `with_weights`, the launch bounds, the in-place retention
snapshots, the `ggout_init` seeding, the zero-shape fixes and the
cublasLt candidate timing -- and the anchor re-entry returned to the
autograd-driven second `bwd` call. The operating point is the section 11.7
hybrid again, with the optimized value chain under it. The experiment sequence
below confirms the removal on every shape (median step, 20 steps, one RTX PRO
6000; Neo runs `DP_TUNE_TRAIN=2`, the large shapes run untuned):

| shape | whole-conv operator  | hybrid before section 12 | hybrid after removal |
| ----- | -------------------- | ------------------------ | -------------------- |
| Neo   | 26.5 ms / 1.27 GiB   | 23.5 ms / 1.30 GiB       | 21.1 ms / 1.28 GiB   |
| Pro   | 134.3 ms / 10.21 GiB | 121.2 ms / 11.19 GiB     | 118.4 ms / 11.23 GiB |
| Max   | 403.9 ms / 25.76 GiB | 356.2 ms / 27.93 GiB     | 351.4 ms / 27.63 GiB |

The value-chain optimizations carry the hybrid 2.3-5% past its own
previous best (and Max's peak memory below it -- the `keep_state` and
zero-shape work lands on both sides of the trade); the whole-conv
operator's ~1 GiB memory edge on Pro/Max came from keeping the fp32
softmax surfaces off the graph, and went back with the span. Against the
dense baseline the hybrid stands at 1.15x (Neo), 1.33x (Pro), 1.26x (Max)
for time at 53-65% of the peak memory in this experiment sequence. Section
11.7 contains the final end-to-end matrix after the subsequent value-path and
graph optimizations.

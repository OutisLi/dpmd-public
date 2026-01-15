# Chemical Conditioning Trunk

A per-atom auxiliary trunk that adds capacity to DPA4/SeZM on the bf16 tensor
cores. It reads a summary of each atom's chemical environment and emits
coefficients that modulate the output of every interaction block. The
descriptor's geometric machinery runs unchanged: the trunk sits alongside it
and adjusts what it produces.

The design follows from two measurements. First, a potential is judged by
derivatives of the energy, and reduced-precision arithmetic is safe on some
quantities and harmful on others depending on how fast they vary with the
coordinates; section 2 makes that quantitative and section 3 turns it into a
placement rule. Second, the per-atom axis reaches about sevenfold more of the
machine's arithmetic throughput than the descriptor's step does end to end,
which is what makes a large trunk affordable; section 5 measures it.

Numbers throughout are from RTX PRO 6000 Blackwell Server Edition (sm_120,
BF16 tensor cores at 1 PFLOPS dense, 1597 GB/s of memory bandwidth), PyTorch
2.13.0+cu130, and the `examples/water/dpa4` training script at `channels=256`,
`batch_size=8` — 1536 atoms and 136652 edges per step, 833.7 ms/step.
Reproduction scripts are in `debug/precision/`.

## 1. What the trunk produces

For each atom `i` the trunk emits a set of coefficients `theta_i`. Those
coefficients scale and shift the node features that each interaction block
produces, so the descriptor's function is conditioned on chemistry while its
geometric evaluation stays exactly as it is. Written schematically, the energy
becomes

```
E_i = F( theta_i , geometry_i )
```

where `F` is the existing descriptor plus fitting network in fp32 and
`theta_i` is the trunk's output. The trunk therefore behaves like a set of
learned, environment-dependent parameters of the potential rather than like
another layer of the potential itself. That distinction is what section 3
shows to be the load-bearing property.

## 2. What the arithmetic has to respect

A machine-learning potential is used through derivatives of its energy with
respect to atomic positions. The first derivative is the force. The second
derivative is the matrix of harmonic force constants, conventionally **FC2**,
from which phonon frequencies follow. The third derivative is the tensor of
cubic anharmonic force constants, **FC3**, which sets phonon-phonon scattering
and hence lattice thermal conductivity.

FC2 and FC3 are obtained by displacing atoms and finite-differencing the
forces. The displacement magnitude is written `h_fd` below; phonopy uses
0.01 Å by default for FC2 and phono3py 0.03 Å for FC3. A second derivative is
one difference of the force divided by `h_fd`, and a third derivative is two
differences divided by `h_fd^2`, so any error in the force is amplified by
`1/h_fd` and `1/h_fd^2` respectively.

Reduced-precision arithmetic contributes exactly such an error, and its
structure matters more than its size. A number stored in bf16 keeps 8 bits of
mantissa, so it is snapped to a grid whose spacing is a relative
`eps = 2^-8 = 3.9e-3` of its own magnitude. Rounding is therefore a *staircase*
in whatever the number depends on. If a quantity `h` enters the
reduced-precision region and the atoms move, `h` crosses one grid line after a
displacement of

```
Lambda = eps * L,        L = |h| / |dh/dx|
```

`L` is the distance over which `h` changes by 100% of itself, i.e. how
sensitive that quantity is to being moved, and `Lambda` is the width of one
flat step of the staircase — the *tread*. Over a displacement shorter than
`Lambda` the rounded value does not change at all; at a tread boundary it jumps
by a relative `eps`.

`tread_demo.py` measures this on the descriptor's own highest radial
frequency. The fast path uses a Bessel basis `phi(r) = sin(w r)/r` with
`w = n_radial * pi / rcut`; at `n_radial = 16` and `rcut = 6` Å the top
component at `r = 3.1` Å has `phi = 0.2397` and `dphi/dr = 1.731` Å⁻¹, and the
bf16 grid spacing at that magnitude is `9.766e-4`, giving

```
Lambda = 9.766e-4 / 1.731 = 5.64e-4 Å        (measured 5.67e-4)
```

Printing the exact and the rounded value on a 1e-4 Å grid shows the staircase
directly: across five consecutive positions the true value climbs from
0.2398971 to 0.2405874 while the bf16 value sits unchanged at 0.2402343750.

A finite difference is unaffected by a staircase only while its step stays
inside one tread, `Lambda > h_fd`. Inverting that condition gives the length
scale a quantity needs in order for bf16 arithmetic on it to leave the
derivatives intact, and it is the number every later section is measured
against:

```
L > h_fd / eps        ->      L > 2.56 Å  (h_fd = 0.01)
                              L > 7.68 Å  (h_fd = 0.03)
```

## 3. Why coefficients are the right place

Section 2 constrains quantities that carry the coordinate dependence. A
coefficient is different in kind, and the difference is worth writing out.

Suppose the energy contracts a coefficient vector `theta` against a geometric
feature vector `phi(x)` evaluated in fp32:

```
E(x) = <theta, phi(x)>
```

Rounding `theta` to bf16 replaces it by `theta + delta`, where
`delta = Q(theta) - theta` is the rounding residual. Because an inner product
is linear in its first argument, the energy splits:

```
E(x) = <theta, phi(x)>  +  <delta, phi(x)>
```

The first term is the intended energy. The second is the error, and it is a
*constant vector* contracted against a *smooth function of position*, hence
itself a smooth function of position. Differentiating any number of times
passes straight through:

```
d^n E / dx^n  =  <theta, d^n phi/dx^n>  +  <delta, d^n phi/dx^n>
```

Both terms contain the same `d^n phi/dx^n`, so the relative error is
`|delta|/|theta| = eps` for every derivative order and for every choice of
`h_fd`. Physically the rounded coefficient is a potential whose parameters are
off by 0.4%, which is an accuracy statement, not a smoothness one.

`tread_demo.py` measures both placements side by side on the same Bessel
component, as relative rms error over 64 sampling centres. Column A rounds the
geometric feature — the coordinate path. Column B rounds a coefficient
multiplying it:

| `h_fd` (Å) | A, 1st deriv | A, 3rd deriv | B, 1st deriv | B, 3rd deriv |
| ---------- | ------------ | ------------ | ------------ | ------------ |
| 1e-5       | 1.23e+1      | 2.4e+9       | 1.12e-3      | 2.31e-3      |
| 1e-4       | 2.26e+0      | 5.0e+6       | 1.12e-3      | 1.11e-3      |
| 1e-3       | 2.06e-1      | 6.6e+3       | 1.13e-3      | 1.13e-3      |
| 1e-2       | 1.45e-2      | 5.20e+0      | 2.17e-3      | 2.65e-3      |

Both carry the same 1.1e-3 of arithmetic error. Column B is flat at that value
until 1e-2, where ordinary truncation error of the difference formula takes
over. Column A has no usable step.

The same holds through a full network. `probe.py` builds a toy potential twice
with identical parameter budgets, once with the reduced-precision trunk on the
coordinate path and once with it producing coefficients, and reports relative
errors over 25 centres:

| trunk placement    | force  | FC2     | FC3     |
| ------------------ | ------ | ------- | ------- |
| on coordinate path | 5.7e-3 | 1.73e-2 | 8.77e-1 |
| on coefficients    | 2.4e-3 | 3.65e-3 | 3.06e-3 |

The force errors are the same order. FC3 differs by 287x. This is the reason
the trunk emits coefficients.

## 4. The trunk's input

Two properties are required of anything the trunk reads.

**Rotational invariance.** The coefficients multiply SO(3) features of the
descriptor, so they must be scalars under rotation; otherwise the modulation
would rotate the features among themselves and the descriptor would stop being
equivariant. Section 6 returns to this.

**A length scale that clears `h_fd / eps`.** The trunk's own output inherits
the sensitivity of its input: if the input crosses bf16 grid lines every
5e-4 Å, so does `theta`, and section 3's assumption that `delta` is constant
fails. Section 2 sets the bar at 2.56 Å for FC2 work and 7.68 Å for FC3.

The second property orders the candidates sharply. The usual way to build a
rotationally invariant descriptor of a neighbourhood is to accumulate
spherical-harmonic moments of degree `l` over the neighbours and then multiply
`k` of them together in a way that contracts the order indices away; `k` is
the contraction order, and it is the axis along which the candidates separate.
`invariant_scales.py` measures `L` by autograd on a real 21-neighbour
environment:

| invariant family                       | order `k` | `L` median | `L` p10 |
| -------------------------------------- | --------- | ---------- | ------- |
| radial count, `l = 0`, wide envelopes  | 1         | 3.559 Å    | 3.239 Å |
| pairwise angular contraction, `l >= 1` | 2         | 0.425 Å    | 0.114 Å |
| three-way angular contraction          | 3         | 0.037 Å    | 0.017 Å |
| four-way angular contraction           | 4         | 0.242 Å    | 0.110 Å |

The ordering is geometric. A displacement `dx` changes a direction `r_hat` at
rate `1/r`, so a degree-`l` angular function changes at rate `l/r`, and an
order-`k` product of such functions at rate `k*l/r`. Hence `L ~ r/(k*l)`, and
richer invariants are systematically sharper — the information content and the
sensitivity grow together.

The trunk's input is therefore built from the first row: **radial occupancy at
`l=0`, from wide envelopes.** Writing `u_m` for a quintic C² cutoff switch
whose transition begins at `s_m` and ends at `rcut`,

```
u_m(r)      = 1 - t^3 (10 - 15 t + 6 t^2),   t = clamp((r - s_m)/(rcut - s_m))
n[i, m, t]  = sum over neighbours j of element t  of  u_m(r_ij)
```

`m` indexes the envelope, of which there are `M`, with `s_m/rcut` spread over
`[0.15, 0.55]` so the transitions are 2.7 to 5.1 Å wide. `n[i, m, t]` is a
smooth count of how many atoms of element `t` sit within scale `m` of atom `i`,
i.e. a coarse-grained radial distribution function resolved by element. The
centre atom's own identity enters through the descriptor's shared type
embedding `z_i`, a learned vector of width `n_te` per element, concatenated to
the counts:

```
summary_i = concat( n[i, :, :].flatten() , z_i )
```

of width `M * n_types + n_te`. This is 46 for water at `M=6, n_te=16`, 336 for
a ten-element alloy, and 2896 for a ninety-element universal model — the
regime where a large trunk has something to learn.

Grouping the sum by neighbour **element** rather than contracting it against
the neighbour's embedding is what keeps the construction cheap. The grouped
form accumulates one scalar per (atom, scale, element), so the per-edge
intermediate is a single number per scale; contracting against an embedding
first would build an `(E, n_te)` intermediate, where `E` is the number of
edges in the batch — at `E = 136652` and `n_te = 32` that is 140 MB,
comparable to the descriptor's own largest tensor. The
information is identical, since any embedding contraction is recoverable as a
linear map applied on the per-atom side, where it becomes part of the trunk's
first GEMM.

Angular information reaches the energy through the descriptor's own SO(3)
blocks, which operate on spherical-harmonic moments in fp32 throughout. The
trunk contributes a complementary view — a wide-range, element-resolved
account of the radial environment — on the one length scale where bf16
arithmetic stays invisible to a finite difference.

## 5. The trunk's shape

The shape is chosen to reach a useful fraction of the bf16 tensor cores, which
requires understanding why the descriptor itself does not.

A GEMM's efficiency is governed by its **arithmetic intensity**, the ratio of
floating-point operations performed to bytes moved. A machine saturates its
arithmetic units only above the ratio of its peak throughput to its peak
bandwidth — on this GPU `1e15 / 1.597e12 = 626` FLOP/byte, the *roofline
knee*. Below that, time is spent waiting for memory.

The descriptor's step sits well below the knee, and shape-resolved profiling
shows why the average is misleading:

| operation                                 | ms    | TFLOPS | note                |
| ----------------------------------------- | ----- | ------ | ------------------- |
| `bmm [[1,136652,1024],[1,1024,1024]]` ×30 | 18.39 | 468    | 47% of peak         |
| `bmm [[136652,9,7],[136652,7,256]]` ×12   | 19.62 | 2.7    | 0.27%, memory bound |
| `add [[1,136652,7,256]]` ×24              | 37.24 | —      | 947 GB/s, 59% of BW |

The wide contractions are efficient. The SO(3) degree contractions have
matrices of size 7×9 against a batch of 136652 edges, so they move far more
data than they compute and reach 0.27% of peak while already using 44% of
bandwidth. Elementwise and copy traffic over the `(136652, 7, 256)` node
tensor — 490 MB in bf16 — is roughly 60% of the step. End to end the step
realizes on the order of 5% of the machine's bf16 throughput.

A per-atom trunk works on a different scale. Its rows are atoms rather than
edges, and the ratio of the two is

```
(E * D_l) / N = (136652 * 7) / 1536 = 623
```

so the same arithmetic costs 623x less here, and its tensors are `(1536, C)` —
6.3 MB at `C = 2048`, four orders of magnitude below the node tensor — so it
leaves the bandwidth pressure that dominates the step untouched.

`roofline.py` measures one residual layer of the form
`x + W2 @ gelu(W1 @ rmsnorm(x))` with `W1: (2C, C)` and `W2: (C, 2C)`, over
`M` rows, in TFLOPS. `M = 192` is one water frame, 1536 is the eight-frame
training batch, 6144 is thirty-two frames:

| width `C` | intensity at M=1536 | M=192 | M=1536 | M=6144 |
| --------- | ------------------- | ----- | ------ | ------ |
| 512       | 154                 | 7.6   | 52.8   | 219.3  |
| 1024      | 279                 | 25.5  | 211.8  | 282.4  |
| 2048      | 473                 | 101.4 | 319.4  | 342.8  |
| 3072      | 614                 | 162.1 | 342.2  | 335.9  |
| 4096      | 723                 | 189.5 | 346.8  | 361.6  |
| 6144      | 878                 | 199.8 | 360.6  | 364.6  |

Three readings set the shape.

**Width starts at 2048.** The intensity column crosses into the useful range
there: 473 at 2048 and 614 at 3072 against a knee of 626, and the measured
throughput tracks it, 319 and 342 TFLOPS. At 1024 the layer spends its time
fetching its own weights and delivers 212; at 512, 53. Beyond 4096 the return
is a few percent for four times the arithmetic.

**Capacity goes into width rather than depth.** A layer's arithmetic grows as
`L*C^2` while the elementwise traffic around it grows as `L*C`, so at equal
arithmetic a wide shallow stack moves strictly less data. `L=6, C=2048` and
`L=24, C=1024` have identical FLOPs and the first moves half the bytes.

**The attainable fraction is about 36%.** Eager PyTorch evaluates the norm, the
GELU and the residual addition as separate passes over memory; a bare GEMM at
this shape measures 381 TFLOPS, so 360 is close to the layer's ceiling without
a fused epilogue. Against the step's end-to-end 5%, this is a sevenfold
improvement in how much of the machine the added capacity actually uses.

The stack follows directly:

```python
h = Linear(summary_width -> width)                              # bf16
for _ in range(n_layers):
    h = h + Linear(2*width -> width)(                           # bf16
                gelu(Linear(width -> 2*width)(rmsnorm(h))))     # bf16
theta = Head(width -> out_width)                                # fp32
```

RMSNorm is preferred to LayerNorm because it needs one reduction rather than
two at equal effect at this width. GELU is preferred to a gated linear unit
because a GLU costs a chunk, a sigmoid and a multiply — three passes over a
`(M, 2C)` tensor — for comparable expressiveness at this depth, while GELU is
a single pass and a candidate for a cuBLAS epilogue later. The expansion is
`C -> 2C -> C` rather than the transformer-conventional `4C`, since the wider
expansion doubles both arithmetic and activation traffic for capacity the
input width calls for. The head runs in fp32: it contributes negligible
arithmetic and it is the last operation before the values become the
potential's parameters. The trunk is per-atom and independent, with no
attention across atoms, which keeps the descriptor strictly local and its cost
linear in system size, as the molecular-dynamics path requires.

At `width=2048, n_layers=12` this is 201 M parameters, 1.94 ms, **+0.23% of
step time**, and 2.8 GB of AdamW optimizer state. At `width=3072, n_layers=16`, 604 M parameters, 5.4 ms, +0.65%, 8.5 GB.

The same accounting opens a trade the trunk makes available. Since it holds
capacity at 1/623 of the fast path's cost, capacity can be moved out of
`channels` and into the trunk, which shrinks the `(E, D_l, channels)` tensor
that dominates bandwidth. Halving `channels` from 256 to 128 halves the
dominant traffic.

## 6. Output and modulation

The head emits four tensors per interaction block `b`, where `N` is the number
of atoms, `D_l` the number of SO(3) degrees the block carries, and `C` the
descriptor's channel width:

```
gamma[b]   (N, D_l, C)     a gain, one value per degree and channel
beta [b]   (N, C)          a shift, one value per channel
U[b],V[b]  (N, C, r)       factors of a rank-r channel mixing, r = 8
```

The node tensor `x` of shape `(N, D, 1, C)` — `D` running over degrees `l` and
orders `m` — is updated after block `b` as

```
x[n,(l,m),c] <- (1 + alpha * gamma[n,l,c]) * x[n,(l,m),c]
              +      alpha * beta[n,c]  if l == 0
              +      alpha * sum_c' (U[n,c,:] . V[n,c',:]) * x[n,(l,m),c']
```

`alpha` is a fixed scalar setting how far the trunk is allowed to move the
block output; section 7 derives its value from a smoothness budget. The head
passes `gamma` through a `tanh`, which keeps the gain inside `(-1, 1)` and so
bounds the whole first term by `alpha` regardless of what the trunk learns.
`U` and `V` are initialized to zero, so the modulation begins as the identity
and enabling the trunk on an existing checkpoint reproduces the parent model at
step zero.

The rank-`r` term is what carries the design past a per-channel rescaling. A
gain can only reweight channels that already exist; `U V^T` is a learned
low-rank correction to how channels mix, conditioned on chemistry, and is
therefore a modification of the block's effective weight matrix rather than of
its output alone.

The modulation is evaluated in `compute_dtype`, i.e. fp32, outside the block's
autocast region. The coefficients are fp32 by then, and keeping the node
tensor in fp32 here is what makes `delta` in section 3 a coefficient residual
rather than a feature residual.

### Preserving equivariance

The descriptor is equivariant: rotating the input structure rotates its
`l > 0` features by the corresponding Wigner matrix while leaving `l = 0`
features fixed. A modulation preserves this as long as it never mixes orders
`m` within a degree and never adds a rotation-varying quantity to an invariant
one. The three terms satisfy this by construction.

`gamma` is indexed by `(l, c)` and broadcast over `m`, so all `2l+1` orders of
a degree receive the same gain. Scaling a whole degree uniformly commutes with
its Wigner matrix.

`beta` is added only at `l = 0`. A constant is invariant under rotation, so it
belongs in the invariant subspace; adding the same constant at `l > 0` would
make those components fail to rotate.

`U V^T` acts on the channel axis with the same matrix for every `(l, m)`. The
channel axis carries no rotation index, so any linear map on it commutes with
the rotation.

The trunk's inputs are invariant for the same reason: section 4's occupancy
counts and type embedding are scalars, so `gamma`, `beta`, `U` and `V` are
scalars, which is what the three rules above assume.

## 7. Setting the modulation strength

Two properties of the trunk fix the smoothness of the result.

**The trunk's input enters detached.** Cutting the gradient at the input
rather than at the output is what makes the trunk both trainable and safe:

```python
theta = self.trunk(summary.detach())
```

This makes `d theta / d x` identically zero while `d theta / d W` is
untouched, so no gradient traverses the reduced-precision region and the
force is produced entirely by the fp32 path. The trunk still learns, because
`theta` enters the value of the energy and the value of the force
`<theta, d phi/dx>`, so both loss terms reach its weights. The alternative,
`self.trunk(summary).detach()`, would leave the trunk permanently at
initialization. Letting the gradient through instead — no detach at all —
degrades FC3 by 44x in `probe.py` (1.35e-1 against 3.06e-3), because the
backward pass through the bf16 region writes high-frequency noise directly
into the force.

**The modulation strength is bounded.** Section 4's input has `L = 3.2` Å,
which clears the 2.56 Å that `h_fd = 0.01` requires but falls short of the
7.68 Å that `h_fd = 0.03` would, so a residual staircase reaches FC3. Its
amplitude is proportional to `alpha`, which makes `alpha` a direct control on
third-order fidelity. `alpha_budget.py` measures this on a 48-atom cluster,
with the wide-envelope input of section 4, detached, recomputed at every
evaluation, at `h_fd = 0.01` for FC2 and 0.03 for FC3, as relative error
against the same model with the trunk in fp32:

| `alpha` | FC2 rms | FC3 rms | FC3 max |
| ------- | ------- | ------- | ------- |
| 1.00    | 6.76e-3 | 4.52e-2 | 8.33e-2 |
| 0.30    | 2.84e-3 | 2.41e-2 | 4.44e-2 |
| 0.10    | 1.07e-3 | 1.03e-2 | 1.90e-2 |
| 0.03    | 3.36e-4 | 3.45e-3 | 6.39e-3 |
| 0.01    | 1.14e-4 | 1.19e-3 | 2.19e-3 |

The response is linear to within the sampling, so the budget is a formula:

```
FC2 rms ~ 0.0068 * alpha          FC3 rms ~ 0.045 * alpha
```

Three readings follow. Even at full strength the arrangement is 19x better on
FC3 than placing the trunk on the coordinate path (4.5e-2 against 8.8e-1),
which is the coefficient placement of section 3 and the detach above doing
their work. FC2 stays at 6.8e-3 even at `alpha = 1`, so phonon accuracy is
comfortable across the whole range and only the third order needs a bound.
And `alpha = 0.03` reaches 3.45e-3 on FC3, matching what section 3's toy
coefficient model achieved, so a fixed scalar is sufficient to recover
coefficient-grade smoothness.

The default is `alpha = 0.05`, which puts FC3 at roughly 2.3e-3. A model
intended for molecular dynamics and phonons can use 0.3; work on anharmonic
properties should use 0.01 to 0.03. `alpha` is a constant rather than a
learnable parameter, because a learnable strength is driven upward by the loss
and would consume the budget it exists to protect.

A 5% multiplicative gain is a modest authority per block, and it is applied
after every block, alongside the shift and the rank-`r` mixing, so the
composition over a six-block descriptor is substantially larger than the
per-block figure suggests.

## 8. Implementation

`deepmd/pt/model/descriptor/sezm_nn/chem_cond.py` holds two modules.

```python
class ChemicalSummary(nn.Module):
    """Wide-envelope l=0 occupancy grouped by neighbour element, plus the
    centre type embedding. Reads edge geometry, so it runs in fp32."""

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
        self, edge_cache, atype_flat, n_nodes
    ) -> Tensor: ...  # (N, n_scales * n_types + n_te)


class ChemicalConditioner(nn.Module):
    """The bf16 trunk and the per-block modulation heads."""

    def forward(
        self, summary: Tensor
    ) -> list[BlockModulation]: ...  # summary arrives already detached


class BlockModulation(NamedTuple):
    gamma: Tensor  # (N, D_l, C)
    beta: Tensor  # (N, C)
    u: Tensor  # (N, C, r)
    v: Tensor  # (N, C, r)

    def apply(self, x: Tensor, alpha: float) -> Tensor:
        """The three-term update of section 6, in compute_dtype."""
```

In `sezm.py`, both `forward` and `forward_with_edges` build the summary once
the edge cache exists, before entering the autocast region:

```python
modulation = None
if self.chem_cond is not None:
    summary = self.chem_summary(edge_cache, atype_flat, n_nodes)
    modulation = self.chem_cond(summary.detach())
```

and `_forward_blocks` applies entry `i` after block `i`:

```python
if modulation is not None:
    block_output = modulation[i].apply(block_output, self.chem_alpha)
```

The trunk's GEMMs are the module's own precision decision rather than a
consequence of the descriptor's setting, so they are wrapped in an explicit
`torch.autocast("cuda", dtype=torch.bfloat16)` and behave identically whether
`descriptor.use_amp` is on or off. `ChemicalSummary` carries the opposite
annotation for the same reason: it reads edge geometry, and an ambient
autocast reaching it would round the summary and shorten the tread of section
2 by whatever margin section 4 established.

bf16 is the format because the trunk's output is the potential's parameter
vector, so the arithmetic error of the trunk becomes the relative error of the
surface. Measured on this GPU, a bf16 GEMM at the trunk's shape carries
2.84e-3 of relative error; the same GEMM in fp8 e4m3 carries 3.74e-2, and
per-tensor, per-row and MXFP8 block scaling all return that same figure,
because the limit is the format's three mantissa bits rather than its dynamic
range. Quantization-aware training recovers roughly a factor of two of it
(`fp8_qat.py`: 4.76e-2 without, 2.44e-2 with, 1.49e-2 with the input
projection and head held at bf16). Through section 7's budget that difference
translates into `alpha = 0.01` for fp8 where bf16 supports 0.05, i.e. a fifth
of the modulation authority, in exchange for 1.2 to 1.45x the throughput on a
module already below 1% of the step. bf16 is the better point on that curve,
and optimizer state bounds the trunk's size before arithmetic does in any
case.

## 9. Configuration

Under `model.descriptor.chem_cond`:

| key           | default        | meaning                                                |
| ------------- | -------------- | ------------------------------------------------------ |
| `enabled`     | `false`        | master switch                                          |
| `width`       | `2048`         | trunk width; 2048 is where intensity crosses           |
| `n_layers`    | `12`           | trunk depth                                            |
| `n_scales`    | `6`            | number of envelopes `M` in the summary                 |
| `scale_range` | `[0.15, 0.55]` | `s_m / rcut` span of the envelope set                  |
| `alpha`       | `0.05`         | modulation strength; sets the FC3 budget               |
| `lowrank`     | `8`            | channel-mixing rank `r`; `0` uses gain and shift alone |

## 10. Verification

**Equivariance.** Rotate a structure by a random SO(3) element and assert the
descriptor transforms correctly with `lowrank > 0` and a non-trivial `beta`.
Two negative controls belong with it: a `beta` applied at `l > 0`, and a
`gamma` indexed by `m`, must each break the assertion. Both are one line away
from the correct code, so an equivariance test that cannot fail carries no
information here.

**Summary precision.** Assert that `ChemicalSummary` produces bitwise
identical output inside and outside an ambient bf16 autocast. This is what
holds the tread width of section 2 at the value section 4 measured, and it
fails silently.

**The alpha budget.** Run `alpha_budget.py` against the real descriptor and
check that FC2 and FC3 track `0.0068*alpha` and `0.045*alpha` within a factor
of two. A larger FC3 error indicates that something sharper than the occupancy
summary has reached the trunk; the envelope range and the summary contents are
where to look.

**Derivative consistency.** FC3 must degrade relative to FC2 by no more than
the table in section 7 predicts, and the analytic second derivative must agree
with the finite-difference one. Agreement between the two estimators is the
direct signature of a surface that is analytic in the coordinates.

**Fine-tune neutrality.** Enabling the trunk on a checkpoint that lacks it
must reproduce the parent model's energies and forces exactly at step zero,
which the zero-initialized `U`, `V` and `tanh` gain guarantee.

**Throughput.** `roofline.py` at the configured width should land within about
10% of section 5's table. A larger shortfall means the layer is not reaching
the tensor cores, and the shape should be revisited before the capacity is
relied on.

## 11. Boundaries

- The trunk's contribution to the force flows through the value of `theta`
  and not through `d theta / d x`, which section 7 cuts. The descriptor's fp32
  path carries the full geometric derivative. Whether the resulting function
  class is rich enough is an empirical question: begin at `width=2048, n_layers=12, alpha=0.05` and compare force MAE before enlarging.
- `alpha` trades expressiveness against third-order fidelity, and section 4's
  measurement says the trade is unavoidable in bf16, since `L = 3.2` Å is
  short of the 7.68 Å that `h_fd = 0.03` asks for. A model used for thermal
  conductivity should stay at the low end of the range.
- Sections 4 and 7 measure a synthetic cluster and a toy modulated potential.
  The mechanism is independent of architecture but the constants are not:
  re-measure `L` and the `alpha` slope on the real descriptor before adopting
  the defaults.
- The `channels`-for-trunk trade at the end of section 5 is a motivation for
  the design and its accuracy-neutral point is unmeasured. There is at present
  no evidence that 128 channels plus a trunk matches 256 channels.
- Section 5's throughput table is a synthetic layer stack. The summary
  construction, the heads and the modulation itself are outside it; the
  modulation touches the full `(N, D, 1, C)` node tensor and is bandwidth
  bound, so it should be budgeted separately.
- Behaviour under `torch.compile` with a varying atom count is untested, and
  the module may need to sit outside the compiled region.

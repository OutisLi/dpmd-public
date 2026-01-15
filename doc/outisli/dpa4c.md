# DPA4C

DPA4C is the **Compact and Compressible** degree-wise member of the DPA4 family. It is a local
atomic descriptor: for each atom, it converts the nearby atomic geometry and chemical species into a
fixed-length numerical vector. A fitting network then maps that vector to an atomic energy, and
differentiation of the total energy gives forces and virials.

DPA4C is intended as a compact student of the larger DPA4 teacher for extreme-speed molecular
dynamics. It retains DPA4's distance representation and type embedding, but removes cross-atom
message passing. The resulting descriptor traverses each directed neighbor edge once, performs one
destination reduction, and applies a center-local invariant readout.

## Overview

### Atomic graph and one-hop computation

The input is a directed neighbor graph. Each atom is a node. A directed edge `j -> i` means that
atom `j` is a neighbor of center atom `i` and lies inside the cutoff radius `rcut`. The same
physical pair can therefore appear in both directions.

A one-hop descriptor reads the atoms directly connected to each center. Many graph neural networks
then perform **message passing**: they repeatedly update each node from features already computed on
neighboring nodes. DPA4C does not do this. Each edge depends only on its distance, direction, and
ordered pair of atom types. All edge contributions for a center are summed once, and every later
operation is local to that center.

The computation is:

```text
positions and atom types
  -> directed cutoff graph
  -> radial, chemical, and angular edge features
  -> one destination sum into degree-wise moments
  -> fixed rotation/reflection-invariant readout
  -> fitting network
  -> atomic energy
```

The **cutoff** limits the local environment to a finite radius. A smooth cutoff envelope brings an
edge contribution continuously to zero as the edge reaches that radius, preventing discontinuities
when an edge enters or leaves the graph.

### Rotations, reflections, and permutations

`O(3)` is the set of all three-dimensional rotations and reflections. A quantity is
**O(3)-equivariant** if it transforms in a predictable way when the atomic coordinates are rotated
or reflected. A vector is the familiar example: rotating the atoms rotates the vector by the same
transformation. A quantity is **O(3)-invariant** if it does not change at all. Energy must be
invariant.

DPA4C first builds equivariant degree-wise moments and then contracts them into invariant scalars.
It is also insensitive to the order in which neighbors are stored, because neighbor contributions
are combined by sums. Reordering center atoms only reorders the corresponding output rows; this
property is permutation equivariance over nodes.

### Main notation

All vectors and matrices contain real numbers. `ℝ^d` denotes a real vector with `d` entries.

| Symbol                        | Meaning                                                     |
| ----------------------------- | ----------------------------------------------------------- |
| `i`, `j`                      | Center atom and one of its neighbors                        |
| `r_i`                         | Cartesian position of atom `i`, in Å                        |
| `a`, `b`                      | Types of center atom `i` and neighbor atom `j`              |
| `N`, `E`, `T`                 | Number of nodes, directed edges, and real atom types        |
| `rho_ij`                      | Regularized distance on edge `j -> i`                       |
| `u_ij`                        | Regularized direction on edge `j -> i`, with shape `(3,)`   |
| `chi_ij`                      | Smooth cutoff envelope `chi(rho_ij)`                        |
| `g`                           | Shared learned radial function with `C_0` output channels   |
| `q`                           | Shared radial mode profiles with `R` outputs                |
| `gamma_ab`, `beta_ab`, `U_ab` | Ordered type-pair scale, shift, and mode-mixing tables      |
| `psi_ij`                      | Edge amplitude before multiplication by the cutoff envelope |
| `phi_ij`                      | Enveloped edge amplitude, with shape `(C_0,)`               |
| `L`                           | Maximum angular degree, configured as `lmax`                |
| `C_0`                         | Scalar and edge width, configured as `channels`             |
| `C_l`                         | Channel width at angular degree `l`                         |
| `K_l`                         | Bispectrum probe rank at nonzero degree `l`                 |
| `R`                           | Number of shared radial modes, configured as `radial_modes` |
| `B_l(u)`                      | Real Cartesian harmonic block of degree `l`                 |
| `X^(l)`                       | Degree-`l` moment tensor                                    |
| `d_i^(0)`, `d_i^(+)`          | Two smooth neighborhood masses                              |
| `n_i^(0)`, `n_i^(+)`          | Their moment-normalization multipliers                      |
| `S`                           | Flat width of all degree-wise moments                       |
| `D_out`                       | Final invariant descriptor width                            |

Native spin (Section 6) adds the following. They appear only when the descriptor is spin
conditioned; every other symbol above keeps its meaning unchanged.

| Symbol                    | Meaning                                                       |
| ------------------------- | ------------------------------------------------------------- |
| `s_i`                     | Raw magnetic moment of atom `i`, in the units of the dataset  |
| `s_hat_i`                 | Conditioned moment actually used, with shape `(3,)`           |
| `m_a`                     | Per-type spin gate: one for a magnetic type, zero otherwise   |
| `s_ref_a`                 | Per-type reference moment magnitude, measured from the data   |
| `sigma`                   | Spin order: how many factors of the moment a term contains    |
| `p`, `t`                  | Signs a term picks up under inversion and under time reversal |
| `C_s`                     | Number of neighbor spin channels, equal to `C_2`              |
| `gamma^s_ab`, `beta^s_ab` | Ordered type-pair spin scale and shift tables                 |
| `psi^s_ij`, `phi^s_ij`    | Spin edge weight before and after the cutoff envelope         |
| `lambda_a`, `mu_a`        | Per-type weights of the two on-site spin channels             |
| `V`, `P`, `Q`, `M0`, `Mw` | The five spin families of Section 6.3                         |

The frame charge state (Section 7) adds the following. `Q_f` always carries a frame subscript and
is unrelated to the degree-two probe matrix `Q_eta` of Section 5.5.

| Symbol   | Meaning                                                              |
| -------- | -------------------------------------------------------------------- |
| `f`      | Frame index; `f(i)` is the frame of node `i`                         |
| `nf`     | Number of frames on the flat node axis                               |
| `Q_f`    | Total charge of frame `f`, in units of the elementary charge         |
| `M_f`    | Spin multiplicity of frame `f`, a positive integer                   |
| `H_pair` | Post-gate hidden width of the ordered pair encoder                   |
| `w_f`    | Condition shift of the center type embedding, with shape `(C_0,)`    |
| `y_f`    | Condition bias of the pair encoder hidden state, shape `(2 H_pair,)` |

The implementation is divided among:

- `deepmd/dpmodel/descriptor/dpa4c.py`: descriptor orchestration, graph ABI, statistics, and
  serialization;
- `deepmd/dpmodel/descriptor/dpa4c_nn/geometry.py`: structural profiles, real Cartesian harmonics,
  moment layout, and degree-two symmetric-traceless conversion;
- `deepmd/dpmodel/descriptor/dpa4c_nn/bispectrum.py`: probe ranks, allowed degree triples,
  normalized Cartesian Gaunt tensors, and independent probe indices;
- `deepmd/dpmodel/descriptor/dpa4c_nn/readout.py`: fixed invariant readout;
- `deepmd/dpmodel/descriptor/dpa4c_nn/pair_film.py`: ordered type-pair conditioning;
- `deepmd/dpmodel/descriptor/dpa4c_nn/spin.py`: native spin conditioning, families, and invariants;
- `deepmd/dpmodel/descriptor/dpa4c_nn/charge_state.py`: frame charge-state embedding and its
  canonicalization;
- `deepmd/pt_expt/descriptor/dpa4c.py`: native PyTorch graph gathers, reduction, autograd, mixed
  precision, and export integration;
- `deepmd/pt_expt/kernels/dpa4c/`: the compressed artifacts, the operator schemas and the
  dispatch, shared by both devices; `source/op/pt/dpa4c/graph_compress*.cu` holds the CUDA
  kernels and `source/op/pt/dpa4c/graph_compress_cpu*` the CPU ones.

## 1. Public structure

Three controls define the degree-wise width profile and radial function class:

```text
channels     = C_0 in {8, 16, 32, 64, 128}
lmax         = L   in {2, 3, 4}
radial_modes = R   >= 0
```

`C_0` is the width of the degree-zero moments, type embeddings, PairFiLM scale and shift, and edge
amplitudes. `L` is the largest angular degree retained. Angular degree is the order of directional
variation: degree zero is scalar, degree one transforms like a vector, degree two describes
quadrupole-like directional structure, and higher degrees resolve progressively finer angular
patterns. The defaults are `channels = 32` and `lmax = 2`.

`R` is the number of shared radial mode profiles. Each ordered atom-type pair mixes these profiles
with its own coefficients, as defined in Section 2.4. The default is `radial_modes = 0`.

Every degree reads the leading channels of one shared `C_0`-wide radial map. The radial function
that can be tabulated for compression therefore has width `C_0`, independent of `L`. Independent
degree-one and degree-two radial heads improved accuracy at a fixed `R`, but at matched inference
cost they lost to increasing `R` on the shared map. They would also enlarge every ordered-pair cache
by

```text
(C_0 + C_1 + C_2) / C_0.
```

The shared map is consequently part of the fixed design.

### 1.1 Degree channels

Each angular degree has `C_l` independent channels. Because every supported `C_0` is a power of two,
the profile is derived using integer shifts. Let

```text
kappa = bit_length(C_0) - 1.
```

Then

```text
C_1 = max(4, 1 << ((kappa + 1) // 2))
C_2 = max(4, C_1 >> 1)
C_l = 1,  l >= 3.
```

The complete channel profile is

```text
degree_channels = [C_0, C_1, C_2] + [1] * (L - 2).
```

Degrees one and two retain several channels because they carry most of the angular information.
Degrees three and four retain one channel each so that increasing `L` adds angular resolution
without allowing the moment state or quadratic readout to dominate the model width.

### 1.2 Bispectrum probe ranks

The exact Gram matrices in Section 5.2 use every channel. Cubic and quartic contractions are more
expensive, so they use lower-rank **probes**: learned linear combinations of the full channels.
Their ranks are fixed by the degree profile:

```text
K_1 = C_2
K_2 = 2
K_l = 1,  l >= 3

bispectrum_ranks = [K_1, K_2] + [1] * (L - 2).
```

For degree one, the exact Gram matrix determines the vectors up to a physical orthogonal
transformation. For degree two, the Gram matrix treats the five packed harmonic components as an
abstract five-dimensional space and is therefore unchanged under the much larger group `O(5)`.
Physical three-dimensional rotations occupy only a three-parameter subgroup of that space. Cubic
contractions recover additional physical orientation information for the probed channels.

`K_2` controls the width of both the cubic and quartic blocks. Raising it from the fixed value to
`C_2` produced a small accuracy gain, but the gain did not justify the larger invariant output.

The five base profiles at `L = 2` are:

| `C_0` | `degree_channels` | `bispectrum_ranks` |
| ----: | ----------------- | ------------------ |
|     8 | `[8, 4, 4]`       | `[4, 2]`           |
|    16 | `[16, 4, 4]`      | `[4, 2]`           |
|    32 | `[32, 8, 4]`      | `[4, 2]`           |
|    64 | `[64, 8, 4]`      | `[4, 2]`           |
|   128 | `[128, 16, 8]`    | `[8, 2]`           |

For `L = 3`, one channel and one probe are appended for degree three. For `L = 4`, the same is done
for degrees three and four. Five channel widths and three choices of maximum degree therefore define
15 structural profiles.

## 2. Edge representation

Consider a directed edge from neighbor `j` to center `i`. The physical displacement is the neighbor
position minus the center position:

```text
Delta_ij = r_j - r_i                     in ℝ^3.
```

The descriptor regularizes its length before forming a direction:

```text
rho_ij = sqrt(|Delta_ij|^2 + eps^2)
u_ij   = Delta_ij / rho_ij
eps    = 1e-7 Å.
```

Here `|Delta_ij|` is the ordinary Euclidean distance. The positive constant `eps` prevents division
by zero, so `u_ij` remains finite even for a coincident or guard edge. For physical separations much
larger than `eps`, `rho_ij` and `u_ij` agree with the usual distance and unit direction to numerical
precision.

### 2.1 C3 cutoff envelope

A cutoff envelope is a scalar weight that equals zero at and beyond `rcut`. It lets a neighbor
contribution disappear smoothly instead of being removed abruptly. `C3` means that the envelope
value and its first three derivatives are continuous at the cutoff.

For any regularized distance `rho`, define the scaled distance and remaining fraction

```text
x = rho / rcut
t = clamp(1 - x, 0, 1).
```

`clamp(y, 0, 1)` limits `y` to the interval from zero to one. With the fixed envelope exponent `p = 5`, DPA4C uses

```text
chi(rho)
  = t^4 [1 + 4x + 10x^2 + 20x^3 + 35x^4].
```

The envelope equals zero when `rho >= rcut`, and its first three radial derivatives join
continuously to zero there. Excluded edges, padding edges, and edges involving a non-physical
padding type receive an additional zero mask.

### 2.2 Parameter-matched radial SwiGLU

A **radial basis** represents one distance by several fixed analytic functions. This gives a small
neural network a richer input than the raw distance alone. DPA4C uses the DPA4 Bessel or Gaussian
basis:

```text
f(rho_ij) in ℝ^(n_radial).
```

The basis is evaluated without a cutoff factor in this descriptor; the single explicit envelope is
applied after radial and chemical information have been combined.

The learned radial map is a bias-free, one-hidden-layer SwiGLU network. `SiLU(x) = x / (1 + exp(-x))`. A SwiGLU layer forms a gate branch and a value branch, applies SiLU to the gate, and
multiplies the two branches element by element:

```text
[h_gate, h_value] = f(rho_ij) W_in
h_ij              = h_value * SiLU(h_gate)
g_ij              = h_ij W_out              in ℝ^(C_0).
```

The post-gate hidden width is

```text
H_radial = 8 * ceil(C_0 / 3),
```

where `ceil` rounds upward. The affine map before the gate emits `2 H_radial` values because it
contains both branches.

### 2.3 Ordered PairFiLM

Different ordered chemical pairs need different edge responses. **FiLM**, or feature-wise linear
modulation, supplies a per-channel scale and shift. It is ordered because a center of type `a` with
a neighbor of type `b` need not use the same table entry as center type `b` with neighbor type `a`.

Let `e_a` and `e_b`, each in `ℝ^(C_0)`, be the type embeddings. Concatenate them and evaluate one
finite type-pair network:

```text
z_ab = concat(e_a, e_b)                    in ℝ^(2 C_0)
h_ab = SwiGLU(z_ab W_pair,in)
[h_ab^scale, h_ab^shift, h_ab^mix] = 0.1 h_ab W_pair,out

gamma_ab = 1 + tanh(h_ab^scale)            in ℝ^(C_0)
beta_ab  = e_a + e_b + tanh(h_ab^shift)    in ℝ^(C_0)
U_ab     = reshape(tanh(h_ab^mix), C_0, R) in ℝ^(C_0 x R).
```

The `h_ab^mix` and `U_ab` blocks exist only when `R > 0`. The PairFiLM post-gate hidden width is

```text
H_pair = 8 * ceil(2 C_0 / 3).
```

The bounded functions keep the cached values well conditioned in `float32`: every component of
`gamma_ab` lies in `(0, 2)`, while the learned residual parts of `beta_ab` and `U_ab` lie in `(-1, 1)`.

Without radial modes, the pre-envelope edge amplitude is

```text
psi_ij = gamma_ab * g_ij + beta_ab         in ℝ^(C_0),
```

where `*` denotes elementwise multiplication.

### 2.4 Pair-conditioned radial modes

With only the scale and shift above, every ordered pair modifies the same shared radial shape
channel by channel. For `R > 0`, a linear head from the shared hidden state produces `R` additional
distance profiles:

```text
q_ij = q(rho_ij) = h_ij W_mode             in ℝ^R.
```

The pair-specific matrix `U_ab` mixes these profiles into every edge channel:

```text
psi_ij,c
  = gamma_ab,c g_ij,c
    + beta_ab,c
    + sum_(mu=1)^R U_ab,c,mu q_ij,mu,

psi_ij in ℝ^(C_0).
```

The mixing covers the complete edge width. Restricting it to scalar channels or to non-scalar
channels caused a large accuracy loss on every measured metric and removed the benefit of increasing
`R`.

At fixed `C_0`, increasing `R` improved the measured energy, force, and virial errors monotonically
until a width-dependent knee. The knee moved to larger `R` as `C_0` increased. After compression,
this mechanism remains one one-dimensional radial table plus one finite ordered-pair cache.

The cutoff envelope multiplies the complete radial and chemical amplitude exactly once:

```text
phi_ij = chi_ij psi_ij                    in ℝ^(C_0),
chi_ij = chi(rho_ij).
```

`psi_ij` is therefore the pre-envelope amplitude, and `phi_ij` is the enveloped amplitude used by
the moment reduction.

### 2.5 Inner bridging switch and distance clamp

Zone bridging adds an analytical short-range pair potential to the learned energy (Section 12.4). For the sum to reproduce that potential at close approach, the learned energy must not depend on the geometry of a close pair at all. DPA4 needs two mechanisms for this, a distance clamp and a gate on every source node, because a message-passing network leaks the pair through the scalar distance, through the edge direction, and through neighbors that have already seen it. DPA4C has none of these channels beyond the edge itself: no node feature crosses an edge, and every term an edge contributes carries the envelope `chi_ij`, namely the amplitude `chi psi`, the two masses `chi^2` and `chi^4`, the non-scalar moments `chi^2 psi B`, and the spin payload `chi^2`. One inner switch on the envelope therefore closes all of them at once, and a clamp on the separation the radial functions read holds the shape of the pair's radial features while the switch closes:

```text
chi_ij = chi(rho~_ij) w(rho_ij),   g_c(rho~_ij),   q_rho(rho~_ij)

t     = clamp((rho - r_inner) / (r_outer - r_inner), 0, 1)
w     = 35 t^4 - 84 t^5 + 70 t^6 - 20 t^7
w'    = 140 t^3 (1 - t)^3 / (r_outer - r_inner)
rho~  = r_mid + (r_outer - r_inner) S(t),   r_mid = (r_inner + r_outer) / 2
S     = 7 t^5 - 14 t^6 + 10 t^7 - 2.5 t^8,   S' = w,  S(0) = 0,  S(1) = 1/2.
```

`w` is the septic smootherstep that DPA4 uses for its source gate. It vanishes with three vanishing derivatives at `r_inner` and reaches one with three vanishing derivatives at `r_outer`, so the descriptor stays `C3` across both radii, like across `rcut`. At and below `r_inner` the edge is multiplied by exactly zero everywhere it appears, which makes the descriptor of both atoms bit-identical to the descriptor of the same graph with that edge deleted, including the neighborhood masses and therefore the normalization. At and beyond `r_outer` the descriptor is bit-identical to the descriptor without a window. Both statements are pinned by tests against an explicit pair exclusion and against the plain descriptor.

The switch reads the true separation `rho`; the cutoff factor, the tabulated radial map `g_c` and the mode profiles `q_rho` read the clamped separation `rho~`. The direction of the edge is the true one throughout. `S` is the antiderivative of `w`, so the clamped separation moves with the true one at exactly the rate at which the switch has opened, `d rho~ / d rho = w`: it is frozen at the window midpoint at and below `r_inner`, meets the true separation with four vanishing derivatives at `r_outer`, and through the lower half of the window rises by only `S(1/2) = 35/512`, under 7 % of the window width. Below the midpoint, where the training-frame filter of Section 12.4 keeps no label, the radial features of the pair are therefore held within the range the retained frames cover and the learned energy of the pair follows the amplitude `w` alone. The force takes the chain rule through the clamp: every table term is differentiated with respect to `rho~` and multiplied by `w`, the envelope by `chi'(rho~) w^2 + chi(rho~) w'`, and the analytical pair term by nothing, since it acts on the true separation. The clamp is a fixed profile without parameters and shares the window of the switch.

The two radii are not absolute lengths. Each is a fraction of the length scale of the pair it applies to, `s = R[a] + R[b]`, with `R` the covalent radius of an element, so the window follows the size of the pair and one setting covers every element combination in the type map. The switch and the clamp are therefore evaluated in the reduced distance `u = r / s`, with `t = (u - f_inner) / (f_outer - f_inner)`, and the frozen separation is `s (f_inner + f_outer) / 2`. A window given directly in Å is the same rule measured against a unit scale: every element takes the radius 0.5 Å, every pair's scale is 1 Å, and the fractions carry the radii themselves, which keeps one code path through the descriptors, the training-data filter and the fused kernels.

The window is descriptor configuration without parameters. The bridging composition injects it under the names `inner_clamp_f_inner`, `inner_clamp_f_outer` and `inner_clamp_scale`, which it uses for every DPA4-family descriptor; the descriptor stores it as `bridging_f_inner`, `bridging_f_outer` and `bridging_scale`, derives the per-element table `contact_radius` from the type map as non-persistent state, and serializes the three from format version 3 on. Like `exclude_types`, it is branch-local under parameter sharing.

## 3. Real Cartesian harmonics

Distance alone cannot describe the arrangement of neighbors around a center. Angular functions
provide that directional information. Spherical harmonics are a standard orthogonal family of
functions on the sphere. DPA4C uses an equivalent real-valued Cartesian form: low-degree polynomials
of the direction coordinates `(u_x, u_y, u_z)`. This avoids complex numbers and is convenient for
explicit CUDA kernels.

`B_l(u)` denotes the block of real Cartesian harmonics at angular degree `l`. It has `2l + 1`
components:

```text
B_l(u) in ℝ^(2l + 1).
```

Degree zero is a constant scalar, degree one is the three-component direction, and degree two has
five independent components. Concatenating degrees zero through `L` gives

```text
B(u) = concat(B_0(u), ..., B_L(u))
       in ℝ^((L + 1)^2),
```

because `1 + 3 + 5 + ... + (2L + 1) = (L + 1)^2`.

The harmonic blocks are equivariant rather than invariant: rotating or reflecting `u` mixes
components within the same degree. Their normalization is fixed by the following identity. For
nonzero vectors `u` and `v`,

```text
B_l(u) dot B_l(v)
  = (|u| |v|)^l P_l((u dot v) / (|u| |v|)),
```

where `P_l` is the degree-`l` Legendre polynomial. The explicit Cartesian polynomials provide the
finite continuation when either vector is zero. Reflection changes only a parity sign:

```text
B_l(-u) = (-1)^l B_l(u).
```

The portable path keeps the explicit powers of

```text
|u_ij|^2 = 1 - eps^2 / rho_ij^2.
```

The compressed kernel substitutes `|u_ij|^2 = 1`. This is exact on the unit sphere. For every
physical separation much larger than `eps`, the relative correction `eps^2 / rho_ij^2` is far below
single-precision resolution. At `Delta_ij = 0`, however, `rho_ij = eps` and `u_ij = 0`, so the
portable and compressed polynomial values are not identical. Such a coincident physical edge lies
outside the intended molecular-dynamics domain.

For degree two, the five packed coefficients can be converted into a `3 x 3` symmetric-traceless
matrix. “Symmetric” means `Q^T = Q`, and “traceless” means the three diagonal entries sum to zero.
The conversion `STF` is isometric:

```text
packed_1 dot packed_2 = STF(packed_1) : STF(packed_2).
```

The colon denotes the sum of elementwise matrix products, also called the Frobenius inner product.
This conversion is used by the `112` bispectrum contraction and the projected quartic in Section 5.

## 4. One-reduction degree-wise moments

A **moment** is a weighted sum over neighbors. It summarizes a variable-size neighborhood into a
fixed-size tensor. Degree zero sums scalar-like information. Degree `l > 0` sums each edge amplitude
multiplied by the corresponding angular block.

The flat width of all moments is

```text
S = sum_(l=0)^L (2l + 1) C_l.
```

The edge amplitude `phi_ij` already contains one envelope factor. Degree zero uses that amplitude
directly. Every non-scalar degree multiplies by one additional `chi_ij`, so its complete edge weight
contains two envelope factors.

The two smooth neighborhood masses match these two cases:

```text
d_i^(0) = sum_j chi_ij^2
d_i^(+) = sum_j chi_ij^4.
```

These are “masses” rather than integer coordination counts because a neighbor near the cutoff
contributes only a fraction. The fixed floor prevents a nearly empty environment from producing a
large normalization:

```text
n_i^(0) = 1 / sqrt(d_i^(0) + 0.25)
n_i^(+) = 1 / sqrt(d_i^(+) + 0.25).
```

`n_i^(0)` and `n_i^(+)` are multipliers. Equivalently, the raw sums are divided by

```text
1 / n_i^(0) = sqrt(d_i^(0) + 0.25)
1 / n_i^(+) = sqrt(d_i^(+) + 0.25).
```

For each edge, DPA4C constructs one flat payload:

```text
payload_ij = [
  chi_ij^2,
  chi_ij^4,
  {phi_ij,c                                      for c < C_0},
  {chi_ij phi_ij,c B_l,m(u_ij)                  for l >= 1,
                                                     m < 2l + 1,
                                                     c < C_l}
] in ℝ^(S + 2).
```

One destination segment sum over `j` produces both masses and every unnormalized moment. The
normalizers are then applied:

```text
X_i,0,c^(0)
  = n_i^(0) sum_j phi_ij,c,

X_i,m,c^(l)
  = n_i^(+) sum_j chi_ij phi_ij,c B_l,m(u_ij),   l >= 1.
```

For each degree,

```text
X^(l) has shape (N, 2l + 1, C_l).
```

The degree-zero harmonic axis has length one. The persistent moment state is stored flat, without
copying:

```text
offset[0]     = 0
offset[l + 1] = offset[l] + (2l + 1) C_l.
```

The descriptor reduces the edge axis exactly once. It constructs no explicit edge pairs,
neighbor-neighbor products, or source-node features.

## 5. Fixed invariant readout

The moments `X^(l)` transform equivariantly under `O(3)`. The fitting network requires invariant
scalars. The readout therefore combines harmonic components only through contractions whose value is
unchanged by rotations and reflections. It contains learned linear channel maps but no learned
nonlinear network and no message passing.

### 5.1 Full channel alignment

Degrees one and two first apply independent full-width residual channel maps:

```text
X_tilde^(l) = X^(l) (I + W_l),    l in {1, 2}.
```

`I` is the identity matrix and `W_l` has shape `(C_l, C_l)`. The multiplication acts only on the
final channel axis, so `X_tilde^(l)` retains shape `(N, 2l + 1, C_l)`.

Degrees three and four each have one channel and retain it directly:

```text
X_tilde^(l) = X^(l),              l >= 3.
```

### 5.2 Exact aligned Gram matrices

A **Gram matrix** stores every pairwise inner product among a set of vectors. For degree `l`, each
channel is a vector of length `2l + 1` over harmonic components. Their Gram matrix is

```text
G_i,c,d^(l)
  = sum_m X_tilde_i,m,c^(l) X_tilde_i,m,d^(l),

G^(l) has shape (N, C_l, C_l).
```

An orthogonal transformation of the harmonic components leaves each inner product unchanged, so
`G^(l)` is O(3)-invariant. The matrix is symmetric. Only its upper triangle is emitted, giving

```text
C_l (C_l + 1) / 2
```

features at degree `l`.

Each strict off-diagonal entry is multiplied by `sqrt(2)`. This isometric half-vectorization,
written `vech_iso`, preserves the Frobenius norm:

```text
|vech_iso(G)|^2 = |G|_F^2.
```

The Frobenius norm `|G|_F` is the square root of the sum of all squared matrix entries. The
`sqrt(2)` factor accounts for the equal lower-triangular entry that is not stored.

### 5.3 Derived low-rank probes

The Gram blocks retain all `C_l` channels. The higher-order contractions use only `K_l` learned
combinations, called probes. When `K_l < C_l`, a projection matrix `A^(l)` with shape `(C_l, K_l)`
forms

```text
Z_i,m,k^(l)
  = sum_c X_tilde_i,m,c^(l) A_c,k^(l),

Z^(l) has shape (N, 2l + 1, K_l).
```

The projection columns are initialized orthonormally, so they initially preserve lengths inside the
selected subspace. If `K_l = C_l`, the projection is the identity and no projection parameter is
stored.

### 5.4 Generic O(3)-even bispectrum

The **bispectrum** used here is a set of third-order invariants. It multiplies three equivariant
harmonic blocks and contracts all angular indices with fixed coefficients. “O(3)-even” means the
result remains unchanged under both rotations and reflections.

Only sorted degree triples satisfying the angular triangle rule and even parity are allowed:

```text
1 <= l_1 <= l_2 <= l_3 <= L
l_3 <= l_1 + l_2
l_1 + l_2 + l_3 is even.
```

The triangle rule selects triples that can couple to a scalar. Even total degree follows from
`B_l(-u) = (-1)^l B_l(u)` and removes reflection-odd pseudoscalars.

For `L = 4`, the allowed triples are

```text
112  123  134
222  224  233  244  334  444.
```

A **Gaunt coefficient** is the integral of a product of three harmonics over all directions on the
unit sphere. For each allowed triple, DPA4C forms a real Cartesian Gaunt tensor:

```text
mathcal_G_m1,m2,m3^(l1,l2,l3)
  = normalize_F [
      integral_S2
        B_l1,m1(u) B_l2,m2(u) B_l3,m3(u) dOmega
    ].
```

`S2` denotes the unit sphere and `dOmega` its surface-angle element. `normalize_F` scales the
complete tensor to unit Frobenius norm. The implementation also fixes a deterministic overall sign.

The invariant contraction is

```text
J_i,alpha1,alpha2,alpha3^(l1,l2,l3)
  = sum_m1,m2,m3
      mathcal_G_m1,m2,m3^(l1,l2,l3)
      Z_i,m1,alpha1^(l1)
      Z_i,m2,alpha2^(l2)
      Z_i,m3,alpha3^(l3).
```

If two or three degrees are equal, permuting the corresponding probe indices gives the same value.
Only non-decreasing representatives are emitted. Multiplicity factors `sqrt(2)`, `sqrt(3)`, or
`sqrt(6)` preserve the norm of the corresponding full symmetric tensor.

The output count for one degree triple is

```text
all degrees distinct:
  K_l1 K_l2 K_l3

exactly two degrees equal:
  K_equal (K_equal + 1) K_other / 2

all three degrees equal:
  K_l (K_l + 1) (K_l + 2) / 6.
```

`D_bispectrum` denotes the sum of these counts over all allowed triples.

### 5.5 Projected Qv quartic

The degree-one probes can be viewed as ordinary vectors, and the degree-two probes as
symmetric-traceless matrices:

```text
v_alpha = Z^(1)[:, :, alpha]             in ℝ^3
Q_eta   = STF(Z^(2)[:, :, eta])          in ℝ^(3 x 3).
```

For every pair of probes, the readout emits

```text
Qv_eta,alpha
  = |Q_eta v_alpha|^2
  = v_alpha^T Q_eta^2 v_alpha.
```

This fourth-order quantity is invariant because it is a squared vector length. The matrix-vector
intermediate `Q_eta v_alpha` is shared with the `112` bispectrum contraction. The quartic therefore
contributes

```text
K_1 K_2
```

features at negligible additional contraction cost.

## 6. Native spin

DPA4C optionally conditions on a per-atom magnetic moment, so that the energy depends not only on
where the atoms are but also on how their moments are oriented. Four interactions dominate that
dependence in a magnetic solid, and the construction below is designed to represent exactly them:

- **exchange**, which favors neighboring moments that are parallel or antiparallel;
- **biquadratic exchange**, a weaker correction that depends on the same relative angle squared;
- **single-ion anisotropy**, which ties a moment to the geometry of its own neighborhood and so
  picks out easy and hard directions;
- **symmetric anisotropic exchange**, also called pseudo-dipolar or Kitaev-like exchange, which ties
  a *pair* of moments to the direction of the bond that joins them.

The last two are of the same spin-orbit order, so a construction that carries one and not the other
is not a controlled approximation. Section 6.4 shows that each emitted coordinate is exactly one of
these sums, not an approximation of one. The rest of the section builds up to that identification.

The moment enters as additional equivariant channels reduced by the same single destination scan,
and the magnetic force is the negative spin gradient of the energy. No virtual atoms are created,
so the type map, the neighbor selection and the type count stay at their physical values.

The feature is enabled by the per-type flags `use_spin`, which the model construction layer injects
from the standard `model.spin` block with `scheme = "native"`. Without them the descriptor is
bit-for-bit the one described above.

A spin-conditioned descriptor requires a moment on every call. Supplying none is an error, not a
demagnetized configuration: the two differ in that the second is a physical state with a defined
magnetic force while the first would report an identically zero one, which in molecular dynamics is
indistinguishable from frozen moments under a perfectly plausible energy. A partially labelled
corpus is admitted through `model.spin.allow_missing_label`, which relaxes the `spin` data
requirement to optional with a zero default, so the data pipeline supplies an explicit zero moment
and the contract still holds.

### 6.1 Conditioned spin

Write `s_i` for the raw moment of atom `i`, in the units of the dataset. The descriptor forms one
conditioned quantity and derives every route from it:

```text
s_hat_i = m_a / s_ref_a * s_i              in ℝ^3.
```

`m_a` is the per-type gate, one for a magnetic type and zero otherwise. `s_ref_a` is the per-type
root-mean-square moment measured by `compute_input_stats`; it is a constant of the type, so
`s_hat` remains linear in the input and the reference is stored with the model. Dividing by it puts
elements whose moments differ by a factor of a few onto a common scale before the fixed
preconditioner sees them, which matters because the spin coordinates below are quadratic and
quartic in the moment.

The gate is **multiplicative** and applied exactly once. A non-magnetic atom therefore has no
magnetic degree of freedom at any derivative order, not merely a vanishing value. This is stronger
than relying on the dataset convention `s = 0`, because the force loss differentiates the magnetic
force a second time and that derivative probes the spin direction even where the value is zero. The
gate and the reference are combined into a single multiplicative table rather than a division, so a
type whose reference is degenerate cannot produce a non-finite intermediate.

The trailing padding type carries a zero gate, so ghost and padding rows contribute nothing.

### 6.2 Symmetry grading

Two operations leave the energy of a zero-field magnet unchanged, and each one constrains what the
descriptor may output.

**Spatial inversion** sends every position to its negative. An edge direction flips, but a magnetic
moment does not: a moment is generated by a circulating current, and reflecting the loop leaves the
circulation unchanged. A vector that behaves this way is called **axial**, as opposed to the
**polar** edge direction. **Time reversal** runs the clock backwards. Now the geometry is untouched
while every moment flips, because reversing time reverses the current that produces it.

Under either operation a term of the descriptor can only come back as itself or as minus itself.
Write `p` for that sign under inversion and `t` for it under time reversal. Since the energy is
unchanged, only terms with `p = +1` and `t = +1` may be emitted.

To predict the two signs, label each channel by two integers: its angular degree `l`, as everywhere
else in this document, and its **spin order** `sigma`, meaning how many factors of the moment the
channel is built from, counted modulo two. A channel linear in the moment has `sigma = 1`, one
quadratic in it has `sigma = 0`, and a purely geometric channel has `sigma = 0`.

Angular couplings are restricted to genuine Gaunt couplings, that is `l_1 + l_2 + l_3` even, which
the readout already enforces and which is unavoidable because the integral of three degree-one
harmonics over the sphere vanishes (Section 5). Under that restriction the two signs are

```text
p = (-1)^l (-1)^sigma,        t = (-1)^sigma.
```

The extra `(-1)^sigma` in `p` is exactly what distinguishes an axial moment from a polar direction:
counting only `l`, as a purely geometric descriptor may, would wrongly admit a term such as
`s_hat_i . u_ij`, whose two degrees sum to an even number yet which changes sign under inversion.

Requiring `prod p = +1` and `prod t = +1` therefore decouples into

```text
sum of l      is even      the rule the readout already enforces
sum of sigma  is even      the only rule native spin adds.
```

Restricting the emitted invariants to even total spin order leaves the descriptor invariant under
the full physical group of a zero-field magnetic energy, and it makes the polar and axial
conventions indistinguishable: an odd number of spin factors never reaches the output, so nothing
changes if the moment is transformed with or without the determinant of an improper rotation.

The rule is realized by the **block structure** rather than by a per-entry filter. Families of even
spin order live in their own blocks whose Grams are emitted, and the families of odd spin order
share one degree-one block whose cross Gram against the geometric degree-one moments is simply never
formed.

### 6.3 Spin families

All five families reuse the shared radial map and one additional ordered type-pair cache. Their
common edge weight mirrors the geometric pair `psi_ij` and `phi_ij` of Section 2:

```text
psi^s_ij,c = gamma^s_ab,c g_c(rho_ij) + beta^s_ab,c
phi^s_ij,c = chi_ij^2 psi^s_ij,c,                       c < C_s.
```

`C_s = C_2` is derived from the degree profile, so every spin channel addresses an already-evaluated
radial channel and the radial table keeps its width `C_0 + R`. The squared envelope matches the
weight of every non-scalar geometric moment, so the neighbour families share the normalization
multiplier `n_i^(+)` of Section 4 and no third neighborhood mass is introduced. Unlike its geometric
counterpart, `gamma^s_ab` is signed: the exchange interaction of an ordered pair may be
ferromagnetic or antiferromagnetic, and a radial map shared across pairs cannot supply that sign.

Each of the two non-scalar blocks also carries one **on-site** channel, the moment of the center
atom itself, weighted by a per-type scalar: `lambda_a` at degree one and `mu_a` at degree two. These
two channels are node local, so they cost no edge work.

| Family | `l` | `sigma` | Neighbor width | On-site channel     | Neighbor channels                              |
| ------ | --: | ------: | -------------: | ------------------- | ---------------------------------------------- |
| `V`    |   1 |       1 |          `C_s` | `lambda_a s_hat_i`  | `n^(+) sum_j phi^s_ij,c s_hat_j`               |
| `P`    |   1 |       1 |          `C_s` | shared with `V`     | `n^(+) sum_j phi^s_ij,c (s_hat_j . u_ij) u_ij` |
| `Q`    |   2 |       0 |            `1` | `mu_a B_2(s_hat_i)` | `n^(+) sum_j phi^s_ij,0 B_2(s_hat_j)`          |
| `M0`   |   0 |       0 |          `C_s` | none                | `n^(+) sum_j phi^s_ij,c \|s_hat_j\|^2`         |
| `Mw`   |   0 |       0 |          `C_s` | none                | `n^(+) sum_j phi^s_ij,c m_b`                   |

`M0` accumulates the moment magnitudes of the neighbors and `Mw` counts, with the same radial
weight, how many of them are magnetic at all. The on-site channels are written after the
destination division, so an invariant that pairs an on-site channel with a neighbor channel carries
exactly one factor of `n^(+)` rather than two.

`B_2` is the same real Cartesian harmonic block of Section 3, evaluated on the conditioned moment
instead of on a direction. Because that construction is a homogeneous polynomial of its argument, it
is smooth at a vanishing moment and no square root of a magnitude ever appears.

**`V` and `P` occupy one block of width `1 + 2 C_s`**: the shared on-site channel first, then the
`C_s` isotropic channels, then the `C_s` bond-projected ones. They may share a block because they
share a grading. `P` carries one moment and two factors of the unit bond direction, so under spatial
inversion both direction factors flip while the moment does not and the product is even; under time
reversal only the moment flips, so it is odd. Those are the two signs of `V`. Sharing the block is
not a convenience: it is why the existing half Gram emits every new invariant with no additional
filtering rule and no special case.

`P` is given the full spin width rather than a narrower one. The readout is linear in the emitted
Gram entries, so the effective radial profile of an interaction is the span of the channel
amplitudes the fitting network mixes: a family of `K` channels reaches a `K`-dimensional space of
radial functions per ordered type pair. `V` reaches `C_s` of them and the single-ion anisotropy
reaches `C_2 = C_s` through its cross Gram, so narrowing `P` would confine an interaction of the
same spin-orbit order to a strictly smaller function space than its neighbours. Reusing the leading
`C_s` entries of the same edge amplitude costs no extra radial evaluation; the price is `3 C_s`
additional accumulators per edge and the quadratic growth of the block Gram.

**`P` reads the edge direction; the other four do not.** The spin branch is therefore no longer
purely radial. Its backward pass carries an angular cotangent alongside the radial one, so the
gradient of a spin coordinate with respect to the atomic positions flows through the direction as
well as through the distance. The direction is the same regularized `u_ij` the harmonics use, so it
is finite for coincident and guard edges and the family inherits the smoothness of Section 3.

### 6.4 Spin invariants

Four contractions of even spin order are emitted, in this order:

| Order | Block                                                        |                    Width |
| ----: | ------------------------------------------------------------ | -----------------------: |
|     1 | Upper triangle of the joint `V`/`P` Gram                     | `(1 + 2C_s)(2 + 2C_s)/2` |
|     2 | Upper triangle of the `Q` Gram without its on-site self-term |                      `2` |
|     3 | Cross Gram of `Q` against the geometric `X^(2)`              |                `2 * C_2` |
|     4 | `M0` and `Mw`                                                |                `2 * C_s` |

A Gram is the table of all pairwise dot products between the channels of a block, so pairing the
on-site channel with a neighbor channel produces a sum over neighbors. Applying the harmonic
addition theorem of Section 3 to the conditioned moment then identifies each such entry with a
physical interaction, exactly rather than approximately:

```text
V/P Gram, on-site x V channel c
  = sqrt(2) lambda_a n^(+) sum_j phi^s_ij,c (s_hat_i . s_hat_j)            Heisenberg exchange

V/P Gram, on-site x P channel c
  = sqrt(2) lambda_a n^(+) sum_j phi^s_ij,c
      (s_hat_i . u_ij) (s_hat_j . u_ij)                                    anisotropic exchange

Q Gram, on-site x neighbor
  = sqrt(2) mu_a n^(+) sum_j phi^s_ij,0
      [3 (s_hat_i . s_hat_j)^2 - |s_hat_i|^2 |s_hat_j|^2] / 2              biquadratic exchange

Q cross X^(2), on-site x channel d
  = mu_a n^(+) sum_j chi_ij phi_ij,d
      [3 (s_hat_i . u_ij)^2 - |s_hat_i|^2] / 2                             single-ion anisotropy
```

The first is the plain angle between two moments and the third the same angle squared; the second
projects both moments of a pair onto the bond that joins them, and the fourth the angle between a
single moment and the direction to a neighbor. The `sqrt(2)` factors come from the isometric
half-vectorization of Section 5, which stores each off-diagonal Gram entry once.

The `Q` Gram omits its on-site self-term. The harmonic blocks are homogeneous, so
`|B_2(s)|^2 = |s|^4` exactly, and that entry is therefore a per-type constant times the square of
the on-site self-term the `V`/`P` Gram already emits.

The remaining entries pair one neighbor channel with another. These are double sums over the
neighborhood and express its collective magnetic order rather than any single pair; the `V`-against-
`P` and `P`-against-`P` entries are the collective counterparts of the anisotropic exchange.

`M0` reports how large the surrounding moments are, irrespective of their orientation. That
longitudinal information is what varies most in a metal whose magnetism comes from delocalized
electrons, where the size of a moment responds to its environment instead of staying fixed.

`Mw` is the one family that does not read the spin value at all, so it is nonzero even when every
moment vanishes. It is retained because every other spin family vanishes with the moments, and
without it the readout cannot distinguish a neighborhood containing no magnetic species from a
magnetic neighborhood that happens to be demagnetized. Being independent of the spin value, it
contributes nothing to the magnetic force.

Two contractions are deliberately absent. The cross Gram of the degree-one spin block against the
geometric degree-one moments has odd spin order and is forbidden by the grading of Section 6.2.

**Antisymmetric spin bilinears, and with them the Dzyaloshinskii-Moriya interaction, are unreachable
at every order.** This is not a two-body-order limitation that a higher term would repair: the
readout contracts channels only through symmetric Grams and through Gaunt couplings that are
symmetric in their channel indices, so a term proportional to `s_i x s_j` — antisymmetric under
exchanging the two moments — is never formed anywhere in the descriptor. Representing it would
require an explicitly antisymmetric contraction, which the present construction does not contain.

### 6.5 Properties

The following hold to machine precision and are covered by property tests rather than stored values:

- **Axial O(3) invariance.** Rotating the structure and its moments together leaves every output
  unchanged, and so does composing that rotation with an inversion. The same holds if the moments
  are instead transformed as polar vectors, that is with the sign of the inversion applied to them
  too: no output has odd spin order, so the two conventions can never be told apart.
- **Exact time-reversal evenness.** Flipping every moment reproduces the descriptor bitwise, and the
  magnetic force changes sign.
- **Exact masking.** Feeding a non-magnetic type an arbitrary moment changes nothing bitwise, and its
  magnetic force and the corresponding rows of the per-type on-site weights are exactly zero through
  the second derivative.
- **Bond-direction resolution.** Permuting the moments of several equidistant neighbors of the same
  type leaves every radial family invariant, because each of them reads the moments only through a
  sum that is symmetric in the neighbor index. The bond-projected family is not invariant, and the
  descriptor separates the two states accordingly.
- **Vanishing magnetic force at zero moment.** Every coordinate that reads the moment value is even
  in it, so `-dE/ds` vanishes identically in the demagnetized limit and the two per-type on-site
  weights receive exactly zero gradient. The ordered spin tables do keep training, because the `Mw`
  family they also weight counts magnetic neighbors rather than their moments. Note that this makes
  a zero moment weaker than a spin-free descriptor: `Mw` reads the per-type mask, so naming a type
  magnetic moves the descriptor even before any moment is supplied. Section 6.6 gives the gate that
  closes the branch outright, which is what a spin-free pretraining actually transfers through.
- **Smoothness.** Every route is polynomial in the conditioned moment or a smooth function of its
  squared magnitude, so all derivatives are continuous as a moment crosses zero.

### 6.6 Execution and calibration

Mixed precision never engages for a spin-conditioned descriptor. The scalar and quadrupole families
are quadratic in the moment and feed a fourth-order readout, and the magnetic force differentiates
them twice, so the eight mantissa bits of bfloat16 are not an acceptable trade for a configuration
whose throughput is not the binding constraint.

The two ordered spin tables are **anchored at initialization**. The geometric heads of Section 2.3
are anchored structurally — `gamma` on the constant one, `beta` on the type embedding — and start at
root-mean-square 1 and 0.25 respectively. `gamma^s` may not take a constant anchor, because the sign
of an ordered exchange amplitude has to stay free, and the pair encoder is bias free, so without an
anchor of its own a spin table emerges from a small centred product and starts near `2e-4`. Because
the calibration below freezes a preconditioner at whatever scale it measures, that would fix the
spin block four orders of magnitude away from the scale it reaches after the first optimizer steps,
swamping in particular the degree-zero coordinates that carry the single-ion anisotropy. Both spin
heads are therefore biased by a learned per-channel offset whose initial value has a fixed magnitude
and a random sign, placing every entry of both tables at exactly `0.5` before training and leaving
the sign of every channel free.

`compute_input_stats` measures the per-type reference magnitudes before it measures the output
coordinates, because the reference rescales the moment the coordinates are built from. It reads the
moment from either `model_spin` or `spin`, the two packings the training pipelines use, and raises
if a spin-conditioned descriptor is calibrated on a system that carries neither. Each output
coordinate is then calibrated over the nodes on which it is **active** rather than over all nodes. A
spin coordinate is exactly zero wherever no magnetic information reaches it, so pooling over all
nodes would scale the preconditioner with the magnetic fraction of the sample, that is with the
stoichiometry rather than with the physics. A coordinate that never activates — the normal state of
every spin coordinate on a demagnetized corpus — takes the identity preconditioner instead of
raising.

That pretrain-then-fine-tune sequence divides the per-type spin quantities into derived and measured
ones, and only the second kind transfers. The gate is a function of `use_spin` alone, so it is
rebuilt from the configuration on every load and is deliberately absent from the checkpoint: a
pretraining that names no magnetic species carries an all-zero gate, and inheriting it would silence
the spin channel and pin the magnetic force at exactly zero for the whole fine-tuning run while
every other diagnostic stays healthy. The reference magnitudes are measured state and therefore do
transfer, so a fine-tune that starts from a demagnetized pretraining inherits the identity
preconditioner rather than the magnitudes of its own corpus; the residual scale is a constant factor
in front of a trainable per-type table, which absorbs it during fine-tuning.

**The branch gate.** One scalar multiplies the calibrated spin block, and it is the only place the
whole branch passes through. It exists because activating a magnetic type on a demagnetized
pretraining otherwise releases weights that never received a gradient: the ordered spin tables sit
at their anchored initialization, the two on-site tables at a unit normal, and the fitting network's
input columns for the spin block were never trained either, since that block was identically zero
throughout the pretraining. Measured on a 118-type demagnetized pretraining transferred to magnetic
FeC, naming `Fe` magnetic moves the energy by up to 4 eV per atom with a configuration-dependent
sign and produces magnetic forces up to 70 eV/uB, which is what forces the output-bias regression to
solve for a per-type constant of several keV.

No weight inside the branch can prevent that. The families reach the fitting network by several
routes and at two spin orders, `Mw` does not read the moment at all, and a factor applied to the
conditioned moment instead would enter the degree-one Grams squared and the quadrupole Grams to the
fourth power, making zero a stationary point from which the branch could never reopen. Applied to
the block the gate is linear, so zero is the exactly spin-free descriptor **and** a point whose
gradient is the branch itself. A fresh descriptor starts there, and so does one that fine-tuning
activates: a pretraining that declares no magnetic type cannot move the gate, because its mask
zeroes every family and the gate's gradient vanishes with them, so the transfer carries the closed
value it was constructed with. The weights inside the branch keep their values, which is what
leaves the gate something to reopen on.

The gate multiplies the block **after** the calibration rather than before it. A closed gate then
feeds the fitting network exactly zero whatever preconditioner was measured, instead of the
constant `-m/s` a pre-calibration gate would leave behind, so the transfer holds even when the
statistics are remeasured on the magnetic corpus.

The gate is learned state and rides the checkpoint. A payload that predates it carries no value to
load and the runtime invents none: such a checkpoint is stamped once, offline, with the gate its own
configuration implies — open for a run that declared magnetic types, since it trained with the block
undamped, and closed for one that declared none. The compiled operator assembles the
invariants without reading the gate, so compression carries it as a factor on the inverse deviation
of the spin columns alone. That single array is what the kernel applies in the forward and what it
pulls every output cotangent back through in the backward, so one factor covers both, and a closed
gate is a zero slope rather than a state the calibration cannot express — a trained gate may
legitimately reach zero, and nothing about the gate restricts what may be compressed.

The compiled CUDA operator carries a spin specialization on every supported profile, selected by a
compile-time flag rather than by a width of its own, so compression covers a spin-conditioned
descriptor under exactly the structural conditions of Section 14. The bond-projected family enters
that specialization as three further per-edge accumulators and widens the degree-one block, so the
compiled profile and the equations above have to be changed together: a kernel built before the
family was added evaluates a different function and its parity against the portable path is what
detects that. Both fused inference paths, the generic composition and the compact canonical
deployment path, return the magnetic force as a value next to the conservative force.

## 7. Frame charge state

DPA4C optionally conditions on the total charge and the spin multiplicity of the frame. The same
nuclear geometry can be a cation, a neutral or an anion, and can be a singlet or a triplet, with
genuinely different energies and forces; a descriptor blind to that pair maps all of them to the
same features and can only fit their average.

The feature is enabled by `add_chg_spin_ebd`. It is unrelated to the native spin of Section 6,
which carries a per-atom magnetic moment: the two are independent inputs and may be used together.
Without it the descriptor is bit-for-bit the one described above.

The condition is a per-frame integer pair, the total charge `Q_f` in units of the elementary charge
and the spin multiplicity `M_f`. It reaches the descriptor as a tensor with shape `(nf, 2)`, and a
single pair is broadcast over the frames. `default_chg_spin` supplies a fallback when a caller
provides none.

### 7.1 Condition embedding

The two integers index their own tables and are mixed by one bias-free SwiGLU trunk whose single
output head is split into two vectors:

```text
[w_f ‖ y_f]
  = SwiGLU([E^Q_{Q_f} ‖ E^M_{M_f}] W_cond,in) W_cond,out,

w_f in ℝ^(C_0),      y_f in ℝ^(2 H_pair).
```

The charge table has 200 rows and is addressed by `Q_f + 100`; the multiplicity table has 100 rows
and is addressed by `M_f`. The trunk hidden width follows the shared rule `8 ceil(C_0 / 3)`.

Charge and multiplicity share one nonlinear pathway rather than contributing two additive
embeddings. They are not independent degrees of freedom: the number of unpaired electrons has the
parity of the electron count, so changing the charge by one flips it, and the structural response
to a spin-state change depends on the oxidation state it happens in. The low-spin to high-spin
elongation of an octahedral iron-nitrogen bond is roughly 0.18 Å at Fe(II) and 0.10 Å at Fe(III);
an additive decomposition would represent neither the parity constraint nor that difference.

`W_cond,out` is zero initialized. An untrained descriptor is therefore independent of the charge
state for every value of it, which keeps the fixed output calibration of Section 11, measured once
before training, free of a random condition offset.

### 7.2 The two injection points

The type embedding enters DPA4C in exactly two places, and the condition follows it into both.

**Center type tail.** The trailing output block of Section 8 becomes

```text
e_a + w_f     for a < T,
e_a           for the padding type.
```

The padding row stays zero: its output row is discarded, and compressed inference conditions only
the real rows of a frozen table, so shifting it would break the parity between the two paths.

**Ordered pair encoder.** The condition biases the hidden pre-activation of the encoder of Section
2.3:

```text
h_ab(f) = SwiGLU([T_a ‖ T_b] W_pair,in + y_f),
```

so `gamma_ab`, `beta_ab`, `U_ab` and, when native spin is present, `gamma^s_ab` and `beta^s_ab`
all become functions of the ordered type pair and of the frame condition. This is the route that
makes the conditioning a property of the descriptor rather than of the fitting network: it changes
how a given geometry maps to the degree-wise moments, and therefore the effective radial response
of every ordered pair.

The condition enters as a bias on the pre-activation rather than as a shift of the encoder input
because `W_pair,in` is linear and bias free. The two are equivalent,

```text
[T_a + s ‖ T_b + s] W_pair,in = [T_a ‖ T_b] W_pair,in + [s ‖ s] W_pair,in,
```

and only the bias form lets one shared projection over the finite type table serve every frame,
instead of duplicating the `(T + 1)^2` pair inputs over the condition axis. Emitting `y_f`
directly, rather than through the image of `[s ‖ s] W_pair,in`, costs the same and removes the
constraint that the bias lie in that image.

Nothing else changes: no moment, no invariant, no output coordinate is added, and `D_out` is
independent of the condition.

### 7.3 Evaluation granularity

The conditioning heads are applied on the coarsest axis over which their argument is constant.
Without a frame condition that axis is the ordered type pair, so the cache of `(T + 1)^2` rows is
built once and gathered per edge, exactly as in Section 2.3. With one, the argument also depends on
the frame, and the product axis has `nf (T + 1)^2` rows, which exceeds the edge count for the
molecular systems a charge state describes: a 60-atom molecule at `rcut = 6 Å` carries a few
thousand edges against 8100 ordered pairs for a 89-element type map. The heads therefore move to
the edge axis, whose cost the training batch size already controls. Both routes evaluate the same
function, and the compression fold below is the first route at `nf = 1`.

The per-edge route gathers the pair pre-activation, of width `2 H_pair`, instead of the scale and
shift, of width `2 C_0`. Its transient is of the same order as the ordered mode-mixing gather of
Section 2.4 at `R = 8`, and it exists only while training: compressed inference reads a folded
table and pays nothing.

### 7.4 Properties

- **O(3) and translation invariance.** The condition is a pair of scalars. It scales and shifts
  edge amplitudes and adds to a node-local block, and never touches `u_ij` or `B_l(u_ij)`, so every
  invariance argument of Section 5 holds unchanged.
- **Grading compatibility.** In the labelling of Section 6.2 the condition has `l = 0` and
  `sigma = 0`, so multiplying any spin family by a condition-dependent factor leaves its parity and
  its time-reversal sign unchanged. Native spin and the frame charge state compose without a new
  selection rule.
- **Frame independence.** Each frame occupies one contiguous block of the flat node axis, and an
  edge inherits the frame of the center it reduces onto, so a batch of mixed charge states agrees
  coordinate by coordinate with the same frames evaluated one at a time.
- **Extensivity.** The condition is embedded from the raw integers, as in DPA4. A total charge is
  an extensive quantity while the descriptor is local, so a model trained on molecules of one size
  applies the same local perturbation on a much larger system. Within the size range of a molecular
  corpus this is the convention the family already uses; across a large size extrapolation it is a
  known limitation, shared with every model that embeds the raw total charge.
- **Unseen conditions.** A charge or multiplicity absent from the training corpus addresses a table
  row that received no gradient. The value is finite but arbitrary; the calibrated range of the
  corpus is the supported range.

### 7.5 Calibration

`compute_input_stats` reads the frame condition of every sampled system under the `charge_spin`
key, falling back to `default_chg_spin` when a system carries none. The preconditioner it freezes
is a single diagonal that has to hold for every charge state the corpus contains, so it is measured
over the sampled distribution of states rather than over one of them.

### 7.6 The two routes at run time

A deployed charge-conditioned artifact serves a state chosen at run time however it was frozen.
Which route the condition takes depends on whether the artifact is compressed, not on the export
format:

| Artifact                        | Condition on the compiled forward | `dim_chg_spin` |
| ------------------------------- | --------------------------------- | -------------: |
| Uncompressed `.pt2` graph lower | An input, `(nf, 2)`               |            `2` |
| Compressed `.pt2`               | None; folded into frozen tables   |            `0` |

**The input route.** An uncompressed export carries a genuine per-frame condition tensor with a
dynamic frame axis. It is the last positional input of the graph lower, the conditional tail after
`fparam` and `aparam`, and the thirteenth slot of the spin-free schema; under native spin the
per-node moment takes the eleventh slot and the tail follows it unchanged, at the fourteenth. Every
call fills the tensor from the condition the caller names, and otherwise from the stored
`default_chg_spin`. Frames in one call may carry different states.

**The fold route.** Compression removes that input rather than ignoring it: the snapshot reports a
zero condition width, which is what routes it onto the compact canonical lower whose argument list
has no conditioning slot. The condition survives inside the frozen tables of Section 14 — the
ordered PairFiLM cache, the ordered mode-mixing cache, the ordered spin scale and shift, and the
type embedding — which reach the compiled lower as AOTI module constants. Serving another state
therefore means rebuilding those four tables and writing them over the constants, and the archive
ships a second compiled artifact, the *charge-state fold*, that performs exactly that rebuild. A
folded snapshot holds one state at a time, so unlike the input route it cannot serve a batch of
mixed states; the portable path, which conditions each frame independently, is what does that.

The rebuild is a separate artifact rather than a stage of the inference graph because the two
differ by a per-step cost that no system size amortizes. Folding inside the graph would evaluate
the ordered pair encoder over its `(T + 1)^2` ordered pairs on every step, for a value that is
constant over a molecular-dynamics run. That work is a function of the type map alone, so it does
not shrink as the system grows: at 8100 ordered pairs for an 89-element type map it is negligible
above roughly `10^5` atoms and dominant below `10^4`. Run once, when the state becomes known, the
same evaluation costs a handful of kernel launches for the whole run.

### 7.7 The charge-state fold

**Archive contract.** A compressed charge-conditioned `.pt2` carries two entries beyond the
ordinary ones:

- `model/extra/charge_state.pt2`, a second AOTI package whose input is the condition as a
  `float32` tensor of shape `(1, 2)` — the layout the inference lower would have received — and
  whose output is the tuple of rebuilt tables;
- `charge_state_constants` in `metadata.json`, a list of strings positionally aligned with those
  outputs. Entry `i` names the constant of the lower that output `i` replaces. An empty entry marks
  an output that no constant receives, which is how a disabled mechanism appears: a spin-free
  descriptor contributes an empty spin table, a descriptor without pair-conditioned radial modes an
  empty mixing table, and neither reaches a constant.

The names are recovered by value at freeze time, by matching each artifact against the lifted
tensors of the exported program. `make_fx` traces a plain function and names every lifted tensor
positionally, so the buffer names are gone by the time the program is exported; an artifact that
matches anything other than exactly one constant is an error rather than a guess. Each lower lifts
its constants independently, so the names hold only for the lower they were resolved against. Only
a compressed DPA4C descriptor folds a charge state, and that family never carries message passing
across ranks, so an archive with a fold holds exactly one lower and the question of a second set of
names does not arise. An archive that declares the constants and cannot supply the fold is
malformed and fails at load rather than degrading to the state it was frozen against.

**The two widths.** A conditioned model has two widths, and they are not the same number.
`dim_chg_spin` is the conditioning width of the compiled forward's argument list. It gates input
construction — whether a forward pass is handed a condition tensor at all — and is legitimately
zero for a compressed model, whose forward reads no condition. The *settable* width is what a
caller may name a state with. On the input route it equals `dim_chg_spin`; when a fold ships it is
the length of the stored `default_chg_spin`, which is both the state the snapshot starts from and
the layout the fold consumes. `DeepPotPTExpt` keeps the two apart as `dchgspin` and
`settable_chgspin`, and publishes the latter through `dim_chg_spin()`; the Python evaluator keeps
`get_dim_chg_spin()` on the former and resolves the settable width inside the fold. Conflating the
two is not a cosmetic error: a `set_charge_spin` that validates against the compiled width rejects
every request on exactly the compressed models the feature exists to serve, and an evaluator that
gates input construction on it alone discards the argument and answers for the frozen state
instead.

Applying a state overwrites the constants of a loaded module, so it is not safe to interleave with
a forward pass, and both inference layers keep it out of the step loop. The C++ layer applies it
once, when the deployment fixes the condition. The Python evaluator receives the condition on every
`eval` call instead, so it caches the last state applied and a loop at one condition pays for the
rebuild once.

### 7.8 Inference backends and the LAMMPS surface

The serving class follows the spin scheme and the message-passing capability of the archive, not
the presence of a charge state:

| Model              | Backend            | Charge-state routes |
| ------------------ | ------------------ | ------------------- |
| DPA4C, no spin     | `DeepPotPTExpt`    | input and fold      |
| DPA4C, native spin | `NativeSpinPTExpt` | input and fold      |
| DPA4, native spin  | `DeepSpinPTExpt`   | input only          |

Native-spin dispatch is `native_scheme && !needs_with_comm`. DPA4C is never message-passing, so it
never ships a with-comm artifact and every native-spin DPA4C archive reaches `NativeSpinPTExpt`,
which carries both routes. DPA4 message-passes across ranks and therefore does ship one, which
keeps it with `DeepSpinPTExpt`; that class serves the input route only, which is all a DPA4 archive
needs, since compression is a DPA4C mechanism.

On the LAMMPS side the condition is a `charge_spin <charge> <multiplicity>` keyword on the
`pair_style` line, accepted by `deepmd`, `deepspin` and `dpa4spin`. It is applied once, in
`settings()`, rather than resupplied per step: a condition named on the pair-style line holds for
the whole run, and it is the fold route that makes the distinction load-bearing, since rebuilding
the tables every step is the cost Section 7.6 rejects. Applying it once is also what lets the
Kokkos device path serve a runtime condition without threading it through: `pair_style deepmd/kk`
inherits `settings()` from the base style, so by the time `compute_edges_gpu` draws the condition
from the model, the model already carries the one the command line asked for.

## 8. Output layout and width

The fitting network receives the blocks in this exact order:

| Order | Block                                                   |                        Width |
| ----: | ------------------------------------------------------- | ---------------------------: |
|     1 | Scalar moments `X^(0)`                                  |                        `C_0` |
|     2 | Upper triangles of aligned exact Grams, in degree order | `sum_(l=1)^L C_l(C_l + 1)/2` |
|     3 | Generic bispectrum, in degree-triple order              |               `D_bispectrum` |
|     4 | Projected `Qv` quartic, tensor probe first              |                    `K_1 K_2` |
|     5 | Spin invariants, in the order of Section 6.4            |                     `D_spin` |
|     6 | Neighborhood-mass divisors `[1/n_i^(0), 1/n_i^(+)]`     |                          `2` |
|     7 | Center type embedding `e_a`                             |                        `C_0` |

The complete width is therefore

```text
D_out
  = 2 C_0
    + sum_(l=1)^L C_l (C_l + 1) / 2
    + D_bispectrum
    + K_1 K_2
    + D_spin
    + 2,

D_spin
  = 0                                                      without native spin
  = (1 + 2 C_s)(2 + 2 C_s)/2 + 2 + 2 C_2 + 2 C_s           with native spin.
```

The spin block precedes the divisors so that the two of them still close the geometric region and
the center-type tail keeps its trailing position. At `C_2 = C_s = 4`, `D_spin` is 63; at
`C_2 = C_s = 8` it is 187.

The two `C_0` terms are the scalar moments and center type embedding. The trailing `+ 2` is
permanent: the two neighborhood-mass divisors are always emitted. `radial_modes` does not appear in
the formula.

Code-derived examples are:

| `C_0` | `L` | `D_out` |
| ----: | --: | ------: |
|    16 |   2 |      86 |
|    32 |   3 |     155 |
|   128 |   4 |     557 |

DPA4C exposes no equivariant fitting channels. Every coordinate passed to the fitting network is
invariant.

## 9. Complexity and memory

Let `E` be the number of directed edges and `N` the number of nodes. Big-O notation describes how
work or storage grows while omitting fixed multipliers. The flat moment width is

```text
S = sum_(l=0)^L (2l + 1) C_l.
```

The portable additive moment path has

```text
time   O(E S)
memory O(E S + N S).
```

`E S` is the temporary edge payload and `N S` is the reduced node state. The one-reduction structure
makes the edge work linear in the number of neighbors.

The learned full-width alignment of degrees one and two costs

```text
O(N [3 C_1^2 + 5 C_2^2]).
```

The exact Gram matrices cost

```text
O(N sum_(l=1)^L (2l + 1) C_l^2).
```

Probe projections cost

```text
O(N sum_(l=1)^L (2l + 1) C_l K_l).
```

The angular widths are at most nine because `L <= 4`. After treating those widths as fixed
constants, the bispectrum rank products dominate its scaling:

```text
O(N sum_allowed_triples K_l1 K_l2 K_l3).
```

The projected quartic costs `O(N K_1 K_2)` up to fixed three-dimensional matrix factors.

Pair-conditioned radial mixing adds

```text
time O(E C_0 R).
```

In the portable tensor path, its dominant practical cost is gathering the ordered table to a
transient shape `(E, C_0, R)`, rather than the multiply-add count itself. The compressed kernel
reads the finite cache directly while scanning edges.

No term scales as the square of the neighbor count. No term requires a source-node feature halo,
because no intermediate node feature is sent to another node.

## 10. Mathematical properties

DPA4C has the following structural properties:

- **Translation invariance.** Only relative vectors `r_j - r_i` enter, so shifting every atom by the
  same vector changes nothing.
- **Neighbor-order invariance.** Reordering the edges entering one center does not change a
  destination sum.
- **Node permutation equivariance.** Reordering nodes and all matching graph indices only reorders
  output rows.
- **O(3) invariance.** Gram and quartic contractions are even, and every Gaunt triple has even total
  parity, so the final descriptor is unchanged by rotations and reflections.
- **Linear edge traversal.** The descriptor performs one destination reduction and no cross-node
  message passing.
- **C3 cutoff continuity.** Edge amplitudes and their first three radial derivatives join
  continuously to zero at `rcut`.
- **Finite coincident-edge values and derivatives.** The direction denominator contains the positive
  regularizer `eps`.
- **Backend-neutral equations.** The array-API implementation supplies the same NumPy, JAX, and
  PyTorch equations; the PyTorch backend replaces selected primitives with native or compiled
  implementations.
- **Time-reversal evenness.** With native spin enabled, every emitted coordinate has even total spin
  order, so the descriptor is unchanged by a global reversal of the magnetic moments and the
  magnetic force changes sign.
- **Scalar frame conditioning.** With the frame charge state enabled, the condition enters only as
  a scalar rescaling and shift of edge amplitudes and as a node-local additive block, so every
  invariance above is preserved and the output width is unchanged.

## 11. Public parameters

A complete descriptor configuration is:

```json
{
  "type": "dpa4c",
  "rcut": 6.0,
  "channels": 32,
  "lmax": 2,
  "basis_type": "bessel",
  "n_radial": 16,
  "radial_modes": 0,
  "use_amp": false,
  "exclude_types": [],
  "precision": "float32",
  "trainable": true,
  "seed": null,
  "use_spin": null,
  "add_chg_spin_ebd": false,
  "default_chg_spin": null
}
```

The parameters are:

- `rcut`: positive outer cutoff radius in Å;
- `channels`: `C_0`, the scalar and edge width in `{8, 16, 32, 64, 128}`;
- `lmax`: `L`, the maximum angular degree in `{2, 3, 4}`;
- `basis_type`: analytic radial basis, either `bessel` or `gaussian`;
- `n_radial`: positive number of analytic radial basis functions;
- `radial_modes`: non-negative integer `R`, the number of pair-mixable shared radial profiles;
- `use_amp`: whether the per-edge training stage uses CUDA bfloat16 autocast; this is an execution
  policy and is not serialized;
- `exclude_types`: ordered atom-type pairs removed from the descriptor graph;
- `precision`: floating-point precision of descriptor parameters;
- `trainable`: whether learned descriptor parameters receive optimizer updates;
- `seed`: deterministic parameter-initialization seed, or `null`;
- `use_spin`: per-type magnetic flags enabling the native spin channels of Section 6, supplied by
  the model construction layer from `model.spin`, or `null`;
- `add_chg_spin_ebd`: whether to condition on the frame charge state of Section 7;
- `default_chg_spin`: fallback `[charge, multiplicity]`, used when a caller supplies none and
  folded into the frozen tables by compression.

`ntypes` and `type_map` are supplied by the model construction layer. Internally, a zero padding
type is appended, so complete type tables contain `T + 1` rows.

### 11.1 Neighborhood masses

The normalized moments alone do not reveal the scale by which their raw sums were divided. DPA4C
therefore emits both divisors:

```text
1 / n_i^(0) = sqrt(d_i^(0) + 0.25)
1 / n_i^(+) = sqrt(d_i^(+) + 0.25).
```

The fitting network can then recover information about effective coordination as well as the
normalized angular structure. The divisors also depend on coordinates, so their derivatives
contribute to force. They close the geometric output block immediately before the center-type
embedding.

The two coordinates cost no extra graph reduction because `d_i^(0)` and `d_i^(+)` are already part
of the common edge payload. They are permanent: there is no `mass_features` switch. All per-atom
multiplicative gates were also removed from the public surface after the experiments in Section 13.3
found no benefit.

Input statistics apply a fixed initialization preconditioner, not a running normalization. Ordinary
invariant coordinates store a mean of zero and are only rescaled from their measured
root-mean-square value toward the type-embedding root-mean-square value. The two divisors are the
only coordinates whose measured mean is subtracted. Their root-mean-square value exceeds that of a
typical invariant by two orders of magnitude, so RMS-only rescaling would leave each near one plus a
small fluctuation. `compute_input_stats` instead records a centered scale for these two coordinates
and leaves every other stored mean at zero.

The estimator draws sixty-four frames from every sampled system on a linear index grid. The count
trades start-up time, which grows linearly with it, against the spread of a sample mean. On a
variable-size store the frames available per system follow the training batch-size specification and
`data_stat_nbatch`; under the settings used for OMat24 the pool is far larger than the drawn count,
so the grid is never truncated. Neither the residual spread at a given count nor its effect on
trained accuracy has been quantified. The preconditioner is fixed at initialization and a constant
diagonal rescaling is within reach of the fitting network.

The measured accuracy effect is small. Across six seed-paired C16 and C32 configurations, the mass
block improved energy error by 0.78% and force error by 0.27% on average; four of six pairs improved
for each metric. A paired t-test over the six differences did not reach significance.
Standardization rather than RMS-only rescaling produced no measurable difference, ruling out feature
conditioning as the explanation for the small effect. The block is retained because its inference
cost is 0.04% and the average sign is consistent, not because the gain is statistically established.

Per-atom diagnostics on three trained checkpoints provide additional motivation. After force
magnitude was regressed out, force error had a partial correlation of 0.32 to 0.36 with the angular
mass. The highest coordination decile had six to ten times the error of the lowest decile. This
diagnostic cannot distinguish information lost through normalization from the separate possibility
that dense environments are intrinsically harder to fit.

## 12. Training contract

### 12.1 Mixed precision

Floating-point formats trade numerical range and precision against speed and memory. `float32`
stores more mantissa precision than `bfloat16`; bfloat16 uses half as many bits and is efficient on
supported CUDA hardware. **Autocast** is a PyTorch execution context that automatically runs
eligible operations in the selected lower precision while leaving operations that require full
precision unchanged.

During training, `use_amp` places exactly `build_edge_features` under bfloat16 autocast on CUDA.
This region contains the radial network and pair-conditioned mode mixing, including the tensors
whose size scales with `E`. It excludes:

- the destination reduction, which accumulates a complete neighborhood;
- the invariant readout, which raises moments to fourth order;
- the ordered pair cache, which is evaluated over the finite type table rather than over edges.

A charge-conditioned descriptor is the one exception to the third exclusion. Section 7.3 moves its
conditioning heads onto the edge axis, so they fall inside the region and their bounded outputs are
produced in bfloat16 along with the amplitude they scale. Nothing downstream of the region changes:
the boundary still restores the compute precision before the reduction.

The boundary converts the returned edge features back to descriptor compute precision, so bfloat16
values do not enter the destination reduction.

Mixed precision is an execution policy rather than learned model state. `use_amp` is absent from the
serialized descriptor. Training reads it from the run configuration. Evaluation and inference
instead read `DP_AMP_INFER` once when the descriptor is constructed, so a traced graph cannot change
behavior after an environment-variable mutation. A checkpoint trained in full precision can
therefore be evaluated with mixed precision, and the reverse.

The switches are independent:

```text
                 training        evaluation
use_amp             on               off
DP_AMP_INFER        off              on
```

PyTorch autocast alone is insufficient for the radial network. DeePMD's `NativeLayer` ordinarily
restores its input dtype after each affine map, which would undo reduced precision at every layer.
DPA4C disables that restoration only for layers inside the autocast region: the radial trunk and,
when present, the mode head. PairFiLM and the readout keep the default behavior. A descriptor for
which neither training nor inference mixed precision can engage is bit-identical to one built
without the option.

One-H20 forward-plus-backward measurements at `E = 245760` are:

```text
config          fp32 ms   amp ms   speedup   fp32 MiB   amp MiB   saved
C=32,  R=0         7.35     7.65     0.96x        951       720     24%
C=32,  R=8        10.31    11.75     0.88x       1198       844     30%
C=64,  R=8        15.66    14.94     1.05x       2203      1485     33%
C=128, R=8        27.44    24.09     1.14x       4203      2784     34%
```

Peak memory fell by 24% to 34% across the measured configurations. Step time improved only when the
channel width was large enough for lower-precision matrix products to offset conversion overhead.
Narrow configurations were slightly slower. Mixed precision is therefore primarily a memory control
and becomes a throughput improvement only at the wide end. The default is off.

### 12.2 Optimizer routing

An optimizer converts parameter gradients into parameter updates. DPA4C training uses HybridMuon in
`slice` mode. HybridMuon combines Adam, which updates each parameter coordinate from running
gradient statistics, with Muon, which applies an orthogonalizing update to matrix-like weights.

Routing examines the final segment of each parameter name:

- a segment containing `bias` or beginning with `adam_` goes to Adam;
- every other parameter of effective rank two or greater goes to Muon.

All descriptor linear weights use `(fan_in, fan_out)` layout and contain no bias. This agrees with
Muon's correction for rectangular matrices.

Scale-like learned arrays place `adam_` on the parameter itself because routing reads only the leaf
name. In DPA4C, this convention covers the type embedding `adam_type_embedding` and radial
frequencies `adam_freqs`. It follows the DPA4 convention that norm scales, layer scales,
frequencies, and embeddings are not suitable for an orthogonalizing update.

No DPA4C descriptor parameter has rank three or greater, so `slice` mode reduces to ordinary
two-dimensional Muon for its matrix weights. Readout matrices for different degrees are stored
separately because their widths differ; this is equivalent to slicing a hypothetical stacked tensor
by degree.

### 12.3 Fitting-network architecture

The fitting network maps each invariant descriptor row to an atomic energy. Removing descriptor
message passing also removes repeated learned nonlinear transformations from the descriptor path, so
sufficient depth is required on the fitting side.

At C16, three hidden layers of width 128 reached an energy error of 53.25 meV/atom and a force error
of 193.39 meV/Å. A single hidden layer of width 384 reached 61.99 meV/atom and 211.03 meV/Å despite
having fewer parameters. The result identifies depth, rather than a single wide layer, as the
relevant capacity: depth moved from the descriptor to the fitting network when message passing was
removed.

`resnet_dt` is a learnable per-neuron multiplier on each fitting residual branch. It is initialized
to 0.1 and is equivalent to LayerScale, which starts a residual contribution small and learns its
scale. This mechanism was harmful for DPA4C fitting. Across two seeds, removing it improved energy
error by 1.25% and force error by 0.64%, while removing 256 parameters. Shipped configurations
therefore set `resnet_dt` to `false`.

At C16, widening a depth-three fitting network from 128 to 192 improved energy error by 4.38% and
force error by 2.40% at an 8.4% throughput cost. This accuracy-throughput exchange rate is
comparable to widening the descriptor itself; fitting width is therefore a genuine inference-budget
choice rather than free capacity.

### 12.4 Zone bridging

`model.bridging_method: "zbl"` or `"nlh"` adds an analytical screened nuclear repulsion to a DPA4C model, in the form DPA4 uses: the plain pair energy on every valid edge inside the cutoff, with no switching function of its own, so that its force is the analytical force everywhere and carries no cross term of a switch. The concise key expands to the canonical `linear_ener` composition over the DPA4C model and an `inner_potential` sub-model with `weights: "sum"`, and the composition hands the window to the descriptor as two fractions of each pair's own length scale together with the scale they measure against (Section 2.5); `bridging_r_inner` with `bridging_r_outer` states that window in Å instead. Below `r_inner` a pair therefore interacts through ZBL alone, and the learned energy is the energy of the two atoms in an environment that lacks the other one.

The composition evaluates both children on one neighbor graph, so force and virial of the sum come from one edge backward. It answers the native-spin capability through its learned child, because the analytical term is a function of the separation alone: it accepts the moments, ignores them, and leaves the magnetic force to the learned model, which is why a bridged model is a valid native-spin model and freezes, compresses and runs in LAMMPS as one. It owns the output bias, the exclusions and the preset bias. The learned child computes no output bias of its own, but it receives the preset of the composition, which decides whether its fitting references the isolated atoms (`vacuum_ref`), so an isolated atom of a bridged model carries exactly its preset energy, as in the plain model. The composition also trains like the plain model under HybridMuon, because the tensors its learned descriptor declares for AdamW keep that route below the path of the child, and a plain checkpoint fine-tunes into the bridged composition and back, the learned descriptor and fitting net moving between their two paths.

The training-frame filter of a bridged model sits at the midpoint of the window, `(f_inner + f_outer) / 2` of the pair's length scale. Below the inner radius the model has no freedom at all, and between the inner radius and the midpoint the switch is less than half open, so a label there asks the throttled descriptor for the residual divided by the opening: a Si-Si dimer label at 0.79 Å meets an opening of 0.012 under the covalent window and demands a learned force of -187 eV/Å. Labels of that kind steer the fit inside the window, and the dimer scan of a model trained on them shows spurious maxima of the total force between the inner radius and the first label. With the filter at 0.50 of the covalent sum, just inside the midpoint of 0.53 that the rule prescribes, a 500k-step fine-tuning on the SiC set (bulk 3C-SiC frames plus C-C, Si-C and Si-Si dimer scans down to 0.79 Å) keeps the total force monotone below the first discontinuity of the DFT labels at every one of its 25 checkpoints, and the smallest ratio of a total-force step to the analytical-force step on a 5 mÅ grid ends at 0.68 / 0.71 / 0.39 for the three pairs. The rule presupposes dimer labels that reach into the window: with the dimer data truncated at 1.0 Å the curve is still monotone after 40k steps but about 10 eV/Å off inside the window, and truncated at 1.2 Å, which leaves the C-C window without a single label, a C-C force maximum of 61 eV/Å appears. The midpoint is where the switch is exactly half open, which holds for every window because the septic smootherstep is antisymmetric about its centre; the rule is that point by construction, not a position tuned on these runs.

The filter alone does not make the result reproducible; the distance clamp of Section 2.5 does. Three seeds of the same 500k-step fine-tuning were run with the switch alone and three with the switch and the clamp, each scored at every one of its 25 checkpoints on the 5 mÅ dimer grid below the first discontinuity of the DFT labels. The reading of a checkpoint is the margin, the smallest ratio of a total-force step to the analytical-force step inside the window, which is one where the learned term adds no curvature and crosses zero exactly where the total force stops decreasing. With the switch alone the final margins for C-C / Si-C / Si-Si were 0.69 / 0.63 / 0.53, 0.58 / 0.59 / 0.10 and 0.38 / 0.60 / -0.10: one seed ended with a persistent Si-Si force maximum of 1-3 eV/Å present from 260k steps on, another with a Si-Si margin of 0.10, and two of the three had grown Si-Si bumps of about 3 eV/Å at 20k-40k steps. With the switch and the clamp the final margins were 0.71 / 0.77 / 0.80, 0.73 / 0.84 / 0.91 and 0.58 / 0.75 / 0.70, the smallest margin at any of the 75 checkpoints was 0.43, and no checkpoint showed any upward step of the total force. The clamp costs nothing where labels are kept: the force RMSE against the dimer labels between the filter and the outer radius was 0.76-0.89 / 0.08 / 0.29 eV/Å with the clamp against 0.73-1.07 / 0.08 / 0.29 without, and outside the window 0.12-0.15 / 0.36 / 0.21-0.23 against 0.06-0.19 / 0.36 / 0.21-0.22. In the dropped zone between 0.41 and 0.53 of the covalent sum, where no label is fitted, the Si-Si force was 18.6-20.1 eV/Å from the DFT values with the clamp against 24.5-39.2 without. The clamp works because below the midpoint the pair's inputs reduce to the amplitude of the switch and a drift of the clamped separation of under 7 % of the window width, so the learned energy of the pair descends along one fixed profile instead of along whatever the radial network extrapolates to separations no retained frame contains.

The pair-clearance filter of `training.training_data.min_pair_dist` removes those frames from training and from the data statistics, and a bridged model takes that filter from its own window instead of from the dataset section: setting the key on such a model is rejected, because a second copy of the filter radius could disagree with the first. What a frame carries is a margin rather than a distance, the smallest ratio of a pair separation to the threshold of that pair, so the filter keeps a frame whose margin reaches one whichever length scale sized the window. The margin is a derived data requirement: the LMDB reader and both read paths of the NPY reader compute it from the frame geometry. A batch read derives the whole batch at once, which is where the cost of the requirement is paid, and the scan of a frame orders its real atoms along the lattice direction whose planes lie furthest apart and pairs each atom with the successors that lie within one slab, instead of visiting every pair. A margin of one bounds that scan: the value settles which side of the window a frame lies on and equals the exact margin whenever the frame fails it, so a frame the filter accepts may carry a larger margin that it reaches. On a batch of the OMat24 store this costs 22 ms against 130 ms for a scan of every pair of every frame. The pt_expt trainer drops the too-close frames of every training batch and replaces a batch left empty by the next one. The retry is local to a rank and involves no collective, so every rank still contributes one batch per optimizer step and distributed training keeps its lockstep. The ranks of one step may then hold different numbers of frames, and because each rank averages its loss over its own frames before the gradients are averaged, a frame of a thinned batch weighs more in that step than a frame of a full one. Validation batches are never filtered. The data statistics draw their batches under the same filter: a batch left empty is replaced by the next one of its system until `data_stat_nbatch` batches are kept or one pass over the system has been scanned, so a system contributes whenever that pass draws a valid frame, and a system without one is skipped. A pass serves every frame of an LMDB system; the NPY reader serves whole batches and reshuffles a set before serving it again, so its pass leaves out the frames of a set that do not fill a last batch. A data source that does not derive the margin raises instead of accepting every frame, and a dataset without any valid frame is reported when the statistics are collected.

## 13. Design rationale

### 13.1 Objective and parameter surface

DPA4C is optimized for extreme-speed molecular dynamics. The two primary deployment quantities are
fused-kernel inference throughput and the largest system that fits in device memory. Training cost
is not a design constraint.

A structural mechanism is retained only when its accuracy gain justifies its inference cost, using
roughly two percent of throughput per one percent of error reduction as the working exchange rate,
and only when no cheaper mechanism reaches the same accuracy.

The numeric scaling surface is

```text
rcut, channels, lmax, n_radial, radial_modes.
```

`basis_type` selects the Bessel or Gaussian analytic family rather than scaling a width. `use_amp`
is an execution argument. `exclude_types`, `precision`, `trainable`, and `seed` are infrastructure
arguments. Beyond the discrete basis choice and the numeric controls above, degree channels and
bispectrum ranks are derived, neighborhood masses are always emitted, and per-atom gates are absent.

### 13.2 Channels against radial modes

Two widths bound the deployment trade-off:

- the equivariant node state width `S` depends on `channels` and `lmax`;
- the invariant output width `D_out` depends on the same two parameters;
- neither width depends on `radial_modes`.

Consequently:

- increasing `L` adds `(2L + 1) C_L` moment values for the newly included degree and one harmonic
  block to every edge;
- increasing `C_0` widens the node state, radial table, output, and ordered-pair caches together;
- increasing `R` adds `C_0 R` cache reads and products per edge plus `R` radial-table channels, but
  leaves the node state and output unchanged;
- enabling mixed precision changes no dimension.

At `C_0 = 16`, the `L = 2` moment state has `S = 48`. Degree three adds seven values and degree four
adds nine more, giving `S = 55` and `S = 64`, respectively. The cost of another angular degree is
therefore fixed by its harmonic width, while its relative cost decreases as the lower-degree channel
blocks become wider.

Channels and radial modes are not interchangeable. At matched accuracy, widening channels is the
cheaper purchase per edge, while increasing radial modes is the cheaper purchase per byte of
persistent node state. The following recorded comparison reaches nearly the same energy mean
absolute error:

```text
                reported energy MAE      S     per-edge radial values
C_0=32, R=12                  44.46      76     C_0 (1 + R) = 416
C_0=64, R=0                   44.57     108     C_0 (1 + R) =  64
```

The C32 configuration keeps the smaller node state but pays about six times the per-edge radial
traffic. Radial modes are therefore the accuracy compensation used when memory, rather than
throughput, is the binding constraint. Their measured useful range ends at twelve; no channel width
improved beyond that point.

### 13.3 Mechanisms not adopted

Three conditioning choices were tested for a multiplicative gate on the finished geometric invariant
vector: center type embedding, calibrated scalar moments, or both. If `condition_i` denotes the
selected conditioning vector and `V` the learned projection, the gate has the form

```text
D_geo,i <- D_geo,i * (1 + tanh(condition_i V)).
```

All three choices were inert across three channel widths and four radial-mode ranks.

A separate experiment gated the equivariant moments before the invariant readout:

```text
X_i,m,c^(l) <- omega_i,l,c X_i,m,c^(l),   l >= 1.
```

This is not a reparameterization of the output gate. Diagonal channel scaling does not commute with
channel alignment or probe projection, so a pre-readout gate changes every Gram and probe
contraction; an output gate can only rescale a completed invariant. Despite the larger function
class, the pre-readout gate was inert at the operating point and was measurably worse at the
radial-rank knee.

The experiments shared one pattern: the force-error tail decreased slightly, but mean errors did not
improve, and the tail effect vanished once the radial function class was adequate. The fixed
multilinear readout and deep fitting network already absorb this form of conditioning. No per-atom
gate is retained or configurable.

## 14. Compression contract

**Tabulation** replaces repeated evaluation of a one-dimensional learned function by a precomputed
lookup table. Values and derivatives are stored on a regular distance grid, and a polynomial
interpolates between adjacent grid points. DPA4C uses quintic Hermite polynomials, which are
degree-five polynomials fixed by the function value and its first two derivatives at both interval
endpoints.

This is useful only because the edge calculation separates cleanly into distance-only, finite
type-table, and angular parts:

1. The analytic radial basis and radial SwiGLU depend only on the scalar distance. Their composed
   outputs `g(rho)` and `q(rho)` can be tabulated on the closed interval `[0, rcut]`.
1. PairFiLM depends only on the finite ordered pair of atom types, and on the frame charge state
   when Section 7 is enabled. Its `gamma_ab`, `beta_ab`, and `U_ab` values can be cached once per
   charge state.
1. The C3 envelope remains analytic. It multiplies the tabulated radial and type-conditioned
   amplitude once for degree zero and twice in each non-scalar moment.
1. Distances at or beyond `rcut` map to zero, so the table needs no extrapolation region.
1. Cartesian harmonics remain explicit low-degree polynomials of the regularized direction.
1. Moment reduction, channel alignment, exact Grams, bispectrum, and projected quartic remain exact
   node operations outside the radial table.

Compression therefore replaces continuous radial-network evaluation without changing the degree-wise
moment or invariant-readout equations.

### 14.1 Compiled CUDA specialization

The portable descriptor accepts every non-negative integer `R`. The compiled CUDA operator has
specializations for

```text
C_0  in {8, 16, 32, 64, 128}
L    in {2, 3, 4}
R    in {0, 2, 4, 8}.
```

A configuration outside this set can train and evaluate through the portable path.
`enable_compression` rejects it rather than silently returning an uncompressed model.

`C_0` and `L` select compile-time kernel profiles. `R` is a runtime argument within its supported
set. Extra angular degrees add moment accumulators that must remain outside the register budget of
the production `L = 2` profile; radial modes only lengthen an inner reduction.

The immutable compression snapshot contains:

```text
radial quintic table          [ceil(rcut / stride), 6 (C_0 + R)]
radial metadata               [stride, table_max, rcut, eps, degree_floor]
ordered PairFiLM cache        [(T + 1)^2, C_0, 2]
ordered mode-mixing cache     [(T + 1)^2, C_0, R]
type embedding                [T + 1, C_0]
alignment/probe matrices      [8, C_1, C_1], zero-padded
sparse coupling layout        int32 metadata and packed coordinates
sparse coupling values        Gaunt entries and isometric probe scales
output mean                   [D_out]
output inverse standard dev.  [D_out].
```

The radial table stores `g` in its leading `C_0` channels and `q` in its trailing `R` channels. The
mode axis is innermost in `U_ab`, so all coefficients needed for one edge channel are contiguous. A
supported mode rank requires at most one 128-bit and one 64-bit load.

For each radial interval and output channel, the interpolant has six physical-coordinate
coefficients:

```text
y(rho) = c0 + c1 tau + c2 tau^2
              + c3 tau^3 + c4 tau^4 + c5 tau^5,
```

where `tau` is the distance from the interval's left endpoint. The table stores `[c0, c1, c2, c3]`
for every channel in one quartet block and `[c4, c5]` in one pair block. The six coefficients arrive
in one 128-bit and one 64-bit load rather than three 64-bit loads, with identical byte traffic.
Forward evaluation and backward differentiation use the same polynomial, so CUDA backward is the
analytic derivative of the compressed forward function.

Degrees one and two contain the wide channel blocks. The `112` and `222` couplings are evaluated in
closed form. Degrees three and four have one channel each; their Grams reduce to squared norms, and
their remaining triples use a compact sparse Cartesian Gaunt table. That table is generated from the
same Lebedev sphere-quadrature construction as the portable readout.

The forward operator saves

```text
state_i = [
  X_i^(0), ..., X_i^(L),
  n_i^(0), n_i^(+)
] in ℝ^(S + 2).
```

The state stores the normalization multipliers `n`, whereas the descriptor output stores their
reciprocals `1/n`. This is the minimum node state that permits one exact backward edge scan without
recomputing destination moments. At `L = 2`, `S + 2` is:

| `C_0` | Saved values per node |
| ----: | --------------------: |
|     8 |                    42 |
|    16 |                    50 |
|    32 |                    78 |
|    64 |                   110 |
|   128 |                   218 |

The autograd-facing operator preserves this state. In the explicit level-two inference composition,
the moment-gradient state overwrites the same allocation after the saved moments reach their last
use.

Backward differentiation has two fused stages. First, a node kernel computes the vector-Jacobian
product, or VJP, of the invariant readout: it propagates an output gradient back to the moment
state. The closed-form `222` contraction is fused into this node VJP. One eight-lane group owns one
node, so one 32-lane CUDA warp handles four nodes concurrently.

Each neighborhood mass reaches the output along two paths, and the node VJP combines them before
the mass gradient leaves the kernel. Through the moments it normalizes, the mass contributes
`-n^2/2` times the moment-cotangent contraction; through its own emitted divisor `sqrt(m + eps)`,
it contributes `n/2` times that output's cotangent. Both terms share the factor `n/2`, so the
combination costs one extra fused multiply-add and one extra output-gradient load per node.

Second, an edge kernel recomputes radial interpolation, PairFiLM amplitudes, mode mixing, the
analytic C3 derivative, and Cartesian-harmonic derivatives from the saved node state. It saves no
per-edge moment tensor. The basis VJP is applied before the lane-group reduction, so only three
Cartesian cotangents are reduced.

Forward and backward edge-group widths are fixed per channel profile. A narrower group keeps more
edges active in one warp but increases each lane's channel and moment accumulators; an angular
channel wider than the group must be tiled. Scalar channels are tiled per lane, allowing one warp to
process several edges without floating-point atomics. The scalar cotangent, which every edge of a
node rereads, is held in shared memory rather than registers.

The generic compressed descriptor requires:

- `float32` descriptor parameters;
- evaluation mode;
- no descriptor-level excluded type pairs;
- a destination compressed-sparse-row graph, whose row pointers delimit the contiguous incoming
  edges of each node.

The frame charge state of Section 7 needs no compiled variant at all. It reaches only the ordered
pair encoder and the center type table, neither of which depends on distance, so compression
evaluates the two condition vectors once and folds them in: `y_f` biases the pair encoder that
produces the ordered caches, and `w_f` shifts the real rows of the frozen type table. Every
artifact keeps its shape, the compiled kernel is unchanged, and inference costs nothing.

The price is that the snapshot holds one charge state at a time, exactly as it holds one set of
weights. `enable_compression` therefore requires `default_chg_spin`, builds the snapshot against
it, and the compressed descriptor reports a runtime condition width of zero, which is what routes
it onto the compact canonical lower. The state is not sealed in, however: the four artifacts it
touches are the complete image of a charge state in the snapshot, so rebuilding exactly those four
re-specializes it to another state. A `.pt2` export ships that rebuild beside the inference lower
as the charge-state fold of Section 7.7, which is how a deployed artifact serves a state chosen at
run time. Evaluating several states at once still means the portable path, which conditions each
frame independently.

The native spin channels of Section 6 are a compiled variant of the same kernels rather than a
separate operator, and they impose no additional requirement: every spin width follows the
degree-two width, so presence is the whole choice. A spin-conditioned descriptor freezes two further
tables, the ordered spin scale and shift and the per-type spin scalars, and its backward returns the
magnetic force alongside the edge gradient. The on-site half of that force closes inside the node
kernel; the neighbour half is emitted per edge and reduced onto source nodes through the source
compressed-sparse-row graph, in the same segment sum that assembles the conservative force.

The compiled spin profile is not independent of the model layer: the degree-one spin block and the
per-edge payload change width whenever a spin family is added or removed, so the kernel and the
portable equations must move together. The compressed-against-portable parity tests are what hold
them in step, and a kernel that has not been rebuilt after such a change fails them rather than
returning a plausible wrong answer.

Generic graph edge vectors may have any floating dtype. The operator converts them to `float32` at
its boundary and returns edge gradients in the input dtype. Generic edge indices may be `int32`,
`uint32`, or `int64`.

The canonical level-two energy/force composition has a stricter compact graph contract. It requires
no descriptor, atomic-model pair, or atomic exclusions; `float32` edge vectors; `uint32` source and
source-order arrays; and `int64` source and destination CSR row pointers. Every public boundary
rejects edge storage above the `2^32 - 1` slot limit.

The canonical path also requires a supported fused fitting network. The graph-fitting operator is
descriptor-independent and does not fix the hidden width to 64. It accepts positive hidden widths
that are multiples of four, including 32, 64, 128, 192, and 256; one or more hidden layers with one
uniform `tanh` or `silu` activation; and a linear scalar head. Identity residuals are supported.
Width-doubling residuals use the portable path, and per-layer residual timesteps are unsupported.
Residual preservation and the head gradient seed are fused into backward activation VJPs.

### 14.2 Zone bridging in the fused kernels

A bridged compressed model keeps the level-two energy-force route and the compact canonical deployment. Two things enter the kernels, and both live in a compile-time specialization of the edge scans. Zone bridging is dispatched at launch like the angular degree, the mode residual and the native spin, on CUDA and on the CPU alike: a launch that carries a bridging window or a pair table runs the bridged scans, and every other launch runs plain scans that contain neither the switch arithmetic nor the pair code. A model without a bridging composition therefore runs the edge-loop arithmetic it would run if the feature did not exist. Where the switch equals one, the bridged scans reproduce the plain ones on CUDA bit for bit in most profiles; the compiler schedules the two specializations and contracts their multiply-adds independently, so in the others they agree to within `1e-7` of the largest component, in either direction (a sweep over the 240 compiled configurations, with a window below every edge and with a pair table of vanishing cotangent, finds one forward and 41 backward configurations of that kind, at most `8.2e-8` apart). On the CPU the forward is bit-identical as well, while the two backward instantiations compile to different fused multiply-add forms and agree to the rounding of single precision, within `2e-7` of the largest gradient component. On the CPU the helpers an edge scan calls per edge are inlined unconditionally, because the doubled set of scan instantiations exhausts the unit-growth budget of the inliner, which would otherwise leave helpers such as the degree-three harmonics behind a call on every edge.

The bridging window multiplies the edge envelope. Every operator takes the two radii after `degree_floor`, and the host passes them to the scans together with the reciprocal width of the window; a bridged launch without a window, which only a direct operator call produces, keeps the outer radius at zero, which no edge lies below. The bridged scans evaluate the switch only for an edge inside the window: in a simulation nearly every edge lies beyond the outer radius, so the edges of a warp almost always take the common path together, and that path costs one comparison. Inside the window the switch follows the Horner form of the portable descriptor, so that the three implementations agree term by term, and the radii come from the descriptor configuration rather than from the table metadata, which keeps its five entries. The radial slope of the envelope follows from the product rule over the cutoff factor and the switch under the same comparison, and the backward scan evaluates it where it closes the radial cotangent of an edge.

The analytical pair potential is evaluated once per step, in the backward edge scan. It has no forward state, and its energy and radial slope share their exponentials, so the scan that already visits every edge together with the energy cotangent of its destination produces both: it adds `seed_i V'(rho) / 2` to the radial cotangent of each edge of node `i` and accumulates `V(rho) / 2` onto the energy of node `i`. ZBL enters as the series

```text
V_ab(rho)  = (1 / rho) sum_k A_abk exp(-c_abk rho)
V_ab'(rho) = -(1 / rho) sum_k A_abk (c_abk + 1 / rho) exp(-c_abk rho),

A_abk = k_e Z_a Z_b a_k,   c_abk = b_k / a_ab,   k = 1..4,
```

whose eight constants per ordered type pair form the table `pair_table` with shape `((T + 1)^2, 8)`, indexed like the PairFiLM cache and zero in the rows of the padding type. The analytical model derives the table from the type map together with its nuclear charges, so it follows a type-map change and is never stored in a checkpoint. On CUDA the lanes of an edge group split the four terms between them, lane `l` of `EdgeWidth` taking the terms `k = l (mod EdgeWidth)`, which is the idiom of the single-channel high degrees: the per-lane radial cotangents ride the reduction that already merges the lanes of an edge, the per-lane energies are merged by one reduction over the warp after the loop, and no lane evaluates more than two exponentials per edge. The series depends on the geometry alone, so the scan evaluates it as soon as the edge is loaded and adds its slope to the radial cotangent only when it closes the edge; the exponentials then overlap the channel scan instead of lengthening its chain of dependent instructions. The CPU scan evaluates the series on the edge and holds the exponent at eighty, where a term lies at least nineteen orders of magnitude below the float32 resolution of the sum for every element pair within 6 Å, and fourteen within 10 Å, because beyond it the exponential reaches the subnormal range, whose arithmetic stalls the core. An edge whose envelope vanishes, beyond the cutoff or inside the window, leaves the descriptor but keeps the pair term, so the CPU resolver rejects only padding neighbors and the scans read the vanishing envelope themselves.

The bridged scans share the launch bounds of the plain ones, and both follow the register budget of their profile. A launch bound names the resident blocks a scan is compiled for, and because a block is one warp, it fixes the registers per thread. A multiprocessor of compute capability 8.9 or 12.x holds at most twenty-four blocks, so eighty registers is the tightest budget a bound can set there, and the radial-mode scans of sixty-four channels and more are compiled for it. Without the budget the compiler gives them 86 to 128 registers; with it they keep all twenty-four warps resident for a spill of at most 80 bytes, which shortens the plain scans by 3% to 13% on an RTX PRO 6000 Blackwell and gives the bridged scans of Air and Plus the occupancy of the plain ones. The other profiles run fastest on the register count the compiler chooses and request no budget there: their plain scans compile to the edge-loop arithmetic of a build without zone bridging and return bit-identical results. In the budgeted profiles the schedule changes with the budget, which moves the plain results by single-precision rounding, at most 5e-6 of a force component.

The generic backward operator takes `pair_table` and the energy cotangent `pair_seed` after its geometry scalars and returns the node energies as a fourth result; the fused canonical operator takes `pair_table` last, reuses its fitting seed, and adds the pair energy onto the energy rows of each node tile. An empty table denotes a model without an analytical term, which is also what the registered autograd of the descriptor passes: on the level-one route the portable analytical model evaluates ZBL itself. The model layer resolves all of this through one capability of the atomic model, `fused_decomposition()`, which answers with the learned descriptor-fitting part and the analytical pair potential when the model is exactly their sum. The fused energy-force route, the canonical eligibility, the canonical lower and the charge-state fold read it instead of reaching for a descriptor, and the per-type bias of the fused pipelines is the fitting bias plus the output bias of the composition and of its learned part. An exported archive records every operator call with its full argument list, so a compressed archive whose calls lack the bridging radii and the pair table does not load against this operator library; it is exported again from its checkpoint.

Measured end to end in LAMMPS with Kokkos on an RTX PRO 6000 Blackwell with `debug/cuda_bench/zbl_compare.py` (the five released grades of Section 15, diamond carbon of 0.25, 1 and 3 million atoms, 100 NVT steps at a vanishing time step so that both variants step through the same configurations, the plain and the bridged model alternating on one device, the median of three repeats), a bridged model costs 3.3% to 4.5% of the MD step for Nano, 2.2% to 4.0% for Mini, at most 0.7% for Neo and Air, and nothing measurable for Plus. The pair term adds the same four exponentials to every edge, which the wider grades hide behind their longer channel scan. The distance clamp of Section 2.5 adds nothing measurable to this: its polynomial and the second evaluation of the cutoff factor run only on the edges inside the window, the plain scans compile to machine code that does not contain it, and the bridged step time of every grade at one and three million atoms is the same with and without it to within the 0.6 % run-to-run spread, with equal or fewer registers and no additional spill in the canonical bridged scans. The time of these latency-bound scans also depends on the order of the neighbors and on where the buffers of a process land: the same plain Mini model on the same input of one million atoms runs its forward scan in 15.5 ms per evaluation in one LAMMPS process and in 16.8 ms in another, so a difference of one or two percent between single runs lies within that spread, and only alternating repeats resolve it; at 32 thousand atoms the step is short enough that repeats scatter by several percent. The specialization doubles the edge-scan instantiations, which adds 7 MB to the operator library and raises the compile time of a per-width translation unit from about one minute to about two. On the CPU (AVX-512, 16 pinned threads, 4096 atoms of the same graph) the plain scans stay within 1% of such a build for the forward and the backward together.

## 15. CUDA benchmark protocol

The end-to-end benchmark uses one NVIDIA H20, LAMMPS 4 Jul 2026 with Kokkos, and the canonical
compressed PT2 lower. Kokkos supplies the accelerator execution layer. Each point executes 10
warm-up molecular-dynamics steps followed by 30 measured NVT steps. NVT denotes a trajectory with
fixed atom count, volume, and temperature. Throughput is the complete MD-step atom count divided by
LAMMPS loop time, in atoms/ms.

LAMMPS hosts the DeePMD-kit styles either compiled into its library or loaded at startup from the DeePMD-kit plugin, and the scan uses the plugin. Both integrations link the same installed DeePMD-kit libraries and run the same operators; on an RTX PRO 6000 Blackwell every grade, plain and ZBL-bridged, steps within 0.5% of the built-in integration from 250 thousand atoms up, with no systematic sign, so a curve measured either way is comparable.

The DPA4C half of the comparison is the five released grades. Every grade uses `n_radial = 16`,
`rcut = 6 Å`, `float32` descriptor and fitting parameters, and three SiLU fitting hidden layers
without a timestep:

| Grade | `channels` | `lmax` | `radial_modes` | Fitting hidden width | `D_out` |
| ----- | ---------: | -----: | -------------: | -------------------: | ------: |
| Nano  |          8 |      2 |              0 |                   96 |      70 |
| Mini  |         32 |      2 |              0 |                  192 |     144 |
| Neo   |         64 |      2 |              0 |                  256 |     208 |
| Air   |         64 |      3 |              4 |                  256 |     219 |
| Plus  |        128 |      3 |              4 |                  384 |     541 |

Grades Nano, Mini and Neo hold the angular degree at two without radial modes and differ only in
scalar width and fitting width. Grades Air and Plus raise the angular degree, whose degree-three
block carries one channel, and add rank-four modes.

The diamond-carbon systems retain the logarithmic size grid below one million atoms. Capacity search
starts at one million atoms, advances in two-million-atom increments until an allocation failure,
and then bisects the final interval at one-million- and half-million-atom resolution.

The comparison reuses compact canonical DPA1 S/M/L curves with fitting widths 64/128/256. The NEP89
baseline is the 89-element NEP4-with-ZBL potential `models/nep89_20250409.txt` run in GPUMD on the
same geometries; GPUMD uses 10 warm-up and 100 measured steps because its native speed report is
inexpensive. Both report complete-step atoms/ms.

Capacity is measured on an otherwise idle device. A co-tenant process shifts the bracket by one or
two grid steps, so a capacity figure is only comparable against others taken under the same
condition; throughput is far less sensitive.

Each result CSV has an identity manifest containing the model hash, input script, geometry
generator, GPU UUID, DeePMD libraries, optional preload, and scan protocol. Shared diamond systems
are published under a process lock. Only an explicit CUDA/Kokkos allocation failure closes a
capacity bracket; timeouts, schema failures, and other process errors abort the run.

### 15.1 H20 results

Saturated throughput is the arithmetic mean of all successful points containing at least one million
atoms. Capacity reports the largest successful diamond supercell and the first failed supercell:

| Model       | Atoms/ms | Largest success | First failure |
| ----------- | -------: | --------------: | ------------: |
| NEP89       |    8,548 |      10,455,280 |    11,036,032 |
| DPA1-S/F64  |    8,873 |      11,539,560 |    11,956,320 |
| DPA1-M/F128 |    7,012 |       8,489,664 |     8,998,912 |
| DPA1-L/F256 |    4,692 |       6,025,656 |     6,501,040 |
| DPA4C-Nano  |   16,814 |      14,051,520 |    14,522,880 |
| DPA4C-Mini  |   10,191 |      14,051,520 |    14,522,880 |
| DPA4C-Neo   |    6,969 |      14,051,520 |    14,522,880 |
| DPA4C-Air   |    3,800 |      14,051,520 |    14,522,880 |
| DPA4C-Plus  |    2,165 |      14,051,520 |    14,522,880 |

The DPA1 curves are reused measurements; the DPA4C grades and NEP89 were measured in this campaign.

Every grade reaches the same largest system. This is the intended consequence of evaluating the
pipeline over node tiles: the descriptor, its cotangent, the moment state and the fitting
pre-activations are all retired one tile at a time, so nothing that scales with the model width
survives at system scale. What remains is the graph and the edge cotangent, which depend only on
the cutoff and are therefore identical across grades. A fitted per-atom budget of 6,838 bytes plus
1,881 MiB predicts 14.66 million atoms against the 14,051,520 measured; Section 11.9 of
`doc/outisli/dpa1&4c_cuda.md` derives that budget, explains why each remaining term stays under a
materialized graph, and estimates what a streaming graph would be worth.

Throughput then separates the grades on their own. Nano exceeds NEP89 by 97% and DPA1-S by 89%
while running 34% more atoms than NEP89. Mini exceeds NEP89 by 19% and DPA1-M by 45%. Neo doubles
Mini's scalar and fitting widths for 68% of its throughput and matches DPA1-M to within 1%. Air
adds degree three and rank four on top of Neo for 55% of its throughput, and Plus is the widest
configuration measured; both run the same system size as Nano, which before tiling they could not
approach.

The grade figures carry the two reductions of Section 11.11 of
`doc/outisli/dpa1&4c_cuda.md`, which remove a fixed cost per step and therefore lift the cheaper
grades most: Nano by 5.4% and Plus by 0.7%.

The DPA4C benchmark models use deterministic random weights. DPA1 uses MatPES EMA checkpoints, and
NEP uses a trained potential. These measurements isolate execution cost and do not compare
predictive accuracy. Different model weights can also produce different NVT trajectories and
neighbor-list rebuild histories.

The same compressed snapshot also serves a CPU host, through a second set of hand-written operators
that share the schemas and the Python front end of the CUDA ones. `doc/outisli/dpa4c_cpu.md`
specifies them and records the measurements: on two 45-core Sapphire Rapids sockets a released grade
evaluates one 8000-atom step in 4 to 18 ms, between 57 and 4978 times faster than the same model
frozen without compression, and the deviation from that model stays at float32 rounding.

### 15.2 Structural parameter sweep

Two structural parameters change the compiled kernel: radial mode rank `R` and maximum angular
degree `L`. Each parameter is swept at every descriptor width while the other remains at its
production value. Every point uses the same atom grid, fitting network, and adaptive out-of-memory
protocol as the capacity curves. The common corner `L = 2, R = 0` is scanned once.

The numbers below were measured before the descriptor gained its two neighborhood divisors and
before inference moved to node tiles, so their absolute throughput is lower than the grades reach
today and their capacities are the pre-tiling ones. The ratios they establish are what the section
is for and are unaffected: both parameters change the per-edge and per-node arithmetic without
changing the tiling. `parameter_sweep.py` without `--grades-only` reproduces them.

Saturated throughput in atoms/ms is shown below. Parentheses give the fraction of the production `L = 2, R = 0` throughput.

|      |       `R = 0` |       `R = 2` |       `R = 4` |       `R = 8` |
| ---- | ------------: | ------------: | ------------: | ------------: |
| C8   | 16,699 (1.00) | 14,228 (0.85) | 13,303 (0.80) | 10,598 (0.64) |
| C16  | 15,124 (1.00) | 12,181 (0.81) | 11,256 (0.74) |  9,975 (0.66) |
| C32  | 10,988 (1.00) |  8,429 (0.77) |  7,541 (0.69) |  6,476 (0.59) |
| C64  |  6,917 (1.00) |  5,096 (0.74) |  4,392 (0.64) |  3,709 (0.54) |
| C128 |  4,583 (1.00) |  3,293 (0.72) |  2,808 (0.61) |  2,261 (0.49) |

|      |       `L = 2` |       `L = 3` |      `L = 4` |
| ---- | ------------: | ------------: | -----------: |
| C8   | 16,699 (1.00) | 11,595 (0.69) | 6,765 (0.41) |
| C16  | 15,124 (1.00) | 10,408 (0.69) | 6,153 (0.41) |
| C32  | 10,988 (1.00) |  8,247 (0.75) | 5,275 (0.48) |
| C64  |  6,917 (1.00) |  5,586 (0.81) | 3,789 (0.55) |
| C128 |  4,583 (1.00) |  3,712 (0.81) | 2,605 (0.57) |

Radial modes are close to free in capacity: the largest successful system is identical at every `R`
for a fixed `C_0`. The ordered mixing cache is a type-table quantity rather than a per-edge or
per-node state. Its throughput cost grows with scalar width. At `R = 8`, the retained fraction falls
from 0.64 at C8 to 0.49 at C128 because mixing coefficients are read for every edge and channel.

Angular degree has the opposite width trend. Degrees three and four add a fixed total of sixteen
single-channel moment components, independent of `C_0`. Their relative cost is therefore largest at
narrow widths: `L = 4` retains 0.41 of baseline throughput at C8 and 0.57 at C128.

Capacity is unchanged across `L` for C8, C64, and C128. The recorded largest-success point is lower
for C16 at `L = 4` and for C32 at `L = 3` and `L = 4`.

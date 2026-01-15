# DPA4 / SeZM Implementation Notes

:::{note}
**Supported backends**: PyTorch {{ pytorch_icon }}
:::

DPA4 is the DPA-series name for SeZM (Smooth Equivariant Zone-bridging Model), implemented in the PyTorch backend of DeePMD-kit. This document is the detailed implementation record; the public user-facing model guide is `doc/model/dpa4.md`. In input files, use `model.type: "dpa4"`; `SeZM` and `sezm` remain equivalent compatibility aliases for the same implementation. Under the DPA4 model scaffold, `descriptor.type` defaults to `dpa4` and `fitting_net.type` defaults to `dpa4_ener`. Invariant property prediction is selected explicitly with `fitting_net.type = "property"`.

Training examples: `examples/water/dpa4/input.json` for energy and `examples/water/dpa4/input_property.json` for invariant property prediction.

**Source files:**

- `deepmd/pt/model/descriptor/sezm.py` — top-level descriptor (`DescrptSeZM`)
- `deepmd/pt/model/descriptor/sezm_nn/` — all submodules
- `deepmd/pt/model/model/sezm_model.py` — model scaffold, compile path, ZBL injection
- `deepmd/pt/model/model/sezm_property_model.py` — property model variant

______________________________________________________________________

## 1. Overview

SeZM is an SO(3)-equivariant message-passing descriptor designed for molecular dynamics workloads where inference speed and physical correctness jointly matter. The model predicts per-atom energies, derives forces by differentiating the energy with respect to coordinates (conservative forces), and guarantees a smooth potential energy surface (PES) through C³-continuous cutoff envelopes on every edge.

The descriptor maps local atomic environments to rotationally invariant scalar features through a stack of interaction blocks, each containing an SO(2) convolution operating in a per-edge local frame and an equivariant feed-forward network (FFN). Edge geometry — distances, radial basis, Wigner-D rotation matrices — is computed once per forward call and shared across all blocks, avoiding the cost multiplier that plagues per-block geometry recomputation.

The descriptor and model are registered primarily as `dpa4`, with `SeZM` and `sezm` kept as equivalent aliases. The model-level scaffold `SeZMModel` handles energy/force/virial output, optional analytical short-range repulsion, and an end-to-end `torch.compile` path that supports second-order coordinate derivatives through Inductor.

### 1.1 Key Innovations

**Zone Bridging framework.** SeZM introduces a general-purpose inner-bridging mechanism that allows any analytical short-range potential to be additively composed with the learned energy, while guaranteeing that the descriptor contribution on every bridging-zone pair is strictly frozen. The mechanism composes two parameter-free modules — InnerClamp (C³ distance saturation) and SFPG (Source Freeze Propagation Gate, a multiplicative per-node gate) — on the same radial window. Together they close all three leakage channels (scalar distance, direction, multi-hop propagation) that would otherwise break the additive decomposition. The default analytical potential is ZBL, but the framework is agnostic to the specific formula: any pair potential that accepts a pairwise distance can be plugged in.

**Compiled double-backward training.** SeZM achieves the first end-to-end `torch.compile` training path for an equivariant ML potential whose force loss requires second-order coordinate derivatives (`∂²E/∂x∂θ`). The approach uses `make_fx` symbolic tracing to capture `autograd.grad(create_graph=True)` as ordinary FX nodes, then hands the flat graph to Inductor with dynamic shapes. This yields a 2–3× wall-clock speedup over eager training while keeping the full conservative-force guarantee intact. A catalogue of compile invariants (§12.2) documents each non-obvious choice that makes this work under PyTorch's current tracing constraints.

**Model architecture:**

- **Environment Initial Embedding (FiLM).** A physics-motivated 4D environment matrix `[s, s·r̂]` is aggregated into a low-rank local descriptor `D = env_agg^T @ env_agg`, then projected to FiLM scale/shift logits that condition the scalar backbone. This provides geometric inductive bias at layer 0 through a dedicated type-embedding branch, decoupled from the main type embedding, reducing the number of interaction blocks needed to capture local geometry.

- **Envelope-gated softmax attention.** The message aggregation uses `edge_env² × exp(logit)` in the numerator and a learnable positive bias `ζ` in the denominator, so that the attention weight of each edge smoothly decays to zero at the cutoff boundary together with its derivatives. A post-aggregation output-side head gate (query-dependent, per-head sigmoid) further modulates the aggregated message. This attention replaces degree-based normalization when active and naturally integrates with SFPG through the `src_weight` parameter.

- **Multi-focus SO(2) convolution.** Multiple parallel focus streams process the same geometric context inside the SO(2) operator. A cross-focus softmax competition (driven by `l=0` scalar invariants, with label smoothing to prevent dead focuses) re-weights the streams before rotate-back. Unlike MHA, which attends across sequence positions, multi-focus attends across parallel equivariant sub-channels on the same edge. Unlike sparse MoE, all focuses are computed and then soft-weighted, preserving SO(3) equivariance.

- **Configurable trainable radial basis.** The default Bessel basis uses trainable frequencies initialized as integer harmonics `nπ/rcut`; the optional Gaussian basis uses trainable centers over `[0, rcut]`. Both share the same `adam_freqs` parameter shape and downstream `n_radial` interface.

**Training methodology:**

- **HybridMuon with slice mode.** SeZM uses `muon_mode="slice"` for its primary optimizer routing: 2D weight matrices go to Muon, while 3D `SO3Linear` weights `(lmax+1, C_in, C_out)` are sliced along the degree axis so each `l`-block receives an independent Muon Newton-Schulz update with correct rectangular scaling. The Muon learning rate adjustment (`lr_adjust`) is set to 0, letting the base learning rate schedule control the effective step size directly.

- **Magma-lite damping.** A per-block cosine-alignment score between Muon momentum and current gradient is EMA-smoothed and mapped to a damping scale in `[0.1, 1.0]`, reducing update magnitude on blocks with noisy or misaligned gradients. No stochastic Bernoulli masking is used, keeping the damping dense and stable for force-field objectives.

- Norm scales, layer scales, radial frequencies, and type embeddings are routed to Adam (via `adam_` naming prefix) for stability.

______________________________________________________________________

## 2. Code Organization

```
deepmd/pt/model/descriptor/
├── sezm.py                    # DescrptSeZM: config parsing, edge-cache construction,
│                               #   block scheduling, forward(), serialize/deserialize
├── sezm_nn/
│   ├── __init__.py            # public re-export layer
│   ├── utils.py               # NVTX helpers, dtype promotion, serialization helpers
│   ├── edge_cache.py          # EdgeFeatureCache, edge construction, SFPG gate,
│   │                           #   build_edge_cache / build_edge_cache_from_edges
│   ├── indexing.py            # packed (l, m) indexing, reduced-layout maps,
│   │                           #   rotation projection helpers, inv_rescale
│   ├── radial.py              # C3CutoffEnvelope, InnerClamp, BridgingSwitch,
│   │                           #   RadialBasis, RadialMLP
│   ├── activation.py          # GatedActivation, SwiGLU
│   ├── projection.py          # BaseGridProjector, S2GridProjector,
│   │                           #   SO3GridProjector, grid resolution helpers
│   ├── grid_net.py            # S2GridNet, SO3GridNet, SwiGLU grid ops
│   ├── lebedev.py             # Packaged Lebedev-rule loader for S2 quadrature
│   ├── embedding.py           # SeZMTypeEmbedding, GeometricInitialEmbedding,
│   │                           #   EnvironmentInitialEmbedding
│   ├── norm.py                # EquivariantRMSNorm, ReducedEquivariantRMSNorm,
│   │                           #   ScalarRMSNorm
│   ├── attention.py           # segment_envelope_gated_softmax
│   ├── attn_res.py            # DepthAttnRes
│   ├── so3.py                 # ChannelLinear, FocusLinear, SO3Linear
│   ├── so2.py                 # SO2Linear, DynamicRadialDegreeMixer, SO2Convolution
│   ├── ffn.py                 # EquivariantFFN
│   ├── block.py               # SeZMInteractionBlock
│   ├── dens.py                # ForceEmbedding, denoising/direct-force heads
│   ├── wignerd.py             # build_edge_quaternion, WignerDCalculator
│   ├── triton/                # Triton inference kernels (DP_TRITON_INFER);
│   │   ├── so2_rotation.py    #   block-diagonal SO(2)/Wigner rotation
│   │   ├── radial_mix.py      #   fused dynamic radial degree mixer
│   │   ├── flash_atten.py     #   fused attention value-aggregation (rotate-back
│   │   │                      #   + envelope-softmax weight + segment reduction)
│   │   ├── so2_value_path.py  #   fused SO(2) value path (rotate+mix front end,
│   │   │                      #   gated mixing stack, focus competition)
│   │   ├── wigner_monomials.py#   quaternion monomial bases for Wigner-D blocks
│   │   ├── force_assembly.py  #   segmented force/virial assembly from edge grads
│   │   ├── so2_stack_fp16x3.py#   fp16x3 tensor-core mixing stack (level 3)
│   │   ├── tile_configs.py    #   launch-config lookup (runtime + built-in layers)
│   │   ├── tile_config_data.py#   built-in per-GPU launch tables (data only)
│   │   └── sweep_tile_configs.py  # sweep CLI + freeze auto-tuner
│   └── cute/                  # Experimental CuTe-DSL rotation kernels
│       └── so2_rotation.py
deepmd/pt/model/model/
└── sezm_model.py              # SeZMModel, InnerPotential,
                                #   trace_and_compile, make_fx path
```

Each submodule is self-contained. `sezm.py` imports everything through `sezm_nn/__init__.py`, which re-exports the public API of each submodule. Tests are split by responsibility: descriptor tests in `test_descriptor_sezm.py`, model/compile tests in `test_sezm_model.py`.

______________________________________________________________________

## 3. Forward Pass Overview

A single forward pass through SeZM proceeds as follows. The text diagram shows the main data flow; subsequent sections expand each stage.

```
SeZMModel main forward path (core_compute):
  Inputs: coord, atype, box
    ├─ build edge schema → coord, atype, edge_index, edge_vec,
    │                       edge_scatter_index, edge_mask
    └─ graph inputs: local atom types + edge-vector geometry

  EdgeFeatureCache (built once per forward via build_edge_cache_from_edges):
    ├─ edges: (src, dst) global indices, edge_vec
    ├─ edge_type_feat: per-edge type embedding (src + dst)
    ├─ edge_rbf: configured radial basis × C³ envelope
    ├─ edge_env: C³ cutoff envelope (flattened to valid edges)
    ├─ D_full, Dt_full: block-diagonal Wigner-D matrices up to MP lmax
    ├─ inv_sqrt_deg: inverse sqrt smooth degree for normalization
    └─ edge_src_gate: SFPG per-edge gate η_src (when bridging is active)

  Radial embedding (computed once in fp32+):
    └─ radial_feat: (E, node_l_schedule[0]+1, C) via RadialMLP(edge_rbf)
       └─ type features fused once after GIE; per-block slices prebuilt

  Node initialization:
    ├─ l=0: Type embedding + optional charge/spin condition embedding
    ├─ optional EnvironmentInitialEmbedding (FiLM)
    └─ l>0: Zonal (m=0) initial embedding via node-level zonal Wigner coupling

  Interaction blocks (pyramid schedule):
    for block i:
      ├─ slice node features to ebed_dim(l_schedule[i] + extra_node_l)
      ├─ optional depth AttnRes from unit/block history
      ├─ MP prefix: coefficients up to l_schedule[i]
      ├─ EquivariantRMSNorm (pre-SO2, MP prefix only)
      ├─ SO(2) Convolution
      │  ├─ pre_focus_mix: full-channel projection
      │  ├─ rotate to edge-local frame via Wigner-D
      │  ├─ radial modulation / dynamic radial degree mixing + multi-layer SO2Linear stack
      │  ├─ optional cross-focus softmax competition
      │  ├─ rotate back to global frame
      │  └─ scatter-aggregate (envelope-weighted or attention)
      ├─ pad SO2 update back to the node tensor; l > l_schedule[i] is unchanged
      ├─ FFN subblocks (ffn_blocks iterations)
      │  ├─ EquivariantRMSNorm (pre-FFN, node degree)
      │  └─ SO3Linear → Activation → SO3Linear (zero-init output)
      └─ optional depth AttnRes updates

  Output (read-out controlled by so3_readout):
    └─ so3_readout="none": extract l=0, scalar FFN with residual x[l=0] + FFN(x[l=0])
    └─ so3_readout="glu"/"mlp": full SO3 Wigner-D grid FFN at the last block's node
       degree over the complete node tensor, then extract l=0 (residual on l=0)
    └─ Reshape to (nf, nloc, channels)

  Post-process in SeZMModel:
    ├─ fitting network + output statistics
    ├─ optional ZBL energy on the same pair-filtered sparse-edge path
    ├─ final atom mask over the complete physical output
    ├─ edge-force scatter: one autograd.grad(E, edge_vec) → force / virial / atomic virial
    └─ local outputs (energy, force, virial)
```

The deployable lower interface uses the same compact edge representation directly. A frozen `.pt2` SeZM model takes `(coord, atype, edge_index, edge_vec, edge_scatter_index, edge_mask, ...)` rather than a padded neighbor matrix. `edge_index` addresses flattened local owners and drives descriptor message passing; `edge_vec` is the displacement source of truth; `edge_scatter_index` addresses the force/virial scatter domain. For Python inference, Vesin and Toolkit-Ops builders construct `edge_vec` directly from their integer PBC shifts. For LAMMPS, ghost coordinates are already periodic-image coordinates, so the same schema is populated with zero-shift edge vectors `x_ghost - x_center` and extended scatter indices for reverse communication.

Two `forward` entry points exist in `DescrptSeZM`:

- **`forward(...)`** — the standard DeePMD descriptor interface. Accepts `extended_coord`, `extended_atype`, `nlist`, `mapping`, and optional frame-level `charge_spin`. Builds a padded edge cache from the neighbor list. Zone bridging is handled only by `SeZMModel` on the sparse-edge path.
- **`forward_with_edges(...)`** — the sparse-edge interface used by `SeZMModel`. Accepts pre-built `(edge_index, edge_vec, edge_mask)` and optional frame-level `charge_spin`. Supports zone bridging and returns both the scalar descriptor and the full equivariant latent for downstream heads.

Both paths share `_forward_blocks(...)` for the actual interaction-block loop.

______________________________________________________________________

## 4. Edge Feature Cache

### 4.1 Design Rationale

Edge geometry computation — coordinate gathering, distance calculation, quaternion construction, Wigner-D evaluation, radial basis expansion — is the most expensive per-edge workload. SeZM computes the SO(2) message-passing Wigner-D matrices exactly once per `forward()` and packs them into an `EdgeFeatureCache` dataclass. Every interaction block reads from this shared MP cache; no block is allowed to recompute MP geometry. When `extra_node_l > 0`, GIE separately computes only the node-level zonal `m=0` coupling it needs, avoiding a full node-level `(D, D)` Wigner-D cache.

### 4.2 Cache Contents

| Field             | Shape              | Description                                       |
| ----------------- | ------------------ | ------------------------------------------------- |
| `src`, `dst`      | `(E,)`             | Flattened node indices in `[0, N)`                |
| `edge_type_feat`  | `(E, C)`           | Per-edge type embedding (src + dst lookup)        |
| `edge_vec`        | `(E, 3)`           | Displacement vectors in Å                         |
| `edge_rbf`        | `(E, n_radial)`    | Configured radial basis × C³ envelope             |
| `edge_env`        | `(E, 1)`           | C³ cutoff envelope weights                        |
| `deg`             | `(N, 1)`           | Smooth degree: `Σ_e edge_env²` per destination    |
| `inv_sqrt_deg`    | `(N, 1, 1)`        | `rsqrt(deg + eps)` for normalization              |
| `D_full`          | `(E, D_mp, D_mp)`  | Block-diagonal Wigner-D for MP (global→local)     |
| `Dt_full`         | `(E, D_mp, D_mp)`  | Transpose of `D_full` (local→global)              |
| `D_to_m_cache`    | dict               | Cached m-major projections keyed by `"lmax:mmax"` |
| `Dt_from_m_cache` | dict               | Cached inverse projections keyed by `"lmax:mmax"` |
| `edge_src_gate`   | `(E, 1)` or `None` | SFPG per-edge gate `η_src(e)` (bridging only)     |

### 4.3 Edge Construction

**Neighbor-list path** (`build_edge_cache`): Takes the DeePMD padded neighbor list `(nf, nloc, nnei)`. Padding entries (`nlist == -1`) and excluded type pairs are filtered out before any distance computation. Edges with `r ≥ rcut` are kept — their `edge_env = 0` from the C³ envelope naturally zeros their messages. This avoids the dynamic-output-size `torch.nonzero` kernel that distance filtering would require, and it keeps the smooth degree `Σ edge_env²` free of discontinuous jumps at the cutoff boundary.

**Sparse-edge path** (`build_edge_cache_from_edges`): Takes pre-built `(edge_index, edge_vec, edge_mask)` from `SeZMModel.core_compute`. Masked edges (`edge_mask=False`) have their displacement vector reset to `(0, 0, 1)` to provide a safe normalization target — this prevents NaN gradients from zero-length vector division while contributing zero downstream thanks to `edge_env = 0`.

### 4.4 Smooth Degree Normalization

After cache construction, `_finalize_edge_cache` computes:

```
deg[j] = Σ_{e: dst(e)=j} edge_env[e]²
inv_sqrt_deg[j] = 1 / sqrt(deg[j] + eps)
```

The squared envelope ensures the degree is a smooth function of atomic positions (C⁶ regularity from squaring a C³ function). `inv_sqrt_deg` is applied at every aggregation site in the non-attention path, providing graph-style normalization analogous to GCN's `D^{-1/2}`.

### 4.5 Dtype Dispatch

Geometry computations always run in fp32+ (`compute_dtype = get_promoted_dtype(dtype)`) regardless of the model's working precision. This ensures accurate distances, quaternions, and Wigner-D matrices for stable training convergence.

`build_edge_cache` computes the geometry/RBF chain (`gather → edge_vec → edge_len → edge_env → edge_rbf`) with the eager PyTorch path, preserving the full autograd graph for force/virial higher-order derivatives. The `edge_cache_to_dtype` helper converts float fields to the working dtype when entering the interaction blocks, and clears the rotation projection caches to prevent dtype mismatches.

______________________________________________________________________

## 5. Radial Functions

### 5.1 C³ Cutoff Envelope

The `C3CutoffEnvelope` (in `radial.py`) enforces a smooth transition to zero at the cutoff radius. For normalized distance `x = r / rcut` and cutoff distance `u = 1 - x`:

```
E_p(x) = u⁴ * Σ[k=0..p-1] binom(k+3, 3) * x^k    for x ∈ [0, 1)
E_p(x) = 0                                        for x ≥ 1
```

The explicit `u⁴` factor guarantees `E(1) = E'(1) = E''(1) = E'''(1) = 0`. The positive-coefficient series is evaluated with Horner's rule, avoiding cancellation between order-one polynomial terms when `x` is one float32 ULP below the cutoff. The exponent `p` (default 5 for edge envelope, 7 for radial basis envelope) controls how steeply values decay near `rcut`: larger `p` keeps values closer to 1.0 over more of the range before dropping.

With the default `p = 5`:

```
E_5(x) = u⁴ * (1 + 4x + 10x² + 20x³ + 35x⁴)
```

The C³ envelope is applied at two points:

1. **Inside `RadialBasis.forward()`**: multiplied into each radial basis function, making the basis itself vanish at `rcut`.
1. **As `edge_env`**: applied to all edge messages during aggregation.

This double application ensures that a message, its first derivative, its second derivative, and its third derivative with respect to distance all reach exactly zero at `rcut`. Conservative forces (first derivative of energy) and force-loss training (second derivative) therefore see no discontinuity at the cutoff boundary.

### 5.2 Trainable Radial Basis

`RadialBasis` (in `radial.py`) produces `n_radial` basis functions evaluated at each edge distance. The `basis_type` setting selects the basis family.

For `basis_type="bessel"` (default), the basis uses a sinc form:

```
φ_n(r) = w_n · sinc(w_n · r / π) = sin(w_n · r) / r
```

where `w_n = n · π / rcut` for `n = 1, ..., n_radial`. The sinc form is chosen for numerical stability near `r → 0`: unlike `sin(w·r)/r`, `sinc` is well-defined at zero and produces stable gradients.

For `basis_type="gaussian"`, the basis uses trainable centers `c_n` initialized uniformly on `[0, rcut]`:

```
φ_n(r) = exp(-0.5 · ((r - c_n) / σ)²),   σ = rcut / max(n_radial - 1, 1)
```

Both basis families are multiplied by the C³ envelope (with `exponent` typically 7) before output.

The trainable radial parameters are stored as `adam_freqs` with the `adam_` prefix so that HybridMuon routes them to the Adam optimizer without weight decay. In Bessel mode they are frequencies; in Gaussian mode they are centers. The parameter name and shape are unchanged across the two modes, so existing Bessel checkpoints continue to load with the default `basis_type="bessel"`.

### 5.3 RadialMLP

`RadialMLP` maps the `n_radial`-dimensional radial basis vector to a `(node_l_schedule[0]+1) × channels`-dimensional per-edge feature. Each hidden layer is `Linear → RMSNorm → SiLU` when `edge_norm=True` and `Linear → SiLU` when `edge_norm=False`; the final layer is always a bare linear projection without normalization or activation.

All linear layers use `bias=False`. This is a deliberate choice: with zero bias, a zero input (from masked or padding edges) produces exactly zero output. Bias terms would leak a constant offset into masked edges, which is particularly problematic on the `torch.compile` path where padding edges must contribute exactly zero.

**Cutoff-vanishing normalization and smoothness (`edge_norm`).** The input `edge_rbf` carries the C³ cutoff envelope and therefore vanishes at `rcut`. A hidden `RMSNorm` normalizes each edge's radial features by their own RMS, which divides that envelope out (recovering an O(1) direction) until the variance reaches the norm's `eps` floor. On a sparse neighborhood — a dimer being the extreme case — the resulting high-gain transition produces a narrow, non-physical force feature just inside the cutoff even though the positive `eps` keeps the operation formally smooth. Setting `edge_norm=False` removes the hidden `RMSNorm`, so the radial features retain the envelope amplitude. Feeding the raw (un-enveloped) basis into the norm instead is not equivalent: the raw Bessel `sinc` basis is oscillatory, and the norm amplifies those oscillations near `rcut`, so removing the norm is preferred over relocating the envelope.

`edge_norm` is the single switch for every normalization that acts on a cutoff-vanishing input. It governs four branches:

1. **Radial MLP** (above): the hidden `RMSNorm` on `edge_rbf`.
1. **Environment-seed FiLM** (`film_scale_norm`, `film_shift_norm`, §6.3): the FiLM logits come from `D = envᵀenv`, which vanishes as `edge_env²` at the cutoff. With `edge_norm=False` both FiLM norms become identity pass-throughs; the logits then reach `tanh` un-normalized and vanish smoothly (`scale → 1`, `shift → 0`) at `rcut`. The env-seed projection is zero-initialized and each FiLM branch has a learnable strength, so dropping the norm costs no expressiveness.
1. **Cross-focus competition** (`focus_compete_norm`, §7): the competition scalars are the `l = 0` of the per-edge features after radial modulation, so once the radial features vanish they vanish too. With `edge_norm=False` the competition norm is an identity pass-through and the softmax logits `einsum(focus_scalars, w)` decay to zero, so the focus weights approach uniform smoothly at `rcut` (label smoothing already blends toward uniform); the message vanishes there via `edge_env²` regardless, so the near-cutoff effect is negligible.
1. **Post-SO(2) residual scaling** (`post_so2_norm`, §8.1): the aggregated SO(2) message also vanishes with the edge envelope. Standard RMSNorm gives this small residual a gain approaching `1/sqrt(eps)`. With `edge_norm=False`, the same equivariant centering, degree balancing, affine scale, and scalar bias are retained, but the denominator becomes `sqrt(1 + variance)`. This unit-floor mode passes infinitesimal messages linearly and smoothly limits large messages without changing any other RMSNorm site.

The remaining norms act on persistent node states (`pre_ffn_norms`, `attn_qk_norm`) or on FFN outputs that are not tied directly to a cutoff-vanishing edge amplitude. They retain standard RMS normalization in both modes.

The output is reshaped to `(E, node_l_schedule[0]+1, C)` — one radial feature vector per degree `l` and per channel. The `l = 0` slice feeds the scalar branch; `l ≥ 1` slices feed the Geometric Initial Embedding (GIE). Each SO(2) block receives the prefix `radial_feat[:, :l_schedule[i]+1, :]` that matches its message-passing degree.

______________________________________________________________________

## 6. Initial Feature Construction

### 6.1 Type Embedding

`SeZMTypeEmbedding` (in `embedding.py`) maps discrete atom types to continuous vectors. It stores a learnable embedding table of shape `(ntypes, channels)` named `adam_type_embedding`. The `adam_` prefix routes it to the Adam optimizer in HybridMuon. An optional padding row (index `ntypes`) is zeroed out for masked atoms.

The type embedding provides the sole `l = 0` initial feature:

```
x[:, 0, 0, :] = type_embedding(atype)    # (N, C)
```

All higher-degree coefficients (`l ≥ 1`) start at zero and are seeded by the Geometric Initial Embedding described below.

### 6.2 Geometric Initial Embedding (GIE)

`GeometricInitialEmbedding` (in `embedding.py`) seeds `l > 0` features at layer 0 by projecting radial features through the zonal (`m = 0`) local-to-global coupling. This gives the initial backbone non-trivial angular information from the start, reducing the number of interaction blocks needed to capture directional dependence.

For each degree `l ≥ 1`, the local `m=0` to global-row coupling at row `(l, m')` gives the real spherical harmonic `Y_l^{m'}` evaluated along the edge direction. When `extra_node_l = 0`, it is gathered from the MP `Dt_full`; when `extra_node_l > 0`, GIE reuses the MP part from `Dt_full` and `WignerDCalculator.forward_zonal(lmin=l_schedule[0]+1)` computes only the extra node degrees. The GIE multiplies this value element-wise with the radial feature `radial_feat[:, l, :]` (shape `(E, C)`), then scatters the result to destination nodes:

```
For each edge e = (src → dst):
  For each degree l ≥ 1:
    msg_l[m'] = m0_to_global[e, l²+l+m'] × radial_feat[e, l, :]    for m' = -l..+l
  scatter_add msg to node dst
Normalize by inv_sqrt_deg
```

The implementation avoids advanced-index writeback (which can silently produce zero gradients on some PyTorch builds under `make_fx`). Instead, it uses `index_add_` into a compact buffer and then assigns back to the output tensor at the appropriate row indices.

When SFPG is active, each edge message is multiplied by `edge_src_gate` before scatter, so any edge from a frozen-zone source contributes exactly zero.

### 6.3 Environment Initial Embedding (FiLM Conditioning)

`EnvironmentInitialEmbedding` (in `embedding.py`, optional via `use_env_seed=True`) provides a physics-motivated inductive bias for the scalar features by conditioning them on a local environment matrix. The conditioning uses Feature-wise Linear Modulation (FiLM): the environment embedding produces per-atom scale and shift parameters that modulate the type embedding.

**The 4D environment vector.** For each edge, a 4-component vector `r_tilde` encodes both radial decay and angular information in the global frame:

```
s = edge_env / r
r_hat = edge_vec / r
r_tilde = [s, s·r_hat_x, s·r_hat_y, s·r_hat_z]
```

Unlike the edge-local frame used by SO(2) convolution, `r_tilde` uses the global frame direction. This preserves full angular information (three independent components) rather than projecting onto a local axis.

**G-network.** A two-layer MLP processes per-edge features:

1. RBF projection: `Linear(n_radial → rbf_out_dim) → SiLU → Linear(rbf_out_dim → rbf_out_dim)`
1. Concatenate with source and destination type embeddings from an independent `env_type_embed`
1. G-MLP: `Linear(concat_dim → hidden_dim) → SiLU → Linear(hidden_dim → embed_dim)`

The environment type embedding is deliberately independent from the main type embedding, allowing `env_seed` to learn its own type representations optimized for the environment-matrix pathway.

**Aggregation and D-matrix.** The outer product `r_tilde ⊗ g` (shape `(E, 4, embed_dim)`) is scattered to destination nodes and normalized by `inv_sqrt_deg`. The resulting aggregated matrix `env_agg` (shape `(N, 4, embed_dim)`) captures the local geometry around each atom. The matrix product `D = env_agg^T @ env_agg[:, :, :axis_dim]` (shape `(N, embed_dim, axis_dim)`) reduces this to a compact descriptor.

**FiLM application.** `D` is flattened and projected to `2 × channels` logits, split into scale and shift:

```
scale = 1 + scale_strength · tanh(norm(scale_logits))
shift = shift_strength · tanh(norm(shift_logits))
x0 = x0 · scale + shift
```

The strengths are learnable log-parameters initialized to `log(0.01)`, so the initial FiLM effect is near-identity (`scale ≈ 1`, `shift ≈ 0`). The `tanh` bounds prevent large-magnitude modulation early in training. The output projection is zero-initialized, making the entire module a no-op at initialization — the network learns when and how to use the environment conditioning from the loss signal.

`norm` above is `RMSNorm` when `edge_norm=True` and identity when `edge_norm=False`. The logits derive from `D = envᵀenv`, which vanishes at the cutoff, so normalizing them shares the radial network's floor-crossing problem (§5.3); the identity path lets the FiLM logits vanish smoothly at `rcut`. The `tanh` bound and the learnable strength keep the modulation well-scaled either way, so the norm is not needed to prevent scale collapse.

### 6.4 Spin Embedding (native spin scheme)

`SpinEmbedding` (in `embedding.py`, enabled when per-type `use_spin` flags are provided) injects a per-atom spin vector `s_i` into the node features as an equivariant extension of the type embedding. It is the descriptor-side mechanism of the native spin scheme (§12.2.1) and is built only for spin models; ordinary energy models never construct it. The spin enters both an atom's **own** features (on-site) and, so that neighbour spins are available before any interaction block, the atom's **initial neighbourhood embedding** (the l=1 backbone seed and the l=0 env-seed FiLM).

**On-site contributions.**

- **`l = 0` (invariant magnitude).** A small network of the squared magnitude `|s_i|²` yields a per-channel scalar added to the scalar type embedding before the edge cache is built. Like the type embedding, it then flows into the per-edge type features `edge_type_feat = type_ebed[src] + type_ebed[dst]` (so a neighbour's `|s_j|²` already reaches the centre through the `src` term) and every block's radial features. The squared magnitude is used rather than `|s_i|` so the feature is `C^∞` at `s = 0`; its gradient there vanishes, keeping the magnetic force continuous as a spin crosses zero.

- **`l = 1` (equivariant direction).** The Cartesian vector `s_i` is mapped to the packed `l = 1` coefficients (orders `m = -1, 0, +1`) and scaled by a per-type per-channel weight, then added to the node backbone after the geometric initial embedding. The Cartesian-to-`l=1` projection is derived from `build_cartesian_basis(1)`, so it intertwines with the same Wigner-D block the descriptor uses: `cart_to_l1(R s) = D¹(R) cart_to_l1(s)`. The atom's own spin is a global-frame vector, so it is written directly into the global packed rows with no Wigner-D rotation (unlike GIE, which rotates an edge-local zonal feature). The map is linear in `s`, so the contribution vanishes at `s = 0` and rotates correctly under SO(3).

**Neighbour-spin aggregation.** A neighbour's spin direction and the neighbourhood's spin–spin correlation are injected into the centre's initial embedding, so spin coupling is available at depth zero rather than only through message passing:

- **`l = 1` neighbour aggregation (`SpinEmbedding.edge_l1` → GIE).** `SpinEmbedding.edge_l1` builds the per-edge packed `l = 1` of the source spin, `cart_to_l1(s_j)`, scaled by a per-source-type per-channel weight and gated by the C³ envelope. The geometric initial embedding then folds this message into its l=1 rows (the first three packed non-scalar rows), so the neighbour spin shares GIE's source gate, scatter and degree normalization with the geometric message — it is the exact spin analogue of GIE rather than a parallel re-implementation, and the bridging source gate now applies to it consistently. It seeds an atom's `l = 1` backbone with `Σ_j s_j`, which then couples with the on-site `s_i` inside the blocks to express `s_i · s_j`-type interactions. It is linear in spin (zero at `s = 0`) and, because `cart_to_l1` intertwines the Wigner-D block, transforms as a proper `l = 1` object. Because it rides on GIE, it is active only when `use_gie` is set (i.e. `use_env_seed` with `node_init_lmax ≥ 1`); the on-site l=1 (above) is added unconditionally.

- **`l = 0` env-seed spin channels (`EnvironmentInitialEmbedding`, per-type `use_spin`).** The neighbour spin is appended to the environment matrix as three extra **coordinate channels**, `env_ij · α · s_j · mask_j` (envelope-gated, masked by source type, with an isotropic learnable scale `α`), so the per-edge coordinate becomes `r̃_ij = [s_ij, s_ij·r̂_ij, env_ij·α·s_j·mask_j]`. The env-matrix invariant `D = Mᵀ M` then automatically carries the neighbour spin–spin invariants `Σ_{j,j'} (s_j · s_{j'})` (with the `j = j'` diagonal giving `Σ |s_j|²`), weighted by the type/radial embedding, which feed the l=0 FiLM. Because spin and geometry rotate under the same SO(3), `D` stays rotation-invariant; the `output_proj` is zero-initialized, so the spin contribution starts neutral. This route is active only when `use_env_seed` is set.

**Invariants of the design.** Every spin route — on-site, the l=1 neighbour aggregation, and the env-seed channel — is gated by the **same per-type spin mask**, applied multiplicatively to the source spin. This is essential, not cosmetic: a non-magnetic atom has no spin degree of freedom, so its magnetic force `-dE/ds` must be **exactly** zero. Relying on the dataset convention `s = 0` for non-magnetic atoms is not enough, because the force is a derivative and the second backward probes the spin direction even where the value is zero; the multiplicative mask makes `∂E/∂s` vanish identically for non-magnetic types on every route. Every spin route is also either linear in `s` (`l = 1` on-site and neighbour) or quadratic via `|s|²` / `MᵀM` (`l = 0`), so all spin features and their spin gradients vanish at `s = 0` — the magnetic force is continuous as a spin crosses zero. All neighbour routes are envelope-gated, so a neighbour's spin (and its derivatives) decays to zero at `rcut`. The per-type `l = 1` weights (on-site and neighbour) carry the `adam_` prefix so HybridMuon routes them to Adam, matching the type-embedding table treatment; the magnitude network's leading `1 → channels` layer carries a singleton input dimension that HybridMuon also routes to Adam.

______________________________________________________________________

## 7. Wigner-D Rotation System

SeZM operates on SO(3)-equivariant features by rotating them between a global frame and edge-aligned local frames. The rotation is realized through real-basis Wigner-D matrices, computed from edge-aligned quaternions by `WignerDCalculator` (in `wignerd.py`).

### 7.1 Edge-Aligned Quaternion

`build_edge_quaternion(edge_vec)` returns a quaternion `q_edge` whose rotation matrix `R(q_edge)` maps the global frame to a local frame where the edge direction aligns with `+Z`:

```
R(q_edge) @ (edge_vec / ‖edge_vec‖) = (0, 0, 1)
```

The quaternion is built from two exact charts:

- **Chart A**: regular away from the `−Z` pole, handling edges that point roughly upward.
- **Chart B**: regular away from the `+Z` pole, handling edges that point roughly downward.

A C^∞ normalized-linear blend (`quaternion_nlerp` with sign alignment for shortest arc) is applied only inside the overlap region of the two charts. This avoids the singularity that any single chart exhibits at one of the poles, producing a smooth edge rotation across all directions.

**Numerical stability.** Vector and quaternion normalizations clamp squared norms before taking the square root — `sqrt(clamp(‖x‖², eps²))` — to prevent NaN gradients at zero or near-zero vectors. For fp16/bf16, the norm computation is promoted to fp32 before the square root.

### 7.2 Random Roll (random_gamma)

When `random_gamma=True` (default), SeZM samples an independent roll angle `γ ~ U[0, 2π)` per edge per forward call, constructs a local `+Z` roll quaternion `q_γ`, and left-composes it with the edge quaternion:

```
q_total = q_γ · q_edge
```

A local `+Z` roll leaves the edge-direction invariant (`(0,0,1)` is unchanged by rotation about `Z`) while randomizing the in-plane gauge. This data augmentation breaks the correlation between the arbitrary choice of local `x/y` axes and the training data, improving generalization without affecting equivariance.

### 7.3 Block-Diagonal Wigner-D

`WignerDCalculator.forward(quaternions)` returns `(D_full, Dt_full)` where `D_full` is block-diagonal with one `(2l+1) × (2l+1)` block per degree `l`:

```
D_full = diag(D⁰, D¹, D², ..., D^lmax)
```

The block for `l = 0` is the `1 × 1` identity. Higher-degree blocks are computed as follows:

| Degree     | Method                                                                                                                                                      |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `l = 1`    | Direct construction from the quaternion-induced 3×3 rotation matrix, permuted and sign-adjusted to match real spherical harmonic ordering `(m = -1, 0, +1)` |
| `l = 2`    | Dedicated degree-4 quaternion tensor contraction kernel                                                                                                     |
| `l = 3, 4` | Dedicated quaternion monomial kernels; when both are needed, they share one degree-8 matrix multiply                                                        |
| `l ≥ 5`    | Generic quaternion polynomial evaluator using precomputed Wigner polynomial coefficient tables                                                              |

All Wigner-D blocks are stored in the real spherical harmonic basis with `m = -l, ..., +l` ordering within each `l` block. The coefficient tables exploit the symmetry relation `D^l_{-m',-m} = (-1)^{m'-m} conj(D^l_{m',m})` to store only half the coefficients.

`Dt_full = D_full^T` is the inverse rotation (local → global), valid because real orthogonal representations satisfy `D^{-1} = D^T`.

### 7.4 Reduced Rotation Projections

Inside SO(2) convolution, only coefficients with `|m| ≤ mmax` are retained. Rather than extracting a subblock at runtime, SeZM precomputes the row/column index sets and caches the projected blocks:

- `project_D_to_m(D_full, coeff_index_m)` → `D_to_m` of shape `(E, D_m, D)` — row subset of `D_full`
- `project_Dt_from_m(Dt_full, coeff_index_m)` → `Dt_from_m` of shape `(E, D, D_m)` — column subset of `Dt_full`

These projections are cached in the `EdgeFeatureCache` keyed by the string `"lmax:mmax"` and reused across blocks that share the same `(lmax, mmax)`.

When `DP_TRITON_INFER >= N` and the model is in eval mode, the SO(2) message, attention aggregation, force assembly, and Wigner-D basis are routed through fused Triton operators (`triton/`). The gate is a cumulative numeric level: `1` universal kernels (block-diagonal rotation, radial degree mixing, `SO2Linear` block GEMM, Wigner-D monomials, attention aggregation, force assembly); `2` the table-configured fused SO(2) value path (`so2_rotate_mix` + `so2_mixing_stack`) and edge-block backward schedules; `3` the fp16x3 split-compensated tensor-core mixing stack on fp64-validated shapes. Levels 1–2 are exact fp32; level 3 perturbs forces at the 2⁻²² rounding scale. Every operator fuses a span of the reference path (folding gathers, block-diagonal Wigner rotations over structural non-zeros only, gate/residual epilogues, and destination-segmented CSR reductions that replace atomic scatters), provides an exact hand-written backward, and falls back to the eager reference for unsupported layouts.

The operators are registered as functional `torch.library.triton_op`s launched through `torch.library.wrap_triton`, so they differentiate under the traced (`make_fx`) force-autograd path and, unlike opaque `custom_op`s, are visible to Inductor and bake into AOTInductor `.pt2` packages — required for LAMMPS, which loads `.pt2` through the C++ `AOTIModelPackageLoader` with no Python operator registration. Shape-dependent launch configurations come from per-GPU tables (`triton/tile_config_data.py`, keyed by `torch.cuda.get_device_name()`) with automatic fallback on untuned GPUs and shapes; the `.pt2` freeze path auto-tunes uncovered keys on the local GPU before tracing and bakes them in.

The complete reference — per-operator algorithms and backwards, the fp16x3 numerical scheme, the launch-table and freeze auto-tuning machinery, the non-obvious failure modes, measured per-kernel profiles, and studies that were rejected — is in [`doc/outisli/triton_op_record.md`](./triton_op_record.md).

### 7.5 Inverse Rescale for Truncated Rotation

When `mmax < lmax`, the rotate-to-local step discards `|m| > mmax` coefficients, and the rotate-back step treats them as zero. For degrees `l > mmax`, this truncation loses energy: the `(2l+1)` full coefficients are reduced to `(2·mmax+1)` retained coefficients. To compensate, the rotate-back output is multiplied by a per-coefficient rescale factor:

```
rescale(l) = sqrt((2l+1) / (2·min(l, mmax)+1))
```

This factor restores the expected norm of each degree block after the truncated round-trip. The rescale vector `rotate_inv_rescale_full` is precomputed as a buffer in `SO2Convolution` and applied element-wise after the rotate-back `bmm`.

______________________________________________________________________

## 8. Interaction Blocks

### 8.1 Block Structure

Each `SeZMInteractionBlock` (in `block.py`) follows a two-path residual architecture operating on node features of shape `(N, D, 1, C)`:

```
SeZMInteractionBlock:
  Path 1 — SO(2) Convolution:
    x_pre = pre_so2_norm(x)                          # EquivariantRMSNorm
    y = so2_conv(x_pre, edge_cache, radial_feat)     # message passing
    y = post_so2_norm(y)                              # standard RMSNorm or unit-floor scaling
    x = x + y

  Path 2 — FFN subblock sequence:
    for i in range(ffn_blocks):
        x_pre = pre_ffn_norm[i](x)                   # EquivariantRMSNorm
        y = ffn[i](x_pre)                            # EquivariantFFN
        y = post_ffn_norm[i](y)                       # optional
        if layer_scale:
            y = y × adam_ffn_layer_scale[i]           # per-channel, init 1e-3
        x = x + y

  return x
```

The `sandwich_norm` config `[so2_pre, so2_post, ffn_pre, ffn_post]` controls which norms are active. The default `[False, True, True, False]` applies post-norm to the SO(2) residual branch and pre-norm to each FFN subblock.

When SO(2) post-norm is active, `edge_norm` selects its variance floor without changing any other sandwich-normalization site. `edge_norm=True` uses the standard denominator `sqrt(variance + eps)`. `edge_norm=False` uses `sqrt(variance + 1)`, so the centered message is unchanged to first order near zero and large residuals remain smoothly bounded. The affine per-degree scale and scalar bias are identical in both modes. The two centered branch maps are related by a constant rescaling of the upstream SO(2) message, so the unit-floor mode does not remove a radial response that the standard mode can represent.

### 8.2 SO(2) Convolution

`SO2Convolution` (in `so2.py`) is the message-passing engine. It operates in the edge-aligned local frame where only `|m| ≤ mmax` coefficients are retained, converting the full SO(3) rotation to a sequence of SO(2) operations that scale linearly in `lmax` instead of cubically.

**Step-by-step flow:**

1. **Pre-focus mixing.** A `ChannelLinear` (n_focus=1) mixes channels on the node tensor `(N, D, C)` before any edge operation.

1. **Gather and rotate to local frame.** Source features are gathered per edge and rotated: `x_local = bmm(D_to_m, x_src)`, producing `(E, D_m, C)` where `D_m = Σ_l (2·min(l, mmax)+1)`.

1. **Radial modulation or dynamic radial degree mixing.** In the default `radial_so2_mode="none"` path, each coefficient is multiplied by its degree-specific radial feature: `x_local *= radial_feat[:, degree_index_m, :]`. The `degree_index_m` buffer maps each position in the m-major layout to its degree `l`.

   When `radial_so2_mode` is enabled, `DynamicRadialDegreeMixer` replaces this diagonal modulation by an edge-conditioned cross-degree kernel in the local SO(2) frame:

   ```
   degree:         z_e[l_out, m, c] = sum_l_in W_e[l_in, l_out, |m|](r) * x_e[l_in, m, c]
   degree_channel: z_e[l_out, m, c] = sum_l_in W_e[l_in, l_out, |m|, c](r) * x_e[l_in, m, c]
   ```

   `degree` uses a kernel shared across channels. `degree_channel` uses a per-channel kernel, optionally low-rank when `radial_so2_rank > 0`. Channel mixing is handled by the following SO2Linear stack. For `|m| > 0`, the same dynamic degree kernel is applied to the `-m` and `+m` signed blocks, preserving SO(2) equivariance.

1. **Optional node-wise grid net** (`node_wise_s2=True` or `node_wise_so3=True`). Destination features are rotated into the same edge-local frame as the source features. The source side is the query branch and the destination side is the context branch. The scalar path uses a SwiGLU product of the two `l=0` slices, while a separate scalar gate is generated from their concatenation. The branch projects both operands to the selected grid (S2 or Wigner-D SO3), applies the grid operation, projects back, applies the scalar gate, merges the scalar branch, and adds the result as a small residual before the SO2Linear stack:

   ```
   scalar = silu(radial_fused_src_l0) * dst_l0
   gate = sigmoid(linear(concat(radial_fused_src_l0, dst_l0)))
   cross = gate * from_grid(grid_op(to_grid(radial_fused_src), to_grid(dst_local)))
   cross_l0 += scalar
   x_local = radial_fused_src + adam_node_wise_s2_scale * cross
   ```

   With `node_wise_grid_branch > 0`, `grid_op` is scalar-routed polynomial branch mixing: router weights come from the concatenated `l=0` scalar source/destination features, while each branch is a quadratic product between channel-mixed source and destination grid fields. With `node_wise_grid_mlp=True` instead, the source and destination grid fields are combined by a polynomial point-wise MLP. The integer value of `node_wise_grid_branch` is the number of polynomial product branches. With `node_wise_so3=True`, an internal per-degree frame assignment loads channels into the configured SO3 frame set before projection, and an internal per-degree frame contraction maps the Wigner-D coefficients back to ordinary `(l,m,c)` features. These maps do not mix `m`. The source input is already radial/envelope fused, so this branch also vanishes smoothly at `rcut`. The destination input is not independently radial-fused, avoiding an unintended squared radial envelope.

1. **Reshape to multi-focus layout.** `x_local` is reshaped to `(E, F, D_m, Cf)` where `F = n_focus` and `Cf = focus_dim` (or `channels` when `focus_dim = 0`). The hidden width is `H = F × Cf`.

1. **Multi-layer SO(2) stack.** For each of `so2_layers` iterations:

   - Save residual: `residual = x_local`
   - Pre-norm (when `so2_norm=True`): `ReducedEquivariantRMSNorm` on the reduced layout. Identity for the last layer.
   - `SO2Linear`: block-diagonal matmul (see §8.3)
   - Bias correction on layer 0 only: the l=0 bias is modulated by `radial_l0 × edge_env − 1` to keep the bias consistent with the radial/envelope scaling.
   - Nonlinearity (between layers, Identity for last):
     - Default: `GatedActivation` — scalar activation on `l=0`, sigmoid gate from `l=0` applied to `l>0`
   - S2 path (`s2_activation[0]=True`): `S2GridNet` — scalar SwiGLU branch + sigmoid gate + S2-grid point-wise multiplication
   - LayerScale + residual: `x_local = residual + scale × x_local`
   - Optional SO(2)-internal `DepthAttnRes` from local layer history

1. **Cross-focus competition** (when `n_focus > 1`). A softmax over focus streams is computed from `l=0` scalars:

   ```
   logits = focus_compete_proj(ScalarRMSNorm(x_local_l0))
   alpha = softmax(logits / tau, dim=focus)
   alpha = (1 − eps) × alpha + eps / n_focus     # label smoothing
   x_local *= alpha[:, :, None, None]
   ```

   Label smoothing (default `eps = 0.02`) prevents dead focuses.

1. **Rotate back.** `x_message = bmm(Dt_from_m, x_local) × rotate_inv_rescale_full`, producing `(E, D, C)`.

1. **Aggregate to destination nodes.**

   - **No attention** (`n_atten_head = 0`): multiply by `edge_env`, scatter-sum by `dst`, multiply by `inv_sqrt_deg`.
   - **With attention** (`n_atten_head > 0`): see §8.4.

1. **Optional message-node grid net** (`message_node_s2=True` or `message_node_so3=True`). After aggregation, the hidden message `M_i` and the destination node hidden state `x_i` are already in the same packed SO(3) basis. The message is the query branch and the node state is the context branch. The branch applies the same scalar/grid net on node tensors, then adds a small residual before the output projection:

   ```
   scalar = silu(M_i,l=0) * x_i,l=0
   gate = sigmoid(linear(concat(M_i,l=0, x_i,l=0)))
   cross = gate * from_grid(grid_op(to_grid(M_i), to_grid(x_i)))
   cross_l0 += scalar
   M_i = M_i + adam_node_wise_s2_scale * cross
   ```

   Since `M_i` is a sum of envelope-gated edge messages, an edge contribution and its radial derivative vanish at `rcut` before this nonlinear node-level product is evaluated. The product therefore does not create cutoff-outside edge contributions; it only transforms the already aggregated message.

1. **Post-focus mixing.** A `ChannelLinear` mixes the full `C` channels back.

### 8.3 SO2Linear

`SO2Linear` (in `so2.py`) implements the core SO(2)-equivariant linear map. It operates on the m-major reduced layout and applies a single block-diagonal matmul that processes all `|m|` groups simultaneously:

- **`m = 0` block**: An unconstrained linear map over all `l = 0..lmax` coefficients (plus optional additive bias).
- **`|m| > 0` blocks**: A constrained 2×2 complex coupling on each `(-m, +m)` pair, treated as `(Re, Im)`:
  ```
  [out_neg_m]   [W_u^T  -W_v^T] [in_neg_m]
  [out_pos_m] = [W_v^T   W_u^T] [in_pos_m]
  ```
  This is the real-number realization of a complex linear map, preserving SO(2) equivariance.

The `|m| > 0` weights are initialized with an extra `1/sqrt(2)` scaling factor to preserve the coupling energy between the real and imaginary parts, compensating for the doubled parameter count relative to the `m = 0` block.

In eval mode with `torch.no_grad()`, SO2Linear caches the assembled block-diagonal weight matrix (`_cached_weight`) to avoid reassembling it every call. The cache is invalidated when gradients are needed.

### 8.4 Attention Aggregation

When `n_atten_head > 0`, SO(2) convolution uses an envelope-gated grouped softmax (`segment_envelope_gated_softmax` in `attention.py`) instead of plain envelope-weighted scatter.

**Logit computation:**

```
q = attn_q_proj(ScalarRMSNorm(x_l0[dst]))     # (E, F, H, Dh)
k = attn_k_proj(ScalarRMSNorm(x_l0[src]))     # (E, F, H, Dh)
logits = dot(q, k) / sqrt(head_dim) + attn_radial_logit_proj(radial_l0)
```

By default, attention is independent for each SO(2) focus stream. When `atten_f_mix=true`, all focus streams are viewed as one attention stream after rotate-back: `F_attn=1` and `C_attn=n_focus * focus_dim`. A degree-aware `SO3Linear` first mixes the full hidden width after rotate-back, then the same attention code path is used. Each head splits the mixed multi-focus hidden width instead of a single focus stream.

**Destination-wise softmax with envelope gating:**

```
edge_log_mass = logits + 2 × log(edge_env) + log(src_weight) + log(edge_mask)
null_log_mass = log(softplus(z_bias) + eps)
grouped_max = max(null_log_mass, scatter_max(edge_log_mass, dst))
edge_mass = exp(edge_log_mass − grouped_max[dst])
null_mass = exp(null_log_mass − grouped_max)
alpha = edge_mass / (null_mass + scatter_sum(edge_mass, dst))[dst]
```

The physical denominator is `softplus(z_bias) + eps + Σ edge_env² × src_weight × exp(logit)`. The factor logarithms are evaluated before multiplication so squaring a small envelope cannot underflow first. For the optional SFPG source gate, a stop-gradient copy supplies the log-domain magnitude while the identity ratio `src_weight / detached(src_weight)` carries derivatives outside the logarithm; this preserves the forward mass and its linear source-gate Hessian without forming `1 / src_weight²`. `z_bias` is learnable and initialized to `softplus⁻¹(1)`; adding `eps` gives the null mass a finite lower bound. The null log-mass participates directly in the segment maximum, so every denominator contains a shifted term equal to one even for empty or fully masked segments. Zero factors use a safe log input and then set the effective edge log-mass to `−∞`, yielding exact zero output and finite zero gradients.

**Output-side head gate:** After scatter-summing the attention-weighted messages, an output gate is applied per head. Optional value and output projections can be enabled independently with `atten_v_proj` and `atten_o_proj`:

```
value = attn_v_proj(message)                       # only when atten_v_proj=true
output = scatter_sum(alpha × value, dst)
gate = sigmoid(attn_output_gate_proj(ScalarRMSNorm(x_l0)))
output = output × gate
output = attn_o_proj(output)                       # only when atten_o_proj=true
```

This query-dependent gate allows the model to suppress or amplify each attention head based on the destination atom's scalar features. The optional projections are degree-aware, per-attention-focus `SO3Linear` maps: each degree `l` has its own channel projection, while all `m` components within the same degree share the same weights. The defaults keep the legacy value and output path without these projection parameters.

When attention is active, `inv_sqrt_deg` is not applied — the softmax normalization replaces degree-based normalization.

### 8.5 Equivariant FFN

`EquivariantFFN` (in `ffn.py`) provides per-node channel mixing with degree-aware gating. Three variants exist:

**Standard gated path** (default):

```
h = SO3Linear(x)              # (N, D, C) → (N, D, hidden)
h[l=0] = activation(h[l=0])   # scalar activation (SiLU)
gate = sigmoid(gate_proj(h[l=0]))   # per-l independent gates
h[l>0] = h[l>0] × gate              # gate higher-degree features
out = SO3Linear(h)             # (N, D, hidden) → (N, D, C), zero-init
x = x + out                   # residual
```

Each degree `l` has an independent gate derived from the scalar `l = 0` features. The gate is expanded via `expand_index` to all `2l+1` m-components within that degree.

**Grid-net path** (`s2_activation[1]=True` or `ffn_so3_grid=True`):

The first `SO3Linear` prepares two grid operands. In the S2 FFN grid path this is `2 × hidden` channels. When `ffn_so3_grid=True`, the same FFN grid operation is evaluated with the SO3 Wigner-D projector and the first linear emits `2 × (2*kmax+1) × hidden` channels, where the extra factor loads channels into the Wigner-D frame set `[0, -1, 1, ..., -kmax, kmax]`. The grid net then:

1. Extracts the `l = 0` slice to build a scalar SwiGLU branch and a sigmoid gate.
1. Projects coefficients to the selected grid.
1. Applies a quadratic grid operation.
1. Projects grid features back to coefficients.
1. Applies the scalar gate and merges the scalar branch back to `l = 0`.

The projector is selected as follows:

- **S2 grid**: used by the S2 FFN grid path when `s2_activation[1]=True` and `ffn_so3_grid=False`.
- **SO3 Wigner-D grid** (`ffn_so3_grid=True`): uses matrix elements `D^l_{m,k}` with `|k| <= kmax` (default `kmax=1`). This option takes precedence over the FFN grid path and ignores `s2_activation[1]`. The second `SO3Linear` contracts the frame-expanded channels back to the ordinary channel width; its weights are still per-degree and do not mix `m`.

Each grid path selects its operation independently. `grid_mlp` and `grid_branch` are length-three lists `[node_wise, message_node, ffn]` (a scalar broadcasts to all three): the `node_wise` and `message_node` entries control the SO(2) convolution cross-grid paths, and the `ffn` entry controls the block-internal FFN grid path. On each path the operation is selected by priority:

- **Quadratic tensor product** (default grid op): multiply the two projected grid fields point-wise, `h(g) = u(g) * v(g)`. This is the direct replacement for the previous S2 activation.

- **Polynomial point-wise MLP** (the path's `grid_mlp` entry is `True`): channel-project the two grid fields, multiply them point-wise, and project back to grid channels. In FFN self mode both projections see the concatenated fields; in cross grid nets the query and context sides are projected separately:

  ```
  # FFN self mode
  z(g) = concat(u(g), v(g))
  h(g) = W_out((W_a z(g)) * (W_b z(g)))
  # cross grid net
  h(g) = W_out((W_a u(g)) * (W_b v(g)))
  ```

- **Scalar-routed polynomial branch mixer** (the path's `grid_branch` entry is `> 0`): it computes router weights from the `l=0, k=0` scalar branch only, while each branch is a quadratic product of channel-mixed grid fields. The integer value of that entry is the number of polynomial product branches. In FFN self mode both operands come from the same node field after the first `SO3Linear`; in cross grid nets the message/source side is the query branch and the node/destination side is the context branch:

  ```
  alpha_r = softmax_r(W_s scalar_pair)
  value_r(g) = (W_a^r u(g)) * (W_b^r v(g))
  h(g) = sum_r alpha_r value_r(g)
  ```

  On each path, `grid_branch` takes precedence over `grid_mlp`.

On the selected grid, the field values are rotational scalars attached to grid arguments. Therefore the grid operation can mix channels freely and does not need per-degree handling. The non-scalar path uses only channel mixing and quadratic products; the sigmoid/SwiGLU/softmax nonlinearities are driven by the invariant scalar branch. The only per-degree equivariant maps are the surrounding coefficient-space `SO3Linear` projections and, for SO3 grid nets, the frame assignment/contraction.

The S2 projector has two quadrature backends:

- **Tensor-product sphere grid**: the SO(2) path uses the automatic grid `[2*mmax + 4, ceil_even(3*lmax + 2)]`. In the FFN path this is lifted to a square grid `[max(R_phi, R_theta), max(R_phi, R_theta)]`, matching the EquiformerV3 sampling rule.
- **Lebedev quadrature** (`lebedev_quadrature=true` or `[so2, ffn]`, default): uses packaged Lebedev rules stored as Cartesian unit points and normalized weights. A scalar bool is broadcast to both S2 branches, while the list form controls SO(2) and FFN separately. The Burkardt source files use physics convention (`theta` is azimuth, `phi` is polar); the packaged `.npz` stores only Cartesian points to avoid runtime convention ambiguity. SeZM selects the smallest packaged rule with algebraic precision at least `3*lmax`, which is sufficient for the low-degree projection of the bilinear S2 product.

The SO3 projector uses a Lebedev sphere grid times equiangular samples in the third Euler angle. The gamma count is resolved internally from `kmax`: one sample is enough for `kmax=0`, otherwise SeZM uses `3*kmax+1` samples. This resolves the `k1 + k2 - kout` frame-frequency range introduced by quadratic grid products. `kmax=1` is the low-cost default that opens odd/antisymmetric couplings such as `1⊗1→1`; larger `kmax` increases coverage at higher cost.

In all variants, the output projection is zero-initialized (weights and bias), so the FFN starts as a no-op and the residual path begins near-identity.

**Descriptor read-out.** After the interaction blocks, a stack of `readout_layers` `EquivariantFFN` blocks (default 1) maps the node tensor to the invariant `l = 0` descriptor. The `so3_readout` switch selects each block's form:

- `so3_readout="none"` (default): a scalar read-out. Only the `l = 0` slice enters a degree-0 FFN (`lmax=0`), so `l > 0` coefficients are discarded before the read-out.
- `so3_readout="glu"` / `"mlp"`: a full FFN whose degree equals the read-out node degree (`node_readout_lmax` — the last interaction block's node degree, or `lmax + extra_node_l` for the zero-block descriptor), driven by the SO3 Wigner-D grid (`ffn_so3_grid=True` internally, frame order `min(kmax, node_readout_lmax)`). It consumes the complete node tensor `(N, D, 1, C)`, so the quadratic grid product (`glu`) or the polynomial point-wise grid MLP (`mlp`) folds `l > 0` geometry into `l = 0` before the scalar is extracted.

Each block forms the residual `x + FFN(x)` on the full coefficient tensor. With `readout_layers > 1` the intermediate blocks return the full SO(3) tensor so high-degree geometry keeps folding into `l = 0` across layers; only the final block slices the `l = 0` channel from its residual sum. Slicing the summed tensor rather than the FFN output directly keeps the saved degree-axis stride static under `torch.compile` dynamic shapes — slicing the SO(3)-grid FFN output saves its degree-first complement with a stride that scales with the dynamic node count, which Inductor specializes to a concrete value and then rejects when the node count changes. The zero-initialized output projection keeps every read-out block near-identity at initialization. This lets a model use cheaper S2 activations inside the blocks (less overfitting) while still mixing high-degree information into the descriptor at the read-out, where only `l = 0` survives.

### 8.6 Equivariant RMS Normalization

`EquivariantRMSNorm` (in `norm.py`) normalizes SO(3)-equivariant features across all degrees simultaneously, with degree balancing to prevent high-multiplicity degrees from dominating the norm.

**Degree balancing.** Each coefficient from degree `l` is weighted by `1 / ((2l+1) × (lmax+1) × C)` in the RMS computation. This ensures that every degree contributes equally to the norm regardless of its multiplicity `(2l+1)`. Without this balancing, degree `l = 3` (with 7 coefficients) would contribute 7× more to the norm than `l = 0` (with 1 coefficient), causing the scalar features to be suppressed.

**Scalar centering.** Before computing the RMS, the `l = 0` slice is mean-centered across channels. This allows the normalization to be invariant to a uniform shift in the scalar features while preserving the zero-mean property of `l > 0` features (which are already zero-mean by equivariance).

**Per-degree affine.** After normalization, a learnable per-degree scale (`adam_scale`) is applied. The scale is expanded to all coefficients via a precomputed `expand_index`. Additive bias is applied only to the `l = 0` slice.

`ReducedEquivariantRMSNorm` is the variant for the m-major truncated layout inside SO(2) convolution, using `n_coeff_l = 2·min(l, mmax)+1` instead of `2l+1` for the degree weights.

Both norms disable autocast (`@torch.amp.autocast("cuda", enabled=False)`) to ensure the RMS is always computed in full precision.

### 8.7 Pyramid Schedules

**L-schedule.** SeZM supports a non-increasing sequence of message-passing `lmax` values across blocks: `l_schedule = [3, 3, 2, 1]`. The SO(2) branch in block `i` uses only the packed prefix up to `l_schedule[i]`. When the schedule decreases between blocks, node features are sliced to the corresponding node schedule described below. The final block does not need to end at `l = 0` — the output always extracts only the `l = 0` features regardless of the final block's MP degree.

**Extra node degree.** `extra_node_l` keeps additional node-only degrees around each block:

```
node_l_schedule[i] = l_schedule[i] + extra_node_l
```

The SO(2) branch still uses `l_schedule[i]`, so edge-local rotation, reduced SO(2) mixing, and message aggregation keep their original cost. The block pads the SO(2) update back into the node tensor, leaving `l > l_schedule[i]` unchanged by message passing. The FFN and its S2 activation operate on `node_l_schedule[i]`, allowing local tensor-product mixing between the MP degrees and the extra node degrees.

**M-schedule.** An independent sequence controls `mmax` per block: `m_schedule = [2, 2, 1, 0]`. Each entry must satisfy `m_schedule[i] ≤ l_schedule[i]`. Unlike `l_schedule`, the m-schedule does not change the global node tensor shape `(N, D, 1, C)`. It only affects the edge-local coefficient set retained during SO(2) operations and the corresponding rotation projections.

When both schedules are `None`, SeZM uses constant `lmax` and `mmax = lmax` across all `n_blocks` blocks. `extra_node_l = 0` exactly recovers the baseline node shape.

**Canonical backbone degrees.** The descriptor references three degrees that must remain valid for any block count, resolved once at construction rather than indexed off the per-block schedules:

- `mp_init_lmax` — the message-passing degree at initialization, driving the Wigner-D calculator and the GIE message-passing coupling rows.
- `node_init_lmax = lmax + extra_node_l` — the node backbone degree at initialization, driving the radial-embedding width, the initial state dimension, and the GIE.
- `node_readout_lmax` — the node backbone degree fed to the read-out FFN.

With blocks these equal `l_schedule[0]`, `node_l_schedule[0]` and `node_l_schedule[-1]`; with zero blocks all three collapse onto `lmax` (plus `extra_node_l` on the node side).

**Zero-block descriptor (`n_blocks = 0`).** Setting `n_blocks = 0` (equivalently `l_schedule = []`) builds the interaction-free descriptor. No `SeZMInteractionBlock` is constructed and the block loop is a no-op, so the forward path reduces to

```
type embedding (+ optional charge/spin)
  → optional env FiLM scalar conditioning (l = 0)
  → optional GIE geometric seed (l ≥ 1)
  → SO(3) read-out FFN → l = 0 scalar descriptor
```

This is a learnable NEP/SOAP-like invariant descriptor: the GIE projects the local neighbour density onto real spherical harmonics per degree, and the `so3_readout="glu"`/`"mlp"` FFN learns the invariant folding from `l > 0` back to `l = 0`, replacing the hand-written fixed many-body contractions of classical descriptors. It is substantially cheaper than the full model — no SO(2) convolution, attention, block FFN, or per-edge high-degree aggregation — at the cost of expressivity.

Because the interaction blocks are the only message-passing stage, in the zero-block descriptor geometry enters **only** through the GIE. The GIE is gated by `use_env_seed`, so a meaningful zero-block descriptor requires `use_env_seed = True` together with `lmax + extra_node_l > 0`; otherwise the output depends solely on the atom types and is coordinate-independent (zero forces). With `so3_readout="none"` the `l ≥ 1` GIE channels are discarded and only the env-seed scalar conditioning survives, so `"glu"`/`"mlp"` is the recommended read-out to retain the geometric information.

### 8.8 SO(3) Linear Layers

Three linear layer variants handle different dimensionality requirements:

- **`ChannelLinear`**: A shared linear map on the last (channel) axis. Weight shape `(C_in, C_out)`, contraction via `einsum("...i,io->...o")`. Used for pre/post-focus mixing and scalar projections.

- **`FocusLinear`**: Per-focus linear mixing. Weight shape `(C_in, F×C_out)` with the focus dimension folded on the output side. Runtime view `(C_in, F, C_out)`, contraction `einsum("bfi,ifo->bfo")`. Used for gate projections in `GatedActivation`.

- **`SO3Linear`**: Per-degree independent mixing. Weight shape `(lmax+1, C_in, F×C_out)`, expanded to `(D, C_in, F, C_out)` via `index_select` on a precomputed `expand_index` buffer. Contraction `einsum("ndfi,difo->ndfo")`. Bias is applied only to `l = 0`. This is the workhorse linear layer for the FFN and SO(2) channel mixers.

All three store weights in `(in, out)` layout (rows = fan_in, cols = fan_out) to match the Muon optimizer's rectangular correction scaling.

### 8.9 Cartesian Tensor Products (edge_cartesian / node_cartesian)

The Cartesian rank-2 tensor product is an SO(3)-equivariant cross-degree mixer for message-passing degree `1` or `2`, offered in two orthogonal placements selected by independent flags. Both share one scaffold in `cartesian.py` (`_CartesianTensorProduct`: a stack of `mixing_layers` layers, each `SO3Linear` per-degree channel mixing → `3×3` product → gated activation → residual) and differ only in the right operand of the product:

- **`edge_cartesian`** (per edge, before aggregation) replaces the rotate-to-local / `SO2Linear` stack / rotate-back core of the SO(2) convolution. For every block with degree `1` or `2` the convolution dispatches to `_cartesian_message` instead of `_so2_message`; blocks with degree `0` or `≥ 3` keep the SO(2) path. Its cost scales with the number of edges.
- **`node_cartesian`** (per node, after aggregation) applies the product to the aggregated message, after the optional message-node grid product and before `post_focus_mix`. It leaves the per-edge message path (SO(2) or `edge_cartesian`) untouched, so its cost scales with the number of nodes.

The flags are independent: either, both, or neither may be set. The surrounding pipeline (`pre_focus_mix`, the radial edge condition, cross-focus competition, attention aggregation, `post_focus_mix`) is shared unchanged.

**Mixing depth (`mixing_layers`).** `mixing_layers` (legacy alias: `so2_layers`) is the number of learnable mixing layers in the per-edge message core; `0` keeps only the edge-condition modulation and drops the learnable channel-mixing stack. The edge-aligned rotation of the SO(2) path is only built when some operation lives in the local m-major frame — `mixing_layers > 0`, a `radial_so2_mode` degree mixer, or a node-wise grid product — tracked by `_needs_local_frame`. When none of these hold, `_so2_message` is replaced by the rotation-free `_radial_message`: per-degree scalar radial scaling commutes with rotation, so the message reduces to a source gather, an elementwise per-degree scale, the cross-focus competition, and the aggregation. The fastest configuration is therefore `mixing_layers = 0`, `radial_so2_mode = "none"` (a rotation-free radial message) combined with `node_cartesian` for the cross-degree mixing.

**Cartesian isomorphism.** For `lmax ≤ 2` the per-channel spherical-harmonic feature `(l = 0, 1, 2)` is isomorphic to a rank-2 Cartesian tensor (a `3×3` matrix) that decomposes into a scalar (trace), a vector (antisymmetric part), and a symmetric-traceless tensor. `build_cartesian_basis(lmax)` returns an orthonormal `(D, 3, 3)` basis aligned with the packed `(l, m)` layout, so the change of basis is a transpose-free exact round trip. Its signs and ordering within each degree are fixed by the `WignerDCalculator` convention (`l = 1` follows the `l1_perm`/`l1_sign` permutation), so the map intertwines the packed Wigner-D rotation with the Cartesian rotation `X → R X Rᵀ`. The per-degree overall scale is immaterial because the learnable layers absorb it. Either matrix product is SO(3)-equivariant because `(R X Rᵀ)(R Y Rᵀ) = R (X Y) Rᵀ`. The edge path uses a one-sided product `Y @ T_e`, which suffices for SO(3) and enables the channel-shared factorization below. The node path selects the form by its `node_cartesian` mode: `default` uses the one-sided product `Y N`, while `parity` uses the symmetrized product `Y N + N Y`, which gives each irreducible part a definite parity under spatial inversion (even scalar and symmetric-traceless parts, odd skew-symmetric part) at the cost of a second matmul that the per-channel node operation absorbs cheaply.

**Edge product (`EdgeCartesianTensorProduct`).** The right operand is the edge tensor `T_e = f_iso · I + f_aniso · A(r̂) + f_sym · S(r̂)`, where `A(r̂) = skew(r̂)` and `S(r̂) = r̂ r̂ᵀ − I/3` (`f_sym` is dropped for `lmax = 1`). The per-degree radial weights `f_*` come from the same radial features as the SO(2) path (after `radial_hidden_proj`), so the edge condition is preserved; the dynamic radial degree mixer is bypassed because the radial dependence is carried by `T_e`. Because `T_e` depends only on the edge direction it is shared across channels, so the product is computed without materializing a `3×3` matrix per channel: `y → from_cart(to_cart(y) @ T_e)` is linear in the packed coefficient and splits, by linearity of `T_e`, into `m = (f_iso/√3) y + f_aniso (K_A y) + f_sym (K_S y)`, where `K_A`, `K_S` are `(D, D)` packed operators for right-multiplication by `A(r̂)`, `S(r̂)`. Each edge builds them once via the fixed projection `W[p, d, k, j] = Σᵢ B[p, i, j] B[d, i, k]` (`K_G = Σₖⱼ W[·,·,k,j] G[k,j]`), after which the per-layer geometry is a single channel-batched `bmm` (the `A`/`S` operators are stacked so one matmul covers both, `S` dropped for `lmax = 1`); the identity component reduces to the scalar `f_iso/√3` rescaling because `B` is orthonormal. This removes the two per-channel basis changes per layer that a literal round trip would run — memory-bandwidth and launch bound in eager mode, and fully fused by Inductor under `use_compile`. With `mixing_layers = 0` the message is the single modulation `x @ T_e` (the same channel-shared `bmm`, no learnable channel-mixing layer); `mixing_layers > 0` refines it with the residual stack. The exact equivalence to the dense `Y @ T_e` is asserted in `test_edge_cartesian_matches_dense_reference`.

**Node product (`NodeCartesianTensorProduct`).** The Cartesian counterpart of the `message_node` grid product: it couples the aggregated message with the destination node feature `x`. The node feature is the fixed operator (the per-node counterpart of the edge tensor `T_e`) and the message is the residual stream, so each layer forms the product of `linear(message)` with `x` lifted by the orthonormal basis. It is configured by `node_cartesian = "<mode>:<layers>"`: `mode` is `default` (one-sided `linear(message) @ x`) or `parity` (symmetrized `linear(message) @ x + x @ linear(message)`), and `layers` is the stack depth (independent of `mixing_layers`); a bare integer `N` is shorthand for `default:N`, and `"none"` (or `0`) disables it. Both operands are per-channel features, so this is the literal per-`(node, channel)` `3×3` product; the channel-shared factorization does not apply here because the node operand is not direction-only. There is no per-edge geometry, so the cost scales with the number of nodes — the regime where the Cartesian form is cheaper than the per-edge SO(2) rotation. It composes with `message_node_s2/so3` (both are post-aggregation node products and may be enabled together).

**Wigner-D elision.** `_need_full_wigner` is computed once in `SeZM.__init__` and is `False` exactly when every block takes the `edge_cartesian` path. In that case `build_edge_cache` skips the full `(E, D, D)` Wigner-D construction (`D_full = Dt_full = None`); the geometric initial embedding reconstructs its zonal coupling from the edge quaternion via `WignerDCalculator.forward_zonal` (the m=0-only path), so no full Wigner-D block is ever materialized. `node_cartesian` does not affect this, since it leaves the per-edge message path unchanged.

**Constraints.** Both paths require `lmax ∈ {1, 2}`. `edge_cartesian` is additionally incompatible with the S2/SO(3) grid product branches (`node_wise_s2/so3`, `message_node_s2/so3`), which would otherwise reintroduce grid rotations into the replaced core; enabling them together raises a `ValueError`. `node_cartesian` carries no such restriction and composes with the grid branches.

**Relation to `radial_so2_mode`.** The Cartesian path keeps edge-direction-conditioned cross-degree mixing: in `edge_cartesian` the edge direction enters through `T_e`'s `A(r̂)`/`S(r̂)` blocks and the matrix product couples the degrees. What it does not replicate is the `radial_so2_mode` cross-degree kernel (`DynamicRadialDegreeMixer`), which weights each `l' → l` coupling path with a separate radial-conditioned, per-channel weight `K_{l,l',|m|}(ρ̃)`. That kernel lives in the rotated local m-major frame, where mixing degrees of equal `|m|` is equivariant; it has no equivariant action on global-frame Cartesian tensors, so it is bypassed on the `edge_cartesian` path. There the radial condition reduces to the three per-channel scalars `f_*` that gate the `I`/`A`/`S` blocks of `T_e`. This is a deliberate accuracy-for-speed exchange.

______________________________________________________________________

## 9. Depth Attention Residuals

SeZM provides three independent attention-residual mechanisms that allow later computation stages to selectively attend to earlier representations. All three are implemented by `DepthAttnRes` (in `attn_res.py`) with the same core algorithm but different scopes.

### 9.1 Core Mechanism

`DepthAttnRes` computes a weighted average over a list of source tensors, using softmax attention on `l = 0` scalar features:

```
keys = [ScalarRMSNorm(scalar_extractor(source)) for source in sources]
query = learned_pseudo_query           # "independent" mode
      | ChannelLinear(scalar_extractor(current_x))   # "dependent" mode
logits = Σ_c query_c × key_c          # dot product per source
alpha = softmax(logits)                # over sources
output = Σ_s alpha_s × sources[s]     # weighted sum, full equivariant tensor
```

The query projection is zero-initialized (`init_std=0`), so initial attention weights are uniform across all sources — equivalent to a simple average. The model learns to specialize the attention as training progresses.

When only one source is available, `DepthAttnRes` short-circuits and returns it directly.

### 9.2 Three Attention Scopes

**`so2_attn_res`** — inside each SO(2) convolution. The SO(2) layer stack maintains a local history of intermediate states. Before each SO(2) layer, attention over this local history provides a skip-connection mechanism within the convolution itself. This helps gradient flow through deep SO(2) stacks (`so2_layers` > 2).

**`full_attn_res`** — across the entire descriptor. A global `unit_history` list accumulates every intermediate representation: the initial `x0`, each SO(2) output, and each FFN output across all blocks. Before the SO(2) unit and each FFN unit, attention over this growing history provides long-range skip connections. A final attention aggregation runs before the output FFN.

**`block_attn_res`** — across blocks. A `block_history` list accumulates one summary per block (the sum of all unit outputs within that block). Each FFN unit also attends to `block_history + [partial_block]`, where `partial_block` is the running sum of unit outputs inside the current block.

`full_attn_res` and `block_attn_res` are mutually exclusive. Both accept modes `"none"` (disabled), `"independent"` (learned pseudo-query), and `"dependent"` (query derived from current state).

______________________________________________________________________

## 10. ZBL Zone Bridging

SeZM supplements its learned energy with an analytical short-range repulsion (ZBL potential) that protects MD simulations from unphysical close contacts. The total energy decomposes as:

```
E_total = E_ZBL(r) + E_model(r̃)
```

where `r` is the true pairwise distance and `r̃` is the effective distance seen by the descriptor after the bridging pipeline. The goal is to make the two-body repulsive wall indistinguishable from a pure ZBL potential inside the bridging window.

### 10.1 The Invariance Requirement

For any frozen pair `(j, k)` with `r_{jk} < r_inner`, the model contribution `E_model` must be constant under all motions that keep `(j, k)` in the frozen zone. Only then does differentiating `E_total` reproduce the analytical ZBL force without a parasitic residual from the learned model.

A scalar-distance clamp alone is insufficient. Two channels still leak trajectory-dependence into the descriptor:

1. **Direction channel.** Even with clamped `‖r_j − r_k‖`, the unit direction `r̂_{jk}` still rotates freely. The Wigner-D operator inherits this rotation, and any `l > 0` path picks up angular dependence.

1. **Multi-hop channel.** A third atom `ℓ` connected to both `j` and `k` through unclamped edges acquires frozen-pair information after one message-passing layer. After a second layer, this information propagates back to `j`, making `x_j` depend on the frozen-pair geometry.

### 10.2 InnerClamp

`InnerClamp` (in `radial.py`) maps true distances to effective distances via a C³-continuous septic Hermite interpolant:

```
r̃(r) = r_inner                                             if r ≤ r_inner
      = r_inner + (r_outer − r_inner) × h_clamp(t)         if r_inner < r < r_outer
      = r                                                   if r ≥ r_outer

t = (r − r_inner) / (r_outer − r_inner)
h_clamp(t) = 20t⁴ − 45t⁵ + 36t⁶ − 10t⁷
```

Boundary conditions: `h_clamp(0) = 0`, `h_clamp(1) = 1`, `h_clamp'(0) = h_clamp'(1) = 0`, `h_clamp''(0) = h_clamp''(1) = h_clamp'''(0) = h_clamp'''(1) = 0`.

Both the scalar distance and the displacement vector are clamped. The vector is rescaled to preserve direction while matching the clamped length:

```
scale = r̃ / max(r, eps)
diff = diff × scale
```

This closes the scalar-distance channel: all downstream quantities derived from `length` (radial basis, cutoff envelope, edge type features) see constant geometry for any frozen-zone displacement.

### 10.3 Source Freeze Propagation Gate (SFPG)

SFPG closes both the direction and multi-hop channels. `BridgingSwitch` (in `radial.py`) produces a per-edge C³ switching amplitude:

```
w(r) = 0                                                   if r ≤ r_inner
w(r) = h_switch((r − r_inner) / (r_outer − r_inner))       if r_inner < r < r_outer
w(r) = 1                                                   if r ≥ r_outer

h_switch(t) = 35t⁴ − 84t⁵ + 70t⁶ − 20t⁷
```

The per-node gate is the product of switch values over all outgoing edges:

```
η_j = Π_{e ∈ E(j)} w(r_e)
```

This product is computed with `torch.scatter_reduce(..., reduce="prod", include_self=True)` directly on non-negative reals. No `log`/`exp` detour is needed, so `η_j = 0` is exact when any neighbor enters the frozen zone. Padded/excluded edges contribute the multiplicative identity `w = 1`.

The per-edge broadcast `edge_src_gate[e] = η_{src(e)}` is applied at every aggregation site:

1. **GeometricInitialEmbedding**: zonal message × `edge_src_gate` before `index_add_`
1. **EnvironmentInitialEmbedding**: outer product × `edge_src_gate` before scatter
1. **SO2Convolution (plain path)**: `edge_env × edge_src_gate` in the scatter
1. **SO2Convolution (attention path)**: `edge_src_gate` enters the softmax as `src_weight`, multiplying `edge_env²` in both numerator and denominator. A muted source contributes zero to the normalization, preventing the denominator from "seeing" frozen-zone edges.

**Equivariance.** `η_j` is a product of functions of pairwise distances (SO(3) scalars). Multiplying an equivariant message by this scalar preserves the transformation behavior.

**Smoothness.** `w` is C³. The product `η_j` is C³ in coordinates. `scatter_reduce("prod")` uses the leave-one-out product rule in its backward, which is exact for non-negative inputs including zero.

**Freezing correctness.** By induction on layer index: if `η_j = 0` for frozen nodes `j, k`, then no non-frozen atom `i` can receive frozen-pair information at any layer, so `E_i^{GNN}` is constant under frozen-pair motions. The proof applies identically to both the plain and attention aggregation paths.

### 10.4 Configuration and Defaults

| Parameter          | Type  | Default  | Description                            |
| ------------------ | ----- | -------- | -------------------------------------- |
| `bridging_method`  | str   | `"none"` | `"none"` to disable, `"ZBL"` to enable |
| `bridging_r_inner` | float | 0.8      | Inner radius in Å                      |
| `bridging_r_outer` | float | 1.2      | Outer radius in Å                      |

The default window `[0.8, 1.2]` Å is tuned for general-purpose pretraining on large materials/molecule corpora (OMat24, OMol, MPtraj-class datasets). Three considerations shape the window:

1. **Data coverage.** `r_inner = 0.8 Å` places the lower edge of the transition zone where the combined datasets carry edges. Choosing lower would leave the sub-0.8 Å region as pure extrapolation.

1. **Bond sensitivity.** Common short bonds land on healthy parts of `dr̃/dr`: O-H (0.96 Å, `h'≈1.29`), N-H (1.01 Å), C-H (1.09 Å). Raising `r_inner` above 0.8 would push these into the low-derivative shoulder.

1. **Numerical smoothness.** The 0.4 Å width keeps `max|d²r̃/dr²| ≈ 24 Å⁻¹`, keeping second-derivative training stable. Narrower windows inflate the curvature.

### 10.5 InnerPotential (ZBL)

The `InnerPotential` module (in `sezm_model.py`) computes the Ziegler-Biersack-Littmark screened nuclear repulsion:

```
V_ZBL(r) = (ke × Zi × Zj / r) × φ(r / a)
a = 0.88534 × a_bohr / (Zi^0.23 + Zj^0.23)
φ(x) = 0.18175·exp(−3.1998x) + 0.50986·exp(−0.94229x)
      + 0.28022·exp(−0.4029x) + 0.02817·exp(−0.20162x)
```

Each pair `(i, j)` contributes `V_ZBL / 2` to atom `i` (symmetric neighbor list). `InnerPotential.forward(edge_vec, edge_index, ...)` evaluates the pair sum directly from the descriptor's sparse `edge_vec`, so the ZBL energy depends on coordinates **only through the same `edge_vec` leaf** as the learned descriptor — its force and virial therefore flow through the single edge backward (§12.4), guaranteeing conservation without a second autograd pass. The descriptor and ZBL receive the same pair-exclusion mask; virtual spin types are additionally masked out via `real_type_count`. The final atom mask is applied after adding ZBL, so excluded and padded atomic outputs remain zero for the complete physical energy.

### 10.6 Output Statistics and Fine-Tuning

Residual output-bias calibration must evaluate the same physical atomic output as training and inference:

```
descriptor.forward_with_edges
→ fitting network
→ output statistics
→ InnerPotential
→ final atom mask
```

`core_compute` is the single implementation of this pipeline. Its normal conservative path creates the `edge_vec` autograd leaf and performs the edge-force scatter. `predict_atomic_outputs_for_stat` calls the model's polymorphic `forward_common(atomic_output_only=True)` under deterministic evaluation and `torch.no_grad()`, returning immediately after complete atomic outputs are assembled. It does not enter the compile cache and does not calculate force or virial. Restoring the original training state invokes the normal module cache-invalidation hooks before training resumes.

Virtual-spin statistics keep two representations with distinct responsibilities: expanded `atype`/`natoms` provide the regression metadata, while the original `coord`/`atype`/`spin` are retained for physical prediction. The predictor therefore follows the production topology exactly — real-atom neighbor construction followed by `extend_nlist` — rather than rebuilding a neighbor list from pre-expanded virtual atoms.

For `change-by-statistic`, the least-squares target is `label − complete_model_prediction`; labels equal to the current model output therefore produce exactly zero bias change. `set-by-statistic` keeps its label-only initialization semantics and does not call the model predictor.

Statistics use the same `training.training_data.min_pair_dist` frame filter as optimization. Systems with no valid sampled frame are skipped. In distributed training, every optimization step enters one fixed all-reduce whenever any configured task enables this filter. If any rank has no valid frame in its locally selected task batch, every rank skips the step, so task-dependent control flow cannot misalign collectives and no below-threshold fallback frame enters the loss. Multi-task auxiliary display keeps its existing per-task metrics through the local unwrapped DDP module; if no valid local frame exists, or in-place FSDP2 provides no collective-free local module, the corresponding instantaneous training metric is reported as `nan`.

______________________________________________________________________

## 11. Fitting Network

The default SeZM fitting net maps the scalar descriptor `(nf, nloc, channels)` to per-atom energies. User input may spell it as `dpa4_ener`, while the internal implementation and serialized fitting type remain `sezm_ener`. It uses the same configuration keys as the standard DeePMD energy fitting (`neuron`, `activation_function`, `precision`, `seed`, ...).

- `neuron = []` produces a direct linear projection from `channels` to scalar energy (no hidden layers).
- When `neuron` is non-empty, each hidden layer is a GLU block: `Linear(in, 2×hidden) → split → value × act(gate)`. The internal hidden width is therefore double the user-specified value (e.g., `hidden=256` creates a 512-wide layer before the split).
- In shared-fitting multitask runs, `dim_case_embd` keeps the standard DeePMD meaning: the width of the per-branch one-hot case vector. By default (`case_film_embd = false`) the one-hot vector is concatenated to the fitting input exactly like the standard PyTorch fitting path.
- When `case_film_embd = true`, the one-hot vector is not concatenated. It is passed through a SeZM-only case FiLM conditioner inside `sezm_ener`: `K → round_up_to_32(4K) → 4*dim_descrpt`, then each modulation target uses its own projector `4*dim_descrpt → 2*target_dim`. The first target modulates only the descriptor slice of the fitting input, leaving `fparam` and `aparam` unchanged; subsequent targets modulate each GLU hidden activation.
- Case FiLM is intentionally a fitting-side operation. The descriptor remains a case-independent feature extractor `φ(R, Z)`, while the case controls the map from descriptor features to the PES. Per-element zero-point shifts remain handled by `bias_atom_e`, so the SeZM output layer does not need an additional bias for case conditioning.

The `dens` path (direct-force / denoising head, inspired by EquiformerV3 DeNS) reuses the same fitting configuration for its scalar energy branch and adds a parallel direct-force head. This path is activated by `loss.type = "dens"` and materialized lazily — training creates it when the loss type requires it, and checkpoint loading recreates it when `dens` weights are present. The `dens` path is retained for experimental purposes but is not the primary training mode.

Invariant property prediction uses the standard PyTorch `property` fitting on top of the SeZM scalar descriptor. Set `fitting_net.type = "property"`, provide `property_name`, and train with `loss.type = "property"`. The factory builds `SeZMPropertyModel`, a subclass of `SeZMModel` that reuses the sparse-edge descriptor path and compile cache but calls `core_compute(..., conservative=False)` so no edge-force autograd or virial path is constructed. Output statistics follow the standard property convention: predictions are scaled by label standard deviation and shifted by label mean before atom masking and reduction.

______________________________________________________________________

## 12. Model Integration

### 12.1 SeZMModel

Set `model.type = "dpa4"` to select the DPA4 / SeZM model scaffold. `SeZM` and `sezm` are accepted as compatibility aliases. Internally it is built as `make_model(SeZMAtomicModel)`.

`descriptor.type` defaults to `dpa4` inside this scaffold. `fitting_net.type` defaults to `dpa4_ener`, which builds the SeZM energy fitting implementation (`sezm_ener`). Explicit `fitting_net.type = "property"` builds `SeZMPropertyModel` and predicts an invariant reduced property instead of energy.

**Mode routing** is selected by `loss.type`:

- `loss.type = "ener"` → conservative energy/autograd-force path
- `loss.type = "dens"` → parallel direct-force/denoising path
- `loss.type = "property"` → invariant property path

The LAMMPS-style interface (`forward_lower`) supports the conservative `ener` path and the invariant `property` path. The `dens` path uses the standard neighbor-list lower interface because its force-conditioning tensor is not part of the compact edge ABI.

### 12.2 Spin Models

SeZM spin is selected by keeping `model.type = "dpa4"` and adding the standard DeePMD `model.spin` block. Two implementation schemes coexist, chosen by `model.spin.scheme`:

- `"native"` — the per-atom spin enters the descriptor as an equivariant feature; the magnetic force is the negative spin gradient of the energy. No virtual atoms are created. Built as `SeZMNativeSpinModel`.
- `"deepspin"` (default) — the classical DeepSpin virtual-atom representation. Built as `SeZMSpinModel`. This is the default so that a `model.spin` block without an explicit `scheme` reproduces the classical behaviour, matching every non-SeZM spin model.

Both are subclasses of `SeZMModel`, so they reuse SeZM's `core_compute`, sparse-edge construction, fitting, output statistics, and serialization rather than wrapping SeZM in the generic `SpinEnergyModel`. Both produce the same public output keys (`force`, `force_mag`, `mask_mag`) and train against the unchanged `ener_spin` loss with the same `spin` / `force_mag` dataset convention, so a system can be trained either way without touching the data pipeline or the loss.

The shared spin contract is the magnetic-force identity

```
force_mag = -dE/dspin
```

which both schemes satisfy. In the deepspin scheme it is reached indirectly through `F_virtual * virtual_scale` (the `virtual_scale` factor cancels analytically); the native scheme differentiates the energy with respect to the spin input directly. The dataset `force_mag.npy` is `-dE/dspin` in the same units as `spin.npy`, so the two schemes are label-compatible.

#### 12.2.1 Native scheme (`SeZMNativeSpinModel`)

The native scheme treats the per-atom spin vector `s` as an **equivariant extension of the type embedding** (see §6.4). Because the spin is an intrinsic node property, the model keeps the real type map, the real neighbor selection, and the real type count — there is no doubling and no virtual neighbor capacity.

- **Magnetic force.** On the conservative path `core_compute` promotes the per-atom spin to a second autograd leaf alongside the per-edge `edge_vec` leaf, and `edge_energy_deriv` runs a single `autograd.grad(energy, [edge_vec, spin])`. The edge gradient is scattered into force / virial as usual; the spin gradient becomes `energy_derv_r_mag = -dE/dspin`. `create_graph=self.training` keeps the magnetic force differentiable for the force-loss second backward, exactly like the conservative force.
- **Masking.** The descriptor's spin embedding and the model both gate by a per-type spin mask, so non-magnetic types contribute zero spin feature and report exactly zero magnetic force; `mask_mag` marks the magnetic atoms for the loss.
- **Equivariance and smoothness.** The spin enters as a global-frame `l = 1` feature, so under a rotation `R` the energy is invariant and both `force = -dE/dx` and `force_mag = -dE/dspin` rotate as vectors. Every spin route is linear in `s` (`l = 1` on-site and neighbour) or quadratic via `|s|²` / `MᵀM` (`l = 0`), so all spin contributions are smooth and vanish at `s = 0`. A neighbour's spin reaches an atom in two ways: directly in the initial embedding (the envelope-gated `l = 1` neighbour aggregation and the env-seed spin–spin invariants of §6.4) and through the message passing; all routes are envelope-gated, so they decay to zero at `rcut` with no information leakage.
- **Spin-free pretraining and transfer.** Because every spin route vanishes at `s = 0` and so do its parameter gradients (the `l = 1` weight gradient carries a factor of `s`; the `l = 0` magnitude network is bias-free in `|s|²`), the native model at zero spin is bit-for-bit the spin-free energy model, with the spin parameters receiving exactly zero gradient. Setting `model.spin.allow_missing_label` admits training systems that lack a `spin` data file — their per-atom spin is filled with zeros — so the shared backbone can be pretrained on a large spin-free corpus (the spin parameters stay dormant at their initialization) and then activated by fine-tuning on spin-labelled data. The model class, type map and parameter layout are identical across the two stages, so the fine-tune is an ordinary `--init-model` continuation with the loss switched from `ener` to `ener_spin`; no virtual-atom scheme can transfer this way, because its expanded type map and neighbor capacity change the descriptor graph. The same flag also permits mixing spin-labelled and spin-free systems within one training.
- **ZBL bridging** works unchanged on all real edges — there are no virtual atoms to mask out of the pair sum.
- **Interfaces.** The native scheme exposes only the public `forward(coord, atype, spin, ...)` (training / eval, via the inherited `forward_common`) and the `.pt2` deployment lower (`forward_common_lower_exportable`, below). There is no separate eager neighbor-list `forward_lower`: it would duplicate the descriptor's own edge-schema construction with no consumer, since SeZM deploys through `.pt2` rather than the TorchScript LAMMPS path.
- **Magnetic-force layout.** The magnetic force is always expressed in the extended (nall) layout so it shares the conservative force's reduce / fold-back path — `communicate_extended_output` (Python eval) and `select_map` / `fold_back` (C++) treat force and magnetic force identically, avoiding a spin-specific reduction. Single-rank, only owned spins are autograd leaves, so the per-local-atom magnetic force is zero-padded to nall (the padding reduces only structural zeros, so the round-trip is exact). Multi-rank with-comm (§12.2.1 deployment), the **extended** spin is the leaf — LAMMPS forward-comms ghost spins — so the magnetic force is genuinely extended: each ghost row is that rank's cross-rank contribution to the owner's magnetic force, folded onto the owner by the LAMMPS spin reverse-comm exactly as ghost conservative forces fold onto owners.
- **Compile and export.** The native scheme shares the compiled `core_compute` path: the per-atom spin is threaded into `trace_and_compile` as a second traced leaf, and `make_fx` unfolds the same single `autograd.grad(energy, [edge_vec, spin])` it already captures for the edge-vector force, so the magnetic force is produced by the compiled graph with no extra backward. The compiled callable is cached under the standard `(training, has_coord_corr)` key — the descriptor's spin-embedding submodule is folded into the module-level shared-cache structure key so a spin and a non-spin model never alias. For `.pt2` deployment the native scheme reports `export_lower_input_kind() == "edge_vec"` and **reuses the energy edge ABI** with one extra per-local-atom spin input: `forward_common_lower_exportable(coord, atype, edge_index, edge_vec, edge_scatter_index, edge_mask, spin, ...)`. `freeze_sezm_to_pt2` writes `lower_input_kind = "edge_vec"` (alongside `is_spin`, `use_spin`, `ntypes_spin`) into `metadata.json`, so `DeepSpinPTExpt` builds the neighbour topology exactly like `DeepPotPTExpt` (`createEdgeTensors` + `compactEdgeTensors`) and folds the extended conservative force and the magnetic force back to local atoms. A spin and a non-spin `.pt2` archive therefore share one C++ inference path, differing only by the extra spin input.
- **Multi-rank deployment.** Because the native scheme uses the rank-decomposable `edge_vec` contract, it ships the second multi-rank with-comm artifact alongside the energy model. The freeze gate is `with_comm = (export_lower_input_kind() == "edge_vec") and supports_edge_parallel()`, which now admits native spin and excludes only the deepspin scheme (nlist) and analytical bridging. `forward_common_lower_exportable_with_comm` traces the parallel lower with the **extended** spin leaf plus the eight `border_op` communication tensors, returning the extended magnetic force; `DeepSpinPTExpt::compute` routes single-rank to `run_model_edges` (owned-atom spins, `fold_to_local=true`) and multi-rank to `run_model_edges_with_comm` (extended spins, `fold_to_local=false`), mirroring `DeepPotPTExpt` including the empty-subdomain phantom-atom path. The per-node spin gradient is exact across ranks through `border_op`'s exact-VJP backward (§23.7); single- vs multi-rank LAMMPS parity (energy, force, magnetic force) confirms the deployment.

#### 12.2.2 DeepSpin virtual-atom scheme (`SeZMSpinModel`, default)

The virtual-atom scheme follows DeePMD's classical convention:

- For every real atom, a virtual spin atom is placed at `coord + spin * virtual_scale`.
- Virtual atom types are stored internally as `atype + ntypes_real`.
- Public metadata (`get_type_map`, `get_sel`, `get_nsel`) reports the real system, while the internal descriptor uses an expanded neighbor capacity of `2 * real_nsel + 1`.
- The standard forward path first builds the real-atom neighbor list, then expands it to include the self spin atom plus real and virtual neighbors. The lower interface follows the same contract: callers provide the real-atom extended inputs and real neighbor list; `SeZMSpinModel.forward_lower` performs the spin expansion internally.
- Energy is retained only on real local atoms. Coordinate derivatives are split into real force and magnetic force (`force_mag`) by combining real and virtual derivatives with the spin mask.
- ZBL bridging remains available, but the analytical pair potential is masked to real-real pairs only via `real_type_count`. Virtual spin atoms still participate in the learned descriptor as neighbours, but they are not treated as nuclei by `InnerPotential`.

The compile path is shared with `SeZMModel`. For the deepspin scheme, the expanded lower inputs include an `extended_coord_corr` tensor that corrects virial contributions for the virtual displacement; this tensor is passed through both eager and compiled `core_compute`. Unlike the native scheme, the deepspin scheme reports `export_lower_input_kind() == "nlist"`: virtual atoms are placed and the neighbor list is expanded *inside* the traced graph, so the `.pt2` export keeps the DeepSpin extended-input signature `(extended_coord, extended_atype, extended_spin, nlist, mapping, fparam, aparam)`. `freeze_sezm_to_pt2` writes `lower_input_kind = "nlist"` (plus `is_spin`, `use_spin`, `ntypes_spin`) into `metadata.json` so `DeepSpinPTExpt` calls the C++ spin backend with that 7-input contract. Spin type masks are registered as non-persistent buffers, so CPU/GPU export moves them with the model without adding checkpoint state.

Both schemes support only the conservative `ener` / `ener_spin` path. `dens + spin` is intentionally unsupported and fails during mode selection: the direct-force denoising head has a different output semantics from the magnetic-force decomposition.

### 12.3 Compiled Execution Path

SeZM supports end-to-end compiled `ener` and invariant `property` paths. Energy training lowers through `torch.compile` and is designed to survive Inductor's compilation of second-order coordinate derivatives — the gradient of the force loss with respect to model parameters passes through both the energy-to-force `autograd.grad` and the optimizer's backward pass. Property training reuses the same sparse-edge compile scaffold but disables the coordinate-gradient endpoint, so the compiled graph is forward-only. Eval/inference lowers the same `make_fx` graph through AOTAutograd's forward-only path instead (see *Backend lowering* below). The implementation lives in `sezm_model.py` and `sezm_property_model.py`.

**Enabling compile:** Set `model.use_compile = true`. Training follows `use_compile`; eval/inference defaults to eager. To opt evaluation into the compile path — which covers `model.eval()` calls in regular `disp_freq` validation, `validating.full_validation`, and `validating.ema_full_validation` — use either of:

- **Recommended, from input.json:** set `validating.compiled_infer = true`. The trainer translates this into `DP_COMPILE_INFER=1` at startup *before* any model is constructed, so `SeZMModel.__init__` samples the right value.
- **Environment variable:** `export DP_COMPILE_INFER=1` before launching training. Useful when you want to flip the switch without editing input.json. A manually exported env var takes precedence over `validating.compiled_infer` (the trainer uses `os.environ.setdefault`), so shell-level overrides stay authoritative.

Because the compile cache is multi-slot (see the cache invariant below), opting eval into compile does not evict the training compile product on every `train → eval → train` flip.

**Eval precision policy:** `DP_TF32_INFER` controls CUDA TF32 matmul precision inside the eval/inference compute path, for both eager and compiled execution. It defaults to `0` (`highest`, TF32 disabled). Set `validating.tf32_infer = true` to request `high` from input.json; the trainer translates this into `DP_TF32_INFER=1` before model construction. A manually exported `DP_TF32_INFER` takes precedence and may be set to `0` (`highest`), `1` (`high`), or `2` (`medium`). Training TF32 remains controlled by `model.enable_tf32`.

`DP_AMP_INFER=1` enables the descriptor's existing bf16 autocast region during eval/inference when `descriptor.use_amp=true`. Set `validating.amp_infer = true` to request this from input.json; the trainer translates it into `DP_AMP_INFER=1` before model construction. AMP and TF32 can be enabled together, but AMP casts selected eligible operations to bf16, so enabling TF32 on top of AMP gives little additional benefit on those operations. Like TF32, inference AMP usually leaves aggregate MAE nearly unchanged but can make the potential energy surface much less smooth.

**The unified kernel** `core_compute()` handles both eager and compile paths. It consumes the edge schema produced by the caller, runs the descriptor and fitting, and then selects the post-processing requested by the model variant. Energy keeps `conservative=True`: `edge_vec` is detached into a fresh autograd leaf, optional ZBL is added through `InnerPotential`, and `edge_energy_deriv` performs a single `autograd.grad(energy, edge_vec)` with `create_graph=self.training` to scatter the per-edge gradient into force, global virial and per-atom virial (§12.4). Property uses `conservative=False`: it leaves `edge_vec` as a forward tensor and reduces the fitting output through the standard DeePMD output definition, with no conservative-force derivative.

Frame and atomic fitting parameters are canonicalized before entering the compiled callable so the traced signature is tensor-only. The semantics match DeePMD's standard PyTorch fitting path: when `fitting_net.numb_fparam == 0`, any incidental `fparam` tensor in the batch is ignored; when `numb_fparam > 0`, the tensor is reshaped to `(nf, numb_fparam)` and participates in the fitting network; when it is missing and `default_fparam` is configured, that default is expanded across frames. `aparam` follows the same "ignored when unused, required when configured" rule with shape `(nf, nloc, numb_aparam)`.

**Trace strategy for `ener`:** `make_fx(tracing_mode="symbolic", _allow_non_fake_inputs=True)` captures the inner `autograd.grad` as ordinary FX nodes. The `make_fx` step is essential: it turns the second-derivative `autograd.grad` into a flat FX graph that Inductor can lower without needing to re-derive the second backward at compile time.

**Backend lowering differs by mode.** The captured graph is handed to a different backend for training versus inference:

- **Training** uses `torch.compile(dynamic=True, backend="inductor")`. The Dynamo frontend builds the optimizer's second backward through the materialised first derivative.
- **Inference** uses `aot_module_simplified` — AOTAutograd's forward-only path — and bypasses Dynamo. Routing inference through `torch.compile` is harmful in two ways. (1) Dynamo re-runs shape-guard production on the forward-only graph, where an intermediate view's extended-atom (`nall`) axis becomes a backed symbol with no input source and `produce_guards` aborts with `sources must not be empty for symbol s...`. (2) The materialised first-derivative graph (the edge backward baked into the forward) makes AOTAutograd treat the call as forward+backward and retain the whole activation set — about 3× the eager peak memory, which OOMs on large inference sweeps. `aot_module_simplified` compiled under `torch.no_grad` builds no backward (peak memory matches eager), still functionalizes the graph so Inductor reuses buffers, and consumes the `make_fx` symbolic shapes directly so a single artifact serves all `(nframes, nall, edge-count)` with no recompiles. Four details wire the hand-built path correctly: the dict output is flattened to a tuple (AOTAutograd requires flat outputs); `select_decomp_table()` aligns the decomposition set with Inductor's fallback set (else `aten._to_copy` raises a decomp/fallback clash); the compile runs under `torch.device(model device)` because AOTAutograd's `PhiloxStateTracker` allocates an RNG-seed tensor without an explicit device; and `triton.max_tiles=1` keeps the data-dependent edge axis on Triton's `x` grid (limit 2³¹) rather than the `y`/`z` grid (limit 65535, which overflows past ~2×10⁴ atoms with `CUDA error: invalid argument`).

**Deterministic inference.** The descriptor's random local-Z roll (`random_gamma`) is gated to training, so the inference graph contains no `aten.rand`. The model is roll-equivariant, which makes the roll a pure training augmentation and keeps inference deterministic (the roll changes energy/force only within the model's inherent ~1e-5 GPU-atomic non-determinism).

**Key compile invariants** (each maps to a `NOTE:` tag in `sezm_model.py`):

| Invariant                                                          | Why                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Trace inputs use safe prime dimensions**                         | Before `make_fx`, real batch tensors are padded or trimmed so `nf`, `nloc`, and `nall` are pairwise-distinct primes ≥ 5 and do not collide with fixed model dimensions (`1`, `2`, `3`, `9`, `nsel`, fparam/aparam widths, charge-spin width, or promoted task-buffer dimensions). `nlist` and `mapping` are clamped after shape coercion so trace-only gathers stay in range. Trimming returns a **contiguous** copy: a sliced trim keeps the pre-trim length in its stride, which `make_fx` records as a free symbol and duck-shaping then fuses with any size symbol of equal trace value (a trimmed `atype` stride equal to the edge count when both land on the same prime previously corrupted the eval graph). |
| **`silu_backward` decomposed**                                     | PyTorch has no registered higher-order derivative for SiLU; the opaque `silu_backward` would crash Inductor's double-backward                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **Eval trace disables `make_fx` duck-shaping**                     | Inference lowers the `make_fx` graph verbatim through AOTAutograd, so its symbolic sizes and strides are baked into the single artifact (training re-derives them through Dynamo from real contiguous inputs, so it keeps the default). The eval trace runs under `torch.fx.experimental._config.use_duck_shape = False`, giving every size and stride an independent symbol. Otherwise a first validation frame whose `nloc` equals a trace axis size (e.g. the edge-count prime) unifies an unrelated axis onto that symbol, and `assert_size_stride` later fails once the real edge count differs — a rare, data-dependent crash that looks config-independent.                                                   |
| **Detach chains stripped in training**                             | `make_fx` under `create_graph=True` wraps saved activations in `fwd → detach → detach → bwd` chains that sever gradient flow. The stripper uses graph topology to remove chain-inner detaches while preserving user-explicit `.detach()`                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| **FX graph rebuilt after stripping**                               | `Graph.erase_node` leaves stale C-level pointers on some builds; a fresh `node_copy` pass prevents segfaults                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| **DDPOptimizer disabled**                                          | Set at import time. DDPOptimizer splits graphs at bucket boundaries, producing subgraphs with symbolic-integer outputs that crash AOT Autograd                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| **`edge_vec` detached into an autograd leaf**                      | The force autograd endpoint is `edge_vec`, not the coordinates: the coordinate gather is a pure forward op outside the differentiated region. Gather / advanced indexing *inside* the differentiated region can silently truncate the second-order gradient under `make_fx`, so keeping it out reduces the AD region to the pure function `(edge_vec, θ) → E`.                                                                                                                                                                                                                                                                                                                                                       |
| **Two dummy edges appended**                                       | `torch.nonzero(valid_mask)` is data-dependent; an empty result cannot be traced symbolically. Two masked dummies (`edge_mask=False`) keep the edge axis at length ≥ 2, avoiding both empty-edge branches and Inductor `E == 1` batched-matmul layout guards.                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| **Compiled callable via `object.__setattr__`**                     | `nn.Module.__setattr__` would register the wrapper as a submodule, exposing duplicate parameter views to FSDP2/DDP                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| **Trace inputs detached (`edge_vec` is the grad endpoint)**        | The grad endpoint is the `edge_vec` leaf created and detached inside `core_compute`, so the runtime forward passes coordinates through as-is. The trace closures (`_prepare_coord_for_trace` for compile, `lower_fn` for export) still `.detach()` their coordinate input so the make_fx / export graph roots cleanly at `edge_vec` with no stray coordinate grad path.                                                                                                                                                                                                                                                                                                                                              |
| **Multi-slot compile cache keyed on `(training, has_coord_corr)`** | `training` toggles the second-derivative branch and `has_coord_corr` toggles the spin virial-correction input. The cache is a `dict` with one slot per key (stored outside the `nn.Module` tree via `object.__setattr__`), so flipping between train and eval reuses the other slot's compile product instead of evicting it.                                                                                                                                                                                                                                                                                                                                                                                        |
| **Shared multi-task compile callable**                             | Branches whose descriptor/fitting structure and graph-state fingerprint match after `share_params(level=0)` reuse one module-level compiled callable. The fingerprint includes descriptor exclude/conditioning/compile flags, fitting atom-exclude state, model atom-exclude hooks, bridging mode, and type map. Per-task tensors (`out_bias`, `out_std`, `bias_atom_e`, `case_embd`) are promoted to FX placeholders.                                                                                                                                                                                                                                                                                               |
| **Conservative Inductor fusion limits**                            | The conservative path sets `max_fusion_size=8` and `triton.persistent_reductions=False` for both training and eval. This keeps dynamic edge reductions from forming oversized persistent Triton kernels while preserving the same compile option envelope across train/eval cache slots.                                                                                                                                                                                                                                                                                                                                                                                                                             |
| **1-D Triton launch grids (`triton.max_tiles=1`)**                 | Inductor's default tiling may put a data-dependent axis (edge / node count) on Triton's y/z launch grid (limit 65535); a dense batch that exceeds it launches an out-of-range grid and dies with `CUDA driver error: invalid argument` (illegal memory access at the next sync). Forcing 1-D grids keeps that axis on the x grid (limit 2³¹−1). Applies to **both** training and eval — the force second backward emits exactly these large fused kernels, so a training-only omission crashes only on dense batches and looks random/config-dependent.                                                                                                                                                              |

**Inductor options** are centralised in `build_inductor_compile_options` (`deepmd/pt/utils/compile_compat.py`): `max_autotune=False`, `shape_padding=True`, `epilogue_fusion=False`, `triton.cudagraphs=False`, `max_fusion_size=8`, `triton.persistent_reductions=False`, `triton.mix_order_reduction=False`, and `triton.max_tiles=1`. **Training** passes the set to `torch.compile` as `options=`; **inference** applies the identical set via `torch._inductor.config.patch` around `compile_fx_inner`. `triton.max_tiles=1` must cover *both* paths: the default 2-D/3-D tiling can place a data-dependent axis (edge / node count) on Triton's y or z launch grid (limit 65535), and a dense batch whose count exceeds 65535 then launches an out-of-range grid and dies with `CUDA driver error: invalid argument` (an illegal memory access surfaced at the next sync). Forcing 1-D grids keeps that axis on the x grid (limit 2³¹−1). The training graph needs this just as much as inference, because the second-order force backward emits exactly these large fused (`bmm`/`index_add`/grid) kernels.

Each setting addresses a specific failure mode under dynamic shapes or higher-order gradients.

**`dens` compile:** Since the direct-force head needs no second coordinate derivative, `dens` compiles `core_compute_dens` directly with `torch.compile` (no `make_fx`). Limitation: `dens` does not support analytical bridging potentials.

### 12.4 Edge-Force Scatter (force / virial / atomic virial)

SeZM derives force, global virial and per-atom virial with the *edge-force scatter* (`edge_energy_deriv` in `transform_output.py`). Every SeZM term — the equivariant descriptor and the analytical ZBL bridge — depends on coordinates only through the per-edge displacement vectors `edge_vec`, so the energy is a pure function `E(edge_vec, θ)` and a single `autograd.grad(energy, edge_vec)` yields the per-edge gradient `g_e = ∂E/∂edge_vec_e`. Force, both virials, and the spin correction are then assembled from `g_e` with explicit scatter and outer-product ops, so one backward produces all of them.

**Mechanism.** With edge `e` running from receiver/centre `dst(e)` to sender/neighbour `src(e)` and `edge_vec_e = r_{src} − r_{dst}`, the chain rule `∂edge_vec_e/∂r_k = (δ_{k,src} − δ_{k,dst}) I` gives:

```
force:         F_k = Σ_{dst(e)=k} g_e − Σ_{src(e)=k} g_e        (two index_add)
global virial: W   = −Σ_e g_e ⊗ edge_vec_e                      (sum of outer products)
atom virial:   Wᵢ  half-split: 0.5·(−g_e ⊗ edge_vec_e) to src(e) and dst(e)
```

The force equals `−∂E/∂x` (the same chain rule, contracted explicitly), so it is conservative to within atomicAdd summation order. The global virial is the pairwise form `Σ rᵢⱼ ⊗ fᵢⱼ`, which is PBC-correct because `edge_vec` is the true minimum-image displacement: Python builders include integer shifts, while LAMMPS obtains the same vector by subtracting already-shifted ghost coordinates. The per-atom virial uses a symmetric **half-split gauge** (each per-edge tensor shared equally by its two endpoints); the total is gauge-invariant.

**Periodic and non-periodic systems** use the same code. The scatter differentiates the real per-edge displacement, so it is boundary-agnostic: under PBC `edge_vec` is the minimum-image vector, and for an open-boundary cluster (`box = None`) it is the direct neighbour displacement. The scatter index is independent of the message-passing index: Python single-domain inference scatters to local owners, while LAMMPS scatters to local-plus-ghost slots so reverse communication can reduce ghost contributions to owners.

**Index spaces.** The edge schema carries two aligned index sets: `edge_index` (owner-local `[0, nf·nloc)`, for descriptor message passing) and `edge_scatter_index` (the force/virial scatter domain). Python single-domain inference uses local owner scatter indices, while LAMMPS uses local-plus-ghost scatter indices so reverse communication can reduce ghost contributions to owners.

**Precision.** `edge_vec` carries the coordinate precision `GLOBAL_PT_FLOAT_PRECISION` (the descriptor casts it to its `compute_dtype` internally for the geometry pipeline). The per-edge gradient `g_e` and the assembled force / virial therefore share that coordinate dtype — the dtype `communicate_extended_output` and the reduced energy expect — and the global virial is summed in `GLOBAL_PT_ENER_FLOAT_PRECISION`. This keeps the physics quantities (coordinates, force, virial) at full coordinate precision even when the network compute runs in a lower `compute_dtype`; the gradient *values* still carry the network's `compute_dtype` precision, exactly as differentiating the coordinates would.

**Spin.** The `extended_coord_corr` virtual-displacement correction adds `force ⊗ coord_corr` per extended atom, reusing the force already produced by the scatter.

______________________________________________________________________

## 13. Weight Layout and Optimizer Routing

### 13.1 Unified (in, out) Convention

All learnable weight matrices store rows as fan_in and columns as fan_out. Focus-aware modules fold the focus dimension `F` on the output (cols) side:

| Module          | Stored shape                     | Runtime contraction                                           |
| --------------- | -------------------------------- | ------------------------------------------------------------- |
| `ChannelLinear` | `(C_in, C_out)`                  | `einsum("...i,io->...o")`                                     |
| `FocusLinear`   | `(C_in, F×C_out)`                | view `(C_in, F, C_out)` → `einsum("bfi,ifo->bfo")`            |
| `SO3Linear`     | `(lmax+1, C_in, F×C_out)`        | expand to `(D, C_in, F, C_out)` → `einsum("ndfi,difo->ndfo")` |
| `SO2Linear`     | `(D_m×C_in, F×D_m×C_out)` per \` | m                                                             |

This layout ensures the Muon optimizer's rectangular correction `scale = sqrt(max(1, rows/cols))` stays at 1.0 when `C_in ≤ F×C_out` (typical), avoiding step-size inflation. Each semantically independent weight block gets its own Muon Newton-Schulz update.

### 13.2 HybridMuon Routing

HybridMuon uses name-based routing to separate optimizer paths (case-insensitive matching on the final effective parameter name segment):

| Pattern              | Optimizer | Weight decay |
| -------------------- | --------- | ------------ |
| Contains `bias`      | Adam      | none         |
| Starts with `adam_`  | Adam      | none         |
| Starts with `adamw_` | AdamW     | decoupled    |
| 2D tensor (default)  | Muon      | —            |
| Other shape          | AdamW     | decoupled    |

SeZM parameters use `adam_` prefixes for norm scales, layer scales, trainable radial basis parameters, and type embeddings (`adam_scale`, `adam_so2_layer_scales`, `adam_ffn_layer_scales`, `adam_freqs`, `adam_type_embedding`) so they route to Adam without weight decay.

**Recommended mode:** `muon_mode = "slice"`:

- 2D weights (ChannelLinear, SO2Linear, FocusLinear): standard Muon
- 3D SO3Linear `(lmax+1, C_in, C_out)`: per-(l) independent Muon with correct rectangular scale
- `adam_`/`bias` parameters: Adam (name-based routing takes priority)

### 13.3 Optional Magma-lite Damping

HybridMuon supports optional Muon-path damping via `training.magma_muon = true`:

1. **Alignment.** Compute per-block cosine alignment between Muon momentum `m_t` and current gradient `g_t`.
1. **Temperature sigmoid.** `raw = sigmoid(cos / tau)` with `tau = 2.0`, then stretch to `[0, 1]`.
1. **EMA smoothing.** `score_t = 0.9 × score_{t-1} + 0.1 × raw`.
1. **Final scale.** `scale = 0.1 + 0.9 × score_t`, always in `[0.1, 1.0]`.
1. **Apply.** `delta_muon = delta_muon × scale`.

Strong momentum-gradient agreement yields larger scales; poor alignment damps the update. The `0.1` floor prevents complete starvation of hard-to-optimize blocks. No Bernoulli skip masking is used (unlike the original Magma paper) because force-field training is sensitive to intermittent block freezing.

______________________________________________________________________

## 14. Initialization Strategy

### 14.1 Deterministic Seeding

All submodules derive seeds from `child_seed(seed, idx)`. Repeated structures (blocks, SO2 layers, FFN subblocks) include loop indices in the seed derivation. When `seed=None`, initialization follows the global RNG.

### 14.2 Weight Initialization

| Component                         | Method                                                                                       |
| --------------------------------- | -------------------------------------------------------------------------------------------- |
| SO2Linear                         | `TruncatedNormal(0, 1/sqrt(fan_in+fan_out), ±3σ)`. `\|m\|>0` blocks: extra `1/sqrt(2)` scale |
| SO3Linear                         | `TruncatedNormal(0, 1/sqrt(fan_in+fan_out), ±3σ)`                                            |
| ChannelLinear, FocusLinear        | Same truncated normal scheme                                                                 |
| EnvironmentInitialEmbedding MLPs  | `TruncatedNormal(0, sqrt(2/(fan_in+fan_out)), ±3σ)`                                          |
| Output projections (FFN, SO2Conv) | Zero-initialized (weights + bias) via `init_std=0`                                           |
| Gate projections                  | `Normal(0, 0.01)` for weights, zeros for bias                                                |
| LayerScale                        | Init to `1e-3`                                                                               |
| FiLM strength logs                | Init to `log(0.01)`                                                                          |
| DepthAttnRes query                | Zero-initialized                                                                             |

### 14.3 Near-Identity Start

The zero-initialization of output projections and the small-value initialization of gates and layer scales together ensure that SeZM starts training near identity:

- Each residual branch `x = x + output_proj(...)` starts as `x = x + 0 = x`.
- Gates start at `sigmoid(0) ≈ 0.5`, providing maximum gradient flow (`sigmoid'(0) = 0.25`).
- LayerScale starts at `1e-3`, keeping residual contributions small initially.
- FiLM starts as `scale ≈ 1, shift ≈ 0`, preserving the type embedding.
- DepthAttnRes starts as uniform average over sources.

This strategy prevents early training instability from large random perturbations in the equivariant feature space.

______________________________________________________________________

## 15. Precision and Numerical Safety

### 15.1 Compute Dtype Promotion

SeZM separates the working dtype (for interaction blocks) from the compute dtype (for geometry and critical operations):

```
dtype = PRECISION_DICT[precision]          # e.g., float32
compute_dtype = get_promoted_dtype(dtype)  # float32 if dtype is float32, else float32
```

Geometry computations (edge distances, quaternions, Wigner-D, GIE, radial basis, environment embedding, norms, output FFN) always run in `compute_dtype` (fp32+). Interaction blocks use the working `dtype`. This prevents the accumulation of half-precision rounding errors in geometric quantities that directly affect the PES shape.

### 15.2 Automatic Mixed Precision

When `use_amp=True` (default) and the model is training on CUDA, interaction blocks run under `torch.autocast("cuda", dtype=torch.bfloat16)`. The same autocast region is used during eval/inference only when `DP_AMP_INFER=1` (or `validating.amp_infer=true` during training validation) is set before model construction. This reduces memory usage for the large edge-level tensors inside SO(2) convolution without usually changing aggregate accuracy; autocast-eligible operations (matmul, bmm) use bf16 while reductions and norms stay in fp32.

All norm modules explicitly disable autocast to prevent precision loss in RMS computation. Inference AMP can coexist with `DP_TF32_INFER`, but bf16 autocast dominates the eligible operations it covers, so TF32 generally adds little throughput there. Both lower-precision modes can reduce PES smoothness even when validation MAE changes little.

### 15.3 Numerical Safety Measures

| Situation                      | Protection                                                                 |
| ------------------------------ | -------------------------------------------------------------------------- |
| Vector normalization near zero | `sqrt(clamp(‖x‖², eps²))`                                                  |
| Quaternion normalization       | Same clamped sqrt; fp16/bf16 promoted to fp32                              |
| Division by distance           | `max(r, eps)` or `r.clamp(min=eps)`                                        |
| Degree normalization           | `rsqrt(deg + eps)`                                                         |
| Softmax stability              | Per-destination max subtraction                                            |
| Softmax denominator            | `softplus(z_bias) × exp(−max)` prevents zero denominator                   |
| Padding edges                  | Displacement reset to `(0, 0, 1)`, `edge_env = 0`                          |
| RadialMLP bias                 | `bias=False` throughout to prevent constant leakage from padding           |
| SO2Linear bias                 | First-layer bias modulated by `(radial_l0 × edge_env − 1)` for consistency |

### 15.4 Conservative Force Guarantee

The entire geometry chain — `edge_vec → quaternion → Wigner-D → rotation → aggregation` — is fully differentiable. The only `.detach()` is the deliberate one that turns the per-edge `edge_vec` into the autograd leaf (the edge-force-scatter endpoint, §12.4); no `.detach()` is applied to edge rotations or geometric quantities *downstream* of that leaf. Forces are then `g_e = ∂E/∂edge_vec_e` scattered back onto atoms — exactly the chain-rule rearrangement of `−∂E/∂x` — so they remain truly conservative (path-independent, derivable from a scalar potential). The finite-difference checks in `test_sezm_model.py::TestSeZMEdgeForceScatter` (`F = −dE/dx`, `W = −dE/dε`, and per-atom virial summing to the global virial) pin this.

______________________________________________________________________

## 16. Tensor Layouts

### 16.1 Node Features

The backbone tensor at block `i` is `(N, D_node, 1, C)` (contiguous), where:

- `N = nf × nloc` — flattened batch×atom
- `D_node = (l_schedule[i] + extra_node_l + 1)²` — node SO(3) embedding dimension
- `1` — singleton focus axis (kept for module reuse; real multi-focus lives inside SO2Convolution)
- `C = channels` — per-coefficient channel width

Features are packed by increasing `l`. Within each `l` block, `m` runs from `−l` to `+l`:

| Degree      | Indices         | Count  |
| ----------- | --------------- | ------ |
| `l=0`       | `[0:1]`         | 1      |
| `l=1`       | `[1:4]`         | 3      |
| `l=2`       | `[4:9]`         | 5      |
| `l=3`       | `[9:16]`        | 7      |
| general `l` | `[l² : (l+1)²]` | `2l+1` |

View conventions inside blocks:

- `x[:, :D_mp].view(N, D_mp, C)` for SO(2) message passing, where `D_mp = (l_schedule[i]+1)²`
- `x.view(N, D_node, 1, C)` at block boundaries and inside FFN
- Inside SO2Convolution: `(E, F, D_m, Cf)` for multi-focus SO(2) stack

### 16.2 Edge Cache Tensors

All edge cache tensors hold **valid edges only** (padding and excluded type pairs removed). The edge count `E` varies per forward call. Key shapes:

| Tensor              | Shape               | Notes                               |
| ------------------- | ------------------- | ----------------------------------- |
| `src`, `dst`        | `(E,)`              | Node indices in `[0, N)`            |
| `edge_vec`          | `(E, 3)`            | Displacement in Å                   |
| `edge_type_feat`    | `(E, C)`            | `type_embed[src] + type_embed[dst]` |
| `edge_rbf`          | `(E, n_radial)`     | Radial basis with envelope          |
| `edge_env`          | `(E, 1)`            | C³ cutoff envelope                  |
| `D_full`, `Dt_full` | `(E, D_mp0, D_mp0)` | Block-diagonal MP Wigner-D          |
| `inv_sqrt_deg`      | `(N, 1, 1)`         | Degree normalization                |
| `edge_src_gate`     | `(E, 1)`            | SFPG gate (or `None`)               |

Edges with `r ≥ rcut` are kept in the cache (not filtered) because their `edge_env = 0` naturally zeros their messages. This avoids the `torch.nonzero` dynamic-shape kernel and keeps the smooth degree free of discontinuous jumps.

### 16.3 Reduced SO(2) Layout

Inside SO(2) convolution, only coefficients with `|m| ≤ mmax` are retained. The reduced dimension is:

```
D_m = Σ_{l=0}^{lmax} (2 × min(l, mmax) + 1)
```

Coefficients are stored in m-major order: all `l`-values for `m=0`, then all `l`-values for `m=−1,+1`, then `m=−2,+2`, etc. Two precomputed buffers map between the full and reduced layouts:

- `coeff_index_m`: maps reduced positions to full `D` indices (for row-select in `D_to_m`)
- `degree_index_m`: maps reduced positions to degree `l` (for radial feature lookup)

______________________________________________________________________

## 17. VRAM Estimation

### 17.1 Notation

| Symbol | Meaning                                    | Source       |
| ------ | ------------------------------------------ | ------------ |
| P      | Total trainable parameters                 | training log |
| N      | Atoms per frame                            | system       |
| nnei   | Max neighbors (sel)                        | config       |
| E      | Edges = N × nnei                           | derived      |
| L      | max(l_schedule)                            | config       |
| L_node | L + extra_node_l                           | config       |
| D      | (L+1)²                                     | derived      |
| D_node | (L_node+1)²                                | derived      |
| C      | channels                                   | config       |
| F      | Effective FFN hidden dim                   | config/auto  |
| B      | Number of blocks                           | config       |
| b      | Bytes per element (4 for fp32, 2 for bf16) | dtype        |

### 17.2 Inference

```
M_infer ≈ E × b × [2D² + D_node×C + B×D×C + max(D×C, D_node×2F)]
                   ^^^^   ^^^^^^^^   ^^^^^   ^^^^^^^^^^^^^^^^^
                 Wigner   GIE/radial  SO2     transient peak
```

The three terms:

1. **MP Wigner-D matrices** `2 × E × D² × b`: `D_full + Dt_full`, shared across all SO(2) blocks.
1. **Node radial / GIE features** `E × D_node × C × b`: radial output and optional zonal coupling scale linearly with the node degree.
1. **SO(2) radial slices** `B × E × D × C × b`: one truncated MP slice per block.
1. **Transient peak**: the larger of the SO2Conv intermediate and the node-degree FFN up-projection. Only one block is active at a time during inference.

### 17.3 Training

During training, autograd saves intermediate tensors across all blocks simultaneously:

```
M_train ≈ E × b × [2D² + B × k × D×C + B × k × D_node×C]     (k ≈ 5)
```

The factor `k ≈ 4–6` accounts for saved tensors: pre-norm input, rotation result, message, FFN up-projection, residual inputs. Training uses roughly **5–8× inference memory** due to these saved activations.

### 17.4 Scaling Summary

| Factor       | Scaling                           | Note                         |
| ------------ | --------------------------------- | ---------------------------- |
| N (atoms)    | Linear                            | E = N × nnei                 |
| nnei         | Linear                            | E = N × nnei                 |
| C (channels) | Linear                            | dominates E×D×C              |
| L (MP lmax)  | **Quadratic**                     | D = (L+1)², Wigner-D is E×D² |
| extra_node_l | Linear-to-quadratic in node FFN   | No node-level Wigner-D cache |
| B (blocks)   | Linear (train) / Constant (infer) |                              |
| F (ffn)      | Linear                            | only transient peak          |

**Bottleneck:** Edge-level tensors (E×D×C) and MP Wigner-D matrices (E×D²). `extra_node_l` increases node FFN/GIE capacity without increasing the MP Wigner-D cache.

______________________________________________________________________

## 18. Serialization

`DescrptSeZM.serialize()` produces a flat dictionary. The internal implementation payload keeps the SeZM type tag; user input and `model_def_script` should use `dpa4`.

```python
{
    "@class": "Descriptor",
    "type": "SeZM",
    "@version": 1,
    "config": { ... all constructor arguments ... },
    "@variables": { key: numpy_array for key, array in state_dict },
    "env_mat": DPEnvMat(rcut, rcut, eps).serialize(),
}
```

`@variables` contains the full `state_dict()` payload, including all `register_buffer` tensors (precomputed index tables, S2 projection matrices, reduced-layout maps, Wigner coefficient tables, interface-compatibility buffers). Serialization is flat at the descriptor level — no recursive per-submodule packing.

`DescrptSeZM.deserialize(data)` reconstructs the descriptor from `config`, then restores the state dict. Transient buffers rebuilt at construction time are dropped by `_load_from_state_dict` when loading from older checkpoints.

______________________________________________________________________

## 19. DeePMD Interface Compatibility

SeZM follows the new-style descriptor interface (same as `dpa3`), using `extended_coord` / `extended_atype` parameter names.

**Implemented interfaces:**

- `forward()` with the standard 5-tuple return `(descriptor, rot_mat, g2, h2, sw)`. Only `descriptor` is meaningful; the rest are empty tensors.
- `get_rcut()`, `get_sel()`, `get_dim_out()`, `get_dim_emb()`
- `mixed_types() → True` (unified neighbor list, no type distinction)
- `serialize()` / `deserialize()`
- `compute_input_stats()` — no-op (SeZM uses learnable RMSNorm)
- `update_sel()` for automatic neighbor selection

**Interface compatibility details:**

- `_ENV_DIM = 1` (se_r style) for `EnvMatStatSe` compatibility
- `ndescrpt = nnei × 1`
- `mean` and `stddev` buffers are maintained but not used in the forward pass
- Output: only `l=0` features as descriptor `(nf, nloc, channels)`

**Not implemented:** `share_params()`, `change_type_map()`, `enable_compression()`.

______________________________________________________________________

## 20. Configuration Reference

### 20.1 Descriptor Parameters

| Parameter             | Type                | Default     | Description                                                                                                                                                                                                                           |
| --------------------- | ------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rcut`                | float               | —           | Cutoff radius in Å                                                                                                                                                                                                                    |
| `sel`                 | int \| list[int]    | —           | Max neighbors (int: total, list: per-type)                                                                                                                                                                                            |
| `env_exp`             | list[int]           | `[7, 5]`    | Envelope exponents `[rbf_env_exp, edge_env_exp]`                                                                                                                                                                                      |
| `channels`            | int                 | 64          | Total channels per `(l,m)` coefficient                                                                                                                                                                                                |
| `basis_type`          | str                 | `"bessel"`  | Radial basis family: `"bessel"` or `"gaussian"`                                                                                                                                                                                       |
| `n_radial`            | int                 | 10          | Number of radial basis functions                                                                                                                                                                                                      |
| `radial_mlp`          | list[int]           | `[64]`      | Hidden sizes for radial MLP; use `0` as placeholder for `channels`                                                                                                                                                                    |
| `edge_norm`           | bool                | True        | Standard RMSNorm on cutoff-vanishing branches; `False` removes the radial/FiLM/focus norms and selects unit-floor post-SO(2) residual scaling (see §5.3)                                                                              |
| `use_env_seed`        | bool                | True        | Enable FiLM conditioning from environment matrix                                                                                                                                                                                      |
| `random_gamma`        | bool                | True        | Random edge roll for data augmentation                                                                                                                                                                                                |
| `lmax`                | int                 | 2           | Max degree (when `l_schedule` is None)                                                                                                                                                                                                |
| `n_blocks`            | int                 | 2           | Number of blocks (when `l_schedule` is None)                                                                                                                                                                                          |
| `l_schedule`          | list[int] \| None   | None        | Pyramid schedule of lmax per block (non-increasing)                                                                                                                                                                                   |
| `mmax`                | int \| None         | None        | Max SO(2) order (when `m_schedule` is None)                                                                                                                                                                                           |
| `kmax`                | int                 | 1           | Max Wigner-D frame order for SO3 grid branches                                                                                                                                                                                        |
| `m_schedule`          | list[int] \| None   | None        | Schedule of mmax per block                                                                                                                                                                                                            |
| `n_focus`             | int                 | 1           | Parallel focus streams in SO(2) convolution                                                                                                                                                                                           |
| `focus_dim`           | int                 | 0           | Per-focus hidden width (0 = use `channels`)                                                                                                                                                                                           |
| `n_atten_head`        | int                 | 1           | Attention heads in SO(2) aggregation (0 = plain scatter)                                                                                                                                                                              |
| `so2_norm`            | bool                | False       | Pre-norm between SO(2) layers                                                                                                                                                                                                         |
| `so2_layers`          | int                 | 4           | SO2Linear layers per convolution                                                                                                                                                                                                      |
| `so2_attn_res`        | str                 | `"none"`    | SO(2)-internal depth attention (`none`/`independent`/`dependent`)                                                                                                                                                                     |
| `radial_so2_mode`     | str                 | `"none"`    | Dynamic radial degree mixer (`none`/`degree`/`degree_channel`)                                                                                                                                                                        |
| `radial_so2_rank`     | int                 | 0           | Low-rank channel factorization for `degree_channel`; `0` means full per-channel kernel                                                                                                                                                |
| `ffn_neurons`         | int                 | 0           | FFN hidden width (0 = auto from channels)                                                                                                                                                                                             |
| `grid_mlp`            | bool or list[bool]  | False       | Per-path polynomial grid MLP `[node_wise, message_node, ffn]`; a scalar broadcasts to all paths                                                                                                                                       |
| `grid_branch`         | int or list[int]    | 0           | Per-path scalar-routed branch count `[node_wise, message_node, ffn]`; `0` disables branch mixing                                                                                                                                      |
| `ffn_blocks`          | int                 | 1           | FFN subblocks per interaction block                                                                                                                                                                                                   |
| `sandwich_norm`       | list[bool]          | `[T,F,T,F]` | `[so2_pre, so2_post, ffn_pre, ffn_post]`                                                                                                                                                                                              |
| `mlp_bias`            | bool                | False       | Bias in equivariant layers                                                                                                                                                                                                            |
| `layer_scale`         | bool                | False       | Learnable LayerScale (init 1e-3)                                                                                                                                                                                                      |
| `full_attn_res`       | str                 | `"none"`    | Descriptor-level full attention residual                                                                                                                                                                                              |
| `block_attn_res`      | str                 | `"none"`    | Descriptor-level block attention residual                                                                                                                                                                                             |
| `s2_activation`       | list[bool]          | `[F, F]`    | `[so2_s2_enabled, ffn_s2_enabled]`; S2 grids are resolved automatically                                                                                                                                                               |
| `ffn_so3_grid`        | bool                | False       | Use the SO3 Wigner-D grid for the block-internal FFN grid path                                                                                                                                                                        |
| `node_wise_s2`        | bool                | False       | Edge-local source-destination S2 product branch inside SO(2) convolution                                                                                                                                                              |
| `node_wise_so3`       | bool                | False       | Edge-local source-destination SO3 Wigner-D grid branch inside SO(2) convolution                                                                                                                                                       |
| `message_node_s2`     | bool                | False       | Post-aggregation hidden message-node S2 product branch before SO(2) output projection                                                                                                                                                 |
| `message_node_so3`    | bool                | False       | Post-aggregation hidden message-node SO3 Wigner-D grid branch before SO(2) output projection                                                                                                                                          |
| `so3_readout`         | str                 | `"none"`    | Final read-out FFN mode: `none` scalar (l=0 only), or `glu`/`mlp` full SO3 Wigner-D grid FFN folding l>0 into l=0                                                                                                                     |
| `lebedev_quadrature`  | bool \| list[bool]  | True        | Scalar switch for both branches, or `[so2_lebedev, ffn_lebedev]` for separate S2 projector control                                                                                                                                    |
| `activation_function` | str                 | `"silu"`    | Base activation                                                                                                                                                                                                                       |
| `glu_activation`      | bool                | True        | Base GLU switch for FFN                                                                                                                                                                                                               |
| `use_amp`             | bool                | True        | AMP with bf16 during training on CUDA; also gates `DP_AMP_INFER` during eval/inference                                                                                                                                                |
| `add_chg_spin_ebd`    | bool                | False       | Add frame-level charge/spin condition embedding to scalar type features                                                                                                                                                               |
| `default_chg_spin`    | list[float] \| None | None        | Default `[charge, spin]` condition used when `charge_spin` is not provided                                                                                                                                                            |
| `use_spin`            | list[bool] \| None  | None        | Per-type spin flags; when provided, builds the native `SpinEmbedding` (on-site l=0 magnitude + l=1 direction, plus envelope-gated l=1 neighbour aggregation) and the env-seed neighbour-spin channels. Set by the native spin factory |
| `precision`           | str                 | `"float32"` | Working precision for interaction blocks                                                                                                                                                                                              |
| `eps`                 | float               | 1e-7        | Numerical stability epsilon                                                                                                                                                                                                           |
| `exclude_types`       | list[tuple]         | `[]`        | Excluded pairs for standalone descriptor use; full models should prefer `model.pair_exclude_types`                                                                                                                                    |

When `add_chg_spin_ebd` is enabled, `charge_spin` is a frame-level tensor with shape `(nf, 2)`. The first column is charge and the second column is spin. The SeZM model canonicalizes this tensor before sparse-edge descriptor execution; the standard descriptor `forward()` canonicalizes it at the descriptor boundary. The condition embedding is added once to the scalar type embedding of every local atom in the frame, before the edge cache is built. The fitting `fparam` remains independent and is not used for descriptor condition embedding.

### 20.2 Model Parameters

| Parameter                  | Type  | Default  | Description                                                                                                         |
| -------------------------- | ----- | -------- | ------------------------------------------------------------------------------------------------------------------- |
| `model.type`               | str   | —        | `"dpa4"` (`"SeZM"` / `"sezm"` compatibility aliases)                                                                |
| `model.use_compile`        | bool  | False    | Enable torch.compile path                                                                                           |
| `model.pair_exclude_types` | list  | `[]`     | Excluded type pairs; copied into the descriptor edge mask and must match descriptor `exclude_types` if both are set |
| `model.bridging_method`    | str   | `"none"` | `"none"` or `"ZBL"`                                                                                                 |
| `model.bridging_r_inner`   | float | 0.8      | Inner radius in Å                                                                                                   |
| `model.bridging_r_outer`   | float | 1.2      | Outer radius in Å                                                                                                   |
| `model.lora.rank`          | int   | —        | LoRA rank (enables LoRA fine-tune; see §22)                                                                         |
| `model.lora.alpha`         | float | `rank`   | LoRA scaling numerator; effective scaling is `alpha / rank`                                                         |

### 20.3 Validation Runtime Parameters

| Parameter                   | Type | Default | Description                                 |
| --------------------------- | ---- | ------- | ------------------------------------------- |
| `validating.compiled_infer` | bool | False   | Opt eval/inference into the compile path    |
| `validating.tf32_infer`     | bool | False   | Use TF32 `high` for eval/inference forwards |
| `validating.amp_infer`      | bool | False   | Use bf16 AMP for eval/inference forwards    |

### 20.4 Environment Variables

| Variable              | Effect                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DP_TRITON_INFER=N`   | Triton inference kernel level, `N` in `{0, 1, 2, 3}` (eval only; composes with compile and `.pt2` freeze). Levels are cumulative. `0`: disabled. `1`: universal kernels that need no launch-configuration table — block-diagonal rotation, radial degree mixing, the `SO2Linear` block GEMM, Wigner-D monomial bases (`wigner_monomials`), segmented attention value-aggregation (`flash_atten`), segmented force/virial assembly (`force_assembly`). `2`: adds the kernels configured from the per-GPU tables served by `triton/tile_configs.py` — the fused SO(2) value path (`so2_value_path`) and the edge-block backward kernels; an unresolved key (untuned GPU or untuned shape) falls back to the level-1 kernel or a spill-safe configuration, and the `.pt2` freeze path auto-tunes uncovered keys on the local GPU before tracing. `3`: adds the fp16x3 tensor-core mixing stack (`so2_stack_fp16x3`) on shapes whose configuration passed the fp64 validation sweep; unswept shapes keep the fp32 stack. Only the numeric levels are accepted. |
| `DP_COMPILE_INFER=1`  | Opt eval/inference into the compile path                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `DP_TF32_INFER=0/1/2` | Eval/inference TF32 policy: `0=highest`, `1=high`, `2=medium`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `DP_AMP_INFER=1`      | Enable bf16 AMP inside the descriptor interaction blocks during eval/inference when `use_amp=true`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |

### 20.5 FFN Width Auto-Resolution

When `ffn_neurons = 0`, the hidden width is computed from `channels`:

- With GLU: `ceil_to_32(8/3 × channels)` — e.g., channels=64 → 192, channels=128 → 352
- Without GLU: `ceil_to_32(4 × channels)` — e.g., channels=64 → 256, channels=128 → 512

______________________________________________________________________

## 21. Testing and Validation

### 21.1 Test Organization

| File                                            | Scope                                                                                                  |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `source/tests/pt/model/test_descriptor_sezm.py` | Descriptor: forward pass, serialization, smoothness, SFPG                                              |
| `source/tests/pt/model/test_sezm_model.py`      | Model: compile path, energy/force correctness, output-bias statistics, edge-force-scatter finite diff. |

### 21.2 PES Smoothness Validation

The primary smoothness test uses direct **total-energy scans** rather than force-vs-finite-difference checks. A force check can pass even when the PES contains non-physical kinks.

**Probe setup:** An eight-atom two-sublattice template in fractional coordinates:

```
[0,0,0], [0,½,½], [½,0,½], [½,½,0], [½,½,½], [½,0,0], [0,½,0], [0,0,½]
```

The cubic lattice is scaled so the nearest-neighbor distance matches three boundary conditions:

1. **Near-cutoff** (`r_nn = 4.95 Å`, `rcut = 5.0 Å`): probes the cutoff envelope
1. **Inner boundary** (`r_nn = r_inner`): probes the InnerClamp transition (with ZBL)
1. **Outer boundary** (`r_nn = r_outer`): probes the bridging-to-identity transition (with ZBL)

Atom 0 is displaced along `x` over `[−0.1, 0.1]` Å. The test validates:

1. The second derivative maintains one sign across the scan (no inflection points)
1. The first derivative changes sign once at the center (single extremum)
1. The energy curve is a single smooth bowl

With randomized model weights, the non-bridged curve may be either concave or convex; both are acceptable. Bridged probes should be convex (ZBL repulsion dominates).

### 21.3 SFPG Invariance Test

A three-atom setup verifies that the SFPG correctly freezes information propagation:

1. Fix the frozen-partner atom on a constant-radius sphere inside the frozen zone
1. Slide the partner along the sphere through multiple angular positions
1. Assert the anchor atom's energy is invariant (span < `1e-10` in fp64)
1. Ablation: clear `bridging_switch` and verify the span jumps by orders of magnitude

### 21.4 Caching Strategy

Triton rotation-kernel correctness is validated by comparing the block-diagonal Triton output against the eager PyTorch reference (`rotate_to_local` / `rotate_back`) with tight numerical tolerances, covering forward and backward for both rotation directions across `lmax` 2–5 (`mmax=1`), plus a composability check that the compiled (`make_fx`) forces match eager.

## 22. LoRA Fine-Tuning

LoRA low-rank adapters let you fine-tune a pre-trained SeZM checkpoint on a downstream dataset (domain shift, new chemistries, condensed-phase data) while keeping the bulk of the backbone frozen. Implementation lives in `deepmd/pt/model/descriptor/sezm_nn/lora.py`.

### 22.1 Design Principles

- **Injection sites**: only the two large equivariant operators, `SO3Linear` and `SO2Linear`, receive LoRA adapters. Every other trainable layer is either fully unfrozen (small params, see the freeze/unfreeze table below) or kept frozen.
- **Shape isomorphy**: adapter parameters `A`/`B` share the base weight's batch structure, so the existing `HybridMuon` `muon_mode="slice"` routing applies uniformly: every `l`-block (SO3) / `m`-group (SO2) gets its own Newton-Schulz update.
- **Forward-path transparency**: `LoRASO3` / `LoRASO2` override the single weight-construction entry point of their base class and fold `ΔW = BA · scaling` into the *effective* weight before the base's single large `einsum`. Forward FLOPs match the base; overhead comes only from an `O(rank)` weight-side matmul that is independent of the edge/node count (≤ ~0.4 % for SO3 and ~0.1 % for SO2).
- **Equivariance**: per-`l` LoRA preserves SO(3) equivariance by construction (no cross-`l` mixing). Per-`|m|`-group LoRA preserves SO(2) equivariance because the block-diagonal 2×2 coupling `[[W_u, −W_v], [W_v, W_u]]` stays a legal complex linear after absorbing `ΔW_u = B_u A` and `ΔW_v = B_v A` (where `B = [B_u; B_v]` splits along the output axis).

### 22.2 Freeze / Unfreeze Policy

| Module / parameter                                                                                                                                                                                                                                            | Treatment             | Rationale                                                                                                                                                                                                                                             |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SO3Linear.weight`                                                                                                                                                                                                                                            | LoRA (base frozen)    | Main equivariant matrix                                                                                                                                                                                                                               |
| `SO2Linear.weight_m0`, `weight_m[i]`                                                                                                                                                                                                                          | LoRA (base frozen)    | Main equivariant matrices                                                                                                                                                                                                                             |
| `fitting_net` (incl. `dens_fitting_net` if present)                                                                                                                                                                                                           | Fully unfrozen        | Absorbs energy/force scale & zero-point shift                                                                                                                                                                                                         |
| `radial_embedding` (`RadialMLP`)                                                                                                                                                                                                                              | Fully unfrozen        | RBF response drifts with bond-length distribution                                                                                                                                                                                                     |
| `env_seed_embedding` (except `env_type_embed.adam_type_embedding`)                                                                                                                                                                                            | Fully unfrozen        | Local geometry → FiLM, domain-sensitive                                                                                                                                                                                                               |
| `film_scale_norm`, `film_shift_norm`, `film_*_strength_log`                                                                                                                                                                                                   | Fully unfrozen        | FiLM branch                                                                                                                                                                                                                                           |
| descriptor-level `final_full_attn_res` / `final_block_attn_res`                                                                                                                                                                                               | Fully unfrozen        | Small attention residuals                                                                                                                                                                                                                             |
| each block's `full_attn_res_so2` / `full_attn_res_ffns`, `block_attn_res_so2` / `block_attn_res_ffns`, `so2_conv.attn_q_proj` / `attn_k_proj` / `attn_qk_norm` / `attn_output_gate_norm` / `focus_compete_norm` / `radial_hidden_proj` / `so2_layer_attn_res` | Fully unfrozen        | Small attention projections + norms                                                                                                                                                                                                                   |
| leaf name ∈ {`adam_scale`, `adam_so2_layer_scales`, `adam_ffn_layer_scales`, `adamw_attn_logit_w`, `adamw_attn_z_bias_raw`, `adamw_attn_gate_w`, `adamw_focus_compete_w`, `adamw_pseudo_query`, `focus_compete_bias`}                                         | Fully unfrozen        | Small routed params, zero-cost. Matching is model-wide: every RMSNorm scale named `adam_scale` (per-block `pre/post_so2_norm`, `pre/post_ffn_norms`, `so2_inter_norms`, `key_norm` in every `DepthAttnRes`, radial-MLP norms, etc.) becomes trainable |
| any leaf name containing `bias`                                                                                                                                                                                                                               | Fully unfrozen        | Constant offsets absorb domain mean shift. Includes the LoRA-preserved `SO3Linear.bias` / `SO2Linear.bias0` and every norm bias (`EquivariantRMSNorm.bias`, `ReducedEquivariantRMSNorm.bias0`) across the model                                       |
| `type_embedding.adam_type_embedding`                                                                                                                                                                                                                          | **Frozen** (override) | Already converged on all elements                                                                                                                                                                                                                     |
| `radial_basis.adam_freqs`                                                                                                                                                                                                                                     | **Frozen** (override) | Trainable radial basis parameters already converged                                                                                                                                                                                                   |
| anything inside a `GatedActivation` submodule                                                                                                                                                                                                                 | **Frozen** (override) | Downstream gate patterns are stable; also shields `gate_linear.bias` from the generic "any bias unfrozen" rule                                                                                                                                        |
| `GIE`, `InnerClamp`, `BridgingSwitch`, `C3CutoffEnvelope`, `WignerDCalculator`, `InnerPotential`                                                                                                                                                              | No trainable params   | —                                                                                                                                                                                                                                                     |

### 22.3 Optimizer Routing

`HybridMuon.get_adam_route` decides the route by the trailing non-numeric segment of the parameter name. The LoRA parameter names (`A_by_l`, `B_by_l`, `A_m0`, `B_m0`, `A_m`, `B_m`) deliberately do **not** start with `adam_` / `adamw_` and do not contain `bias`, so they fall into the `muon` branch. The slice-mode matrix view then gives:

- `A_by_l` (shape `(L+1, R, C_in)`): `batch=L+1, rows=R, cols=C_in` → per-`l` Newton-Schulz on a squat rectangular matrix.
- `B_by_l` (shape `(L+1, F·C_out, R)`): `batch=L+1, rows=F·C_out, cols=R` → per-`l` NS on a tall rectangle.
- `A_m0`, `B_m0`, `A_m[i]`, `B_m[i]`: 2D matrices, single-matrix NS (`batch=1`).

`B` is initialised to zero, so the first forward is exactly the base forward; backward produces `grad_A = 0` and non-zero `grad_B`, and Muon updates `B` first. From the second step onward both `A` and `B` receive non-zero gradients, each processed by the standard rectangular-NS path with the existing `lr_adjust` / Magma damping logic.

### 22.4 Checkpoint Handling

Two distinct save paths coexist during a LoRA fine-tune run:

- **mid-train `latest_model-{step}.pt`** — written by `Trainer.save_model`. Contains the LoRA `A`/`B` keys, optimizer state, EMA shadow. Meant for resume. `_extra_state.model_params` keeps the `lora` block so `Trainer.__init__` re-triggers LoRA injection after loading.
- **best top-K ckpt from full validation** — written by the new `Trainer.save_model_merged`. Produces a *plain SeZM checkpoint*: `A`/`B` keys removed, `ΔW` folded into every `weight` / `weight_m0` / `weight_m.*`, `_extra_state.model_params.lora` stripped, no optimizer state, no EMA. Ready for `deep_pot` / LAMMPS deployment or for re-training a plain SeZM from the merged starting point.

The merged save path is wired automatically: `Trainer.__init__` caches `self._lora_enabled` after injection, and the training loop passes `save_model_merged` as the `save_checkpoint` callback to `full_validator` and `ema_full_validator` whenever LoRA is active.

Offline merge is also available via `merge_lora_into_base(model)` (destructive, rewrites submodules in place) or `build_merged_state_dict(module)` (non-destructive, produces a flat state dict). `strip_lora_from_extra_state` is the complementary helper for cleaning `_extra_state.model_params`.

### 22.5 Configuration Example

```json
"model": {
  "type": "dpa4",
  "type_map": ["O", "H"],
  "descriptor": {...},
  "fitting_net": {...},
  "lora": {
    "rank": 16,
    "alpha": 16.0
  }
}
```

`rank` is required; `alpha` is optional and defaults to `rank` (scaling `= 1.0`). Typical usage:

```bash
dp --pt train lora_ft.json --finetune pretrained.pt
```

See `examples/water/dpa4/lora_ft.json` for a full example input.

### 22.6 Test Coverage

- `TestLoRASO3Adapter` / `TestLoRASO2Adapter` — merge numerical parity (`merge_into_base` forward equals LoRA forward), SO(2) z-rotation equivariance after LoRA injection.
- `TestApplyLoRAToSeZM` — end-to-end injection on a minimal SeZM: subclass replacement, base weights frozen, adapters trainable, full-unfreeze list, override-freeze for type embedding / radial frequencies / `GatedActivation`.
- `TestBuildMergedStateDict` — merged-state-dict key set equals a never-LoRA'ed sibling, weight values equal `W + ΔW`.

## 23. Exporting SeZM to .pt2 for LAMMPS

SeZM compiles its force-loss graph through `make_fx` + `torch.compile`, a pipeline that TorchScript cannot represent because a nested `autograd.grad(create_graph=True)` lives inside `edge_energy_deriv`. AOTInductor (`.pt2`) is the official successor to `torch.jit.script` on PyTorch ≥2.11 and is the format `DeepPotPTExpt::init` already consumes, so the pt backend routes SeZM checkpoints to an AOTInductor freeze path while every non-SeZM checkpoint keeps the standard `torch.jit.script` branch.

### 23.1 CLI Usage

```bash
dp --pt freeze -c run_dir -o frozen_model
```

`main.freeze` calls `is_sezm_checkpoint`; when the `type` field in `_extra_state.model_params` is `"SeZM"` it rewrites the output suffix to `.pt2` and hands off to `freeze_sezm_to_pt2`. The dispatch is transparent to the user.

The resulting archive:

```
frozen_model.pt2  (ZIP archive)
├── data/aotinductor/model/*.cubin / *.so   # Inductor AOT shared library
├── extra/metadata.json                     # flat fields for DeepPotPTExpt.cc
└── extra/model_def_script.json             # raw training params (verbatim)
```

`extra/metadata.json` matches the reader contract at `source/api_cc/src/DeepPotPTExpt.cc` lines ~88-137 **and** the metadata-only load path of `deepmd.pt_expt.infer.deep_eval.DeepEval._init_from_metadata`. Emitted fields:

- `type_map`, `rcut`, `sel`: descriptor geometry.
- `dim_fparam`, `dim_aparam`, `has_default_fparam`, `default_fparam`: fitting parameter wiring, including optional frame parameters used by the SeZM fitting head.
- `mixed_types`, `is_spin`: SeZM is `True` / `False`.
- `output_keys`: insertion-order names of the tensors returned by the traced graph (currently `energy`, `mask`, `energy_redu`, `energy_derv_r`, `energy_derv_c`, `energy_derv_c_redu`). `DeepPotPTExpt::extract_outputs` zips this list with the flat output vector of `AOTIModelPackageLoader::run`.
- `fitting_output_defs`: serialised `OutputVariableDef` entries so the C++ side can rebuild `ModelOutputDef` without importing any Python class; the Python `DeepEval._init_from_metadata` consumes the same list.
- `sel_type`: selected atom types (equivalent of `atomic_model.get_sel_type()`); feeds `DeepEval.get_sel_type()` in metadata-only mode.

`extra/model_def_script.json` preserves the training `model_params` verbatim so downstream tooling (follow-up fine-tune, manual inspection) can recover the full knob set that produced the checkpoint.

Note that SeZM does **not** ship `extra/model.json` — it cannot, because the descriptor does not travel through the `dpmodel` array-api-compat layer. The pt_expt `DeepEval` loader treats `model.json` as optional (see `_init_from_metadata`); every public inference path on a SeZM `.pt2` therefore goes through `metadata.json` exclusively.

### 23.2 Dtype Contract

The AOTI package is compiled with **fp64 inputs and fp64 outputs**, regardless of the checkpoint's internal compute dtype. This matches `source/api_cc/src/DeepPotPTExpt.cc:222-225`, where LAMMPS coordinates are unconditionally cast to `torch::kFloat64` — AOTInductor performs no auto-cast, so any mismatch would fail at load time. `freeze` never reads or writes the `descriptor.precision` / `fitting_net.precision` fields; SeZM's own `_input_type_cast` / `_output_type_cast` bridge fp64 I/O to whatever internal dtype the checkpoint uses.

### 23.3 Deployment Paths

A SeZM `.pt2` is consumable by every DeePMD-kit inference front-end that speaks the `.pt2` contract — the freeze step produces exactly one artefact, not three. The three supported entry points are:

**LAMMPS (`pair_style deepmd`).** No C++ change required: the `.pt2` suffix routes to `DeepPotPTExpt`, which forwards to `AOTIModelPackageLoader`.

```
LAMMPS coord (fp64) → DeepPotPTExpt::compute → AOTIModelPackageLoader::run → .pt2 .so
                                                                             ↓
                   DeepPotPTExpt::extract_outputs ← keyed outputs (output_keys)
```

A minimal end-to-end recipe ships in `examples/water/dpa4/lmp/`.

**Python `DeepPot` / `dp test` / ASE calculator.** Transparent through the standard deepmd Python API:

```python
from deepmd.infer import DeepPot

pot = DeepPot("frozen_model.pt2")
e, f, v = pot.eval(coord, cell, atype)  # (nf, 1), (nf, nloc, 3), (nf, 9)
e, f, v, atom_e, atom_v = pot.eval(..., atomic=True)
```

Equivalently:

```bash
dp test -m frozen_model.pt2 -s system/           # numerical validation against labels
```

Under the hood `DeepPot(".pt2")` dispatches to `deepmd.pt_expt.infer.deep_eval.DeepEval`; since the metadata-only-load patch in that module treats `extra/model.json` as optional, the same loader that consumes pt_expt's own `.pt2` also consumes SeZM's. The public deepmd ASE calculator (`from deepmd.calculator import DP`) is a thin wrapper around `DeepPot` and therefore works out of the box.

**Raw AOTI runtime.** For research scripts that want to skip the deepmd stack entirely, `torch._inductor.aoti_load_package` loads the archive directly; the caller builds the lower-interface inputs (extended coords, nlist, mapping) themselves. `source/tests/pt/model/test_sezm_export.py::TestSeZMExportArchive::test_aoti_load_and_run_returns_finite_outputs` demonstrates the pattern.

Device-locking caveat: `.pt2` packages bake in the freeze host's CPU ISA, GPU compute capability, and libtorch ABI. Re-freeze on the host that will run inference (Python or LAMMPS).

### 23.4 Limitations (v1)

- **No multi-task**: `freeze_sezm_to_pt2` rejects any checkpoint carrying `model_dict` in `_extra_state.model_params`.
- **No `head` selection**: `head=None` is the only accepted value.
- **No dpmodel-level introspection**: `dp test` and `DeepPot.eval` work, but methods that need a deserialised `dpmodel` instance (`DeepEval.eval_descriptor`, `eval_fitting_last_layer`, `eval_typeebd`) raise `NotImplementedError` on SeZM `.pt2` archives since SeZM does not ship `extra/model.json`. This is a deliberate trade-off: SeZM's internals are too PyTorch-specific to round-trip through the array-api-compat `dpmodel` layer.

### 23.5 Implementation Layer

- `deepmd/pt/model/model/sezm_model.py` adds `SeZMModel.forward_common_lower_exportable`: detaches the coordinate and reinstates `requires_grad=True` inside the traced closure so LAMMPS's non-leaf fp64 tensors get a fresh grad endpoint for the inner `autograd.grad`, and decomposes `silu_backward` the same way the training compile path does.
- `deepmd/pt/entrypoints/freeze_pt2.py` owns ckpt loading, sample-input synthesis (`_resolve_nframes` picks a duck-sizing-safe `nframes`), the `torch.export.export` call, AOTInductor compilation, and the `extra/` sidecar writes (including `sel_type` — the single field that `DeepEval._init_from_metadata` needs beyond what C++ reads). At `DP_TRITON_INFER >= 2` with a CUDA target it also runs `_tune_triton_configs` before tracing: launch-table keys of the checkpoint's shapes that are uncovered for the local GPU are swept and registered in-process, and the fused value-path entries are rebound, so the compiled package bakes deployment-tuned launches (see [`triton_op_record.md`](./triton_op_record.md) §4).
- `deepmd/pt/model/descriptor/sezm_nn/triton/so2_rotation.py` and `triton/radial_mix.py` expose the Triton inference kernels through `torch.library.triton_op` and launch them with `wrap_triton`. The op remains opaque during the CPU `make_fx` trace, but Inductor sees through it during CUDA AOTI lowering after `move_to_device_pass`, so dynamic edge dimensions stay symbolic while the Triton cubins are packaged for the C++ runtime.
- `deepmd/pt/entrypoints/main.py::freeze` detects SeZM checkpoints via `is_sezm_checkpoint` and dispatches; every non-SeZM path keeps the legacy `torch.jit.script` behaviour.
- On the loader side, `deepmd/pt_expt/infer/deep_eval.py` treats `extra/model.json` as optional: absent → take the metadata-only path; present → reconstruct the dpmodel for dpmodel-level introspection. The same loader now serves both pt_expt-native and SeZM archives without a code branch for "which backend produced the `.pt2`".

### 23.6 Test Coverage

`source/tests/pt/model/test_sezm_export.py` ships self-contained suites (no external checkpoint; a tiny fp64 SeZM is built on the fly):

- `TestSeZMExportPipeline`: traces `forward_common_lower` via `forward_common_lower_exportable`, exports to an `ExportedProgram`, saves / loads `.pte`, and asserts *bitwise* parity (`rtol=1e-10, atol=1e-10`) between eager, traced and loaded outputs on both the trace shape and a different inference shape. Mirrors `pt_expt/model/test_export_pipeline.py`.
- `TestSeZMWithCommExportPipeline`: the same trace / export / `.pte` round-trip for the parallel `forward_common_lower_exportable_with_comm` graph, asserting fp64 parity against the eager parallel forward (self-send `comm_dict`) on the trace shape and a different owned-atom count.
- `TestSeZMExportArchive`: drives `freeze_sezm_to_pt2` end-to-end on a tiny synthetic ckpt, verifies the ZIP layout, the `metadata.json` field coverage (including `sel_type`), and that the AOTI loader produces finite outputs.
- `TestSeZMViaDeepPot`: loads the frozen archive through the standard `deepmd.infer.DeepPot` entry and asserts `eval(coord, cell, atype)` matches eager `SeZMModel.forward(...)` at `rtol=1e-5, atol=1e-7`. This is the contract `dp test` and the ASE calculator rely on. Tolerance is looser than the `.pte` pipeline because AOTInductor fuses pointwise / reduction kernels and the fused accumulation order differs from eager.
- `TestSeZMFreezeGuards`: error paths — `is_sezm_checkpoint` rejects a non-SeZM file, and `freeze_sezm_to_pt2` raises `NotImplementedError` on `head != None` and on `model_dict` checkpoints. The `dens` mode also raises inside `forward_common_lower_exportable`; it is not exercised in CI because driving a `dens` model through LAMMPS is not a supported workflow.

Every suite clears the pt-test sentinel default device (`cuda:9999999`) inside a context manager so PyTorch 2.11's AOTI / export passes can create their internal unnamed tensors without tripping the guard — matching the `pt_expt/test_change_bias.py` pattern. `TestSeZMViaDeepPot` additionally pins `pt_expt.utils.env.DEVICE` to CPU so the CPU-compiled `.pt2` meets the device contract that `DeepEval._prepare_inputs` relies on (AOTI packages are device-locked).

### 23.7 LAMMPS Multi-GPU (MPI Domain Decomposition) Inference

SeZM keeps its edge-based contract under MPI domain decomposition: each rank owns `nloc` local atoms plus a halo of `nghost` ghosts, and the descriptor message passing runs on the extended node set with cross-rank ghost-feature exchange. The force/virial scatter is unchanged — `edge_scatter_index` already addresses the extended `[0, nf·nall)` domain, so LAMMPS reverse-communicates ghost forces under `newton_pair` exactly as for any pair style. The only addition is the descriptor's forward-side ghost exchange.

**Node-set convention.** The single-domain path folds each ghost neighbour onto its local owner via `mapping`, so `edge_index` stays in `[0, nf·nloc)` and the node set is the owned atoms. The parallel path indexes the extended atoms directly: `edge_index = edge_scatter_index` (both `[0, nf·nall)`), the node set spans local owners followed by ghosts, and `extended_atype` `(nf, nall)` supplies ghost types for the edge-type and environment-seed features. `atype` stays `(nf, nloc)` so the owned-atom count remains a clean export dimension and drives fitting / the energy read-out.

**Ghost exchange.** Each interaction block refreshes ghost-node features from their owner ranks through the opaque `deepmd_export::border_op` (`exchange_ghost_features` in `sezm_nn/block.py`), applied at the SO(2) convolution input — the descriptor's only cross-node operation. SeZM node features are SO(3) coefficients in the shared global frame, so a ghost and its owner carry identical features and the per-row owner→ghost copy is exact and equivariance-preserving (inference fixes `random_gamma`, so the frame is deterministic). Placing the exchange at the convolution input (rather than on the block input) is what makes the attention-residual paths correct: the depth-attention aggregation is per-node and may leave stale ghost rows in its output, but those rows are overwritten by the exchange immediately before the convolution reads them, so owned destinations always gather up-to-date neighbours. Initial type/FiLM/GIE features are computed only for owned nodes; ghost rows start partial and are filled by the first exchanging block (see the schedule below).

**Exchange schedule.** A block communicates only when its SO(2) convolution reads neighbour rows the local rank cannot rebuild (`DescrptSeZM._block_comm`). Block 0 reads the initial node state, whose ghost rows a rank reproduces from `extended_atype` (the type embedding), so it exchanges only when env-seed / GIE folds neighbour-environment information into them. Every later block reads a previous block's output, which a rank cannot reproduce for ghosts (they receive no local messages), so it always exchanges. A purely local model — `use_env_seed=False` with a single block — therefore performs zero communication under domain decomposition, retaining its single-pass speed. The exported with-comm artifact still carries the (then unused) comm-tensor inputs, so the C++ dispatch and artifact layout are unchanged.

**Gradient exchange.** `border_op` carries a registered backward (`border_op_backward`) that is the exact reverse of the forward gather. The forward overwrites every ghost row (`g[ghost] = g[owner]`), so the backward routes each ghost row's output gradient to its owner and **zeros the ghost input rows** — a ghost INPUT never reaches the output, so its gradient is exactly zero. The single `autograd.grad(energy, [edge_vec, spin])` in `edge_energy_deriv` then threads correctly through the exchange: with every rank in lockstep, each edge's cross-rank contribution accumulates onto the owning rank's `edge_vec`, and each owned atom collects the spin gradient of all its ghost copies, so forces stay conservative and the per-node magnetic force is exact without a bespoke reduction. The ghost-row zeroing is load-bearing only for per-node leaves: a wrong ghost-input gradient dead-ends for the conservative force (ghost nodes are never edge centres, so their initial features carry no `edge_vec` dependence) but would otherwise corrupt the magnetic force, whose leaf is the per-node spin carried directly in the node state.

**Dual artifact.** `freeze_sezm_to_pt2` compiles a second AOTInductor package, nested as `model/extra/forward_lower_with_comm.pt2`, whenever `SeZMModel.supports_edge_parallel()` holds. `metadata.json` records `has_comm_artifact`. The with-comm graph is traced by `forward_common_lower_exportable_with_comm`, whose positional signature fixes the C++ ABI:

```
coord (nf=1, nall, 3), atype (1, nloc), extended_atype (1, nall),
edge_index (2, E), edge_vec (E, 3), edge_scatter_index (2, E), edge_mask (E,),
[fparam], [aparam], [charge_spin],
send_list, send_proc, recv_proc, send_num, recv_num, communicator, nlocal, nghost
```

The frame axis is fixed at one (LAMMPS single-frame inference); `nall`, `nloc` and `nedge` stay dynamic; the eight communication tensors are static (`nswap` is fixed at LAMMPS init). The optional `fparam` / `aparam` / `charge_spin` slots are present only when their dimensions are non-zero, matching the regular artifact.

**C++ dispatch.** `DeepPotPTExpt::compute` selects the with-comm artifact when `lmp_list.nprocs > 1` and `has_comm_artifact_`. For the edge schema it then builds the extended-index edge tensors (`createEdgeTensors(..., fold_to_local=false)` keeps ghost-neighbour edges instead of folding them onto a local owner), assembles the eight communication tensors with `build_comm_tensors_positional`, and calls `run_model_edges_with_comm` against the nested artifact. Single-rank edge inference keeps the folded path and still requires `atom_modify map yes`. The communication tensors are `clone`d into torch-allocated storage rather than left as `from_blob` views over the LAMMPS swap arrays: the artifact is compiled assuming 16-byte-aligned inputs (the freeze-time samples are torch-allocated), so unaligned views would trigger a per-step "input not aligned, copying" warning and an extra copy inside AOTInductor. The clones are `nswap`-sized, so the one-time alignment cost is negligible, and the pointer values copied into the send-list tensor still address the live swap buffers.

**Device placement.** Each rank loads the artifact on `cuda:(node_rank % visible_gpu_count)`, where `node_rank` is the node-local MPI rank from `PairDeepBaseModel::get_node_rank()`. A multi-GPU launch must therefore expose every device the ranks may use (e.g. `CUDA_VISIBLE_DEVICES=0,1,2,3` for four ranks); restricting visibility to a single device collapses all ranks onto it, which still yields correct forces but serialises the GPU work and multiplies that device's memory.

**Limitations (v1).** Multi-rank is restricted to the conservative non-spin path without analytical bridging:

- **ZBL / Source Freeze Propagation bridging** is excluded — the SFPG gate folds each node's entire outgoing-edge set, which a single rank cannot observe for ghost owners. `supports_edge_parallel()` returns `False` when `inter_potential` is set or the descriptor carries a bridging switch, so no with-comm artifact is emitted and multi-rank C++ inference fails fast.
- **Spin** uses the nlist lower interface (`lower_input_kind = "nlist"`), so the edge with-comm artifact does not apply; multi-rank spin fails fast.

**Validation.** `source/tests/pt/model/test_sezm_parallel.py` emulates one rank in a single process by driving `border_op` with a self-send swap whose send-list maps each ghost slot to its owner, reducing the exchange to the folded gather. It asserts the parallel path reproduces the single-domain path on owned atoms for the descriptor output and for energy / force / virial (CPU to fp64 round-off; CUDA under a looser atomic-scatter tolerance), across the fast, `full_attn_res` and `block_attn_res` paths, and that `supports_edge_parallel()` gates bridging out. The parity tests perturb the weights away from SeZM's near-identity initialization first: an untrained model has near-zero message passing, which would mask any ghost-exchange error proportional to the convolution output (notably in the attention-residual paths). `test_sezm_export.py::TestSeZMWithCommExportPipeline` exports the with-comm graph and asserts `.pte` round-trip parity against the eager parallel forward on the trace shape and a different owned-atom count; the freeze guards verify the nested artifact and `has_comm_artifact` flag are emitted for plain models and withheld for spin.

## 24. Embedding Extraction

### 24.1 Purpose

The `embedding` command exposes three representations of a structure, all produced by a single forward pass with **no force or virial autograd**:

- `descriptor` `(nf, nloc, channels)` — the per-atom scalar descriptor, i.e. the same tensor consumed by the fitting network. It summarizes each atom's local environment.
- `atomic_feature` `(nf, nloc, neuron[-1])` — the activation after the last fitting hidden layer, before the final linear projection `W`.
- `structural_feature` `(nf, neuron[-1])` — the masked atom-sum of `atomic_feature`, a whole-structure summary.

This is deliberately separate from the legacy `eval-desc` / `eval_descriptor` / `eval_fitting_last_layer` hook machinery, which is left untouched.

### 24.2 Pooling Derivation

The atomic energy is an affine map of the last hidden activation, $E_i = \mathbf{w}\cdot \mathbf{a}_i + b_i$, where $\mathbf{a}_i$ is `atomic_feature` of atom $i$. The total energy is therefore

```math
E = \sum_i m_i E_i = \mathbf{w}\cdot\Big(\sum_i m_i \mathbf{a}_i\Big) + \sum_i m_i b_i,
```

where $m_i$ is the atom mask (zero for padding and excluded atoms). The term $\sum_i m_i \mathbf{a}_i$ is exactly `structural_feature`. Pooling before the output projection (instead of after) is valid by associativity and yields a fixed-width descriptor of the whole structure whose projection through the fitting output layer reproduces the bias-free total energy.

### 24.3 Implementation

The path is self-contained (it does not reuse the `eval-desc` hook machinery) and reuses the existing edge-schema forward, branching out before the force computation:

- `GeneralFitting._forward_common` (`deepmd/pt/model/task/fitting.py`) gains a `return_atomic_feature` parameter. When set, it adds the last hidden activation (`call_until_last(...)`) to the result under the `atomic_feature` key. This is independent of the legacy `eval_return_middle_output` / `middle_output` hook (left untouched for the soon-to-be-deprecated `eval-desc`). `SeZMEnergyFittingNet._forward_common` and `_forward_case_film` thread the same parameter.
- `SeZMModel.core_compute(..., embedding_only=True)` (`deepmd/pt/model/model/sezm_model.py`) keeps `edge_vec` detached (no autograd leaf), runs the descriptor, computes the atom mask once (reused by both the embedding pooling and the energy masking), obtains `atomic_feature` via `fitting_net._forward_common(..., return_atomic_feature=True)`, and returns `{descriptor, atomic_feature, structural_feature}` -- all cast to float32 -- before the `edge_energy_deriv` force/virial scatter.
- `forward_common` and `forward_common_lower` thread `embedding_only` through to `core_compute`; `forward_embedding(coord, atype, box, ...)` is the public entry. It reuses the standard `ener` edge-schema construction and is restricted to the `ener` mode (it raises `NotImplementedError` for `dens`).

### 24.4 Compile Routing

The embedding graph honors `DP_COMPILE_INFER` through the same make_fx / AOT machinery as the energy graph, but is eval-only with no coordinate correction, so it needs only a single compiled graph -- not a keyed cache:

- per-instance `compiled_embedding` and `_embedding_task_buf_order` single slots (installed via `object.__setattr__`, kept outside the `nn.Module` tree).
- The energy cache and its `(training, has_coord_corr)` `cache_key` are **unchanged**.
- `trace_and_compile(..., embedding_only=True)` traces `core_compute(embedding_only=True)` (reusing the dict-output repack and task-buffer promotion) and stores the result in the single slot, skipping the cross-task module cache and the pending-compile timing log.
- `reset_head_for_mode("ener")` resets `compiled_embedding` alongside the energy cache because both read the same fitting head.

### 24.5 Inference API and CLI

- `deepmd/pt/infer/deep_eval.py::DeepEval.eval_embedding` builds the input tensors, runs `forward_embedding` under `torch.no_grad()`, and batches over frames with `AutoBatchSize.execute_all` (the three outputs are concatenated on the frame axis). It dispatches on `_uses_edge_schema`; non-SeZM models raise `NotImplementedError`.
- `deepmd/infer/deep_eval.py` adds the abstract `eval_embedding` to `DeepEvalBackend` and the `DeepPot` wrapper method.
- `deepmd/entrypoints/embedding.py` iterates systems and writes a single HDF5 file. Each system is a group holding `descriptor`, `atomic_feature`, `structural_feature` (float32) plus an `atom_types` (nframes, natoms) dataset -- all gzip level 9 + byte shuffle -- an `nframes` attribute, and a `system` attribute recording the source directory; the model `type_map` is a file-level attribute. Group names are de-duplicated from the system base name. The CLI is `dp embed -m <model.pt> -s <system> -o embedding.hdf5`. It operates on the `.pt` checkpoint; the frozen `.pt2` package is not supported.

### 24.6 Scope and Test Coverage

Only SeZM/DPA4 energy models are supported; spin models and standard descriptors raise `NotImplementedError`. `source/tests/pt/model/test_embedding.py` covers, self-contained (without the `eval-desc` machinery):

- the compiled embedding graph (`DP_COMPILE_INFER=1`) matches the eager output for all three embeddings.
- `output_layer(atomic_feature)` reconstructs the per-atom energy and `output_layer(structural_feature)` reconstructs the total energy (with zeroed bias), validating both the atomic feature and the pooled feature against the independent energy forward.
- the `DeepEval` API returns the expected shapes in float32 for a SeZM checkpoint and raises `NotImplementedError` for a standard `se_e2_a` checkpoint.

## 25. Running SeZM in nvalchemi-toolkit (MD)

### 25.1 Purpose and Scope

`deepmd/pt/nvalchemi/dpa4wrapper.py` exposes `DPA4Wrapper`, a thin adapter that lets NVIDIA's [`nvalchemi-toolkit`](https://github.com/NVIDIA/nvalchemi-toolkit) molecular-dynamics engines (NVE / NVT / NPT / FIRE, ...) drive a trained SeZM / DPA-4 model. It is an **adapter, not a port**: the `SeZMModel` runs unchanged as a black-box `nn.Module`; the wrapper only translates between nvalchemi's graph `Batch` and SeZM's sparse-edge lower interface. `nvalchemi-toolkit` is an **optional** dependency — `deepmd/pt/nvalchemi/__init__.py` raises a clear install hint if it is missing, and nothing in the core package imports it.

The wrapper subclasses `(nn.Module, BaseModelMixin)` so it satisfies the nvalchemi model contract: a `ModelConfig`, an `embedding_shapes` property, `compute_embeddings`, and a `forward(batch) -> ModelOutputs` that returns a fully-adapted output dict. The dynamics engine's `compute()` reads `energy` / `forces` / `stress` straight from that dict and detaches — the same pattern the bundled MACE / Lennard-Jones wrappers use.

### 25.2 The Integration Seam: `forward_lower`

The wrapper targets `SeZMModel.forward_lower` (Section 12), the conservative edge-based entry point whose only coordinate-dependent geometry is the per-edge displacement `edge_vec`. Forces, the global virial, and the per-atom virial all come from the single `autograd.grad(E, edge_vec)` inside `edge_energy_deriv` (`deepmd/pt/model/model/transform_output.py`), so the wrapper does **not** need nvalchemi to differentiate through `positions` — it returns analytical conservative `forces` / `stress` directly. This is why `autograd_outputs` is empty (like the LJ wrapper), not `{forces, stress}` (like MACE).

Feeding nvalchemi's own neighbour list into `forward_lower` (rather than letting SeZM rebuild one) is the whole point: it keeps the `-ops` Warp neighbour kernel on the critical path instead of duplicating the work.

**Single-frame batching trick.** An nvalchemi `Batch` concatenates several (possibly heterogeneous) graphs into one node axis with a global `neighbor_list` that already carries per-graph node offsets. The wrapper presents the *entire* batch to SeZM as a single frame with `nf = 1`, `nloc = num_nodes`. Edges never cross graphs, so message passing is correct as-is, and per-graph energy / virial are recovered afterwards by segment-summing the per-atom outputs with `batch_idx`. This makes heterogeneous batches work with no special casing.

### 25.3 Conventions (the part that must be exactly right)

These were validated to float64 round-off against the native `forward`; getting any of them backwards yields plausible-but-wrong physics.

- **Edge direction.** nvalchemi COO `neighbor_list` rows are `[source, target]`, where `source` is the centre atom (its row in the neighbour matrix) and `target` is the neighbour; the integer image `neighbor_list_shifts` belongs to the `target`. SeZM uses `edge_index = [src, dst]` with `edge_vec = r_src - r_dst`, aggregating onto `dst` (the centre). The mapping is therefore `dst = source`, `src = target`:

  ```python
  edge_index = torch.stack([target, source])  # [src=neighbour, dst=centre]
  edge_vec = positions[target] - positions[source] + shift_vec
  ```

  This matches the MACE wrapper's `vectors = pos[receiver] - pos[sender] + shifts` with `(sender, receiver) = (source, target)`.

- **PBC shifts (no ghosts).** Periodic images enter only through `edge_vec`; there are no ghost atoms. The physical shift is `shift_vec[e] = neighbor_list_shifts[e] @ cell[graph_of_source_e]`. A multi-frame batch gathers the per-edge cell and contracts with `torch.einsum("eb,ebc->ec", ...)`; the single-frame case (the usual MD run, `cell.shape[0] == 1`) takes a fast path `shifts @ cell[0]`, a single `(E,3)·(3,3)` matmul that avoids materializing an `(E,3,3)` per-edge cell. `edge_scatter_index` equals `edge_index` because the force/virial scatter domain is exactly the local atoms.

- **Stress sign.** nvalchemi defines the LJ kernel virial as `W = -dE/dε` and the Cauchy stress as `σ = W / V`. SeZM's `energy_derv_c_redu` (`forward_lower["virial"]`) satisfies the same `W = -dE/dε` (pinned by `test_virial_matches_strain_finite_difference`), so the wrapper uses `stress = virial / volume` with **no sign flip**, identical to the LJ wrapper. Per-graph virial is the `batch_idx` segment-sum of `extended_virial` (requested via `do_atomic_virial=True`); volume is `|det(cell)|` per graph.

- **Type mapping.** Atomic numbers are mapped to SeZM type indices via a dense lookup table built from `model.get_type_map()` and the `ELEMENT_TO_Z` periodic table in `sezm_model.py` (a single device-side `index_select`, no host round-trips). Pass `atomic_number_to_type={Z: type_index, ...}` to override for non-element type maps (e.g. placeholder names).

- **Sync-free hot path.** The per-step path does no host-device synchronization. Type validation (the `z.max()` bound check and the unmapped-species `any()` check, the only two `.item()`-style stalls) runs once and is memoized on the `atomic_numbers` tensor identity, since a dynamics run reuses that tensor every step while mutating only positions; the optional charge/spin condition is pre-built into a buffer rather than rebuilt from a Python list each step. This keeps every step fully async-enqueueable.

- **Eval mode.** `__init__` puts the model in `eval()` so the sparse-edge path stays eager and deterministic — `random_gamma` augmentation is training-only, and `should_use_compile()` is `False` in eval unless `DP_COMPILE_INFER` is set.

### 25.4 ModelConfig

```python
ModelConfig(
    outputs=frozenset({"energy", "forces", "stress"}),
    active_outputs={"energy", "forces"},  # + "stress" when compute_stress=True
    autograd_outputs=frozenset(),  # forces/stress are analytical (internal autograd)
    optional_inputs=frozenset({"cell", "neighbor_list_shifts"}),
    supports_pbc=True,
    needs_pbc=False,
    neighbor_config=NeighborConfig(
        cutoff=model.get_rcut(), format=NeighborListFormat.COO, half_list=False
    ),
)
```

`half_list=False` because SeZM consumes a full (directed) neighbour list. Stress requires a periodic `cell`; it is opt-in via `compute_stress=True` (or `model.set_config("active_outputs", {"energy", "forces", "stress"})`) so NVE/NVT runs skip the extra per-atom virial work.

### 25.5 Usage

```python
import torch
from deepmd.pt.nvalchemi import DPA4Wrapper
from nvalchemi.data import AtomicData, Batch
from nvalchemi.neighbors import compute_neighbors
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.integrators import NVE
from nvalchemi.hooks import NeighborListHook

# Load a .pt checkpoint (or pass an existing SeZMModel), or a frozen .pt2.
model = DPA4Wrapper.from_checkpoint("model.pt", device="cuda", compute_stress=True)

# One-shot inference.
data = AtomicData(atomic_numbers=z, positions=r, cell=cell, pbc=pbc)
batch = Batch.from_data_list([data], device="cuda")
compute_neighbors(batch, config=model.model_config.neighbor_config)
out = model(batch)  # {"energy": (B,1), "forces": (N,3), "stress": (B,3,3)}

# Molecular dynamics: register a COO NeighborListHook and run.
nl_hook = NeighborListHook(
    model.model_config.neighbor_config, stage=DynamicsStage.BEFORE_COMPUTE
)
nve = NVE(model, dt=0.5, n_steps=1000, hooks=[nl_hook])
batch = nve.run(batch)
```

`from_checkpoint` handles the standard `.pt` layout (`_extra_state.model_params` → `get_model` → `ModelWrapper.load_state_dict`) and the multi-task `head` selection; pass an already-built `SeZMModel` to `DPA4Wrapper(model, ...)` directly to skip checkpoint I/O.

**Acceleration and the `.pt2` backend.** `from_checkpoint` accepts both formats:

- **`.pt` eager** (default): runs `SeZMModel.forward_lower`.
- **`.pt` compiled**: export `DP_COMPILE_INFER=1` (optionally `DP_TRITON_INFER=2`) *before* `from_checkpoint`; SeZM's make_fx/AOT compile path is reused unchanged. The dynamic-shape graph absorbs the per-step neighbour-count changes with no recompiles, giving roughly 2.9x (compile) / 3.5–3.8x (compile + Triton) per-step speedup after a one-time ~70–95 s compile.
- **`.pt2`**: `from_checkpoint("model.pt2", ...)` loads the AOTInductor package via `aoti_load_package` and reads `model/extra/metadata.json` for the cutoff and type map (`self.model` is `None`, `self._aoti_runner` holds the callable). The package is fp64-I/O and device-locked to its freeze host; freezing with `DP_TRITON_INFER` set bakes the Triton kernels into the package as well, with launch configurations auto-tuned on the freeze GPU when the checkpoint's shape keys are not covered by the built-in tables (see [`triton_op_record.md`](./triton_op_record.md) §4). The wrapper feeds it the same edge schema and renames the raw lower outputs (`energy` / `energy_derv_r` / `energy_derv_c`) before `adapt_output`. Embeddings are not available on this backend.

nvalchemi's own `FusedStage.compile()` is deliberately **not** used: SeZM's inner `autograd.grad(..., retain_graph=True)` conflicts with AOTAutograd's donated-buffer optimization, and even when bypassed the step does not fuse (the inner autograd, `.item()`, and `requires_grad_` break the graph), so it runs slower than eager. Speed therefore comes from the SeZM-native compile above.

### 25.6 Validation

`source/tests/pt/model/test_sezm_nvalchemi.py` (skipped when `nvalchemi-toolkit` is absent) builds a random-weight fp64 model and asserts the wrapper reproduces SeZM's native `forward`:

- **Periodic** single system, including a pair that only interacts across the periodic boundary: energy / forces / virial parity.
- **Non-periodic** cluster (`cell=None`): parity, and no `stress` key.
- **Heterogeneous batch** (two graphs, different sizes *and* different cells): per-graph energy / forces / virial parity, validating the `batch_idx` segment reduction and the global `neighbor_list` node offsets.
- **Embeddings** and the **type-map override** / unknown-element error paths.

Force conservativeness itself is pinned by the native `forward` (FD-validated in `TestSeZMEdgeForceScatter`), so the wrapper tests assert parity against it rather than re-running finite differences.

```bash
OMP_NUM_THREADS=1 DP_INTER_OP_PARALLELISM_THREADS=0 DP_INTRA_OP_PARALLELISM_THREADS=0 \
    pytest source/tests/pt/model/test_sezm_nvalchemi.py -v
```

Runnable MD / single-point / relaxation examples live in `examples/water/dpa4/nvalchemi/` (see its `README.md`); the user-facing guide is `doc/inference/nvalchemi.md`.

### 25.7 Limitations (v1)

- **No nvalchemi-side compile.** Speed comes from SeZM's own compile (`DP_COMPILE_INFER`) or a frozen `.pt2`; nvalchemi's `FusedStage.compile()` is not viable here (see the acceleration note in 25.5).
- **Charge / spin** is forwarded as a single global `[charge, spin]` (`default_charge_spin`) because the batch is presented to SeZM as one frame; per-graph charge/spin in a heterogeneous batch is not supported.
- **`dens` mode** is not supported — `forward_lower` is `ener`-only.
- **Embeddings.** `compute_embeddings` attaches the per-atom descriptor (`node_embeddings`) and its sum-pool (`graph_embeddings`) via the `embedding_only` lower path; the fitting-net hidden features of Section 24 are not exposed through the nvalchemi interface.

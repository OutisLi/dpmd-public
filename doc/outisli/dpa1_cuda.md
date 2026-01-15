# Compressed DPA1 CUDA Inference

## 1. Scope

This document specifies the CUDA inference path for geometrically compressed
DPA1 descriptors (`se_atten_v2`, strip type embedding, `attn_layer == 0`) in the
`pt_expt` graph lower. The target workload is a forward energy evaluation
followed by one analytical backward for force and virial. Training and double
backward are outside this operator contract.

The numerical contract is fp32 model computation with TF32 disabled. Coordinates
and public graph geometry may remain fp64; operator boundaries cast the geometry
to the model precision and return the edge gradient in the input dtype where
autograd requires it.

The implementation consists of:

| Component                              | Location                                                                 |
| -------------------------------------- | ------------------------------------------------------------------------ |
| Compressed descriptor forward/backward | `source/op/pt/dpa1_graph_compress.cu`                                    |
| Fused energy fitting                   | `source/op/pt/graph_fitting.cu`                                          |
| Force and virial reduction             | `source/op/pt/edge_force_virial.cu`                                      |
| Uncompressed end-to-end inference      | `source/op/pt/dpa1_graph_energy_force.cu`                                |
| Python operator integration            | `deepmd/kernels/cuda/dpa1/graph_compress.py`                             |
| Graph contract                         | `deepmd/dpmodel/utils/neighbor_graph/graph.py`                           |
| C++/Kokkos graph ingestion             | `source/api_cc/include/commonPT.h`, `source/api_cc/src/DeepPotPTExpt.cc` |
| CUDA regression tests                  | `source/tests/pt_expt/descriptor/test_dpa1_cuda.py`                      |
| End-to-end benchmark tools             | `debug/dpa1_cuda_bench/`                                                 |

## 2. Mathematical contract

For edge `e = (source, destination)` with displacement
`r_e = x_source - x_destination`, the environment channels are

```text
q      = |r_e| + protection
sw     = quintic_switch(|r_e|; rcut_smooth, rcut)
raw    = [sw/q, sw*r_x/q^2, sw*r_y/q^2, sw*r_z/q^2]
R_e    = (raw - average[type_destination]) * inverse_stddev[type_destination]
```

The compressed geometric network evaluates a quintic Hermite table on `R_e[0]`.
For strip type embedding,

```text
gate_e = type_gate[type_destination, type_source]
G_e    = table(R_e[0]) * (1 + gate_e * sw)    # smooth type embedding
G_e    = table(R_e[0]) * (1 + gate_e)         # non-smooth type embedding
```

The destination moment and descriptor are

```text
M_i[k, c] = (1 / nnei) * sum_{dst(e)=i} R_e[k] * G_e[c]
D_i[c, a] = sum_{k=0..3} M_i[k, c] * M_i[k, a],  a < axis_neuron
```

`D_i` is flattened channel-major. The rotation output is `M_i[1:4, :]`.
The backward differentiates the Gram contraction, table interpolation, type
gate, switch function, and normalized environment channels analytically.

## 3. Graph topology contract

`NeighborGraph` supports both preserved and canonical payload order. Generic
builders retain the incoming stream and omit CSR views by default because
attention and legacy paths may observe that order. A consumer requests
`with_csr=True` when it requires grouped edge reductions; `canonicalize=True`
also requests CSR and applies a stable destination-major permutation. CSR fields
are keyword-only extensions appended after the original graph fields, so the
legacy positional constructor remains unchanged.

```text
destination_order: (E,) index dtype
destination_row_ptr: (N + 1,) int64
source_order: (E,) index dtype
source_row_ptr: (N + 1,) int64
destination_sorted: bool
```

`destination_sorted` is true only when the payload is destination-major and
`destination_order` is the identity. It selects direct addressing in the
compressed descriptor kernel; it does not weaken the edge-mask contract. The
attribute is assigned only by a CSR construction boundary and is not treated as
proof by `canonicalize_neighbor_graph`, which rebuilds the canonical payload.

The row pointers are always int64. Edge indices and order tensors may be int32 or
int64, but each order tensor must use the same dtype as `edge_index`. A
10-million-atom graph with 181 edges per atom contains 1.81 billion edges and
fits signed int32 order entries. A graph with 215 edges per atom exceeds the
signed int32 range; such graphs require int64 edge addressing. CSR offsets remain
int64 in all cases.

The compact deployment ABI uses int64 source indices and source order
unconditionally. The artifact therefore remains valid across deployment system
sizes without requiring the user to estimate an edge bound. Generic graph
operators retain int32 dispatch for internal and eager workloads.

Graph artifacts record `graph_edge_dtype` in metadata. Compressed DPA1 graph
export is available only when the FP32 fused graph operator serves the
descriptor; automatic export selects the nlist lower otherwise, and an explicit
graph request is rejected. A compressed graph artifact therefore consumes FP32
edge vectors directly because its descriptor forward, analytical backward, and
force assembly all compute in FP32. Generic and uncompressed graph descriptors
retain the FP64 geometry ABI. Python builders, DeepEval, the C++ host adapter,
and the Kokkos producer select the recorded dtype; the two representations are
not mixed within one artifact.

Masked edges may occur inside a CSR row. This is required by cached or padded
topologies; every consumer applies `edge_mask` rather than assuming that all row
entries are valid.

`build_edge_csr` constructs the optional destination/source views. A
canonicalization applies the same stable destination permutation to every edge
field, moves masked guards to the suffix, and makes destination order the
identity. Stability preserves incoming order inside each destination segment.
Graph-form export and deployment builders request this layout explicitly; eager
graphs without a CSR consumer do not pay the two sorting passes.

### 3.1 Kokkos device path

Eligible geometrically compressed DPA1 models export
`lower_input_kind = "dpa1_canonical"`. This deployment ABI contains eight
tensors: atom types, total and owned node counts, source index, FP32 edge vector,
destination/source row pointers, and source order. Destination indices are
implicit in the destination rows; no destination order or edge mask is stored.
Physical edges occupy `[0, E)`, while `max(E, 2) - E` storage guards lie outside
every CSR row.

`pair_style deepmd/kk` constructs this compact model-cutoff stream directly on
the GPU. The cutoff fill is destination-major and simultaneously accumulates
source counts; destination/source scans and a counting scatter complete both CSR
views. The C++ bridge wraps the Kokkos buffers with `from_blob` and performs no
index cast, guard concatenation, mask allocation, or CSR rebuild.

Device-edge dispatch is selected only when the artifact declares an edge or
graph lower. `pair_style deepmd/kk` rejects nlist artifacts because its arrays
reside in the Kokkos device memory space; nlist artifacts use the standard
`pair_style deepmd` path. The graph ABI carries both total and owned node
counts: the descriptor runs on local-plus-halo nodes, fitting outputs from halo
nodes are masked from the energy, and extended force and centroid per-atom
virial rows are folded through Kokkos reverse communication. Host-staged and
device-resident communication share an atom-order DualView accumulation buffer
so multi-stage halo exchange preserves intermediate contributions. The buffer
is allocated only when centroid per-atom virial is requested.
Message-passing models use the separate edge ABI and its with-comm artifact.

The host C++ helper canonicalizes arbitrary input payloads before invoking the
exported model. Generic graph artifacts preserve the full masked NeighborGraph
contract for attention, exclusions, arbitrary edge order, and reference
fallbacks; they do not share the compact positional ABI.

`compactEdgeTensors` remains active. The Kokkos pair scans the LAMMPS skin list
and emits only current model-cutoff edges. Carrying the full skin topology
through the descriptor was correct but slower, especially for small systems.
The benchmark uses a 1 Å skin, matching GPUMD's `Neighbor` implementation; the
earlier 2 Å setting reduced whole-step throughput by approximately 6--9%.

## 4. Compressed descriptor kernels

### 4.1 Thread mapping

One warp owns one destination node. For table widths 16 through 64, two
16-lane sub-warps process alternating edges. A lane evaluates one or more
contiguous spline channels:

| Table width | Channels per sub-warp lane |
| ----------: | -------------------------: |
|          16 |                          1 |
|          32 |                          2 |
|          64 |                          4 |

Widths 8, 128, and 256 use one edge per warp. The mapping coalesces the six
coefficients of adjacent channels from
`table[segment, channel, coefficient]`. The source type is read once by the
sub-warp leader and the pair index is computed for arbitrary multi-element
systems.

### 4.2 Forward

Each warp:

1. reads its destination CSR interval;
1. accesses edges directly for canonical CSR or applies the destination
   permutation for a generic graph, and applies `edge_mask` in both forms;
1. loads and broadcasts edge geometry and spline location;
1. evaluates the table and type gate across channels;
1. accumulates the four moment channels in registers;
1. combines the two sub-warp edge streams;
1. writes the moment and Gram descriptor, plus rotation when requested.

The Gram contraction is formed in the same kernel. No global moment atomics and
no separate Gram kernel are required. When the center type embedding is
concatenated to the descriptor, each lane writes channels at stride 32, so the
tail is complete for type-embedding dimensions larger than one warp. The
autograd descriptor path returns rotation; level-2 energy/force inference
suppresses its allocation and stores because no downstream fitting consumes it.

### 4.3 Backward

The backward first differentiates `D = M^T M_axis`. It includes both terms of
the Gram derivative:

```text
dL/dM[:, c] =
    sum_{a < axis} dL/dD[c, a] * M[:, a]
  + 1[c < axis] * sum_i dL/dD[i, c] * M[:, i]
```

The resulting four-channel gradient is broadcast to the edge-processing
sub-warps. Each edge recomputes its spline value and derivative, contracts all
channels in registers, differentiates the switch and normalized environment,
and writes one edge gradient. Masked edges and guard entries are written as
zero.

The spline value and derivative use a single forward-mode Horner recurrence.
Linear extrapolation uses the endpoint value and derivative, preserving the
table's C1 contract.

### 4.4 Width dispatch and resource policy

Compression evaluates the complete embedding network while constructing the
spline table. Activations, timestep factors, and identity or width-doubling
residuals from every embedding layer are therefore already represented in the
table coefficients. The CUDA interpolation kernel does not see the first or
intermediate layer widths; only the final width `neuron[-1]` remains at runtime.

CUDA instantiations exist for widths `{8, 16, 32, 64, 128, 256}`. An arbitrary
positive final width up to 256 uses the smallest containing bucket; table and
gate channels are zero-padded and sliced from the outputs. A non-bucket width
uses the compressed descriptor CUDA operator but not the level-2 end-to-end
mega path. `axis_neuron` must lie in `[1, min(16, neuron[-1])]`. The compressed
graph path additionally requires attention-free strip input, FP32 tables and
statistics, and no excluded type pairs.

The translation unit enables CUDA fast scalar intrinsics but does not impose a
global register cap. Each forward/backward specialization has two resource
policies with identical arithmetic:

- `Balanced`: `__launch_bounds__(256, 2)`, allowing more registers to avoid
  spill;
- `Occupancy`: `__launch_bounds__(256, 4)`, limiting allocation to increase
  resident CTAs.

Either policy launches four or eight warps per CTA (128 or 256 threads). The
operator selects among these four configurations on first use for each device,
direction, width, axis, index type, CSR mode, type/gate mode, descriptor stride,
and coarse type-count/node-count/degree classes. Candidate timing uses CUDA
events on at most a representative node subset; the selection is cached
in-process. Model weights are not part of the key because they do not change
kernel work. Arithmetic and sub-warp mapping do not change, so resource
selection does not alter the descriptor's reduction order.

Volta, Ampere, Ada, low-SM Hopper (including H20), and general Hopper have
built-in safe fallbacks. Unknown devices use the balanced policy. When the input
is too small, event setup fails, or the stream is under CUDA Graph capture, the
appropriate fallback launches directly. Capture fallback is not cached, so a
later uncaptured call can still tune.

The PyTorch CUDA operator library contains native SASS selected by the local
toolchain and, by default, a lowest-supported virtual PTX target
(`DEEPMD_CUDA_PORTABLE_PTX=ON`). CUDA toolkits before 13 use compute 70 PTX,
covering V100 and newer devices through driver JIT; newer toolkits use compute
75 because they no longer provide the same Volta compilation path. A deployable
V100 image must therefore be built with a toolkit that supports SM70.

The operator contains no GEMM, so TF32 is irrelevant inside this file and
remains disabled in the fitting path.

### 4.5 Uncompressed descriptor portability

The uncompressed kernel evaluates the MLP explicitly. It requires
`N1 in {8,16,32,64}`, `N2 in {N1,2*N1}`, `NG in {N2,2*N2}`,
`N2 <= 64`, and `NG <= 128`. Layers 2 and 3 implement identity and
width-doubling residuals. Layer 1 has no fused residual epilogue, so a model
whose native first-layer input/output shape forms an identity or doubling
residual uses the reference path.

The fused uncompressed descriptor stages its embedding tiles in dynamic shared
memory. Its widest `(N1, N2, NG) = (32/64, 64, 128)` stacks require more than
the V100 per-block opt-in limit with the 128-edge forward tile. The launch
queries `sharedMemPerBlockOptin`: devices with sufficient capacity retain the
128-edge tile, while lower-capacity devices use a compiled 64-edge tile.
`cudaFuncSetAttribute` errors are checked before launch. This keeps the H20 path
unchanged and prevents an invalid launch on SM70.

## 5. Force and virial reduction

The force operator consumes both CSR views. One warp owns one node and evaluates

```text
force[i] = sum_{dst(e)=i} dE/dr_e - sum_{src(e)=i} dE/dr_e
atom_virial[i] = sum_{src(e)=i} -dE/dr_e outer r_e
```

Each node writes its force and atom virial once. The hot reduction contains no
global floating-point atomics. The global per-frame virial is reduced from node
virials in two FP64 stages and cast to model precision only at the output.
Frame/component/partial work is flattened onto the CUDA grid, and the number of
partials is derived from the average nodes per frame. The launch therefore has
no grid-y frame limit and does not replicate total-node work for every frame.

Materializing atom virial does not add a second edge pass: the node virial is
required by the global reduction in either case. Isolated benchmarks measured
0.395 ms with and without returning atom virial at approximately 97,000 atoms.
No conversion or freeze CLI option was added for this negligible kernel-time
difference.

## 6. Fitting and precision policy

`graph_fitting` evaluates the energy fitting network through pedantic fp32
cuBLAS GEMMs. Bias, activation, timestep, residual, and saved activation
derivative are fused into layer epilogues. The per-frame energy reduction and
atomic-energy bias remain fp64. The fitting bias and atomic-model output bias
are combined before the fused head, so level 2 reproduces the standard
`apply_out_stat` energy exactly. An ownership mask zeros halo atomic energies
and seeds fitting backward only on owned nodes.

`graph_fitting_backward` derives the node count and descriptor width from its
saved derivative buffer and first weight. It does not retain the descriptor
solely for shape metadata. After the fitting forward, the descriptor reaches
last use and its storage can be reused for the descriptor gradient; fitting
saved state is released immediately after backward. This lifetime contract is
part of the level-2 memory design and does not change fitting arithmetic.

Compensated FP16x3 was evaluated but is not used:

- applying FP16x3 to the whole fitting network was slower for S, M, and L;
- the first forward GEMM was 1.35--1.49x slower;
- backward-only selection improved a large model by approximately 4.9%;
- unseen structures can exceed the safe FP16 head range even when trained-model
  bounds pass.

The marginal backward gain did not justify a second numerical policy and its
range proof. The final path remains fp32 throughout fitting.

## 7. Export and compilation

`DP_CUDA_INFER` selects the graph-lower implementation at trace time:

| Level | Compressed descriptor path                                          |
| ----: | ------------------------------------------------------------------- |
|     0 | Reference `tabulate_fusion_se_atten` formulation                    |
|     1 | `dpa1_graph_compress` with registered autograd                      |
|     2 | Value-returning descriptor + fitting + analytical force composition |

The level-2 compressed path is a Python composition of explicit custom
operators. It invokes descriptor forward, fitting forward/backward, descriptor
backward, and dual-CSR force reduction without retaining an autograd tape. This
keeps the hand-written operators opaque in the exported `.pt2`.

Exported level-2 artifacts use the canonical graph ABI and index destination
segments directly. Eager level-1 and level-2 execution instead follows the
graph's explicit `destination_sorted` property and supports permutation CSR.
Every form applies `edge_mask`, including masked entries inside a canonical CSR
row. The level-2 wrapper converts geometry to model precision once and threads
that tensor through descriptor forward, descriptor backward, and force
assembly; no intermediate fp32-to-fp64-to-fp32 round trip remains.

`forward_(common_)lower_graph_exportable` accepts `destination_sorted` as an
explicit trace-time static argument. Before tracing, it validates that both CSR
orders are permutations and that every active edge lies in the row for its
source or destination. A canonical trace additionally requires an identity
destination order. This check occurs once during export and adds no inference
work. The PyTorch-specific validation lives in
`deepmd.pt_expt.utils.graph_csr`, separate from model construction.

Trace compatibility requires:

- fake implementations with symbolic output shapes;
- CPU implementations for trace-time execution;
- `register_autograd` for descriptor and fitting forwards;
- `SymInt` for dynamic node and edge-dependent sizes;
- device-independent routing so a CPU `make_fx` trace records CUDA operators;
- no host reads from symbolic tensors;
- contiguous statistics and table buffers at the operator boundary;
- a destination-major payload and identity destination permutation for
  level-2 artifacts.

The exported ABI invariant is established by Python graph builders and the C++
host/Kokkos boundaries; eager graphs carry `destination_sorted=False` unless a
CSR builder establishes it. A non-contiguous
`mean[:, 0, :]` view once entered the manually orchestrated level-2 backward and
was read with a contiguous stride assumption. The level-1 path already stored a
contiguous buffer, so narrow random tests did not expose the mismatch. The
level-2 wrapper now materializes the contiguous view and the CUDA entry validates
all statistics, tables, graph tensors, dtypes, and layouts.

## 8. Correctness verification

The CUDA suite covers:

- table widths 8, 16, 32, 64, 100, and 128;
- `axis_neuron` 4 and 16;
- concatenated type-embedding tails wider than one warp;
- one-sided and two-sided type gates;
- smooth and non-smooth type embedding;
- four-element mixed-type graphs;
- int32 and int64 edge addressing;
- canonical and arbitrarily ordered edge payloads;
- masked cutoff edges inside CSR rows;
- destination/source CSR construction;
- force, global virial, and atom-virial parity across 8,192 one-node frames
  and 4,096 heterogeneous frames;
- level-2 canonical/permutation CSR parity with a production-shaped residual
  fitting;
- zero-node descriptor forward and backward;
- automatic first-use resource selection on a non-trivial graph;
- matched FP64/FP32 graph artifacts with bitwise-identical physical outputs;
- `make_fx`, `torch.compile(fullgraph=True)`, CPU trace execution, and dynamic
  graph export.

Observed test results:

```text
42 passed, 2 subtests passed  source/tests/pt_expt/descriptor/test_dpa1_cuda.py
57 passed, 2 subtests passed  NeighborGraph builders and graph utilities
17 passed                      DPA1 graph lower and graph export
14 passed                      graph metadata and export-schema tests
2 passed                       graph-form DeepEval deployment parity
3 passed                       TestEdgeTensorPack.* C++ API tests
```

The S/M/L EMA packages were evaluated on a 512-atom periodic diamond system.
Relative L2 differences between uncompressed and compressed packages are the
normal table-approximation error, not operator-path disagreement:

| Model |  Energy |   Force |  Virial | Atomic energy | Atom virial |
| ----- | ------: | ------: | ------: | ------------: | ----------: |
| S     | 4.99e-6 | 1.54e-4 | 1.78e-5 |       6.07e-6 |     6.99e-5 |
| M     | 2.92e-6 | 1.91e-4 | 1.60e-4 |       5.04e-6 |     1.59e-4 |
| L     | 1.73e-6 | 2.10e-4 | 2.68e-5 |       4.64e-6 |     1.32e-4 |

For the compressed model itself, level 1 and level 2 force, virial, and atom
virial are identical in the trained S reproduction.

## 9. Performance

### 9.1 Model computation on a prebuilt graph

Measurements use trained EMA weights, approximately 158 model-cutoff edges per
atom, TF32 disabled, atom virial enabled, and CUDA-event timing after warmup.

| Model, 97,336 atoms | Original compressed path | First integrated path | Canonical/single-cast path | Prototype best |
| ------------------- | -----------------------: | --------------------: | -------------------------: | -------------: |
| S                   |                24.706 ms |               9.95 ms |                    9.15 ms |        8.57 ms |
| M                   |                26.520 ms |              12.72 ms |                   12.00 ms |       11.36 ms |
| L                   |                35.144 ms |              19.89 ms |                   19.11 ms |  17.5--17.7 ms |

The adaptive resource policy was measured separately on the H20 with the same
97,336-atom, approximately 158-edge-per-atom graph and production S/M/L network
dimensions:

| Model | Automatic |      Best forced candidate | Auto overhead |
| ----- | --------: | -------------------------: | ------------: |
| S     |  9.485 ms |  9.463 ms (`occupancy128`) |         0.23% |
| M     | 12.478 ms | 12.401 ms (`occupancy128`) |         0.62% |
| L     | 19.363 ms | 19.249 ms (`occupancy128`) |         0.59% |

The kernel-level tuner selected `occupancy128/occupancy256` for S
forward/backward and `occupancy256` for both directions of M and L. Its
direction-specific choices differ from the best whole-pipeline forced setting
by less than one percent. The first uncached L call took 47.90 ms and the next
cached call 19.57 ms, giving approximately 28 ms one-time tuning overhead.

The apparent gap between the initial repository benchmark and the prototype was
mostly not a kernel regression. The prototype timed a prebuilt graph with
precast geometry and prebuilt CSR; LAMMPS includes neighbor processing and graph
assembly. Within model computation, the first integrated kernel was 12--16%
slower than the prototype. Canonical CSR access and a single geometry conversion
recovered 4--8 percentage points. The remaining L difference includes the
prototype's optional backward-only FP16x3 fitting policy, which is intentionally
not part of the production path.

At 97,336 atoms, the dual-CSR force/virial stage decreased from approximately
5.01 ms to 0.395 ms. Its hierarchical global virial had relative RMS error
`9.55e-8` against an independent FP64 reduction, compared with `4.98e-6` for
the atomic reduction.

### 9.2 LAMMPS + Kokkos whole-step benchmark

The deployment benchmark uses:

- NVIDIA H20, CUDA 12.8, PyTorch 2.11;
- trained MatPES DPA1-L0 S/M/L EMA checkpoints;
- periodic diamond supercells;
- `pair_style deepmd/kk`;
- 1 Å LAMMPS neighbor skin with displacement-triggered rebuild
  (`every 1 delay 0 check yes`), matching GPUMD's 1 Å skin and `skin/2`
  displacement criterion;
- 10 warmup and 100 measured NVT steps;
- TF32 disabled and atom virial enabled;
- throughput from the LAMMPS `Loop time`, including pair construction,
  communication, integration, and neighbor rebuilds.

Matched FP64/FP32 compressed graph artifacts at 129,168 atoms isolate the edge
ABI change. Three alternating runs give:

| Model | FP64 atoms/ms | FP32 atoms/ms | Speedup | FP64 memory | FP32 memory |
| ----- | ------------: | ------------: | ------: | ----------: | ----------: |
| S     |         6,305 |         6,531 |  1.036x |   3,761 MiB |   3,031 MiB |
| M     |         5,291 |         5,467 |  1.033x |   4,115 MiB |   3,385 MiB |
| L     |         3,810 |         3,886 |  1.020x |   4,585 MiB |   3,855 MiB |

The 730 MiB reduction is independent of model width because the three artifacts
share the same edge count. DeepEval comparison of matched M artifacts is bitwise
identical for energy, force, virial, atomic energy, and atomic virial.

The final seven-curve scan runs each model to its first OOM:

| Curve               | Plateau atoms/ms | Largest successful system |
| ------------------- | ---------------: | ------------------------: |
| GPUMD + NEP         |            8,576 |                 8,489,664 |
| DPA1-S uncompressed |            2,978 |                 1,999,872 |
| DPA1-S compressed   |            7,030 |                 5,025,816 |
| DPA1-M uncompressed |            2,265 |                 1,000,000 |
| DPA1-M compressed   |            5,764 |                 4,516,560 |
| DPA1-L uncompressed |            1,299 |                   499,200 |
| DPA1-L compressed   |            4,060 |                 3,509,376 |

Compression raises plateau whole-step throughput by 2.36x, 2.54x, and 3.13x
for S, M, and L respectively. All three models use `axis_neuron=16`. The
generated plot and per-size CSV files are in
`debug/dpa1_cuda_bench/throughput.png` and
`debug/dpa1_cuda_bench/results/`.

At 129,168 atoms, three-run compressed medians are 6,544, 5,469, and 3,898
atoms/ms for S, M, and L. Relative to the preceding 6,531, 5,467, and 3,886
measurements, the changes are +0.2%, +0.0%, and +0.3%.

For `N=97,336` and `E=15,378,920`, the compact int64 topology retains
412.1 MiB versus 661.5 MiB for the generic graph inputs. The isolated M
level-2 sequence measures 11.022 ms versus 11.665 ms with identical outputs.
At 129,168 atoms, matched generic and compact whole-step measurements are
19.877 and 16.267 ms for S, 23.982 and 20.308 ms for M, and 33.115 and
29.669 ms for L. The corresponding reductions are 18.2%, 15.3%, and 10.4%.
The measured M process GPU-memory peaks are 3,389 and 2,355 MiB.

The 30-step compact scan reaches:

| Model | Plateau atoms/ms | Largest successful system | Generic limit |
| ----- | ---------------: | ------------------------: | ------------: |
| S     |            8,568 |                 9,524,736 |     5,025,816 |
| M     |            6,802 |                 7,526,400 |     4,516,560 |
| L     |            4,546 |                 5,510,880 |     3,509,376 |

An Nsight profile of L at 129,168 atoms reports the following per-step costs:

| Stage                               |                  Time |
| ----------------------------------- | --------------------: |
| Compressed descriptor backward      |               8.64 ms |
| Compressed descriptor forward       |               6.14 ms |
| Fitting GEMMs                       |  approximately 8.3 ms |
| Kokkos cutoff edge count/fill       | approximately 4.58 ms |
| CSR histograms and counting scatter | approximately 1.79 ms |
| Force/atom-virial node reduction    |               0.58 ms |

The difference between model-only and whole-step throughput is the device
neighbor pipeline, CSR construction, graph-input packing, and MD-engine work;
the custom descriptor remains opaque in the AOTInductor package.

## 10. Optimization study

### 10.1 Effective changes

- **Sub-warp edge parallelism.** Two independent 16-lane edge streams keep the
  warp active for widths 16--64.
- **Direct moment and Gram contraction.** Node ownership removes global moment
  atomics and separate Gram kernels.
- **Compile-time validity specialization.** Compact cutoff-filtered graphs
  instantiate mask-free descriptor kernels, while generic cached or excluded
  graphs retain the authoritative edge-mask checks.
- **Forward-mode Horner backward.** One recurrence produces table value and
  derivative.
- **Fast scalar environment arithmetic.** Hardware reciprocal and square-root
  instructions remove scalar division latency; surface and short-NVE checks
  remain at the fp32 reference floor.
- **Adaptive resource dispatch.** Balanced and occupancy-specialized kernels
  remove the H20-specific global register cap. Runtime device/workload tuning
  selects 128/256-thread launches once and caches the result; built-in
  architecture profiles and portable PTX preserve migration fallback.
- **GPUMD comparison.** GPUMD compiles for the native CUDA architecture and
  launches the production NEP force path with a fixed 64-thread block; it does
  not perform occupancy- or event-based runtime tuning. DPA1 retains the same
  bounded-configuration principle, but its width-dependent register range
  requires selecting between two resource policies.
- **Dual-CSR force reduction.** One node owner replaces edge atomics and avoids
  materializing an `(E, 9)` outer-product tensor.
- **Kokkos CSR construction.** Destination identity ordering and source
  counting scatter avoid a destination radix sort on the direct device path;
  the generic C++ host adapter canonicalizes arbitrary payloads separately.
- **Explicit canonical CSR property.** Production artifacts index destination
  segments directly; generic eager graphs retain permutation semantics, while
  every path applies the edge mask.
- **FP32 graph geometry ABI.** Compressed artifacts record FP32 edge geometry;
  Kokkos computes bond vectors in double arithmetic and stores the final FP32
  value directly. This removes the retained FP64 edge buffer and model-boundary
  conversion without changing outputs.
- **Inference lifetime contraction.** Level-2 suppresses the unused rotation
  output. Fitting backward infers its output shape from saved derivatives and
  weights, so descriptor and fitting state reach last use before descriptor
  backward and their storage can be reused.
- **Explicit graph ABI.** Python export, C++ API, and Kokkos use the same
  destination/source topology contract.
- **Aligned neighbor policy.** LAMMPS and GPUMD benchmarks use the same 1 Å skin
  and displacement-triggered rebuild criterion.

### 10.2 Rejected changes

| Experiment                                                            | Outcome                                                                   |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Initial one-edge-per-warp implementation without sub-warp concurrency | Only 1.16--1.18x descriptor gain at 100k atoms                            |
| Packed spline coefficient planes                                      | No forward gain; backward change within approximately 0.5% noise          |
| Saving nine per-edge state scalars                                    | Additional forward traffic exceeded backward recomputation cost           |
| Splitting moment and Gram kernels                                     | Forward increased from 4.52 to 5.40 ms                                    |
| 48-register cap                                                       | Spill traffic regressed the wide descriptor                               |
| One CTA per SM and shared spill staging                               | Lost latency-hiding occupancy                                             |
| Skipping derivative work on selected branches                         | Register growth reduced residency and regressed by approximately 6%       |
| Non-inlined extrapolation correction                                  | Call ABI stack traffic slowed the forward                                 |
| Whole-network FP16x3 fitting                                          | Slower for all S/M/L shapes                                               |
| Backward-only FP16x3 fitting                                          | Approximately 4.9% gain with an undesirable range-policy branch           |
| FP16 moment checkpoint                                                | Saved 48 MiB at 97k L but introduced maximum error 3.17e-3                |
| Exact moment recomputation                                            | Saved 6.5--9.8% eager peak but slowed inference by 22.6--34.4%            |
| CUDA Graph replay at 100k atoms                                       | Approximately 0.04 ms gain                                                |
| Caching the full skin-edge CSR                                        | Correct, but excess masked skin edges regressed small systems             |
| Generic source `argsort`                                              | Approximately 4.3 ms per step at 129k atoms; replaced by counting scatter |
| Compile-time `axis_neuron=16` specialization                          | No S gain and a small M/L regression from added stack pressure            |
| Shared Kokkos/AOTI CUDA stream                                        | Moved waiting from Pair to Modify without reducing whole-step time        |
| Mutation-only descriptor-gradient output                              | 2,305 MiB peak versus 2,291 MiB for allocator reuse; no speed difference  |
| Parallel custom frame-energy reduction                                | Approximately 0.19% gain but changed the 512-atom FP64 total by 1e-4 eV   |

Node-chunked fitting was also evaluated as a memory-pressure path. With FP32
edges and descriptor storage reuse, 24,576-node M chunks reduced the 97k
incremental peak from 527 to 422 MiB with 2.8% extra time; 32,768-node L chunks
reduced 1,051 to 801 MiB with 5.2% extra time. S was already limited by
edge-gradient storage and gained no peak reduction. Fully streaming descriptor,
fitting, and descriptor backward reached 523 MiB for L at a 13% time cost.
These variants remain candidates for an automatic OOM-avoidance path rather
than the default throughput path.

### 10.3 General lessons

1. Profile the complete forward-plus-force path. A compressed embedding removes
   both forward MLP work and its larger descriptor backward.
1. Register pressure is a launch-policy constraint. Channel/edge mapping must be
   chosen together with occupancy.
1. Preserve semantic payload order and represent reduction order explicitly.
1. Build sparse topology with histogram, scan, and scatter when keys lie in a
   bounded node range; comparison sorting is unnecessary.
1. Measure integration overhead separately from model kernels.
1. Treat tensor lifetime and ABI dtype as kernel design parameters. An unused
   output or shape-only dependency can retain several gigabytes at MD scale.
1. Validate production dimensions and strides. Small random tensors do not
   expose every layout contract.
1. Reject marginal mixed-precision gains when they introduce an input-range
   policy into molecular dynamics inference.

## 11. Usage

The established conversion flow remains unchanged:

```bash
# PyTorch checkpoint to an intermediate pt_expt model.
dp convert-backend model_ema.ckpt.pt model_ptexpt.pte

# Geometric compression and graph-form export.
DP_CUDA_INFER=2 dp --pt-expt compress \
    -i model_ptexpt.pte \
    -o model_ptexpt_compress.pt2 \
    -t input.json

rm model_ptexpt.pte
```

No atomic-virial CLI option is introduced by this path.

Adaptive CUDA resource tuning and portable PTX are enabled automatically; no
runtime environment variable or additional CMake argument is required.

Relevant verification commands:

```bash
cmake --build source/build -j64

CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 \
    DP_INTER_OP_PARALLELISM_THREADS=0 \
    DP_INTRA_OP_PARALLELISM_THREADS=0 \
    python -m pytest \
    source/tests/pt_expt/descriptor/test_dpa1_cuda.py \
    source/tests/pt_expt/model/test_graph_export.py \
    -q

source/build/api_cc/tests/runUnitTests_cc \
    --gtest_filter='TestEdgeTensorPack.*'
```

The benchmark scripts under `debug/dpa1_cuda_bench/` generate matching
uncompressed/compressed EMA packages, diamond systems, LAMMPS scans, and plots.

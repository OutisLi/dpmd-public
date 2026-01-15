# Compressed DPA1 and DPA4C CUDA Inference

## 1. Scope

This document specifies the CUDA inference paths for geometrically compressed
DPA1 descriptors (`se_atten_v2`, strip type embedding, `attn_layer == 0`) and
degree-wise compressed DPA4C descriptors in the `pt_expt` graph lower. The
target workload is a forward energy evaluation followed by one analytical
backward for force and virial. DPA4C additionally compiles a native spin
variant, which extends that backward with the magnetic force; section 11.12
specifies it. Training and double backward are outside these operator
contracts.

DPA4C has a second implementation of the same operators for the CPU, specified
in `doc/outisli/dpa4c_cpu.md`. The two share every operator schema, the Python
front end, the compression artifacts and the eligibility predicates; only the
kernels differ, and the dispatcher selects between them. Anything below that
mentions the CPU refers to the portable reference path, not to those kernels.

The numerical contract is fp32 model computation with TF32 disabled. Coordinates
and public graph geometry may remain fp64; operator boundaries cast the geometry
to the model precision and return the edge gradient in the input dtype where
autograd requires it.

The implementation consists of:

| Component                              | Location                                                                 |
| -------------------------------------- | ------------------------------------------------------------------------ |
| Compressed descriptor forward/backward | `source/op/pt/dpa1_graph_compress.cu`                                    |
| DPA4C device templates                 | `source/op/pt/dpa4c/graph_compress{,_kernel}.cuh`                        |
| DPA4C per-width instantiations         | `source/op/pt/dpa4c/graph_compress_c{8,16,32,64,128}.cu`                 |
| DPA4C CUDA kernel registration         | `source/op/pt/dpa4c/graph_compress.cu`                                   |
| DPA4C operator schemas                 | `source/op/pt/dpa4c/ops.cc`                                              |
| Shared operator schemas                | `source/op/pt/graph_ops_schema.cc`                                       |
| Fused energy fitting                   | `source/op/pt/graph_fitting.cu`                                          |
| Force and virial reduction             | `source/op/pt/edge_force_virial.cu`                                      |
| Uncompressed end-to-end inference      | `source/op/pt/dpa1_graph_energy_force.cu`                                |
| Python operator integration            | `deepmd/pt_expt/kernels/cuda/dpa1/graph_compress.py`                     |
| DPA4C Python integration               | `deepmd/pt_expt/kernels/dpa4c/graph_compress.py`                         |
| Graph contract                         | `deepmd/dpmodel/utils/neighbor_graph/graph.py`                           |
| C++/Kokkos graph ingestion             | `source/api_cc/include/commonPT.h`, `source/api_cc/src/DeepPotPTExpt.cc` |
| Native spin C++ serving                | `source/api_cc/src/NativeSpinPTExpt.cc`                                  |
| Native spin LAMMPS pair styles         | `source/lmp/pair_dpa4spin{,_kokkos}.cpp`                                 |
| Spin throughput and capacity scan      | `debug/cuda_bench/in_spin.lammps`, `BENCH_FLAVOR=spin`                   |
| CUDA regression tests                  | `source/tests/pt_expt/descriptor/test_dpa1_cuda.py`                      |
| DPA4C CUDA regression tests            | `source/tests/pt_expt/descriptor/test_dpa4c_cuda.py`                     |
| End-to-end benchmark tools             | `debug/cuda_bench/`                                                      |

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

The compact deployment ABI uses uint32 source indices and source order, while
both CSR row-pointer arrays remain int64. The compact storage limit is therefore
`2^32 - 1` edge slots. Kokkos construction, Python validation, export dynamic
shapes, and the public C/C++ device API reject larger storage before accessing
the topology. On the H20, the remaining edge geometry, gradient, neighbor-list,
and model state exhaust device memory before the uint32 limit.

Canonical artifacts record `graph_edge_dtype = "float32"` and
`canonical_index_dtype = "uint32"` in metadata. Compressed DPA1 graph export is
available only when the FP32 fused graph operator serves the descriptor;
automatic export selects the nlist lower otherwise, and an explicit graph
request is rejected. Python builders, DeepEval, the C++ host adapter, and the
Kokkos producer enforce the recorded contract. Canonical artifacts from the
former int64 ABI must be re-frozen.

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
the GPU. The cutoff fill is destination-major and accumulates an independent
source histogram; a scan and counting scatter complete the source CSR. The two
degree sequences are kept independent because opposite periodic displacements
can straddle the strict cutoff through floating-point rounding. The C++ bridge
wraps the Kokkos buffers with `from_blob` and performs no index cast, guard
concatenation, mask allocation, or CSR rebuild.

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
cuBLAS GEMMs. Bias, activation and residual are fused into layer epilogues.
The per-frame energy reduction and atomic-energy bias remain fp64. The fitting
bias and atomic-model output bias are combined before the fused head, so level
2 reproduces the standard `apply_out_stat` energy exactly. An ownership mask
zeros halo atomic energies and seeds fitting backward only on owned nodes.

The saved state is the pre-activation of every layer, written by the GEMM
itself; the backward re-derives `act'` from it. Section 11.8 records the
traffic argument and the measurement, and why a per-layer timestep is
rejected by the eligibility gate rather than supported.

`graph_fitting_backward` derives the node count and descriptor width from its
saved buffer and first weight. It does not retain the descriptor solely for
shape metadata. After the fitting forward, the descriptor reaches last use and
its storage can be reused for the descriptor gradient; fitting saved state is
released immediately after backward. This lifetime contract is part of the
level-2 memory design and does not change fitting arithmetic.

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
| GPUMD + NEP5        |            8,576 |                 8,489,664 |
| DPA1-S uncompressed |            2,978 |                 1,999,872 |
| DPA1-S compressed   |            7,030 |                 5,025,816 |
| DPA1-M uncompressed |            2,265 |                 1,000,000 |
| DPA1-M compressed   |            5,764 |                 4,516,560 |
| DPA1-L uncompressed |            1,299 |                   499,200 |
| DPA1-L compressed   |            4,060 |                 3,509,376 |

Compression raises plateau whole-step throughput by 2.36x, 2.54x, and 3.13x
for S, M, and L respectively. All three models use `axis_neuron=16`. The
generated plot and per-size CSV files are in
`debug/cuda_bench/throughput.png` and
`debug/cuda_bench/results/`.

At 129,168 atoms, three-run compressed medians are 6,544, 5,469, and 3,898
atoms/ms for S, M, and L. Relative to the preceding 6,531, 5,467, and 3,886
measurements, the changes are +0.2%, +0.0%, and +0.3%.

For `N=97,336` and `E=15,378,920`, the former compact int64 topology retained
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
  output. Fitting backward infers its output shape from the saved
  pre-activations and weights, so descriptor and fitting state reach last use
  before descriptor backward and their storage can be reused.
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

## 11. DPA4C optimization study

### 11.1 Numerical invariants

The compressed DPA4C path preserves the following non-negotiable numerical
contract:

- all descriptor, spline, PairFiLM, mode-mixing, readout, and fitting
  arithmetic is FP32;
- TF32 and FP16 are disabled;
- the radial table uses quintic Hermite interpolation with matching value,
  first derivative, and second derivative at every knot;
- the force is the analytical derivative of the exact compressed forward;
- no performance path may alter reduction order without an explicit numerical
  parity test.

Autograd in `deepmd/pt_expt/kernels/dpa4c/graph_compress.py` is not part of the
inference hot path. Compression setup uses it once to evaluate radial-network
first and second derivatives at table knots. `torch.library.register_autograd`
connects the hand-written CUDA backward to PyTorch; the canonical level-2
composition invokes that backward operator explicitly.

### 11.2 Structure the kernel implements

The kernel reproduces the portable equations exactly, including the three
features that distinguish the current descriptor from its predecessor:

- **Two envelope masses.** Degree zero normalizes against
  `sum_j chi^2` and every non-scalar degree against `sum_j chi^4`; the
  non-scalar payload additionally carries a second envelope factor. The saved
  state therefore holds both normalizers, and the backward produces two
  independent mass cotangents.
- **Pair-conditioned radial modes.** The `R` shared mode profiles depend only
  on the scalar distance and occupy the trailing channels of the same radial
  table. Each ordered type pair mixes them with its own coefficients, read
  from a `[(T + 1)^2, C_0, R]` cache whose mode-innermost layout keeps the
  coefficients of one channel contiguous.
- **Angular degrees three and four.** Both carry a single channel, so their
  moments are distributed across the lanes of an edge group rather than tiled
  by channel, their Grams reduce to a squared norm, and their coupling triples
  are driven by a sparse Cartesian Gaunt table emitted as a compression
  artifact. The `lmax = 2` instantiation compiles none of this.

Parity is established over the full cross product of the public parameter
surface: five channel widths, three angular degrees, four mode ranks, and both
topology forms. The forward is held to `3e-5` absolute and relative against the
portable descriptor, and the analytical backward to `8e-6` absolute and `1e-4`
relative against the autograd of the single-precision reference implementation.

### 11.3 Profile progression

The isolated operator benchmark (`debug/cuda_bench/kernel_profile.py`) runs one
H20 over a fully periodic diamond supercell of 32,768 atoms and 5,177,344 edges
at 158 edges per atom, and reports the fastest of five timed batches. Run-to-run
spread is below 0.1%. "Initial" is the first correct implementation of the
present structure, before any of the changes in Section 11.4.

The production profile, degree two without radial modes:

| Channels |  Initial |    Final | Speedup |
| -------: | -------: | -------: | ------: |
|        8 | 0.924 ms | 0.544 ms |   1.70x |
|       16 | 1.140 ms | 0.850 ms |   1.34x |
|       32 | 1.844 ms | 1.254 ms |   1.47x |
|       64 | 2.487 ms | 1.945 ms |   1.28x |
|      128 | 4.126 ms | 3.323 ms |   1.24x |

The two structural parameters that change the compiled kernel were optimized
separately. "Initial" below is the first correct implementation of each path,
after the production profile had already been tuned:

| Configuration  |  Initial |    Final | Speedup |
| -------------- | -------: | -------: | ------: |
| C8, R = 8      | 1.570 ms | 1.606 ms |   0.98x |
| C16, R = 8     | 2.122 ms | 1.884 ms |   1.13x |
| C32, R = 8     | 3.666 ms | 3.296 ms |   1.11x |
| C64, R = 8     | 6.741 ms | 6.330 ms |   1.06x |
| C128, R = 2    | 7.321 ms | 5.870 ms |   1.25x |
| C128, R = 4    | 9.310 ms | 7.996 ms |   1.16x |
| C128, R = 8    | 12.92 ms | 11.11 ms |   1.16x |
| C32, lmax = 3  | 2.648 ms | 2.362 ms |   1.12x |
| C128, lmax = 3 | 6.147 ms | 5.287 ms |   1.16x |
| C32, lmax = 4  | 4.861 ms | 4.802 ms |   1.01x |
| C128, lmax = 4 | 9.069 ms | 8.401 ms |   1.08x |

Nsight Systems resolves the three kernels of one step at the final production
configuration:

```text
kernel                 C8        C32       C128
forward             243 us     499 us    1359 us
node backward        67 us      79 us     287 us
edge backward       234 us     676 us    1677 us
```

Hardware-counter collection is disabled by the host driver
(`ERR_NVGPUCTRPERM`), so every claim below rests on timeline data, static SASS
resource analysis, and ablation, and no hardware-saturation claim is made.

Two roofline estimates guided the search on this device. At `C_0 = 8` the
forward was estimated to move about 256 bytes per edge through L1 and to execute
roughly 200 FLOP per edge, which is 23% of aggregate L1 bandwidth and 5% of FP32
peak, and the kernel was therefore treated as issue- and latency-bound. At
`C_0 = 128` the same accounting gives about 65% of aggregate L1 bandwidth, so the
widest profile was treated as approaching a memory-throughput limit. The
optimizations below follow that split: narrow widths gain from removing fixed
per-edge cost, wide widths from removing memory instructions and register spill.

Both figures are hand accounting rather than measurement, and the narrow one does
not survive contact with hardware counters. Section 11.13 measures the same
kernels where counters are available and finds the narrow profile at 97.5% of L1
active cycles, not 23% of its bandwidth: every width is L1-cycle saturated there,
and the byte count that saturates it is the compressed radial table, which no
launch-configuration change reduces. The measured widths agree with the estimate
only at the wide end. Read the split above as the reasoning that produced
Section 11.4, and Section 11.13 for what the kernels are actually limited by.

### 11.4 Effective changes

- **Redundant edge state instead of leader broadcast.** Every lane of an edge
  group reloads the edge geometry and spline location rather than receiving
  seven geometry fields and three location fields by shuffle. A leader-only
  branch is still issued warp-wide, so it pays the same slots and adds ten
  shuffles; the addresses are identical inside a group, so the memory system
  serves one transaction either way. Total time improves by 22% at C8 and
  7% at C128.
- **Split coefficient blocks.** The interval row separates into a quartet block
  and a pair block, which delivers the six spline coefficients in one 128-bit
  and one 64-bit load instead of three 64-bit loads at identical traffic.
  C16 gains 13%, C64 7%, and C128 8%.
- **Table row resolved once per edge.** The interval base address is computed
  once and every channel offset becomes a compile-time immediate. Before this
  change integer address arithmetic accounted for roughly 45% of the SASS
  instruction mix, because the runtime table width blocked strength reduction
  of the per-channel wide multiply. C32 gains 8%.
- **Scalar cotangent in shared memory.** The degree-zero cotangent is one
  vector of width `C_0` that every edge of a node rereads. Holding it in
  registers cost one entry per channel tile and spilled 24 bytes at C64 and 40
  bytes at C128; shared memory serves it as a conflict-free broadcast because
  concurrent edge groups address identical channels. C64 gains 8% and C128 4%,
  at a 1--2% cost for the narrow widths that never spilled.
- **Per-profile edge-group width.** A warp keeps `32 / width` edges in flight,
  and the per-edge geometry is amortized over exactly that many edges, so the
  narrowest group whose per-lane accumulators still fit the register budget
  wins. Re-tuning after the
  changes above moved C8 to two lanes (its single largest gain, 22%) and C32
  backward to four lanes (18%), and confirmed the existing widths elsewhere.
- **Compile-time degree, runtime mode rank.** Specializing the angular degree
  keeps the `lmax = 2` register budget free of the degree-3/4 accumulators,
  while leaving the mode rank at runtime avoids multiplying the instantiation
  count by four for a parameter that only extends an inner reduction.
- **Per-width translation units.** The kernel templates live in a header that
  five thin translation units instantiate, one per scalar width. The angular
  degree, mode, and topology specializations therefore compile in parallel, and
  the operator library rebuilds in well under a minute on seven cores instead
  of serializing every instantiation into one unit.
- **Channel-major mode cache with vector loads.** The ordered mixing cache
  stores the mode axis innermost, so the `R` coefficients a lane needs for one
  channel are contiguous and arrive in at most one 128-bit and one 64-bit load
  rather than `R` scalar loads. The traffic is unchanged; only the instruction
  count falls. At C128 this is worth 13% at every rank. A mode-major layout,
  which is the natural choice when the loads are scalar, was the original
  arrangement and is strictly worse once the loads are vectorized.
- **Compile-time mode presence.** The two edge kernels are specialized on
  whether the descriptor has radial modes at all. Sharing one kernel forces the
  register allocator to satisfy the mode path even when the rank is zero, which
  cost the rank-zero profiles 8--17%. Specializing recovers that and leaves
  them 4--7% faster than before the mode work began.
- **Compile-time high-degree components.** Degrees three and four distribute
  their single-channel components across the lanes of an edge group. Iterating
  the component at compile time and selecting with a lane predicate turns the
  harmonic selector into a folded constant, whereas the natural formulation,
  iterating the lane's own tiles, leaves a sixteen-way indirect branch in the
  inner loop. Degree three gains 11--14% and degree four up to 7%.
- **Single-ordering 222 contraction.** The symmetrized `222` invariant is
  `k(tr(Q_a Q_b Q_c) + tr(Q_a Q_c Q_b))`, and for symmetric factors
  `tr(ACB) = tr((ACB)^T) = tr(BCA) = tr(ABC)`, so the two orderings are
  identical and the averaging factor folds into the coupling constant. One
  ordering therefore replaces four 3x3 products per node with two. The block is
  four outputs on four lanes once per node, so the end-to-end effect is below
  the measurement resolution; the change stands because the arithmetic was
  provably redundant, not because it was measurable.
- **Unit-norm angular basis.** The harmonics are evaluated on a unit direction,
  so the squared norm that makes each degree traceless is exactly one.
  Substituting the constant is exact on the unit sphere for the value and for
  the gradient alike: two polynomials that agree on the unit sphere differ by a
  multiple of `|u|^2 - 1`, whose gradient at a unit vector is purely radial and
  is therefore annihilated by the tangential projection that closes the
  coordinate VJP. The regularized direction departs from unit norm by a
  relative `eps^2 / rho^2`, which stays far below single precision at every
  physical separation. This removes three products from the degree-two basis,
  two terms from its VJP, and the norm from every degree-three and degree-four
  expression.

### 11.5 Rejected or non-effective changes

| Experiment                                       | Result                                                                                                                                                                                                                 |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Coarsening the radial table from 0.002 to 0.05 A | Identical time at every width, so the table is not capacity-limited in cache and no accuracy may be traded for it                                                                                                      |
| Simultaneous value/derivative Horner sweep       | Three fewer instructions per channel but one dependency chain of ten instead of two overlapping chains; C128 +9%                                                                                                       |
| Unrolling the edge loop by two                   | Register pressure from two edges in flight; +2% at C8 rising to +13% at C64                                                                                                                                            |
| Relaxing the register cap to 85 (24 warps)       | Slower at every width; the kernel prefers occupancy over scheduling headroom                                                                                                                                           |
| 16-lane backward edge groups at C64/C128         | Halving the in-flight edge count costs 49% and 21% of backward time                                                                                                                                                    |
| 4-lane backward edge groups at C64/C128          | Register pressure returns; C64 +7%, C128 +30% with 32 bytes of spill                                                                                                                                                   |
| 8-lane forward edge groups at C128               | Leaves half of the degree-one lanes idle; forward +24%                                                                                                                                                                 |
| 2-lane edge groups at C16                        | 24 bytes of spill in both directions; +35%                                                                                                                                                                             |
| Padding each channel to eight coefficients       | Two aligned 128-bit loads, but 33% more traffic; superseded by the split-block layout, which costs no extra traffic                                                                                                    |
| Type-grouped factorization of the FiLM cache     | Would remove the per-edge shift and mixing loads, but needs per-source-type accumulators, which do not fit shared memory at production type counts; it would only accelerate few-element systems such as the benchmark |
| Cubic Hermite table                              | Faster, but only C1; rejected because knot second derivatives are discontinuous                                                                                                                                        |
| FP16 or TF32 anywhere                            | Rejected; violates the FP32 smooth-potential contract                                                                                                                                                                  |

### 11.6 End-to-end effect

The whole-step LAMMPS scan of the five released grades reports saturated
throughputs of 16,814, 10,191, 6,969, 3,800, and 2,165 atoms/ms for Nano
through Plus, each on 14,051,520 atoms, against 8,548 for NEP89 and 8,873,
7,012, and 4,692 for compact canonical DPA1 S/M/L. The complete curves are described in `doc/outisli/dpa4c.md`,
together with the mode-rank and angular-degree sweeps, whose ratios predate
node tiling and whose figures are regenerated by a full `parameter_sweep.py`
run.

### 11.7 Remaining high-value directions

These concern the descriptor kernels. Capacity is treated separately in
Section 11.9, and Section 11.10 places these kernels inside the whole-step
budget, where they account for somewhat less than half of a LAMMPS step.

1. Occupancy is capped at 32 warps per SM on this device. One-warp blocks hit the
   32 blocks-per-SM architectural limit, and 64 registers per thread exactly
   exhausts the 65,536-register file at that occupancy, so multi-warp blocks do
   not help either. Going further requires a working-set reduction that lowers
   the per-thread register count, not a launch-configuration change. Both the
   block limit and the resident-warp capacity are device constants -- 24 and 48
   respectively on the part measured in Section 11.13 -- so the arithmetic must be
   redone per device even though the conclusion has so far been the same on both.
1. The widest profile is close to an L1 throughput limit. The only remaining
   lever there is fewer bytes per edge, which the type-grouped factorization
   above could provide for few-element systems if a general form is found.
1. Degree four remains the weakest profile at the narrow scalar widths, where
   its sixteen single-channel components dominate a small channel loop. Both
   directions of widening the edge group were measured and rejected, so a gain
   there requires a different distribution of those components rather than a
   retuning of the existing one.
1. Counter profiling. Timeline data alone is insufficient to claim hardware
   saturation, and `ERR_NVGPUCTRPERM` withholds the counters on this host.
   Section 11.13 carries the profile taken where permissions allow it, and it
   revises the narrow-width attribution above; the H20 numbers in this section
   remain timeline and ablation measurements.

### 11.8 Fitting network study

The descriptor kernels dominate a step only while the fitting stays narrow.
Measured in isolation at one million nodes with the 144-wide descriptor of the
S and M grades and three hidden layers, the fitting costs 5.3 ms at width 64
and 71.7 ms at width 384, against descriptor totals of 0.5 to 3.3 ms per
32,768 nodes. At the production
pairings the fitting is a fifth to two fifths of model computation, so it is
worth the same scrutiny as the descriptor.

#### Where the time is

A timeline profile at width 64 splits the fitting into 70.4% GEMM, 26.3%
elementwise epilogues, and 2.4% head. Neither part is near its roof:

| Quantity                   |                        Width 64 |          Width 256 |
| -------------------------- | ------------------------------: | -----------------: |
| GEMM throughput            | 13.5 TFLOP/s (31% of fp32 peak) | 18.8 TFLOP/s (43%) |
| GEMM effective bandwidth   |         1.14 TB/s (28% of peak) |                 -- |
| Forward epilogue bandwidth |         2.94 TB/s (74% of peak) |                 -- |

The epilogues are bandwidth-bound and close to the roof; the GEMMs are at
neither roof, which is characteristic of a short reduction axis.

#### Pre-activation as the saved state

A layer must materialize its pre-activation, because that is where the GEMM
writes, and must materialize its activation, because the next GEMM reads it.
Storing the activation derivative as well is a third full-width store. The
layout instead points the GEMM directly at the retained buffer and lets the
backward re-derive `act'` from the pre-activation, which is the traffic
minimum of two stores and one load per element.

The measured trade is almost exactly neutral, and instructively so. Removing
the store cut the forward epilogue by 38% per layer instance and the whole
forward by 4-6%; recomputing `act'` cost the backward 1-7%. On this device one
fp32 store and one `tanhf` per element both come to roughly 0.38 ms per 384
million elements, so storage and recomputation are priced the same. Totals
move from -3.3% at width 64 to about zero at width 128.

The layout is retained for its structural consequences rather than its speed:
the operator carries one saved-tensor concept instead of two.

The per-layer timestep was dropped in the same change, and for an unrelated
reason. It is not incompatible with this layout -- the scale is a
length-`dout` vector, so the backward could form `act'(pre + b) * idt` from
the same saved pre-activation -- but no deployed model uses one, and carrying
it costs a load and a multiply per element in the forward and in both backward
epilogues. The eligibility gate therefore rejects it and such models fall back
to the reference network. Since the library default is `resnet_dt=True`, that
gate is load-bearing: it is covered by a negative test, and every entry point
that converts a fitting module into operator arguments goes through
`fitting_operator_arguments`, which raises rather than silently evaluating a
different network.

#### Rejected: a fused square-layer kernel

The roofline argues that a layer of width W should cost
`max(12WN / BW, 2W^2N / FLOPS)`, which at W = 64 is 0.19 ms per layer against
the 0.71 ms that cuBLAS plus an epilogue actually take. A fused kernel that
stages the weights and a tile of input rows in shared memory, keeps the whole
reduction in registers, and writes only the pre-activation and the activation
was written and measured:

| Variant                    | Square layer, width 64 |
| -------------------------- | ---------------------: |
| cuBLAS GEMM plus epilogue  |                 686 us |
| Fused, 64 nodes per block  |         553 us (1.24x) |
| Fused, 128 nodes per block |         545 us (1.26x) |
| Fused, persistent grid     |     slower than either |
| Fused, width 128           |    no gain over cuBLAS |

The kernel reaches 34% of both roofs simultaneously, which locates the limit
in the inner loop rather than in memory: a one-dimensional register tile reads
one weight quartet and four input scalars from shared memory for every sixteen
fused multiply-adds, so roughly a third of issued instructions are loads.
Closing the gap needs a two-dimensional register tile with double-buffered
staging -- a CUTLASS-class implementation. A hand-written GEMM whose measured
return is 7% of the forward at one width, and nothing at the next width up,
does not justify a second numerical path through the layer, so it was removed.

The persistent-grid variant is worth recording separately because it was
expected to win. Staging the weights once per block rather than once per node
tile removes real global traffic, but the `__syncthreads()` that must then
separate staging from computation on every iteration serializes what a
non-persistent grid overlaps across blocks for free.

#### Rejected: rewriting the SiLU sigmoid

`graph_fitting.cu` is compiled without `--use_fast_math`, so its `tanhf` is the
accurate library routine and the sigmoid identity `(1 + tanh(z/2))/2` looked
like an obvious candidate for the direct exponential form. It is not. Measured
per kernel instance at one million nodes and width 64:

| Sigmoid form                    | Forward epilogue | Backward epilogue | Seed epilogue |
| ------------------------------- | ---------------: | ----------------: | ------------: |
| `(1 + tanhf(z/2)) / 2`          |       194,983 ns |        246,065 ns |    223,542 ns |
| `__frcp_rn(1 + __expf(-z))`     |       215,534 ns |        249,064 ns |    224,460 ns |
| `__fdividef(1, 1 + __expf(-z))` |       190,904 ns |        244,937 ns |    221,475 ns |
| `1 / (1 + expf(-z))`, accurate  |     slower still |                -- |            -- |

Three things come out of this. The CUDA `tanhf` is already a branch-free
two-MUFU sequence -- one `MUFU.EX2` and one `MUFU.RCP` with about thirteen
float operations -- so the identity is not the naive choice it looks like. The
accurate `expf` is the worst of the four because its implementation carries a
branchy slow path; thirteen `BRA` instructions appear in the disassembly that
the tanh form does not have. And the correctly rounded reciprocal needs a
Newton refinement, which costs more than the tanh scaling it was meant to
replace.

The only form that wins is the fully approximate one, by 2.1% of the forward
epilogue. That is 0.5% of the fitting forward and roughly 0.1% end to end, in
exchange for an approximate division and an approximate exponential. The
identity is kept.

The activation is a compile-time template parameter, so supporting tanh
alongside SiLU costs nothing at run time and only two extra instantiations of
three small kernels at compile time. There is no throughput argument for
narrowing the operator to SiLU.

#### Not applicable: asynchronous shared-memory copy

`cp.async` on Ampere and later, and the bulk descriptor copies on Hopper,
accelerate a global-to-shared staging pipeline: they bypass the register file
and let stage `i+1` of a copy overlap stage `i` of computation. Neither
precondition holds here.

The fitting epilogues use no shared memory at all; they are streaming
elementwise kernels. The DPA4C kernels use shared memory almost entirely for
computed values -- moments, aligned moments, probes, the mode caches, the
normalizers, the scalar cotangent. The single global-to-shared copy in the
whole path is the node backward staging the saved moment state, which is
consumed immediately after one barrier with no independent work to overlap.

The equivalent idea was also measured directly, in the fused square-layer
prototype above: a persistent grid that stages once per iteration was slower
than a non-persistent grid, because the barrier separating staging from
computation serializes what independent blocks otherwise overlap for free.

L2 residency control through `cudaAccessPolicyWindow` was considered for the
radial table and rejected analytically. The table is 2.6 MB at the widest
production grade against 60 MB of L2, so it is already resident by more than
an order of magnitude and pinning it cannot change its hit rate.

#### Neutral: two-dimensional epilogue indexing

Deriving a channel from a flat element index costs a 64-bit division per
thread, and the seed epilogue paid two. Indexing (channel, node) directly
removes them, and measured neutral to within 0.5% at every width: the
epilogues are bandwidth-bound, and the divisions hide entirely under memory
latency. The indexing is kept because it is no more complex than the flat
form, but it should not be cited as an optimization.

### 11.9 Capacity study

Throughput and capacity are separate axes. Throughput is set by the kernels;
capacity is set by whatever scales with the atom count, and optimizing one
tells you nothing about the other. This section measures the capacity budget,
records what was removed from it, and states why the residue is what it is.

#### Measuring the budget

Device memory is sampled while a LAMMPS step runs and fitted against the atom
count over three system sizes. Two sizes would give a slope; three confirm the
model is linear, which it is to better than half a percent:

| Atoms     | Device peak |       Fit |
| --------- | ----------: | --------: |
| 2,985,984 |   20.75 GiB | 20.85 GiB |
| 5,026,560 |   34.00 GiB | 33.85 GiB |
| 8,998,912 |   59.09 GiB | 59.14 GiB |

That is `6,838 bytes per atom + 1,881 MiB`. Against a 95.22 GiB device the fit
predicts 14.66 million atoms and the scan reaches 14,051,520, the remaining 4%
being allocator fragmentation. A budget this predictive is what makes the rest
of the section arithmetic rather than guesswork. `debug/cuda_bench/memory_profile.py`
performs the sampling and the attribution.

#### What the budget was, and what tiling removed

Before tiling the same measurement gave 9,347 bytes per atom, of which the
model's own node-scale tensors were the majority: at the Neo widths the fitting
pre-activations and activation ping-pong came to 5,120 bytes per atom, more
than the graph and the descriptor together, and the descriptor, its cotangent
and the moment state added a further 2,104.

None of it needs to be node-scale. Inference knows the head cotangent before
the forward runs -- it is the ownership mask -- and both the descriptor and the
fitting are evaluated per node, so a run of consecutive nodes can complete the
descriptor forward, the fitting forward, the head, the fitting backward and the
descriptor backward before the next run starts, retiring everything it
allocated. Nothing is recomputed, which is what separates this from the usual
activation-checkpointing trade.

Destination-sorted CSR is what makes the partition exact. A run of consecutive
destination nodes owns a contiguous span of the edge axis, and every edge has
exactly one destination, so runs neither share work nor split a reduction. A
run receives a slice of the destination row pointer, whose entries stay
absolute, plus its first node index, which is needed only because neighbour
lookups index the atom type table with absolute source indices.

Three operator-level changes implement it:

- `graph_fitting_energy_gradient` walks node tiles, and the cotangent replaces
  the descriptor in place: a tile reads its descriptor rows in the first layer
  and writes their cotangent in the last, and no later tile revisits them. Its
  forward activation ping-pong and its two backward cotangent buffers share one
  allocation, because the backward reads only the saved pre-activations.
- `dpa4c_canonical_compress_energy_gradient` folds the descriptor forward, the
  fitting and the descriptor backward into one operator over the same runs. The
  loop lives in C++ because a Python loop's trip count follows a dynamic node
  count, which export rejects with a `ConstraintViolationError`.
- `DP_NODE_TILE` sets the run length, 131,072 nodes by default. The choice is
  not sensitive: between 64k and 256k the isolated operator moves by 3%, so the
  tile is chosen for the memory it bounds.

#### Effect

Across the five grades, at most 2.0% of throughput for up to 155% of capacity:

| Grade | Atoms/ms before |  after | Largest system before |      after |
| ----- | --------------: | -----: | --------------------: | ---------: |
| Nano  |          15,738 | 15,699 |            11,036,032 | 14,051,520 |
| Mini  |           9,988 |  9,813 |             9,524,736 | 14,051,520 |
| Neo   |           6,919 |  6,783 |             8,000,000 | 14,051,520 |
| Air   |           3,773 |  3,740 |             8,000,000 | 14,051,520 |
| Plus  |           2,139 |  2,146 |             5,510,880 | 14,051,520 |

Both columns were measured before the reductions of Section 11.11, which the
current figures in Section 11.6 include; the pair is kept as measured because
it isolates tiling.

Every grade now reaches the same system size. That is the substantive result:
nothing which depends on the model width survives at system scale, so capacity
is decided by the cutoff alone, which the grades share. The widest grades gain
most because their retired fitting state was largest.

#### What remains

Of the 6,838 bytes per atom, 5,072 are graph tensors and 1,766 belong to
LAMMPS with Kokkos, mostly its neighbour list. Both codes run a 1 Å skin --
`neighbor 1.0 bin` in the LAMMPS input, `rc + 1.0` in the GPUMD neighbour
build -- so the candidate shell is 7 Å, about 127 half-list candidates per atom
at diamond density against the 158 full-list neighbours the model consumes.
The fixed 1,881 MiB is the CUDA context, the cuBLAS workspace, the compression
artifacts and the tile buffers; it is 2% of the budget at capacity.

| Array              | B/atom | Share | Removable              |
| ------------------ | -----: | ----: | ---------------------- |
| `edge_vec`         |  1,896 | 27.7% | No, see below          |
| Edge cotangent     |  1,896 | 27.7% | No, determinism        |
| `source`           |    632 |  9.2% | No, it is the topology |
| `source_order`     |    632 |  9.2% | No, it is the topology |
| Row pointers       |     16 |  0.2% | --                     |
| LAMMPS with Kokkos |  1,766 | 25.8% | Outside this boundary  |

`edge_vec` looks removable, since the kernel could gather ghost coordinates and
subtract. It is not worth it. LAMMPS coordinates are fp64, so a gather costs 24
bytes against the 12 of a sequential `edge_vec` read, and the neighbour
compaction pass already reads them to apply the cutoff, so nothing is saved
there. Per surviving edge the traffic would rise from 36 bytes -- one store and
two sequential reads -- to 48, and the two reads would become gathers where
they are now perfectly coalesced. Storing fp32 absolute coordinates instead
would halve the gather but invites catastrophic cancellation: at fp32 a 400 Å
box carries about 2.4e-5 Å per coordinate, two orders of magnitude worse than
the current geometry and growing with the box.

The edge cotangent cannot be retired per run either. The source side of the
force assembly walks `source_order`, and one atom's outgoing edges belong to
many destination runs, so the full array must exist before the scatter. The
only alternative is accumulating force atomically, which is exactly the
non-determinism removed earlier.

#### The floor is set by the materialized graph, not by physics

The four graph arrays are irreducible only because the graph is materialized
before the model runs. A streaming interface would remove them, and the
argument that rules it out today is narrower than it first appears.

Determinism is not the obstacle. Runs execute in order on one stream, so
accumulating each run's contribution into a force component in run order is
deterministic without atomics. What is missing is a per-run source index: the
source side of the assembly walks `source_order`, a permutation over the whole
edge axis, and restricting it to one run needs a two-level index from source
atom to run to edges rather than the single global permutation. Because
destination order is spatially sorted, an atom's outgoing edges fall in one or
two runs, so such an index would be shallow.

The larger obstacle is the interface. `edge_vec`, `source` and `source_order`
are produced by the Kokkos pair style before the model is called, so making
them transient turns graph construction from "materialize, then evaluate" into
a run loop that builds a run's edges, evaluates it and discards them. Total
build work stays linear in the edge count, since the neighbour traversal is
already per atom, but the pair style, the graph ABI and the force assembly all
change together.

What it would be worth: removing `edge_vec`, `source`, `source_order` and the
edge cotangent takes 5,056 of the 6,838 bytes per atom, leaving Kokkos's own
neighbour list as the floor at roughly 1,800 bytes per atom, or about fifty
million atoms on this device. That is the largest remaining capacity lever by a
wide margin, and it is an interface project rather than a kernel one.

#### An O(N^2) bug worth recording

The kernel that clears the reserved edge slots zeroes everything from the last
destination row of the launch to the end of the edge storage. Under tiling that
is nearly the whole edge array, once per run, so step cost grew as the square
of the atom count: the plateau decayed monotonically, worst for the fastest
grade, 38% across Mini's range.

It was not producing wrong answers, and the reason is the interesting part.
Runs execute in order, so a later run rewrote whatever an earlier one had
erased -- correctness resting on execution order rather than on the partition.
Only the run ending at the last node may clear the padding, and its row pointer
terminates exactly where the padding begins, so the fix is also the precise
statement of the invariant.

### 11.10 Whole-step kernel budget and graph construction

Sections 11.3 through 11.8 optimize the model kernels in isolation. Once those
approached their limits, the question became what else a LAMMPS step spends
time on. The answer reshaped the priority list: a fifth of the step is not
model evaluation at all.

The measurement is an `nsys` timeline of Neo at one million atoms, five warmup
plus twenty timed steps. Instance counts identify each entry unambiguously
over those twenty-five steps: 27 is once per step, 216 is once per node run at
eight runs per step, and 4 is the number of neighbor list rebuilds.

| Stage                                       | Share | Instances |
| ------------------------------------------- | ----: | --------: |
| Descriptor edge backward                    | 24.1% |       216 |
| Fitting GEMM                                | 23.3% |      1269 |
| Descriptor forward                          | 19.7% |       216 |
| Graph construction (three dominant kernels) | 16.6% |   27 each |
| LAMMPS neighbor list rebuild                |  5.6% |         4 |
| Fitting elementwise                         |  2.9% |      1080 |
| Force and virial assembly                   |  2.4% |        27 |
| Descriptor node backward                    |  1.6% |       216 |
| Per-frame energy reduction                  |  1.2% |        27 |

Graph construction is the only large entry that had never been examined. Three
of its kernels dominate: a candidate count pass, a fill pass that writes source
indices and edge vectors, and a scatter that permutes edges into source-major
order. The accompanying prefix scans and cursor initialization are each below
0.1% and are omitted from the table.

#### Why it is slow

The count and fill passes move roughly 9.6 GB in 14.1 ms, which is 680 GB/s, or
17% of what the device sustains. Neither pass is bandwidth limited. The inner
loop explains the gap:

```cpp
const int j = d_neighbors(i, jj) & NEIGHMASK;    // coalesced
const int mj = loc2model(multi ? j : owner(j));  // gather, then gather
const double dx = x(j, 0) - xi;                  // gather
```

Every candidate walks a chain of dependent gathers, and the loop is serial
within a thread, so few of them are ever in flight. Achieved bandwidth rules
out a bandwidth limit, and the remaining time is consistent with gather
latency; settling that attribution requires the memory-dependency stall
counters, which host permissions withhold.

#### Rejected for the count pass: one warp per center

The natural reading of a strided access pattern is that lanes should cooperate
on one center's candidate list rather than each taking a center of its own.
Converted to a `TeamPolicy` of 32 lanes per center, the count pass went from
6,939 to 7,528 microseconds, 8.5% slower.

The premise was wrong. LAMMPS types the device neighbor list as
`tdual_int_2d_dl`, whose device layout is `LayoutLeft`, precisely so that the
conventional thread-per-center traversal coalesces: consecutive `i` are
adjacent in memory at fixed `jj`. Thread-per-center was already the coalesced
mapping. Assigning lanes to candidates converts that read into a stride of
`natoms`, and the team reduction adds overhead on top. The conclusion is specific to a pass that only reads: the `LayoutLeft`
neighbor list already rewards thread per center, and a team adds overhead for
nothing. Section 11.11 finds the opposite balance in the fill pass, whose
stores are what the warp mapping coalesces.

#### Effective: composing the candidate maps

In the folded single-rank representation a candidate resolves to a model node
through the ghost owner and then through the model map, two dependent gathers
in a loop that runs 253 million times per pass. Both tables are built on the
host and synced once per neighbor rebuild, so the composition can be formed in
the same place at O(nall) cost and consulted with a single gather. The
extended multi-rank representation gives ghosts their own nodes, where the
composition degenerates to the model map and the traversal already performed
one gather; that path gains nothing and is unaffected.

The composition also replaces the ownership table rather than joining it. That
table existed only to be consumed by this resolution, so once the resolution
moves to the host nothing reads it on the device, and device memory is
unchanged. This matters because memory sets the maximum system size.

The fill pass went from 14,067 to 13,808 microseconds and the count pass from
6,939 to 6,842, about 1.5% on the two traversals and 0.25% end to end. The
gain is small, and the likely reason is worth recording: the two tables are
4.4 MB each against a 60 MB L2, small enough that the removed gathers were
served largely from cache rather than from HBM. The change is kept because it
costs no memory and removes an indirection, not because it is significant.

Total edge counts are recomputed every step rather than reused between
neighbor rebuilds, and that is required rather than conservative. Reuse would
demand that the edge set include the skin, since a pair can cross the model
cutoff between rebuilds, and carrying `(rcut + skin)^3 / rcut^3` times the
edges costs 59% more edge memory. Capacity is the binding constraint, so the
exact per-step test stands.

#### Headroom that was quantified and declined

The fitting GEMM runs at 21.9 TFLOP/s on shapes of M=131072, N=K=256, moving
only 342 GB/s, so it is compute bound rather than memory bound. With TF32
excluded by the precision policy the ceiling is the SIMT fp32 peak near 44
TFLOP/s, which puts cuBLAS at 50% of peak. Hand-written SGEMM reaches 60% to
70% on large square problems, so the realistic headroom is 1.2x to 1.3x on
23.3% of the step, or 4% to 5% end to end, in exchange for a CUTLASS-class
kernel to maintain.

That 50% is a property of this device. Section 11.13 measures the same shape at
78% of fp32 peak on a part with a different balance, where the hand-written-SGEMM
argument leaves no headroom at all, so the figure has to be re-measured rather
than assumed before the trade is reconsidered.

Relaxing the precision policy would not rescue it either. The accuracy-
preserving tensor-core route is a split representation such as 3xTF32, which
needs three to six tensor-core passes per product. This device peaks near 74
TFLOP/s on TF32, so the split path lands between 12 and 25 TFLOP/s and does not
beat the 21.9 TFLOP/s the SIMT path already delivers. The cut-down tensor cores
remove the usual reason to leave fp32 SIMT, so the GEMM is at its practical
ceiling here whatever the precision policy allows.

### 11.11 Second optimization round: reductions and the fill pass

The budget above put two thirds of a step in the model kernels, which earlier
rounds had already taken close to their limits, and the rest in graph
construction, the neighbor list, and the reductions around them. This round
worked the remainder. It matters more than its share of a Neo step suggests,
because those costs do not scale with the model: at Nano they are more than
half of the step, and Nano is the grade that runs the largest systems.

#### Per-frame energy reduction

The reduction of node energies to frame totals cost 1,758 microseconds, 1.2% of
a Neo step, for arithmetic that is trivial in molecular dynamics. It built a
node-length frame index and scattered through `index_add_`. Molecular dynamics
carries one frame, so that index is identically zero and a million atomic
additions serialize on a single address.

The force path had already solved this problem for the virial, which is the
same reduction over the same contiguous frame segments: a block reduces a
strided slice of one (frame, component) pair into an fp64 partial, and a second
pass folds the partials. Generalizing those two kernels over the component
count lets the energy use them at one component and the virial keep using them
at nine, so the fix adds a template parameter rather than a code path.

| Reduction                               |        Before | After |
| --------------------------------------- | ------------: | ----: |
| Energy kernel time                      |      1,758 µs | 17 µs |
| Distinct results over 20 identical runs | more than one |     1 |

Determinism improves because a segment reduction fixes the summation order from
the launch geometry, while the atomic scatter left it to arrival order. The
saving is a constant 1.74 milliseconds per step, so it is worth more to the
cheaper grades:

| Grade | Step before | Step after | Gain |
| ----- | ----------: | ---------: | ---: |
| Nano  |    63.65 ms |   61.91 ms | 2.8% |
| Mini  |   100.29 ms |   98.55 ms | 1.8% |
| Neo   |   143.43 ms |  141.69 ms | 1.2% |
| Air   |   267.32 ms |  265.58 ms | 0.7% |
| Plus  |   466.89 ms |  465.15 ms | 0.4% |

#### The fill pass: one warp per center

The graph-construction study above rejected a warp-per-center count pass. The
fill pass is a different problem, and the distinction is the point.

An ablation isolates it. Removing the source-count atomic, which is one atomic
per edge and 158 million per step, took the fill pass from 13,816 to 12,930
microseconds. The atomic is worth 886 microseconds, not the several
milliseconds that separate the fill pass from the count pass, so the two
traversals differ in their stores rather than in their counting.

They do. A thread-per-center fill has each thread writing at an offset private
to its center, so the lanes of a warp scatter their twelve-byte edge vectors
across thirty-two unrelated rows. The count pass writes nothing and therefore
pays none of this. Cooperating on one center sends consecutive survivors to
consecutive slots and coalesces the dominant store stream, which is 2.5 GB per
step.

Ordering is preserved exactly. Candidates are taken a warp at a time and each
lane writes at its exclusive prefix within the warp, so lane order is candidate
order within a chunk and chunks advance in candidate order. The edge sequence
is the one the serial fill produces, bit for bit, which is why the change needs
no numerical argument.

| Kernel                                    |    Before |     After |
| ----------------------------------------- | --------: | --------: |
| Fill pass                                 | 13,816 µs | 12,053 µs |
| Count pass (unchanged, thread per center) |  6,888 µs |  6,861 µs |
| Graph construction total                  | 24,106 µs | 22,412 µs |

The count pass keeps its thread-per-center mapping. Applying the same transform
to it costs 8.5%, as recorded above, because it has no stores to coalesce and
only pays the loss of the `LayoutLeft` coalescing on the neighbor list. The two
passes traverse identically and want opposite thread mappings; the store stream
decides.

#### Rejected: staging the compacted edge vectors

An edge vector spans three words, so even with consecutive survivors each lane
issues three four-byte stores at a stride of twelve bytes, and a warp's 384
bytes reach three cache lines per instruction where one would do. Staging the
compacted vectors in scratch and flushing them word by word makes the global
store perfectly contiguous.

It was slower: 12,053 to 12,328 microseconds. The extra shared-memory round
trip and the barrier it requires cost more than the coalescing recovers, which
says the stores after the warp-per-center change are already close enough to
contiguous for write combining to absorb the remainder. Reverted.

#### Rejected: larger node tiles

The tile bounds every node-scale allocation, and at one million atoms on a 96 GB
device it could be raised without any memory pressure. Measured at Neo, raising
it monotonically loses throughput: 142.79 ms per step at 131,072 nodes, 143.16
at 262,144, and 143.61 at 524,288. The launches the tile adds are already
cheaper than the locality it preserves, so the default stands and needs no
size-dependent policy.

#### Rejected: more aggressive spatial sorting

The remaining cost of the traversal is gather latency on coordinates, which
depends on how atoms are ordered in memory. Tightening `atom_modify sort` does
not help: 141.54 ms per step at the default, 142.83 at `sort 100 3.0`, and
144.38 at `sort 20 2.0`. The generated lattice is already in spatial order, so
the sort has no locality left to recover and only adds its own cost.

#### Combined effect

Both changes remove a fixed cost per step, 1.74 milliseconds from the reduction
and 1.69 from the fill pass, so the relative gain scales inversely with the
cost of the grade. Whole-step throughput at one million atoms, measured under
LAMMPS with Kokkos on one device:

| Grade | Step before | Step after | Gain |
| ----- | ----------: | ---------: | ---: |
| Nano  |    63.19 ms |   59.76 ms | 5.4% |
| Mini  |   100.36 ms |   96.93 ms | 3.4% |
| Neo   |   143.28 ms |  139.85 ms | 2.4% |
| Air   |   266.45 ms |  263.02 ms | 1.3% |
| Plus  |   466.24 ms |  462.81 ms | 0.7% |

The "before" column adds the two measured savings back to the measured step, so
the gains carry the run-to-run spread of the benchmark, which is near 1%. A
directly measured Neo baseline of 142.79 ms taken before either change brackets
the modelled 143.28 within that spread.

Those steps are single points at one million atoms. The full capacity scan,
whose saturated throughput averages every successful point of at least one
million atoms, reports 16,814, 10,191, 6,969, 3,800, and 2,165 atoms/ms for
Nano through Plus, each still reaching 14,051,520 atoms and first failing at
14,522,880. Capacity is unchanged, which is the intended outcome: both changes
were required to cost no memory.

Neither change adds device memory at system scale. The reduction replaces a
node-length index tensor with a partial buffer of one fp64 per frame and
parallel slice, bounded by the slice cap rather than by the node count, and the
fill pass adds 896 bytes of team scratch per block, which is shared memory
rather than global. Maximum system size is unaffected.

#### Hardware counters remain unavailable

Every attribution in this section rests on timeline data, achieved-bandwidth
arithmetic, static resource analysis, and ablation. Nsight Compute reports
`ERR_NVGPUCTRPERM` even for a root user, because `RmProfilingAdminOnly` is set
in the driver module and clearing it requires reloading the module on the host.
The ablation of the source-count atomic shows why this matters: it refuted the
hypothesis that atomics dominated the fill pass and redirected the work to the
store pattern, which is what actually paid.

### 11.12 Native spin specialization

Native spin is a compiled variant of the same kernels rather than a separate
operator. `Profile<Channels, Lmax, HasSpin>` gains a spin block, the three
schemas (generic, compact canonical, level-two fused) carry three extra
tensors, and the descriptor backward returns three cotangents instead of one.

#### What the variant adds

The neighbour spin width follows the degree-two width, `C_s = C_2`, so the
variant needs no structural parameter of its own: presence is the whole
choice. The moment row grows by `8 C_s + 5` reduced coordinates -- magnitude,
magnetic coordination, the isotropic spin vector, the bond-projected spin
vector, and the spin quadrupole -- plus eight node-local on-site coordinates
that no edge writes. Because the geometric row is
`C_0 + 3 C_1 + 5 C_2 = 3.75 C_0` and the degree profile gives `C_1 = C_0 / 2`
and `C_2 = C_0 / 4`, the spin block is `2 C_0 + 13`, or a little over half the
geometric row at every width.

Three inputs cross the operator boundary. `spin` is the raw per-node moment,
indexed by absolute node id like `atype`, because neighbour lookups address it
with source indices. `spin_pair` interleaves the ordered spin scale and shift
so one channel arrives in a single 64-bit load, matching the geometric
PairFiLM cache. `spin_type` packs the four per-type scalars a node needs into
one 128-bit row: the gate divided by the reference magnitude, the bare gate
that the magnetic-coordination family counts, and the two on-site weights.

#### The bond-projected family, and why it reaches the angular VJP

Four of the five families read only the neighbour moment. The fifth reads the
edge direction as well,

```
P_c = sum_j phi^s_{ij,c} (s_hat_j . u_hat_ij) u_hat_ij
```

with `s_hat_j` the conditioned neighbour moment, `u_hat_ij` the unit bond
direction and `phi^s_{ij,c}` the spin edge amplitude that every family shares.
`P` joins the on-site moment and the isotropic neighbour channels on the channel
axis of a single degree-one block of width `1 + 2 C_s`, so the half Gram of that
block emits every admissible degree-one spin invariant without a new rule. The
parity matches the isotropic family: inversion flips both the moment-direction
product and the bond vector, so the family is even, and time reversal flips the
moment alone, so it is odd.

Its cotangent is what distinguishes it in the kernel. Writing `w` for the
amplitude, `s` for the conditioned moment, `u` for the direction and `dP` for
the incoming cotangent of `P`, the single dot product `dP . u` carries all three
vector-Jacobian products:

```
dL/dw = (s . u) (dP . u)
dL/ds = w (dP . u) u
dL/du = w [ s (dP . u) + (s . u) dP ]
```

Only the third is new to the spin branch. Until this family existed no spin
moment depended on the direction, so the spin backward produced a radial and an
envelope cotangent and nothing angular. The direction term is accumulated into
the same register triple that the single-channel high degrees fill, and so
passes through the one transverse projection in `basis_vjp` that turns a
direction cotangent into a coordinate gradient -- the projection is what
accounts for `u` being normalized, exactly as the geometric degree-one family
already requires. Joining the two cotangents before that projection rather than
reducing them separately is what keeps one edge producing one coordinate
gradient instead of one per family.

#### Magnetic force across a destination-major scan

The energy gradient with respect to a moment splits in two. The on-site half
belongs to the node the block already owns, so it closes inside the node
backward with no communication. The neighbour half belongs to the source of
each edge, which a destination-major scan does not own; it is emitted per edge
and reduced onto source nodes afterwards through the source CSR, exactly like
the conservative force. The reduction is a segment sum over a gathered edge
axis, so summation order comes from the topology rather than from arrival
order, and the result is bitwise reproducible.

The level-two fused composition tiles over destination runs. The on-site
gradient is node-indexed, so a run addresses its own slice and the host offsets
the pointer by the run origin. The per-edge half spans tiles and is therefore
materialized over the whole edge axis and reduced once after the loop.

#### Cost

Diamond supercell, 32,768 atoms, 5,177,344 edges, 158 edges per node, one H20,
descriptor forward plus backward in isolation, minimum of five timed batches:

| Channels | baseline | native spin | overhead |
| -------- | -------- | ----------- | -------- |
| 8        | 0.547 ms | 0.943 ms    | +72%     |
| 16       | 0.855 ms | 1.172 ms    | +37%     |
| 32       | 1.248 ms | 1.743 ms    | +40%     |
| 64       | 1.942 ms | 2.464 ms    | +27%     |
| 128      | 3.364 ms | 4.115 ms    | +22%     |

From Mini upward the overhead sits between 22% and 40% against a spin block that
is a little over half the geometric row, so the variant costs materially less
than its coordinate count. The five families share the edge amplitude, the
envelope, the table lookup and the neighbourhood normalizer with the geometric
moments, and add no reduction of their own. Nano is the outlier because the
quadrupole and on-site blocks contribute a fixed thirteen coordinates that do
not shrink with the channel count, and at the narrowest width that constant
dominates. This is structural, not an implementation deficit; removing it would
require dropping a spin family, not tuning the kernel.

The overhead also falls as the geometric profile deepens, because the spin block
is fixed by `C_s` while the geometric one grows with the angular degree and the
radial modes: at 32 channels with `lmax = 4` and eight radial modes it is +5.9%,
6.778 ms against 7.175 ms, the smallest of any configuration measured.

#### Re-measuring the two spin edge-group widths

`EdgeMap` carries `SpinForward` and `SpinBackward` beside the geometric pair.
Both were re-measured for this layout rather than carried over, because the
bond-projected family adds three accumulators per lane and tile. They govern
different kernels, so a single build yields one point of each sweep; a group
wider than the scalar width leaves a tile empty and does not compile, which
bounds the narrow profiles from above.

Spin forward, milliseconds by group width:

| Channels | 2     | 4         | 8         | 16        | 32    |
| -------- | ----- | --------- | --------- | --------- | ----- |
| 8        | 0.455 | **0.375** | 0.617     | -         | -     |
| 16       | 0.859 | **0.463** | 0.647     | 1.071     | -     |
| 32       | 1.462 | 0.768     | **0.731** | 1.152     | 1.971 |
| 64       | 2.612 | 1.612     | **0.928** | 1.267     | 2.046 |
| 128      | 6.102 | 3.734     | 2.347     | **1.672** | 2.403 |

Spin edge backward, milliseconds by group width:

| Channels | 2         | 4         | 8         | 16    | 32    |
| -------- | --------- | --------- | --------- | ----- | ----- |
| 8        | **0.568** | 0.651     | 1.142     | -     | -     |
| 16       | 0.860     | **0.709** | 1.233     | 2.141 | -     |
| 32       | 1.395     | **1.013** | 1.390     | 2.443 | 4.314 |
| 64       | 2.345     | **1.536** | 1.697     | 2.716 | 4.803 |
| 128      | 4.826     | 3.359     | **2.444** | 3.487 | 5.537 |

Both curves are unimodal in the group width, which is what a trade-off between
in-flight edges and per-lane accumulators predicts. The optima differ from the
geometric pair in exactly two places. At Nano the spin forward wants one step
wider than the geometric two, because the eightfold spin payload of a two-lane
group tiles into more accumulators than the recovered edge concurrency is worth.
At Air the spin backward wants one step narrower than the geometric eight,
because the bond-projected family's angular cotangent adds per-lane state that
outweighs the geometry recompute a wider group amortizes. Elsewhere the
geometric widths are already optimal for the spin variant and are retained.

#### Three defects and what they teach

**A defaulted template parameter silently selected the wrong layout.**
`Profile` was declared `template <int Channels, int Lmax, bool HasSpin = false>`.
Two node-backward helpers were templated only on `<Channels, Lmax>` and wrote
`using P = Profile<Channels, Lmax>;`, so under a spin instantiation they read a
spin-free profile and used its `OutputWidth` as the row stride of the
descriptor gradient -- 70 instead of 103 at Nano. Every node but the first read
another node's row. The corruption entered through the degree-two probe
gradient and propagated to the whole angular block and to both mass cotangents,
leaving the coordinate gradient 18% wrong while the forward stayed exact.

The fix was to delete the default argument rather than to patch the two call
sites, so the compiler now forces every instantiation to state its layout, and
the two helpers that genuinely touch only geometric widths say
`Profile<Channels, Lmax, false>` with the reason recorded. A default argument on
a parameter that changes memory layout is a silent-corruption hazard: the
compiler cannot warn about a choice the author never had to make.

**A chain factor applied on one path and not the other.** The kernel
differentiates the conditioned moment, which is the raw input scaled by the
per-type gate over the reference magnitude. The node backward applied the
remaining factor when storing the on-site gradient; the edge backward stored
the per-edge cotangent unscaled. The magnetic force was therefore correct on
one half and off by the reference magnitude on the other. The store is now the
single point where the cotangent crosses from the conditioned moment to the raw
input, which is the only place the factor can be applied exactly once.

**A shared array of odd length displaced a vector-aligned one.** The staged
spin cotangent occupies `5 C_s + 5` floats. Declared before the mode cache, it
left that cache on a four-byte boundary, and the mode residual reads its rows as
`float4`. Every profile with radial modes faulted with a misaligned address. The
mode caches now carry an explicit `__align__(16)`, which is robust to whatever
precedes them. Shared-memory alignment is a property of the whole declaration
sequence, not of a single array, so any buffer read as a vector type should
state its alignment rather than inherit it.

#### Isolating a defect in a compiled specialization

A specialization that differs from its base only by a compile-time flag admits
a differential harness, which localizes a fault in one step where reading does
not. The construction is to make the suspected variant the only free variable.
Comparing a spin descriptor against a spin-free one does not qualify, because
the two carry different weights; comparing one descriptor against itself under
two instantiations does.

For a layout fault the harness invokes the operator twice on the same
compressed descriptor, once with the spin tensors present and once with them
replaced by empty tensors, holding every table bit for bit identical and
neutralizing the output calibration so the two column layouts stayed
comparable. Running the backward in its in-place form exposes `moment_gradient`
directly. The forward state matched bitwise while the moment gradient did not,
which pinned the fault to the node backward without reading any kernel code.

For the chain factor the harness disabled one half of the magnetic force at a
time -- zeroing the on-site weights leaves only the neighbour families, zeroing
the ordered spin tables leaves only the on-site ones -- and the neighbour half
returned a relative error of exactly 0.700 at every channel width. A constant,
width-independent ratio is the signature of a missing scalar factor rather than
of a corrupted reduction, and `|k - 1| = 0.7` identified `k = 1.7`, the
reference magnitude configured in the test.

A constant, width-independent ratio is the signature of a missing scalar
factor rather than of a corrupted reduction, which is what distinguishes a
chain-rule omission from a broken accumulation without reading either.

#### Two defects at the operator boundary

Both are properties of the registration rather than of any kernel, and both
affect every DPA4C model.

**Two unannotated outputs shared one allocation.** The backward returns
`(edge_gradient, spin_gradient, edge_spin_gradient)`, and on the spin-free path
the last two are absent. Building one empty tensor and returning it twice makes
`out[1] is out[2]`, an alias the schema does not declare, which is undefined
under functionalization. `torch.library.opcheck` rejects it with "Outputs 1 and
2 alias unexpectedly", for zero-element tensors as much as for populated ones.
Every absent output now receives its own allocation, in the fake and CPU
implementations as well as in the device one, and `opcheck` covers the backward
on both paths so the class of defect is caught mechanically rather than by
inspection.

**A registration that cannot close a gradient must refuse it.** The magnetic
cotangent leaves the operator in two pieces, and reducing the per-edge piece onto
source nodes needs the source CSR, which the schema does not carry.
`register_autograd` therefore genuinely cannot complete it, and returning `None`
in the spin slot made correctness depend on the Python wrapper noticing
`spin.requires_grad` and routing around the registration: a caller reaching the
operator directly received a vanishing magnetic force and no error. The
registration now raises when the moment requires a gradient. An incomplete
derivative is not a missing one, and a registration that cannot express the
difference should decline rather than answer zero.

#### Verification

The portable path is the oracle. `test_spin_compressed_matches_portable`
compares the descriptor, the coordinate gradient and the magnetic force across
`(C, lmax, radial_modes)` in {(8,2,0), (16,2,0), (32,2,4), (64,3,0), (128,3,8),
(32,4,0)}, at the tolerance the geometric backward already uses: `3e-5` for the
descriptor and `8e-6` absolute with `1e-4` relative for the gradients, because
tabulation error reaches the gradients through the table derivative. At the
model level `test_compressed_spin_lowers_match_autograd` holds the fused
generic lower and the compact canonical path to the autograd magnetic force at
the same gradient tolerance.

Three properties of the harness matter.

**The dispatch must be shown to be active.** The compressed path is gated on
`cuda_infer_level() >= 1`, and `DP_CUDA_INFER` defaults to `0`. A test that
enables compression but leaves the variable unset runs the portable code on both
sides and compares it with itself; the residual it measures is atomic
reordering, and an operator returning twice the correct magnetic force passes
every tolerance. Each test therefore sets the variable itself with
`monkeypatch.setenv` and asserts that the compressed descriptor is **not**
bitwise equal to the portable one. Tabulation guarantees a difference when the
kernel runs and an exact zero when it does not, so the assertion cannot rot into
a tautology. A test comparing CUDA level one against level two, or against the
canonical path, is a composition test rather than a kernel test: all three share
the descriptor operator.

**Probing every output column in one comparison** keeps the geometric and spin
blocks coupled through the shared normalizer and the cross Gram, which a
per-block probe would hide.

**The cotangent must stay bounded.** Weighting columns by their index inflates
the gradient magnitude at the widest profile until fp32 tabulation noise alone
exceeds the tolerance, and the spin-free path shows the same noise, so the
failure would be misread as a spin defect.

#### The deployment ABI is a separate gate

Making the kernels spin-capable also made `mega_eligible` accept a
spin-conditioned descriptor, and through it `canonical_model_eligible`. That is
correct for the runtime dispatch, which passes the moment as a keyword and is
verified against autograd. It is not correct for export: the compact canonical
`.pt2` ABI is positional and has no spin slot, whereas the graph ABI carries one
at index 10. Auto-resolution would therefore have exported a native-spin model
through an ABI that cannot receive its moment, producing an artifact that
silently evaluates at a vanishing moment -- wrong energy and no magnetic force,
with no error anywhere. The canonical ABI therefore carries the moment at slot
8, its last slot, and auto-resolution selects it for a native-spin model only
because that slot exists.

The lesson generalizes beyond spin: an eligibility predicate shared between
runtime dispatch and export resolution answers two different questions. Runtime
asks whether the kernels can compute the model; export asks whether the wire
format can carry its inputs. Widening the first silently widens the second.

#### Deployment: the first Kokkos native spin pair style

The compact canonical artifact reaches LAMMPS through `pair_style dpa4spin/kk`,
with `pair_style dpa4spin` serving the same artifact on the host. The serving
chain is:

| Layer    | Element                                                              |
| -------- | -------------------------------------------------------------------- |
| Artifact | `lower_input_kind = dpa4c_canonical`, nine positional inputs         |
| C++      | `NativeSpinPTExpt::compute_canonical_graph_gpu`                      |
| Dispatch | `DeepSpinBackend` virtual, `DeepSpin` forwarder, C API, `deepmd.hpp` |
| LAMMPS   | `source/lmp/pair_dpa4spin{,_kokkos}.{cpp,h}`                         |

Native spin is a distinct scheme from the virtual-atom `deepspin` one.
`NativeSpinPTExpt` is a `DeepSpinBackend` of its own that treats the ABI as an
internal branch, serving both the graph and the canonical lower, and
`PairDPA4Spin` derives from `Pair` rather than from the legacy pair style.

The factory selects it on two metadata fields, `spin_scheme == "native"` and an
absent `has_comm_artifact`. The second conjunct is a capability of the serving
class rather than a property of the scheme: `NativeSpinPTExpt` has no
cross-rank message-passing route, so an artifact that needs one stays with
`DeepSpinPTExpt`, which retains a native-spin implementation of its own for
exactly that case. Native spin therefore has two serving classes today, split
on a domain-decomposition capability. Giving `NativeSpinPTExpt` the
message-passing route collapses the predicate to the scheme alone and leaves
one owner.

Two properties distinguish the spin style from its energy twin. The moment is
built on device in model-node order from `atomKK->k_sp` as direction times
magnitude; ghost rows read their owner's `sp`, which LAMMPS has already
forward-communicated through `AtomVecSpin::fields_comm`, so the moment needs no
exchange of its own and never leaves the device. The scatter then writes `fm`
beside `f`, which doubles the reverse communication to six values per atom and
requires the ghost slots of both arrays to be cleared before the scatter,
because `VerletKokkos::force_clear` clears only the local range under newton
off.

Because the traced lower has no conditional tail, the style rejects `fparam`,
`aparam` and `charge_spin` at `init_style` rather than ignoring them, and it
serves only a native-spin canonical artifact.

The figures below come from one-off runs rather than from a committed
harness, on a 128-atom two-species periodic system with a compressed DPA4C
native-spin artifact. Host `dpa4spin` reproduces the established graph lower
bit for bit: energy, force, magnetic force and
pressure all agree to exactly zero over a ten-step run with neighbour rebuilds,
which exercises the per-atom energy, the reduced-Planck conversion, the
centroid virial and the reverse communication together. The device style agrees
with the host one to 1.2e-07 eV in energy and 4.8e-08 eV/Angstrom in force, and
both agree with the eager model to about 1e-06 relative on force and 3e-07 on
the magnetic force, which fixes the reduced-Planck scaling absolutely rather
than only consistently.

Multi-rank agrees with single rank for both styles. Over the same ten-step run
at two and three ranks the worst deviations are 2.4e-07 eV in energy, 5.2e-08
eV/Angstrom in force and 5e-10 Angstrom in the final positions, which exercises
the extended node set, the ghost moment rows, the combined force and magnetic
force reverse communication, the ghost-slot clearing and the multi-rank
centroid-virial fold.

The device path inherits the energy style's run-to-run reproducibility
characteristic: the atomic cursor in the source-CSR scatter permutes
`source_order`, which changes the fp32 reduction order and moves the total
energy in its ninth significant digit. This is a property of the shared graph
assembly, not of the spin families, whose own reductions are ordered by the
topology.

#### Two shape contracts that differ between the lowers

The compact lower emits `atom_virial` as `(N, 3, 3)` and `virial` as `(F, 3, 3)`,
whereas the graph lower emits `(N, 9)` and `(nf, 9)`. The shared remap to dense
keys indexes into a `(nf, nall, 1, 9)` buffer and therefore requires the flat
form. The spin path normalizes at the canonical seam. The energy path has the
same latent mismatch on its host canonical branch with `atomic=true`; its device
branch is unaffected because it reshapes explicitly.

The Python evaluator dispatches the native-spin canonical artifact through the
graph-spin path with the compact nine-input tuple, so `DeepPot.eval` and the
`dp test` flow work on the deployment artifact rather than only on the graph
one.

#### Resource state

The forward and the edge backward run at the 64-register cap set by their
occupancy target, so their register pressure appears as spill rather than as a
register count. At the benchmarked profile, `lmax = 2` with no radial modes,
spill stores of the spin variant are:

| Kernel        | 8 channels | 32 channels | 128 channels |
| ------------- | ---------- | ----------- | ------------ |
| forward       | 0 B        | 0 B         | 4 B          |
| edge backward | 100 B      | 28 B        | 4 B          |

The edge group is not a lever on spill, and treating it as one is misleading.
The 100 bytes at Nano are introduced by the tuned `SpinBackward = 2`: the same
kernel at width four spills nothing and is 15% slower. The occupancy target is
a second lever on the same spill and does pay; the section on the spin
optimization round below measures both. Halving the group doubles
the edges in flight, and on a narrow profile the per-edge geometry that every
lane of a group recomputes dominates whatever the spill costs. Widening the
backward map to remove spill is slower at every channel width. The larger spills
appear only at `lmax >= 3` with radial modes -- at 32 channels with `lmax = 4`
and eight radial modes, 120 bytes in the forward against 40 spin-free and 96 in
the edge backward against 48 -- and that is the configuration where the spin
overhead is *smallest*, +5.9% against +40% at `lmax = 2`. Those spills therefore
bound nothing that matters.

The binding resource is elsewhere. The node backward spills nothing in either
variant and is not register bound either; spin in fact lowers its register count,
from 97 to 94 at 128 channels with `lmax = 2` and from 102 to 98 at `lmax = 4`.
It is shared-memory bound. At 128 channels with `lmax = 4` the staged moment row
takes it from 12,480 to 14,944 bytes per block, which against a 228 KiB
shared-memory budget lowers the occupancy ceiling from 18 blocks per SM to 15.
The bond-projected family accounts for 768 of those bytes, exactly `3 C_s`
floats at `C_s = 64`. Any further spin family that widens the staged row pays
there first.

#### Throughput and capacity of the spin path

The figures above are correctness figures from one-off runs. The scan below is a
committed harness: `lmp_scan.py` carries a flavor that pairs a LAMMPS input
script with the geometry writer it requires, so the spin curve and the energy
curve place the same atoms in the same box and differ only in the moments the
data file carries and the pair style that serves them. `freeze_dpa4c.py --spin`
freezes the same five grades with `use_spin` on every type.

Moments are held fixed during the scan. LAMMPS integrates spins with
`fix nve/spin`, which has no Kokkos implementation and would place a host-side
per-atom update in every step, hiding the pair-style cost the scan measures.
The style still evaluates the spin-conditioned descriptor and writes `fm` on
every step, so the per-step model cost is the one a spin trajectory pays.

Saturated throughput, the mean of every successful point of at least one
million atoms, and the largest successful system:

| Grade | Energy |   Spin | Spin cost | Energy atoms | Spin atoms |
| ----- | -----: | -----: | --------: | -----------: | ---------: |
| Nano  | 16,814 | 11,199 |     1.50x |   14,051,520 | 11,036,032 |
| Mini  | 10,191 |  8,233 |     1.24x |   14,051,520 | 11,036,032 |
| Neo   |  6,969 |  5,784 |     1.20x |   14,051,520 | 11,036,032 |
| Air   |  3,800 |  3,457 |     1.10x |   14,051,520 | 11,036,032 |
| Plus  |  2,165 |  1,884 |     1.15x |   14,051,520 | 10,455,280 |

The cost falls as the profile widens, because the spin families are tied to the
degree-two width while the geometric work grows with the scalar width and the
angular degree.

#### Why the spin path stops at a smaller system

Capacity is not grade-dependent in either flavor, and the spin limit sits below
the energy one for a single reason. The backward materializes a second edge
cotangent: `edge_spin_gradient` has the shape of `edge_vec`, so at the 158 edges
per atom of this cutoff it adds 1,896 B/atom to the 6,838 B/atom that Section
11.9 derives. The ratio predicts `14,051,520 x 6838 / 8734 = 11.0` million against
11,036,032 measured, so the second cotangent explains the difference to within
the half-million-atom resolution of the scan. Anything else the spin branch
holds at system scale is below that resolution.

Removing it is not a memory-side decision. The magnetic force is a per-source
reduction -- an edge from source `j` carries dE/dspin of `j` -- while the
descriptor backward runs destination-major over node tiles, so the two
groupings do not align. A deterministic reduction therefore needs the edge axis
materialized, and the alternative, a float atomic scatter, would set the
summation order by arrival and is excluded by the precision policy.

#### Effective: the occupancy target of the degree-two spin kernels

The spin branch spills where the geometric one does not. At Nano the edge
backward spills 104 bytes against none, at Mini 32, and at Neo 8 in both
directions. Both edge kernels compiled for a fixed thirty-two blocks per SM; a
block is one warp, so that target divides the 65,536-register file into exactly
sixty-four registers per thread, which the geometric working set fits and the
five spin families do not.

Section 11.5 records that widening the edge group removes this spill and is
slower, and that remains true. The occupancy target is a different lever on the
same spill, and it points the other way. Compiling the degree-two spin kernels
for twenty-four blocks buys eighty registers, removes the spill outright, and
is worth 2.0% to 4.5% at one million atoms:

| Grade | 32 blocks |  24 blocks | 20 blocks |
| ----- | --------: | ---------: | --------: |
| Nano  |    11,394 | **11,911** |    11,768 |
| Mini  |     8,191 |  **8,427** |     8,249 |
| Neo   |     5,767 |  **5,882** |     5,834 |
| Air   | **3,465** |      3,209 |     2,979 |
| Plus  | **1,870** |      1,691 |        -- |

The sign flips at degree three, and the reason is that the relief there is
incomplete. Those profiles overflow the budget on geometry alone -- they spill
16 to 56 bytes with no spin branch at all -- and still spill at eighty and at
ninety-six registers, so the occupancy they give up buys a partial fix and
costs 7.4% to 9.6%. The target is therefore raised only where the spill
disappears, which is exactly `Lmax == 2`. The geometric kernels are compiled
unchanged, so their resource use is identical bit for bit and the energy path
cannot regress.

#### Effective: folding the magnetic reduction into the force assembly

The magnetic cotangent was reduced onto its source nodes in Python, by gathering
the edge axis through `source_order` and then segment-summing the gathered copy.
That reads the edge axis, writes a permuted copy of it, and reads that copy
again -- 5.7 GB per step at one million atoms -- and the gathered copy is a
second edge-scale buffer held at the point where memory decides the largest
system. It cost 3.9 ms per step, 2.3% of a Neo spin step.

The force assembly already walks that grouping: it accumulates the source
contribution of each node from the same `source_order` and the same
`source_row_ptr`, in the same warp-per-node loop. Three more accumulators ride
that loop and the warp fold that closes it, so the reduction costs one read of
the spin cotangent and no separate pass. The gathered copy disappears with it.

| Grade | Before |  After |
| ----- | -----: | -----: |
| Nano  | 11,911 | 12,206 |
| Mini  |  8,427 |  8,578 |
| Neo   |  5,882 |  6,003 |
| Air   |  3,465 |  3,490 |
| Plus  |  1,870 |  1,876 |

The operator returns the per-source total as a fourth output, empty when no
spin cotangent is supplied, and the two spin call sites keep the sign and the
on-site term they already applied. The summation order changes from a linear
segment sum to the lane-strided warp fold the force uses, which is the order
the force and the virial have always used and is fixed by the launch geometry
rather than by arrival.

The eager autograd path keeps the standalone reduction. It never enters the
force assembly, so there is nothing there to fold into.

#### Rejected: retuning the edge group for the raised register budget

The edge group widths of Section 11.5 were tuned when both edge kernels
compiled for sixty-four registers and the spin variants spilled. Raising the
budget to eighty removes that spill, so the tuning was repeated at Neo to see
whether the optimum had moved with it. It had not, and both directions are
worse by a wide margin:

| Change                | Kernel   |   Before |    After |
| --------------------- | -------- | -------: | -------: |
| `SpinBackward` 4 to 8 | backward | 5,878 µs | 6,206 µs |
| `SpinForward` 8 to 4  | forward  | 4,225 µs | 5,825 µs |
| `SpinForward` 8 to 16 | forward  | 4,225 µs | 5,598 µs |

The forward result is the informative one. The hypothesis was that a narrower
group would hide the per-edge moment gather behind more edges in flight, since
the gather costs 441 µs there. Narrowing instead doubles the channels each lane
carries, and the accumulator pressure that adds outweighs the recovered
concurrency by far more than the gather is worth. The widths are a sharp local
optimum and are not a function of the register budget.

#### What the spin backward actually spends

Ablation separates the two effects that `HasSpin` has on the backward, because
the flag changes the edge group as well as the arithmetic. Measured at Neo,
one million atoms:

| Variant                       | Backward |
| ----------------------------- | -------: |
| No spin, width 8 (production) | 4,405 µs |
| No spin, width 4              | 5,362 µs |
| Spin, width 4 (production)    | 5,878 µs |
| Spin, width 8                 | 6,206 µs |

Of the 1,472 µs the spin backward costs over the geometric one, 956 belong to
the narrower group and only 516 to the spin arithmetic. Neither width is
retunable, as the table above shows, so that 956 µs is the price of the edge
concurrency the spin kernel needs rather than a parameter left on the table.

Two further ablations bound the arithmetic. Removing the in-loop spin channel
work entirely -- eight cotangent reads and about thirty fused multiply-adds per
channel -- leaves the backward unchanged at 5,880 µs, so that block is fully
hidden behind memory latency. Removing the per-edge moment gather leaves the
backward unchanged as well but takes the forward from 4,225 to 3,784 µs, which
is the 441 µs quoted above: the backward's eight groups hide the gather that
the forward's four do not.

The remaining structural lever is to compute the per-edge geometry once per
group and broadcast it across the lanes of that group instead of having each
lane recompute it. It is bounded above by the 956 µs, about 3.5% of a Neo spin
step, against a broadcast of the nine basis components and the envelope on the
critical path of both kernels.

#### Combined effect

Against the spin path as first committed, at one million atoms:

| Grade | Before |  After | Gain |
| ----- | -----: | -----: | ---: |
| Nano  | 11,394 | 12,206 | 7.1% |
| Mini  |  8,191 |  8,578 | 4.7% |
| Neo   |  5,767 |  6,003 | 4.1% |
| Air   |  3,465 |  3,490 | 0.7% |
| Plus  |  1,870 |  1,876 | 0.3% |

Over the saturated range, where graph construction takes a larger share and
neither change touches it, the same two are worth 6.3%, 4.8%, 3.0%, 0.7% and
1.6% from Nano to Plus.

The energy path is unchanged. The occupancy target is conditioned on the spin
branch, so the geometric kernels compile to the same registers and the same
spill as before, and the assembly kernel is templated on the presence of the
cotangent, so its geometric instantiation carries neither the accumulators nor
the three warp shuffles that fold them.

The occupancy target only takes effect where the architecture can host it. A
`__launch_bounds__` minimum above the device's blocks-per-multiprocessor limit is
not clamped by `ptxas` -- it is discarded with a warning, leaving the register
allocator unconstrained -- so the 32 this section reasons about applies to devices
that hold 32 blocks. Section 11.13 records how the request follows a lower limit.

### 11.13 Porting the tuning to a second architecture

Sections 11.3 through 11.12 were measured on H20. This section re-measures the
same kernels on an RTX PRO 6000 Blackwell, a graphics-class part with a very
different resource balance, and records which of the earlier conclusions are
properties of the algorithm and which were properties of that first device.

The distinction matters for how the tuning is expressed. A constant that follows
the algorithm belongs in the source; a constant that follows the device has to be
derived from what the device reports, and not from its architecture name --
Blackwell spans both a 100 KB shared-memory budget per multiprocessor and, on
compute-class parts, more than twice that.

#### What the two devices differ in

|                                        | H20 (CC 9.0) | RTX PRO 6000 Blackwell (CC 12.0) |
| -------------------------------------- | ------------ | -------------------------------- |
| Shared memory per multiprocessor       | 228 KB       | 100 KB                           |
| Resident blocks per multiprocessor     | 32           | 24                               |
| Resident warps per multiprocessor      | 64           | 48                               |
| fp32 GEMM, cuBLAS on the fitting shape | 21.9 TFLOP/s | 57.7 TFLOP/s                     |
| bf16 GEMM, cuBLAS on the fitting shape | --           | 242 TFLOP/s                      |

#### Hardware counters, finally

Section 11.7 closed with a request to re-run counter profiling once host
permissions allowed it. They do on this host, so every claim below rests on
counters rather than on ablation. Production geometry, 32,768 atoms and 5,177,344
edges, degree two without radial modes:

| kernel        | width | L1 active |  SM | DRAM | wavefronts/edge | load B/edge |
| ------------- | ----: | --------: | --: | ---: | --------------: | ----------: |
| forward       |     8 |     97.5% | 61% |  48% |             6.6 |         201 |
| edge backward |     8 |     97.5% | 55% |  58% |             5.8 |         197 |
| forward       |   128 |     98.8% | 72% |  14% |            37.3 |        3450 |
| edge backward |   128 |     98.8% | 76% |  12% |            47.9 |        2870 |
| node backward |   128 |     97.7% | 25% |  20% |             5.9 |         118 |

The dominant stall is a scoreboard dependency on an L1TEX access, 5.5 cycles per
warp at the widest width.

**The narrow-width reading in section 11.3 does not hold here.** That section
estimated 23% of aggregate L1 bandwidth and 5% of fp32 peak at `C_0 = 8` and
concluded the kernel was issue- and latency-bound with room to spare. The
counters put the same kernel at 97.5% of L1 active cycles. Both ends of the width
range are L1-cycle saturated on this device, and the narrow end is additionally
carrying about half of DRAM.

The mechanism is visible in the ratio of bytes to wavefronts. At `C_0 = 8` the
kernel moves 201 bytes per edge in 6.6 L1 accesses, which is 30 bytes per access
against a 128-byte line: the two-lane edge group means each access touches one
32-byte sector, the smallest transaction the cache serves. Sector occupancy is
complete -- no byte is fetched and discarded -- so the L1 cost is exactly the
byte count divided by the sector size, and the byte count is the compressed
radial table: eight channels at six coefficients is 192 of the 201 bytes.

That has a practical consequence. On this device the narrow profiles cannot be
improved by removing fixed per-edge work, which is what section 11.4 optimized;
they can only be improved by reading fewer coefficient bytes, and the coefficient
count is fixed by the quintic interpolant that the smoothness contract requires.

#### `__launch_bounds__` follows the block limit

The occupancy target of section 11.12 asks for 32 resident blocks, H20's limit. On
CC 12.0 the limit is 24, and `ptxas` does not clamp a larger request -- it
discards it:

```text
ptxas warning : Value of minnctapersm for entry ... is out of range.
                minnctapersm will be ignored
```

The edge kernels therefore take their minimum block count from
`edge_resident_blocks`, per profile and per architecture. On CC 8.9 and 12.x a
request of 24 blocks sets the tightest budget the hardware allows, 80 registers
per thread, and the radial-mode kernels of 64 and 128 channels take it: the
unconstrained allocator gives them 86 to 128 registers, and budgeted they keep all
24 warps resident for a spill of at most 80 bytes. A sweep of the plain kernels
over widths 8 to 128, degrees two to four, rank zero and four and both spin
settings (jittered diamond, 131,072 atoms, forward and backward together) puts
that class 3% to 13% faster with the budget. Every other kernel requests no
minimum, which is also what the discarded request amounted to: the allocator
settles on 80 registers for most of them, and where the budget binds it costs the
narrow profiles up to 11% and the spin families up to 12%, and Nano 7% of its
model step, because the two-lane edge groups of eight channels keep sixteen edges
in flight per warp and gain little from more warps. A request of zero compiles to
the same instructions as an omitted one; a request of one does not, since it lifts
the allocator's own target and the kernels grow to 92 to 240 registers.

With one warp per block, 24 blocks is 24 of 48 resident warps, so these kernels
are pinned at 50% occupancy by the block limit and not by their registers. Going
higher would need more than one warp per block, which at 24 blocks leaves 42
registers per thread against the 80 the kernel uses. That is the same conclusion
section 11.7 reached from the register file, reached here from the block limit.

#### Effective: the node-group width follows the device

The node backward holds six shared arrays sized by the resident node group count
times the scalar width -- the three geometric quantities of the recompute and
their three cotangents. `NodeWidth` was a single constant, 8, while the edge group
widths were tuned per scalar width in `EdgeMap`. On H20 that is harmless: 228 KB
per multiprocessor keeps the block limit at the architectural maximum at every
width. On a 100 KB budget the same footprint admits eight blocks at `C_0 = 128`,
for 16.7% theoretical occupancy against 50% for the two edge kernels.

Widening the group to 16 lanes halves the footprint and doubles the resident
warps. It also doubles the redundant load work, because every lane of a group
reloads its node's shared state, so the trade is not free and does not hold at
every width. Measured on this device:

| width | narrow blocks | wide blocks | wide effect on the step |
| ----: | ------------: | ----------: | ----------------------- |
|     8 |            24 |          24 | none, tie               |
|    16 |            23 |          24 | +3.6% slower            |
|    32 |            18 |          24 | +1.4% slower            |
|    64 |            15 |          24 | within noise            |
|   128 |             8 |          15 | **2.1% faster**         |

The wide variant pays only where the narrow one is starved of warps outright,
which on this device is a quarter of the warp capacity. Both widths are therefore
compiled and the launcher asks
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` what the running device grants
each of them, widening only when the narrow variant falls below a quarter of the
device's own warp capacity. The threshold is expressed against
`cudaDevAttrMaxThreadsPerMultiProcessor` rather than against an architecture, so a
compute-class part with a generous shared-memory budget resolves to the narrow
width at every scalar width and keeps the H20 tuning unchanged.

The gain is 1.5% of the descriptor at `C_0 = 128` and nothing at the other
widths. It is kept because it removes a device-specific constant from the source,
not because the number is large.

#### Rejected on this device as well

| Experiment                                  | Result                                                                                     |
| ------------------------------------------- | ------------------------------------------------------------------------------------------ |
| 16-lane backward edge groups at `C_0 = 128` | 17% slower, against 21% on H20; the coalescing gain never covers the lost edge concurrency |
| 4-lane edge groups at `C_0 = 8`             | 24% slower, forward and backward alike; the two-lane group remains optimal                 |
| 32-lane node groups at `C_0 = 128`          | 5% slower than 16 lanes; the redundant load work grows faster than the occupancy it buys   |
| A fixed 16-lane node group at every width   | Regresses the narrow widths, which is what motivated the occupancy query                   |
| Tensor-core fitting GEMM                    | See below                                                                                  |

#### The fitting GEMM, re-evaluated

Section 11.10 declined a tensor-core fitting GEMM because H20's TF32 peak left a
split representation between 12 and 25 TFLOP/s against 21.9 for the SIMT path.
The arithmetic is different here and the conclusion survives anyway.

On the production shape, M = 131072 and N = K = 256, cuBLAS reaches 57.7 TFLOP/s
in fp32 with TF32 disabled, which is 78% of this device's fp32 peak -- against
50% on H20, so the hand-written-SGEMM headroom that section 11.10 quotes does not
exist here. A bf16 pass on the same shape reaches 242 TFLOP/s, so a
three-pass split representation is bounded by 81 TFLOP/s, or 1.4x the fp32 path
before any split, merge or accumulation overhead. That is not enough to justify a
CUTLASS-class kernel and an argument about the precision contract, so the GEMM
stays on cuBLAS fp32.

For the same reason cuBLAS's own `CUBLAS_COMPUTE_32F_EMULATED_16BFX9` does not
apply: its documentation states the emulation pays only where bf16 peak exceeds
nine times fp32 peak, and the ratio here is 4.2.

#### A build hazard worth recording

The per-width translation units did not always rebuild when only
`dpa4c_graph_compress_kernel.cuh` or `dpa4c_graph_compress_launch.h` changed, so
an edited header could leave stale objects in the library while the link
succeeded. Several measurements in this section had to be discarded and repeated
once that was found. Any benchmark loop over these kernels should delete the
`dpa4c_graph_compress*.o` objects before building and confirm the instantiation
it expects is present, for example with

```bash
nm -C libdeepmd_op_pt.so | grep -oE 'node_backward_kernel<[0-9a-z, ]+>' | sort -u
```

which shows the template arguments the library actually carries.

#### End to end on the second device

The five released grades rescanned with `parameter_sweep.py --grades-only`,
saturated throughput being the mean over systems of at least one million atoms
and capacity the largest system that completes:

| Grade | H20 (section 11.6) | RTX PRO 6000 Blackwell | Ratio |
| ----- | -----------------: | ---------------------: | ----: |
| Nano  |             16,814 |                 22,029 | 1.31x |
| Mini  |             10,191 |                 12,773 | 1.25x |
| Neo   |              6,969 |                  8,504 | 1.22x |
| Air   |              3,800 |                  5,894 | 1.55x |
| Plus  |              2,165 |                  3,115 | 1.44x |

Capacity is 14,051,520 atoms on both devices, so node tiling continues to reach
the memory bound rather than a kernel limit. The two heavy grades gain most,
which is consistent with their degree-three and radial-mode paths being the most
arithmetic-dense and this device having the larger fp32 throughput.

These ratios are the hardware. The one source change in this section contributes
1.5% of the descriptor at the widest scalar width and nothing at the others,
which is below the resolution of a whole-step scan; it is worth having because it
removes a device-specific constant, not because it moves this table.

#### The NEP89 reference moves more than DPA4C does

GPUMD was built on this host so the reference is measured on the same device
rather than carried over. NEP89 plateaus at 16,340 atoms/ms against 8,548 on H20,
a factor of 1.91, while the DPA4C grades gain between 1.22x and 1.55x. The
comparison therefore reads differently here:

|                 |    H20 | RTX PRO 6000 Blackwell |
| --------------- | -----: | ---------------------: |
| NEP89           |  8,548 |                 16,340 |
| Nano over NEP89 |  1.97x |                  1.35x |
| Mini over NEP89 |  1.19x |                  0.78x |
| NEP89 capacity  | 10.5 M |                 10.5 M |
| DPA4C capacity  | 14.1 M |                 14.1 M |

The direction is consistent with what the counters say about each model. NEP89 is
arithmetic-dense and gains most of the fp32 throughput this device adds over a
compute part with a reduced fp32 rate; DPA4C is L1-cycle saturated at every
scalar width, so it gains far less than the fp32 ratio would suggest. Nano
remains ahead of NEP89 at saturation and Mini no longer is, every grade stays
well ahead below roughly 60,000 atoms where NEP89 has not yet reached its
plateau, and DPA4C holds 34% more capacity.

That is a hardware-balance statement, not a regression: nothing about DPA4C got
slower. It does mean the "fastest at equal accuracy" claim has to be qualified by
device class, and that the L1 saturation documented above is what caps the gain
from a part with more fp32 throughput.

The DPA1 reference curves are not re-measured here and are omitted from the
figure rather than carried over from H20, because a throughput comparison is only
meaningful between curves taken on one device. The previous cross-model figure
remains the H20 record.

## 12. Usage

### 12.1 Which consumers reach the kernels

The kernels are ordinary Torch custom operators, so nothing about them is
specific to LAMMPS. What decides whether they run is the dispatch condition in
`DescrptDPA4C.call_graph`, and all of it has to hold:

- the descriptor is compressed (`enable_compression`), because the kernels read
  the frozen spline tables that compression produces;
- `DP_CUDA_INFER >= 1`;
- the model is evaluated through the graph lower and the caller supplied the
  destination CSR arrays;
- the block layout is one the operator serves (`mega_eligible`).

Verified on the released grades:

| Consumer                                       | Reaches the kernels | Notes                                                                                                    |
| ---------------------------------------------- | ------------------- | -------------------------------------------------------------------------------------------------------- |
| LAMMPS + Kokkos, `pair_style deepmd/kk`        | yes                 | builds the CSR graph on device in `compact_canonical_graph_kokkos.h` and supplies its own neighbour list |
| Python `DeepPot` on a frozen `.pt2`            | yes                 | the freeze traces the graph lower, so the archive calls the operators directly                           |
| ASE, `deepmd.calculator.DP` on a frozen `.pt2` | yes                 | wraps the same `DeepPot`                                                                                 |
| Uncompressed model, any consumer               | no                  | falls through to the array-API reference, which is plain PyTorch                                         |

So the compressed artifact is not LAMMPS-only. A `.pt2` evaluated from Python
executes the same three descriptor kernels; confirming this on the Nano grade
shows `forward_kernel`, `node_backward_kernel` and `edge_backward_kernel` in a
CUDA profile of `DeepPot.eval`. What LAMMPS adds is not kernel access but the rest
of the step: it owns the neighbour list and builds the CSR graph on device, while
the Python path rebuilds a neighbour matrix per call. At four thousand atoms that
neighbour build is the largest single kernel in the Python profile, larger than
the descriptor forward, which is why the two paths are not comparable on
throughput even though they run identical descriptor code.

Without compression there is no hand-written descriptor kernel at all -- the
uncompressed DPA4C descriptor is the reference implementation on whatever backend
executes it. Two accelerations remain available to it, both independent of the
descriptor: the force and per-atom virial assembly of any graph-lowered model
(`DP_CUDA_INFER >= 1`), and the fused cuBLAS fitting network.

### 12.2 Conversion flow

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
    source/tests/pt_expt/descriptor/test_dpa4c_cuda.py \
    source/tests/pt_expt/model/test_dpa4c_graph_lower.py \
    source/tests/pt_expt/model/test_graph_export.py \
    -q

source/build/api_cc/tests/runUnitTests_cc \
    --gtest_filter='TestEdgeTensorPack.*'
```

The benchmark scripts under `debug/cuda_bench/` generate matching
uncompressed/compressed EMA packages, diamond systems, LAMMPS scans, and plots.
`kernel_profile.py` times the descriptor operators in isolation on a periodic
diamond neighborhood and is the fast inner loop for kernel work.
`parameter_sweep.py` owns every DPA4C configuration, both the released grades
of the capacity figure and both structural sweeps; `benchmark_all.py` produces
the DPA1 and NEP89 reference curves. `fitting_profile.py` times the fused
fitting network on its own, which is how the fitting share of a step and the
activation study of Section 11.8 were measured.

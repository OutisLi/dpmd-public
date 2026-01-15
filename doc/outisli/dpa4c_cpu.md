# Compressed DPA4C CPU Inference

This document specifies and justifies the CPU inference path of the compressed
degree-wise DPA4C descriptor in the `pt_expt` graph lower. The target workload
is one forward energy evaluation followed by one analytical backward for force
and virial; training and double backward are outside the operator contract.

The numerical contract is IEEE float32 model computation. No reduced-precision
path exists: bf16, fp16, VNNI and AMX are all excluded, because a
molecular-dynamics potential-energy surface has to stay smooth and eight
mantissa bits do not deliver that.

## 1. Implementation

| Component                              | Location                                                                                       |
| -------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Operator schemas (device neutral)      | `source/op/pt/dpa4c/ops.cc`, `source/op/pt/graph_ops_schema.cc`                                |
| Descriptor CPU binding and dispatch    | `source/op/pt/dpa4c/graph_compress_cpu.cc`                                                     |
| Descriptor CPU kernel body             | `source/op/pt/dpa4c/graph_compress_cpu_kernel.h`, `*_scan.inc`, `*_readout.inc`                |
| Per-instruction-set instantiations     | `source/op/pt/dpa4c/graph_compress_cpu_{scalar,avx2,avx512}.cc`                                |
| Instruction-set selection              | `source/op/pt/cpu/dispatch.h`                                                                  |
| Balanced CSR work partition            | `source/op/pt/cpu/partition.h`                                                                 |
| Threaded counting sort of an edge axis | `source/op/pt/cpu/group.h`                                                                     |
| Fused energy fitting                   | `source/op/pt/cpu/graph_fitting_cpu.cc`                                                        |
| Force and virial assembly, graph CSR   | `source/op/pt/cpu/edge_force_virial_cpu.cc`                                                    |
| Vectorizable float32 transcendentals   | `source/op/pt/cpu/activation.h`                                                                |
| Cell-list search and fused graph build | `source/op/pt/cpu/neighbor_search_cpu.cc`                                                      |
| Heap retention policy                  | `source/op/pt/cpu/allocator_policy.cc`                                                         |
| Host graph assembly for LAMMPS         | `source/api_cc/include/graph_assembly.h`                                                       |
| Python operator front end              | `deepmd/pt_expt/kernels/dpa4c/`, `deepmd/pt_expt/kernels/{graph_fitting,edge_force_virial}.py` |
| Python graph builder                   | `deepmd/pt_expt/utils/cell_graph_builder.py`                                                   |
| Backend selection                      | `deepmd/pt_expt/kernels/utils.py`                                                              |
| Regression tests                       | `source/tests/pt_expt/descriptor/test_dpa4c_cpu.py`                                            |
| Benchmark harness                      | `debug/cpu_bench/`                                                                             |

The CUDA and CPU halves of every operator share one schema and one Python
front end. A schema is declared in its own translation unit, compiled
unconditionally, while each device registers only its own kernels: declaring
a schema beside one of the two would make the operator vanish whenever that
half is absent, and the front end could no longer distinguish "library not
loaded" from "this device has no kernel". That distinction is what
`operator_available` answers, by asking the dispatcher for the backend
device's key.

### 1.1 Selection

The CPU operators replace an Inductor lowering of the same arithmetic and are
faster wherever they apply, so unlike the CUDA operators they carry no level
gate. `fused_operators_enabled` and `fused_energy_force_enabled` resolve
against the backend device: on CUDA they read `DP_CUDA_INFER`, on CPU they are
always true. What remains is availability and eligibility -- the compiled
operator has to exist for the backend, and the model has to be one it
represents exactly:

- `channels` in {8, 16, 32, 64, 128}, `lmax` in {2, 3, 4},
  `radial_modes` in {0, 2, 4, 8}, float32 parameters, no excluded type pairs;
- a compressed snapshot (`enable_compression`), because the operator reads the
  radial table rather than the radial network;
- no native spin. The magnetic families need a source-major counterpart of the
  destination scan, which no CPU deployment asks for yet, so a
  spin-conditioned descriptor keeps the reference path.

The last condition sits in `op_available`, beside the dispatcher probe, rather
than in the structural predicate `mega_eligible`. The two answer different
questions: eligibility decides whether the compression artifacts are worth
building, which a snapshot does once for every device that may later consume
it, while availability decides whether *this* device has a kernel. Folding the
device into eligibility would make a CUDA-bound snapshot refuse to build its
own tables on a CPU host.

A CPU target also keeps the generic graph lower rather than the compact
canonical ABI. The canonical ABI exists so a device-resident neighbor list can
reach the descriptor without a host round trip, which is a property of the
CUDA and Kokkos deployment; on CPU its only effect would be a narrower index
type.

## 2. What the kernel computes

The operator evaluates the whole descriptor: the tabulated radial branch and
its shared mode profiles, the ordered PairFiLM amplitude, one destination
reduction producing both envelope masses and every degree-wise moment, the
invariant readout, and the output calibration. Per edge `e` with destination
`i`, source `j`, displacement `r`:

```text
rho   = sqrt(|r|^2 + eps^2)                 u    = r / rho
chi   = C3envelope(rho) * mask
g     = quintic_table(rho)                  (channels + radial_modes wide)
a     = gamma[pair] * g[:C0] + beta[pair] + sum_mu U[pair,:,mu] g[C0+mu]
M0[i] += a * chi                            Ml[i,m] += B_l,m(u) * a * chi^2
mass2[i] += chi^2                           mass4[i] += chi^4
```

then `X0 = M0 / sqrt(mass2 + 1/4)`, `Xl = Ml / sqrt(mass4 + 1/4)`, and the
readout contracts exact channel Grams, the closed-form `112` and `222`
couplings, the sparse Cartesian Gaunt couplings of the remaining triples, and
the projected quartic. The analytical backward differentiates all of it and
emits one edge cotangent.

The harmonics are evaluated with the squared norm of the regularized direction
substituted by one. The two polynomials agree on the unit sphere and their
gradients differ by a purely radial term, which the tangential projection
closing the coordinate backward annihilates exactly.

## 3. Design, and the measurements that produced it

### 3.1 The baseline had to be repaired first

The graph lower exports on a synthetic system of a few dozen atoms, so every
loop over the node or edge axis reaches Inductor with a size hint of about a
hundred elements. The C++ backend parallelizes a loop only above
`cpp.min_chunk_size` elements per thread, so 30 of the 33 generated kernels
were emitted serially, and the thread count of the two that were not was baked
in from the freezing host. A 4096-atom DPA4C nano step measured 1272 ms
against 111 ms once the loops were parallel, and the artifact would have
pinned itself to the freezing machine's core count wherever it was deployed.

Both are fixed in `build_inductor_compile_options(inference=True)`:
`cpp.min_chunk_size = 1` and `cpp.dynamic_threads = True`. Every baseline in
this document is the repaired one; the 11x that fix recovers is a bug, not a
result.

### 3.2 The radial table wants a different layout than on a GPU

The compression artifact stores the six spline coefficients channel-major, as
a quartet block followed by a pair block, so that a CUDA lane fetches one
channel in one 128-bit and one 64-bit load. On a CPU the vector axis is the
channel axis, so that layout requires transposing a 4x16 tile per block --
about twelve shuffles, all on one port, against the five fused multiply-adds
they feed. Coefficient-major blocking, `(interval, block, 6, lanes)`, turns
the evaluation into six contiguous vector fused multiply-adds and needs no
shuffle at all.

The re-layout is a process-local cache built once per model load, keyed on the
table's data pointer and holding a strong reference to the source storage, not
a second compression artifact. One artifact has to serve both devices, and a
snapshot compressed on a CUDA host then stays runnable on a CPU one. The same
cache carries the ordered FiLM scale and shift as separated, padded planes and
the mode mixing coefficients channel-major, for the same reason.

### 3.3 The destination reduction must not be a scatter

The aten lowering expresses the edge-to-node reduction as `index_add`. On a
CPU that is a locked read-modify-write per component, and at 158 neighbours
per atom it was the single most expensive operation of a compressed step: 88
ms of a 111 ms neo step, 77% of the total. Giving one thread a node and all of
its edges keeps the accumulators in registers for the whole scan and removes
the contention entirely -- the same reduction then costs 0.62 ms.

Threads receive contiguous node ranges of roughly equal *edge* count, found by
one binary search on the row pointers. Splitting the node axis evenly is wrong
whenever the degree distribution is not, and the reduction is a barrier, so
the slowest thread sets the step.

### 3.4 The vector width must be a compile-time constant

This was the largest single lesson, and it contradicted the design the CUDA
path suggests. On a GPU the channel width fixes the thread mapping, so the
kernels are instantiated per width; on a CPU the channel axis is merely the
vector axis, so a run-time trip count looks harmless. It is not. GCC then
emits a generic vectorized loop with an alias-versioned peel and a scalar
tail, and on the 16-iteration blocks of this kernel that scaffolding costs
more than the block. Making the lane count a compile-time constant per
instruction-set unit and writing every channel loop as
`for (base; base < n; base += kBlock) for (lane < kBlock)`:

| grade | forward before | after | backward before | after |
| ----- | -------------: | ----: | --------------: | ----: |
| nano  |        1.18 ms |  0.77 |         2.41 ms |  1.68 |
| mini  |           0.81 |  0.58 |            3.21 |  1.77 |
| neo   |           1.56 |  0.91 |            3.98 |  2.41 |
| air   |           1.95 |  1.31 |            7.09 |  4.99 |
| plus  |           2.79 |  1.80 |           13.51 |  9.75 |

Note that "a run-time multiple of the vector width" is not enough: the
compiler has to see the constant.

### 3.5 The backward reduces along the component axis

The backward needs both directions of the angular contraction: the amplitude
cotangent is a basis-weighted sum over harmonic components, and the basis
cotangent is an amplitude-weighted sum over channels. The second is a
reduction along the vector axis, so a component-major layout costs one
horizontal reduction per component -- eight to twenty-four of them per edge,
each a four-deep shuffle chain.

Keeping a second, channel-major copy of the angular cotangents makes the
component index the vector axis, and the reduction disappears: the loop
accumulates into a component-indexed vector and stores it. The envelope
pullback then comes for free from the identity

```text
sum_k a_k * sum_{l,m} B_lm * dM_lm,k  =  sum_{l,m} B_lm * (sum_k a_k dM_lm,k)
```

so the one remaining channel-axis reduction is also eliminated. Together with
four independent partial sums to cover the fused multiply-add latency of the
two short reductions that remain, the backward improved by 4% to 22% per grade.

### 3.6 The heap must retain the working set

glibc serves an allocation above its dynamic mmap threshold -- capped at 32
MiB -- with `mmap`, and returns it with `munmap` on free. Every buffer of a
graph-lower step that scales with the edge count crosses that cap on a
production system, so each step re-faults its whole working set. The cost is
proportional to the working set rather than to the arithmetic, which is why it
appeared as a throughput cliff exactly where the step outgrew the cap: neo
throughput fell from 1350 atoms/ms at 17,576 atoms to 609 at 125,000, while
the arithmetic per atom is identical.

`source/op/pt/cpu/allocator_policy.cc` raises the mmap and trim thresholds to
256 MiB from a library initializer, because both consumers of the fused path
load this library and neither shares an earlier entry point. The knee is
broad, and the trade is resident memory:

| threshold          | 125,000-atom neo step | resident |
| ------------------ | --------------------: | -------: |
| glibc default      |                205 ms |  2.1 GiB |
| 128 MiB            |                 97 ms |  3.1 GiB |
| 256 MiB (selected) |                 88 ms |  4.9 GiB |
| 1 GiB              |                 89 ms |  7.2 GiB |

`DP_CPU_MALLOC_RETAIN=0` restores the glibc default.

### 3.7 The deployment cost was the neighbor plumbing, not the model

Once the operators were fast, neither deployment path was limited by them. A
LAMMPS step spent about 310 ms turning its neighbor list into a graph against 4
to 18 ms in the model, and an ASE-style `DeepPot.eval` spent about 140 ms in a
single-threaded cell list and another 50 ms building the compressed-sparse-row
views. Three observations removed almost all of it.

**A host neighbor list is already destination-grouped.** LAMMPS hands over one
row per local atom, so after the cutoff filter the surviving edges are still
grouped by destination: the row pointers are a prefix sum over per-atom counts
and the destination permutation is the identity. The adapter was instead
running the generic sort-based construction -- roughly twenty tensor
operations including `nonzero`, three `index_select`, three `cat` and three
sorts, about half a gigabyte of traffic per step, several stages
single-threaded. `assembleGraph` in `source/api_cc/include/graph_assembly.h`
replaces it with two threaded passes over a cached destination-grouped skin
topology that write the final buffers directly, and folds the cutoff filter
and the type exclusion into one predicate so an excluded edge is never
allocated rather than allocated and masked.

**The cell-list search is embarrassingly parallel over destination atoms.**
Its arithmetic is three subtractions and a dot product per candidate, and the
coordinates of a host-sized system stay resident in L2, so the only structural
requirements are a bounded candidate set and a single write. Threading it took
the search on an 8000-atom cell from 91.9 ms to 3.3 ms, with a pair multiset
identical to the reference.

**The search already computes every displacement it tests.** The generic
builder discards them and recomputes the edge vectors with two coordinate
gathers and a per-edge `(E, 3, 3)` broadcast of the lattice, then sorts the
payload it just built in order. Handing the displacements back removes all of
it. That form is not differentiable in the coordinates, which is exactly what
a deployment path can afford: it feeds a frozen artifact whose forces come
from the model's own analytical backward with the displacements as inputs. A
caller that differentiates through the graph keeps the recomputing builder.
`neighbor_graph` returns the whole payload -- endpoints, displacements, mask,
row pointers and the source permutation -- in 4.1 ms against 28.6 ms for the
threaded search followed by the generic construction, and its outputs are
bit-identical.

Only the source-major permutation still needs a pass of its own, because the
compiled graph reads it (`graph_source_csr` in the artifact metadata records
whether it does, probed from the exported graph rather than from a model
predicate). `group_by_node` in `source/op/pt/cpu/group.h` is the threaded
counting sort both consumers share, reached from the host adapter through the
`build_graph_csr` operator so that one implementation serves both.

### 3.8 A benchmark that inherits an OpenMP affinity mask measures nothing

The first LAMMPS numbers were wrong by more than an order of magnitude, and
the failure is worth recording because it is invisible from inside the
measurement. Under `OMP_PROC_BIND=close OMP_PLACES=cores`, libgomp pins the
main thread of a process that has touched an OpenMP runtime to a single core:
after importing PyTorch the benchmark's own affinity mask had shrunk from 180
processors to 2. A forked child inherits the mask of the calling thread, so
the LAMMPS process it launched saw two processors and ran the model on one
thread no matter what `OMP_NUM_THREADS` said. In-process measurements were
unaffected -- worker threads get their own places -- which is why only the
subprocess path was wrong, and why the discrepancy looked like an adapter
cost rather than a thread count. `lmp_scan.py` now restores the full mask in
the child. The symptom to watch for is LAMMPS reporting a low `CPU use`
percentage next to a high thread count.

### 3.9 A scalar accumulator forbids vectorizing the loop that feeds it

The backward's two remaining channel sweeps -- the one that forms the
amplitude cotangent while pulling back the envelope and the radial derivative,
and the one that projects that cotangent onto each radial mode -- were running
entirely scalar. Nothing about the arithmetic prevents vectorizing them; the
accumulators did. Summing into a scalar fixes the order of a floating-point
reduction, and a compiler may not reassociate it unbidden, so GCC declined the
whole loop and fell back to one channel at a time:

```text
graph_compress_cpu_scan.inc:434: missed: not vectorized: complicated access pattern
graph_compress_cpu_scan.inc:441: missed: not vectorized: control flow in loop
```

Accumulating one partial sum per lane and folding them once after the sweep
restores 64-byte vector code. The reduction order changes, which is why the
change is admissible only where the contract is float32 rounding rather than
bit reproducibility; the measured deviation from the reference did not move.

| grade | backward before | after |      gain |
| ----- | --------------: | ----: | --------: |
| nano  |         1.68 ms |  1.63 |     1.03x |
| mini  |            1.77 |  1.61 |     1.10x |
| neo   |            2.41 |  1.94 |     1.24x |
| air   |            4.99 |  3.23 |     1.55x |
| plus  |            9.75 |  4.41 | **2.21x** |

The gain scales with the channel count and the mode rank because that is what
the two sweeps traverse: the narrow grades were never spending much time
there. The lesson generalises past this kernel -- a hand-written loop whose
inner statement accumulates into a scalar is silently serial, and the
vectorizer's own report is the cheapest way to find them. Every reduction in
this kernel is now lane-partitioned; the two that were not were found by
reading `-fopt-info-vec-missed` rather than by reasoning about the algorithm,
which had said the reductions were a small fraction of the work.

### 3.10 A libm call in a loop body costs the loop, not just the call

The fitting network's activation was `std::tanh`, which is an opaque library
call: the loop around it cannot be vectorized at all, so every channel of every
layer paid a scalar transcendental -- and paid it again in the backward, which
re-derived the derivative from the pre-activation.
`source/op/pt/cpu/activation.h` replaces the library calls with the Cephes
single-precision forms written as plain expressions. The larger win came from
asking what the backward needs rather than how fast a tanh can be: for tanh the
derivative is algebraic in the *output*, so the forward now leaves the
activation in the saved buffer instead of the pre-activation and the backward
reads `1 - a^2`. Silu's derivative needs its argument, so its saved state stays
the biased pre-activation -- either way the forward leaves exactly what the
backward's derivative function consumes, which also spares the backward a bias
addition. Measured on one layer's epilogue at 8000 nodes and 256 channels,
best of four runs:

| epilogue                         |      time | effective bandwidth |
| -------------------------------- | --------: | ------------------: |
| copy only (bandwidth floor)      |  0.022 ms |           1103 GB/s |
| `1 - a^2` from the stored output | **0.023** |                1086 |
| `fast_tanh`                      |     0.195 |                 126 |
| `std::tanh`                      |     0.299 |                  82 |

The derivative pass is now at the bandwidth floor, 13x cheaper than either
transcendental, and the arithmetic is unchanged: `1 - tanh(z)^2` evaluated from
a stored `tanh(z)` is bit-identical to computing it from `z`. Across three
hidden layers that removes 0.79 ms from a neo backward.

Getting there needed three separate fixes, each found from the vectorizer's
report rather than from reasoning:

- The minimax **rational** approximation over the whole line, which is the form
  most libraries use for a vectorized tanh, is **not monotone**: over a grid of
  240,000 points it reverses direction 29,910 times, by up to 7 units in the
  last place. `std::tanh` reverses on none of them. A potential-energy surface
  cannot carry that, so the implementation uses the Cephes branch structure
  instead -- an odd polynomial near the origin, `1 - 2/(exp(2|x|) + 1)` beyond
  -- whose outer branch is monotone because the exponential is. Measured
  monotone on the same grid, with a worst relative error of 2.1e-7.
- `std::min` and `std::max` on floats are conditional expressions, not
  `minps`/`maxps`, because they differ on NaN. One of them next to this much
  inlined arithmetic exceeded what the compiler would if-convert and cost the
  vectorization of the whole loop. `std::fmin`/`std::fmax` are single
  instructions and are defined for NaN.
- Selecting between the two branches with a ternary, or with a `bool` cast to
  `float`, is also control flow. Deriving the mask from
  `copysign(1, magnitude - crossover)` is pure data flow and vectorizes.

Two things did not work, and are worth recording:

- **Replacing the library call, on its own, was worth 1.5x on the epilogue and
  nothing end to end.** The activation is 0.2 ms of a step that the wider
  grades measure at 15 ms, against a 30% run-to-run spread on the whole graph.
  The gain is real and measured in isolation; it is simply not where the time
  was. What paid was changing *what is stored*, not how fast the function
  evaluates.
- **`-mprefer-vector-width=512` on the elementwise translation units bought
  nothing.** GCC defaults to 256-bit vectors on Intel targets, and the report
  confirms the flag moves 78 loops in the fitting kernel from 32-byte to
  64-byte vectors -- yet the fitting kernel measured 1.22 ms against 1.20 ms
  without it. These loops are not instruction bound. The flag was reverted
  rather than kept on the theory that wider must be better.

The forward's remaining transcendental keeps the fused epilogue about 15%
slower than the equivalent chain of separate aten operations (neo: 1.13 ms
against 0.95 ms), which do strictly more memory traffic. What aten has is a
hand-written AVX-512 `tanh`; closing that gap means intrinsics, and at 0.2 ms
per model call it is not the next thing to do.

### 3.11 A counting sort's histogram belongs to its chunk, not to its nodes

With the plumbing of Section 3.7 in place, a stage profile of a neo step at
8000 atoms put 1.98 ms of a 2.54 ms graph assembly inside `build_graph_csr`,
and only 0.56 ms in the two passes over the topology. The sort of the source
axis, not the geometry, was the adapter's cost.

The histogram was laid out node-major -- a contiguous row of chunk counters per
node -- so that the two prefix passes would read contiguously. That optimized
the wrong pass. Each chunk's counters were then spread across the whole 5.3 MiB
table at a 664-byte stride, so every increment of the two passes over the edge
axis touched a distinct cache line and missed to the last level. Transposing
the table gives each chunk a contiguous column of tens of kilobytes, resident
in L2, and leaves the two prefix passes strided instead -- they touch the table
once each against the edge axis twice. That took `build_graph_csr` from 1.98 ms
to 1.11 ms and the step from 10.23 ms to 9.60 ms.

A microbenchmark of the sort alone (`/tmp` scaffolding, not retained) then put
the remainder at 0.767 ms for 1.26 million edges: 0.382 in the permutation
write, 0.138 in the histogram, 0.126 in the node totals, 0.063 zeroing, 0.048
in the chunk offsets, 0.033 in the identity permutation. The write is a random
scatter into a 10 MB array, which a radix pre-partition would localize; the
remainder is spread thinly enough that no single change is worth its risk.

### 3.12 What did not help: staging, Kokkos, and the neighbor skin

Three attempts on the LAMMPS path produced nothing, and the reasons are more
useful than the attempts.

**Gathering once instead of twice made the assembly slower.** The two passes
over the topology count and then write, so every candidate's coordinates are
gathered twice. Staging each thread's survivors and copying them into the
payload afterwards gathers once -- and measured 2.09 ms against 0.56 ms. The
coordinates of a host-sized neighbor list stay resident in L2, so the second
gather is nearly free, while staging moves 70 MB through a buffer and reads it
back. The lesson is that a redundant *cached* read is not a cost; a redundant
*write* is.

**Kokkos has nothing to accelerate on a CPU host.** `pair_style deepmd/kk`
exists to keep the neighbor list device-resident and avoid host-device
synchronization; on a CPU the device is the host, so the transfer it removes is
already absent, and the compact canonical ABI it feeds would save only the
difference between 32-bit and 64-bit edge indices. What remains is Kokkos'
parallelization of LAMMPS' own timestep, and the timing breakdown bounds that:

```text
Pair 97.76%   Neigh 0.00%   Comm 0.52%   Modify 1.05%   Other 0.55%
```

Everything outside the pair style is 2.2% of the step, so a perfectly
parallelized remainder is worth at most that. The LAMMPS build available here
is configured `CUDA Serial` and refuses to start without a GPU, so measuring it
would mean a rebuild with an OpenMP host backend for a bounded 2% -- not a
trade worth making. On CUDA the same package remains essential, for the reason
it exists.

**The neighbor skin is nearly flat.** The adapter scans every candidate the
skin list carries, so a thinner skin scans fewer. It also rebuilds more often:

| `neighbor` skin | skin neighbours per atom |  neo step |
| --------------- | -----------------------: | --------: |
| 0.3 A           |                      167 |  14.75 ms |
| 0.6             |                      191 | **10.71** |
| 1.0             |                      274 |     10.80 |
| 2.0             |                      374 |     11.08 |

The two effects cancel between 0.6 and 2.0 A, and a skin thin enough to matter
(0.3 A) rebuilds nearly every step and costs 37%. The benchmarks use 1.0 A.

### 3.13 Retaining the heap spends the capacity that saving an allocation buys

A destination-major payload makes `destination_order` the identity, so the
adapter stops materializing it and hands the consumers an empty permutation;
both force kernels already index the rows directly when the permutation is
absent. That removes eight bytes per edge -- 157 MB on a 125,000-atom system --
of allocation and fill on every step. Both deployment paths produce identical
energies and force norms with it.

It bought nothing measurable. The step time moved inside the run-to-run spread,
and the peak resident set did not fall at all:

| grade @130,000 atoms | peak, retaining | peak, `DP_CPU_MALLOC_RETAIN=0` | step, retaining | step, not |
| -------------------- | --------------: | -----------------------------: | --------------: | --------: |
| nano                 |       2.833 GiB |                      2.235 GiB |        113.6 ms |  305.1 ms |
| neo                  |           3.763 |                          2.987 |           223.8 |     437.5 |
| plus                 |           4.162 |                          3.651 |           475.7 |     617.3 |

The reason is Section 3.6. Retaining blocks that a step reuses is worth 1.3x to
2.7x of throughput at these sizes, and it costs 0.5 to 0.8 GiB of peak resident
set -- several times what the identity permutation occupied. Since capacity is
*defined* by the peak resident set, the retention policy has already spent more
capacity than any single-allocation saving can return, and the saving is
invisible in the metric it was meant to move.

Two consequences worth carrying forward. A per-allocation memory optimization
cannot be validated against peak resident set while the heap is retained; it
has to be measured against the live working set, or with retention disabled.
And raising capacity on this path means either shrinking the live set wholesale
-- the compact canonical payload of Section 7 -- or making retention bounded
rather than accumulating, which trades back the throughput it was introduced
for. The change is kept because it is strictly less work and less memory for
identical results, not because it was measured to help.

### 3.14 Instruction-set selection is a run-time decision

The kernel body is compiled three times -- `x86-64`, `x86-64-v3` and
`x86-64-v4` -- into separate namespaces, and `__builtin_cpu_supports` selects
one on first use. Building the library for the host with `-march=native` would
make it unusable on any older machine, which matters because it ships inside a
wheel and inside the LAMMPS deployment tree.

The architecture level is *pinned* per translation unit rather than added to
it. The project may be configured with `ENABLE_NATIVE_OPTIMIZATION`, and a
trailing `-march` wins, so without pinning the fallback units emit AVX-512 as
well and the dispatch is a fiction that fails with an illegal instruction on
the first older host. `objdump` confirms the isolation: the AVX2 unit contains
no `zmm` reference and the scalar unit no `ymm` or `zmm` reference.

## 4. Measured performance

Machine: two Intel Xeon Platinum 8457C (Sapphire Rapids), 45 cores per socket
with 165 of 180 logical processors online, so 83 physical cores; AVX-512F/DQ/
BW/VL with VNNI and BF16, no AMX exposed under the hypervisor; 48 KiB L1d and
2 MiB L2 per core, 97.5 MiB L3 per socket; two NUMA nodes; 1.7 TiB memory.
All measurements use 83 threads with `OMP_PROC_BIND=close OMP_PLACES=cores`,
a fully periodic diamond supercell at 158 edges per atom, and discard warmup.

### 4.1 End to end

One energy, force and virial step through the frozen AOTInductor package, at
8000 atoms and 1,264,000 edges. The baseline is the uncompressed model frozen
the same way, with the parallelization fix of Section 3.1 applied:

| grade | C0 / lmax / R |  baseline |    this work | speedup | atoms/ms |
| ----- | ------------- | --------: | -----------: | ------: | -------: |
| nano  | 8 / 2 / 0     |  225.8 ms |  **3.59 ms** |     63x |     2229 |
| mini  | 32 / 2 / 0    |  610.9 ms |  **5.02 ms** |    122x |     1594 |
| neo   | 64 / 2 / 0    | 1063.2 ms |  **5.69 ms** |    187x |     1406 |
| air   | 64 / 3 / 4    | 51 626 ms |  **8.04 ms** |   6421x |      995 |
| plus  | 128 / 3 / 4   | 54 131 ms | **11.32 ms** |   4781x |      707 |

Run-to-run spread on the compressed column is about 15%, so treat the speedups
as one significant figure. The two mode-carrying grades are outliers because
Inductor materializes the gathered ordered mixing table: `(E, C0, R)` is 663 MB
at 647k edges and `C0 = 64`. The fused kernel reads the same coefficients as
`R` broadcast weighted passes over contiguous channels and never lands them.

### 4.2 System-size scaling

Compressed package, throughput in atoms/ms:

|   atoms |    edges | nano | mini |  neo | air | plus |
| ------: | -------: | ---: | ---: | ---: | --: | ---: |
|     216 |   34 128 |  224 |  129 |  106 |  97 |   76 |
|   1 000 |  158 000 |  723 |  427 |  359 | 324 |  244 |
|   4 096 |  647 168 | 1490 |  781 |  812 | 677 |  433 |
|   8 000 | 1264 000 | 2009 | 1169 | 1073 | 724 |  529 |
|  17 576 | 2777 008 | 1843 | 1356 | 1158 | 776 |  568 |
|  32 768 | 5177 344 | 1860 | 1494 | 1205 | 736 |  601 |
|  64 000 | 10112000 | 2028 | 1395 | 1049 | 920 |  563 |
| 125 000 | 19750000 | 2076 | 1369 | 1119 | 896 |  583 |

One process sweeps every point in order, so these sit a few per cent below the
isolated best-of figures of Section 4.1; the curve's shape is what they are
for. Throughput saturates around ten thousand atoms and stays flat, which is
the expected behaviour of a kernel whose work is proportional to the edge
count. Resident memory at 125,000 atoms is 4.9 to 6.8 GiB including the
retained heap of Section 3.6; the incremental growth across measured steps is below 10
MiB, so the step allocates nothing it does not reuse. Without the retention
policy the same curve loses more than half its throughput above 32,768 atoms,
which is the shape that made the policy necessary rather than merely
profitable.

### 4.3 Kernel cost, and where it goes

Descriptor operator alone at 8000 atoms, forward and analytical backward:

| grade | forward | backward |   total |
| ----- | ------: | -------: | ------: |
| nano  | 0.73 ms |  1.63 ms | 2.57 ms |
| mini  |    0.58 |     1.61 |    2.46 |
| neo   |    0.72 |     1.94 |    2.81 |
| air   |    1.29 |     3.23 |    4.77 |
| plus  |    1.88 |     4.41 |    7.40 |

Achieved parallelism, measured as process CPU time over wall time, is 80 to 88
of 83 cores throughout.

The backward columns for the wider grades are what Section 3.9 recovered:
before vectorizing its two channel sweeps they read 2.41, 4.99 and 9.75 ms for
neo, air and plus.

Separating the per-node from the per-edge cost by measuring the same node axis
at two cutoffs (4000 and 158,000 edges over 1000 nodes) gives, in cycles at
2.6 GHz:

| grade | forward / edge | forward / node | backward / edge | backward / node |
| ----- | -------------: | -------------: | --------------: | --------------: |
| nano  |             66 |          1 744 |             325 |           3 005 |
| neo   |             92 |          2 041 |             420 |           3 770 |
| plus  |            243 |          5 530 |            1653 |          11 189 |

The per-node term is the invariant readout, and at 158 edges per atom it is 4%
to 8% of the *descriptor operator*; the scan dominates. These two columns were
separated before Section 3.9, so they overstate the backward's per-edge term by
the factor that section recovered. The forward matches an instruction-count
estimate closely, while the backward runs at roughly half that instruction
rate, which locates its remaining limit in the per-edge critical path rather
than in issue throughput or memory.

### 4.4 Threads and simultaneous multithreading

One step of the compressed neo package at 8000 atoms, one process per point:

| threads |    1 |    2 |    4 |    8 |   16 |   32 |   48 |   64 |   83 |  165 |
| ------: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|      ms |  336 |  173 | 87.7 | 44.9 | 24.9 | 12.6 |  9.7 |  7.5 |  7.7 |  6.7 |
| speedup | 1.00 | 1.94 | 3.83 | 7.49 | 13.5 | 26.8 | 34.6 | 44.6 | 44.0 | 50.1 |

Scaling is close to linear to 32 threads and saturates near 64, where the step
is 7.5 ms and the fixed per-call cost -- allocation, the small node-axis
tensors, the readout -- starts to show. The last column enables simultaneous
multithreading, which is worth 14% here.

That last point is the opposite of what the *baseline* does. The uncompressed
Inductor package at 4096 atoms measures 102 ms on 83 physical cores and 153 ms
on 165 logical ones, a 1.5x loss, because it is bandwidth bound on the
intermediates it materializes and a second thread on a core only adds pressure
on one L1. The fused path has no such intermediates, so a second hardware
thread finds idle issue slots. The measurements in this document use one thread
per physical core because that is the conservative setting for both.

### 4.5 Capacity and throughput against system size, through LAMMPS

The CUDA scan finds a model's ceiling by doubling the system until the first
out-of-memory failure and bisecting the last interval. A CPU host has more
memory than a job of this shape usefully occupies, so the CPU sweep
(`debug/cpu_bench/lmp_sweep.py`) replaces the failure with a resident-memory
budget -- the number a deployment plans against -- and otherwise keeps the same
shape: double from 128 atoms, then three bisections. Largest system inside 4
GiB, and the whole-step throughput at each size in atoms/ms:

|           atoms |        nano |        mini |         neo |         air |        plus |
| --------------: | ----------: | ----------: | ----------: | ----------: | ----------: |
|             128 |         105 |          77 |          65 |          61 |          44 |
|             512 |         217 |         207 |         181 |         168 |         126 |
|           2 048 |         600 |         449 |         423 |         393 |         301 |
|           8 000 |         968 |         760 |         734 |         644 |         426 |
|          16 224 |        1248 |         985 |         861 |         669 |         477 |
|          32 768 |        1239 |         935 |         782 |         667 |         338 |
|          66 528 |        1094 |         950 |         822 |         683 |         313 |
|         130 000 |         974 |         742 |         571 |         543 |         525 |
| ceiling (4 GiB) | **229 680** | **162 400** | **146 016** | **146 016** | **124 200** |

Throughput peaks between 16 000 and 66 000 atoms and falls by a quarter to a
third beyond it, where the step's working set leaves the 195 MiB of last-level
cache. Above roughly 100 000 atoms a single 30-step run varies by 30%, so read
the large columns as a trend rather than as points; the ceilings come from the
peak resident set, which is monotone in system size and reproducible.

Each grade's ceiling is set almost entirely by the graph rather than by the
model: nano and plus differ by 24 times in model width and by only 1.8 times in
capacity, because what scales with the system is the edge payload and the
cached skin topology, which depend on the cutoff alone.

### 4.6 Whole-step throughput of the two deployment paths

The numbers above hold the neighbor graph fixed, which is the right boundary
for judging the operators. A deployment also pays for the graph. Both paths at
8000 atoms, after Section 3.7:

| path                       | grade | baseline |   this work | speedup |
| -------------------------- | ----- | -------: | ----------: | ------: |
| LAMMPS `pair_style deepmd` | nano  | 150.0 ms | **7.19 ms** |     21x |
|                            | mini  |    563.7 |    **8.75** |     64x |
|                            | neo   |   1121.1 |    **9.43** |    119x |
|                            | air   |        — |   **12.14** |       — |
|                            | plus  |        — |   **16.25** |       — |
| `DeepPot.eval` (ASE)       | nano  | 283.0 ms | **8.29 ms** |     34x |
|                            | mini  |    732.2 |    **9.43** |     78x |
|                            | neo   |   1208.1 |   **10.27** |    118x |
|                            | air   |   52 884 |   **13.10** |   4037x |
|                            | plus  |   55 563 |   **16.66** |   3335x |

The two uncompressed mode-carrying rows were too slow to measure through
LAMMPS. Both paths now sit within 5 ms of the model itself and within a
millisecond of each other: the LAMMPS step carries 4.0 to 5.6 ms of adapter and
integration over the model graph, the ASE call 4.4 to 5.3 ms of graph
construction and conversion. That is what a molecular-dynamics host should
show, and it is the reverse of what the same measurement gave before Section
3.7, when the plumbing cost an order of magnitude more than the model on both
paths.

The two paths agree on the physics through completely different graph
construction -- a C++ assembler over a cached LAMMPS skin list against a fused
Python cell-list search. On the same 8000-atom configuration:

| grade | LAMMPS `PotEng` | `DeepPot.eval` |     difference |
| ----- | --------------: | -------------: | -------------: |
| nano  |  -71798.0120 eV |    -71798.0120 | 1.6e-9 eV/atom |
| neo   |     -72102.3160 |    -72102.3162 |         2.1e-8 |
| plus  |     -72208.5300 |    -72208.5297 |         4.0e-8 |

Force norms agree to the six digits LAMMPS prints. The residual is float32
rounding: the two paths sum the same edges in different orders.

## 5. Accuracy

Compared against the uncompressed model in its own float32 arithmetic, on a
512-atom diamond supercell displaced by 0.1 A per atom so that forces reach
28 eV/A:

| grade | dE per atom |      dF max | dVirial (relative) |
| ----- | ----------: | ----------: | -----------------: |
| nano  |   4.0e-6 eV | 3.4e-5 eV/A |             8.0e-6 |
| mini  |      1.2e-6 |      2.8e-5 |             4.2e-6 |
| neo   |      9.2e-7 |      3.9e-5 |             3.2e-6 |
| air   |      5.7e-7 |      2.0e-5 |             2.2e-6 |
| plus  |      4.1e-7 |      1.9e-5 |             4.6e-6 |

A table stride of 0.002 A and one of 0.01 A give the same deviation to within
the run-to-run spread, so the quintic interpolation error is below float32
rounding and the default `dp compress` stride of 0.01 needs no revision.

Against the portable reference of the *same* compressed equations, over the
cross product of five channel widths, three angular degrees and two mode
ranks, the forward agrees to 2e-6 relative and the analytical backward to
3e-6 relative -- float32 rounding.

## 6. Rejected or ineffective changes

| Experiment                                                                | Result                                                                                                                                                                                                                                                                                            |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Mode residual inside the channel loop, keeping the amplitude in registers | 2.5x slower on the mode-carrying grades: the inner trip count becomes a run-time value the vectorizer cannot carry an accumulator across. The amplitude fits L1 several times over, so the repeated passes cost nothing.                                                                          |
| Coarsening the radial table from 0.002 to 0.05 A                          | Within run-to-run spread at every width. The first-principles estimate had made the table the dominant traffic and predicted an L2-capacity cliff; the measurement refuted it, and the byte count is not what limits the scan.                                                                    |
| Four independent partial sums in the two backward reductions              | 5% to 13%, not the 3x a pure latency-chain model predicted. Worth keeping, but the dependency chains were not the limit either.                                                                                                                                                                   |
| Single socket                                                             | 45 cores on the local socket reach 62% of the throughput of 83 cores across both, and 38 cores on the *remote* socket reach 71%. Remote memory is faster than local here, so there is no NUMA placement to fix and the throughput cliff it was meant to explain was the allocator of Section 3.6. |
| Minimax rational tanh over the whole line                                 | Non-monotone by up to 7 units in the last place, see Section 3.10. Rejected on smoothness, not on speed.                                                                                                                                                                                          |
| `-mprefer-vector-width=512` on the elementwise units                      | Moves 78 loops from 32-byte to 64-byte vectors and changes nothing measurable; those loops are not instruction bound.                                                                                                                                                                             |
| bf16, fp16, VNNI, AMX                                                     | Rejected by contract: a molecular-dynamics potential-energy surface has to stay smooth.                                                                                                                                                                                                           |

## 7. Remaining directions

1. The fitting network is still the largest block of a step: 2.31 ms of neo's
   5.69 ms model graph, against 2.81 ms for the descriptor's forward and
   backward together. Only 0.35 ms of that is the matrix products. Section 3.10
   took the backward's share to the bandwidth floor; the forward's single
   transcendental is what remains, and aten's hand-written vectorized `tanh`
   runs the same elementwise pass roughly six times faster than the
   compiler-vectorized form. Reaching that needs intrinsics.
1. The backward's per-edge critical path, which is now 1.6 to 4.4 ms against a
   0.6 to 1.9 ms forward. Section 3.9 removed the scalar sweeps; what remains
   is a dependency chain. The structural fix is software pipelining across
   edges -- two edges in flight so their chains interleave -- which the CUDA
   path rejected for register pressure but which a CPU with 32 architectural
   vector registers can afford.
1. Hand-written intrinsics. The channel loops rely on GCC's vectorizer, which
   the reports confirm produces 64-byte vector code but which also inserts an
   alias-versioned entry per loop. Explicit intrinsics would remove the
   versioning and pin the register allocation.

A note on capacity, since the sweep of Section 4.5 raises it. The per-atom
budget fits at 16.2 KB for nano and 30.8 KB for plus, against 6.8 KB on CUDA,
and the width-dependent share is 9% to 17% for every grade but plus, where it
is 47%. So node tiling -- which on CUDA makes capacity independent of model
width -- would mostly fix plus here and barely move the rest; the fitting half
is already tiled, only the descriptor is not. The shared 16 KB baseline is the
larger target, and it is redundancy rather than necessity: int64 indices where
uint32 suffices, and an edge destination column that the row pointers already
imply. Together about 4.4 KB per atom, which is what the compact canonical
payload specifies. That payload was declined for the CPU on the grounds that
its purpose -- letting a device-resident neighbor list reach the descriptor
without a host round trip -- is vacuous on a host; the reasoning conflated its
latency purpose with its memory contract, and the memory contract is worth 28%
of the baseline. Section 3.13 is the caveat: with the heap retained, that
saving has to be large enough to outweigh the retention itself before the
capacity metric moves.

The invariant readout is *not* on this list, though it is the one part of the
descriptor that is still scalar. Its degree blocks are at most 16 channels
wide, so its Gram matrices are 16x16 and its whole per-node cost is 0.22 ms of
a 7.5 ms step. Vectorizing the triangular contraction would recover a fraction
of that, which is below the run-to-run spread of the measurement that would
have to justify it.
4\. The source-major permutation is built on every step because the compiled
graph reads it, even for a model whose descriptor never does: it reaches
`edge_force_virial` as an operator argument that the spin-free path ignores.
Splitting that argument out of the non-magnetic schema would remove a
threaded pass over the edge axis from both deployment paths.
5\. Native spin. The magnetic families need a source-major scan beside the
destination-major one; no CPU deployment asks for it yet.
6\. A CPU counter profile. This host denies the performance monitoring unit
(`perf_event_paranoid = 2`, a 64 KiB `memlock` cap), so every attribution
above rests on ablation, compiler reports, static instruction counts, and a
two-point separation of per-node from per-edge cost. The conclusions that
survived contact with measurement are marked as such; the roofline estimate
in Section 6 did not.

## 8. Tests and reproduction

Regression tests: `source/tests/pt_expt/descriptor/test_dpa4c_cpu.py`. They
cover forward and backward parity against the portable reference over the
whole compiled parameter surface and both topology forms, the masked-edge
contract, the isolated-atom limit, the force and virial assembly against the
array-API scatter it replaces, the fused fitting against the dense network,
and the table cache.

Benchmarks: `debug/cpu_bench/`.

```bash
cd debug/cpu_bench
# Freeze the baseline and the compressed artifact of every released grade.
OMP_NUM_THREADS=83 python freeze.py --grade all --variant plain
OMP_NUM_THREADS=83 python freeze.py --grade all --variant compress --stride 0.01

# End to end, one row per (grade, path).
OMP_NUM_THREADS=83 OMP_PROC_BIND=close OMP_PLACES=cores \
    python bench.py --grade all --paths plain,compress --atoms 8000

# Operator alone, and against the same math under Inductor.
OMP_NUM_THREADS=83 python bench_kernel.py --grade neo --atoms 8000 --reference

# Numerics.
python check_kernel.py --sweep          # operator against the reference
python check_model.py --atoms 512       # compressed path against the model

# Size scaling, and thread scaling one process per point.
OMP_NUM_THREADS=83 python scaling.py --grades nano,mini,neo,air,plus
for t in 1 2 4 8 16 32 48 64 83 165; do
    OMP_NUM_THREADS=$t python scaling.py --quiet --grades neo --sizes 8000
done

# Whole-step deployment paths at one size.
OMP_NUM_THREADS=83 python lmp_scan.py --grades nano,neo --atoms 8000
OMP_NUM_THREADS=83 python bench_deeppot.py --grades nano,neo --atoms 8000

# Capacity and throughput sweep: double from 128 atoms until the resident set
# leaves the budget, then bisect the last interval three times.
OMP_NUM_THREADS=83 python lmp_sweep.py --budget-gb 4 --steps 30
```

The `infer` test suite is bound by artifact compilation rather than by
execution -- each case freezes a `.pt2`, and AOTInductor's C++ compilation is
single-threaded at 20 to 40 s per artifact -- so run it across cores:

```bash
pytest source/tests/pt_expt/infer -q -n 16 --dist loadfile
```

The Inductor baseline analysis of Section 3.1 is reproduced by
`probe_inductor_threads.py`, which freezes one grade under several option sets
and reports how many generated kernels carry an OpenMP region, and by
`profile_baseline.py`, which freezes with per-kernel profiling and attributes
the baseline step to its generated kernels.

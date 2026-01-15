# LAMMPS performance for DeePMD-kit

## Scope and baseline

The OutisLi LAMMPS fork is a general execution engine for DeePMD-kit potentials, including ordinary molecular dynamics and native-spin dynamics. The public LAMMPS release remains a supported alternative. This study measures the additional benefit of the fork while holding the DeePMD-kit library, frozen model, input structure and simulation settings fixed. It complements [the compressed DPA4C CUDA study](dpa4c_cuda.md), whose ordinary-model comparisons use the public LAMMPS release to isolate improvements in DeePMD-kit itself.

The starting fork is `/nas/outisli/Software/lammps/lammps-outisli`, commit `43dcef6`. Its tracked files are clean; the untracked installation directories are preserved. The DeePMD-kit checkout starts at `9812b06f1`. The existing `build_kk` uses CUDA 13.3, Kokkos 5.2.1, the `BLACKWELL120` architecture and the Open MPI installation of the existing `dpmd` environment. The baseline executable and shared library are copied to `/tmp/lmp-perf-20261021/base`; each benchmark explicitly selects the corresponding shared library as well as the executable.

The handover reads the Claude Code sessions named `dpa4c_perf_base` and `dpa4c_perf_wpe` and the existing CUDA study. The fork already contains a device-resident Nose-Hoover chain for ordinary NVT, a neighbor check in the memory space that owns the coordinates, reproducible atom sorting, and a Kokkos Newtonian-spin integrator. These are part of the baseline. The WPE comparison motivates measuring execution overhead and neighbor work; its implementation is not an input to these changes.

## Measurement and correctness

GPU 0 through GPU 3 of the workstation are available for this task. Measurements compare the two builds on the same RTX PRO 6000 Blackwell GPU, with the order alternating between A/B and B/A. Model archives are the trained `debug/cuda_bench/models/v20260911_trained_rcf` files, rather than the random-weight archives that the older harness selects by default. Ordinary runs use the existing carbon and BCN inputs at 512, 10,296 and 100,800 atoms. The primary metric is LAMMPS loop time per step after warm-up; profiler times establish attribution and are reported separately. CUDA-context initialization and profiling do not overlap the timed part of another measurement.

Native-spin timings use the existing random-weight archives under `models/spin_random` and `spin_newton/in_spin_newton.lammps`, with `fix nve/spin/kk` and the established 0.000002 ps timestep. These measure the actual Newtonian spin integrator, including its changing Cartesian moments. They are not trained magnetic-model accuracy benchmarks.

Integration changes are checked through LAMMPS inputs, including the final positions, velocities and forces, thermodynamic values, group and mass variants, repeated runs and restart behavior where applicable. Unchanged arithmetic is expected to reproduce the baseline bit for bit on this build. A changed reduction tree requires an explicit numerical assessment rather than a relaxed comparison hidden in the harness. Existing related tests are run where their build dependencies are available. Each accepted optimization receives an independent review using `gpt-6.1-sol` at `high` reasoning effort; findings are resolved before the corresponding code commit. Private benchmark material and this document belong in a separate `doc` commit in the DeePMD-kit checkout.

## Prioritized work

The estimates below are hypotheses, not measured speedups. The list prioritizes expected benefit relative to implementation and validation cost. An experiment without a reproducible benefit is recorded and removed from the production change.

| Priority | Work                                                                               | Expected benefit                                                                       | Difficulty                                                      | Status                               |
| -------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | --------------------------------------------------------------- | ------------------------------------ |
| 1        | Reduce a scalar kinetic energy with one scalar accumulator                         | Several microseconds of potential saving on small systems                              | Low to medium; check launch geometry and summation order        | Committed as `fee7c14`               |
| 2        | Fuse the initial NVT velocity scaling, half-step kick and full-step drift          | Two fewer host kernel launches; several percent on small systems                       | Low to medium; retain arithmetic and subclass behavior          | Committed as `7c28131`               |
| 3        | Remove the duplicate stream wait in the neighbor displacement check                | Approximately one host synchronization call per checked step, for ordinary and spin MD | Low to medium; preserve device/host selection and MPI decisions | Committed as `6218c78`               |
| 4        | Store shared neighbor-bin metadata in its integer types                            | Reduce conversions and shared storage during neighbor rebuilds                         | Medium; validate all three team builders and bin alignment      | Committed as `f3e7a95`               |
| 5        | Reject candidates with a conservative coordinate bound                             | Reduce double-precision work on candidates outside the cutoff                          | Medium; prove the bound against encoded cutoffs and rounding    | Rejected after a measured slowdown   |
| 6        | Store atom-image indices in the compact graph and decode bond vectors in the model | Remove 12 emitted bytes per edge; benefit depends on scan cost and device bandwidth    | Medium to high; align plugin, API and model arithmetic          | Rejected after the whole-step matrix |
| 7        | Pack narrow atom IDs and local indices into scalar sort keys                       | Reduce the 17-kernel comparison sort used by the atom map                              | Medium; preserve wide IDs and complete duplicate chains         | Design reviewed; lower measured cost |

Neighbor-rebuild profiling identifies the shared-bin metadata layout and double-precision candidate checks as the next targets. Bin overflow already grows to the observed requirement, and combining coordinate-history storage with binning would couple separate subsystems. Changes to skin or rebuild frequency are outside this implementation study.

Fusing the final kick with the temperature reduction is a later candidate. A temperature compute may use a group different from the integrator, and its reduction covers all local atoms even when the integrator uses the first-group prefix. The coupling and validation cost exceed those of eliminating the unused scalar-reduction components, so it is not the first approach.

## Baseline profile

On 21 October 2026, Nano at 512 carbon atoms takes 0.1501 ms per step under Nsight Systems, with 0.1055 ms of device work, 26 kernels and 14 host kernel launches per step. The integrator launches seven kernels; the temperature scalar reduction takes approximately 9.8 microseconds, and the neighbor displacement reduction approximately 4.1 microseconds. There are two stream-synchronize calls per step around the host result of the neighbor check. This is a profiled measurement, not a plain-run throughput result. The report and SQLite trace are under `debug/cuda_bench/work_nsys/lmp_perf_base_nano_512`.

`ComputeTempKokkos::reduce_scalar` uses the six-component `CTEMP` accumulator although its scalar operator only updates `t0`. `FixNHKokkos::initial_integrate` launches separate chain-scaling, velocity-update and position-update kernels after the chain update. These source facts motivate the first two experiments.

## Scalar temperature reduction

The scalar host result, device result and thermostat kinetic-energy storage use one `double`. The pressure tensor retains its six-component accumulator. Kokkos infers the reduction value type from the tagged operator, so no extra dispatch, state or public option is needed. The per-atom formula, mask, range, MPI reduction, degrees-of-freedom handling and temperature-compute invocation state are unchanged.

At 512 atoms, both builds launch two blocks of 256 threads. The scalar accumulator reduces registers from 48 to 26 and dynamic shared memory from 12,384 to 2,064 bytes. The temperature kernel takes 9.741 microseconds in the baseline and 4.299 microseconds with the scalar accumulator. These are Nsight Systems kernel measurements; whole-step A/B measurements are required separately. The two reports are `work_nsys/lmp_perf_base_nano_512` and `work_nsys/lmp_perf_scalar_nano_512` under `debug/cuda_bench`.

The first integration check used ordinary interacting `lj/cut/kk` and failed its bitwise comparison. An unchanged-baseline repeat fails as well: at timestep zero the coordinates and velocities agree but the pair forces already differ in their last bits. This precedes any integration update and reflects the ordering of the LJ force accumulation. It is not evidence of a temperature regression. The controlled test uses a cutoff below all interatomic distances and a constant nonzero `fix addforce/kk` force, retaining the kick while removing this unrelated force-order variability. All eleven controlled configurations agree bit for bit in every recorded thermodynamic row and particle state: ordinary NVT, multiple chain loops with drag and a temperature ramp, a subgroup, firstgroup, another temperature group, per-atom masses, a biased temperature, dynamic degrees of freedom, isotropic and anisotropic NPT, and restart. The source and runner are `debug/lmp_perf/check_integrator.py`; raw results are `/tmp/lmp-perf-20261021/controlled_base` and `controlled_scalar`.

The independent reviewer also compares the compiled PTX of the per-atom scalar accumulation: the multiply and three fused multiply-add instructions are identical between builds. Together with the matched launch geometry and controlled trajectories, this supports preserving the arithmetic on the measured build without adding rounding operations or a custom reduction implementation.

Further checks compare two-rank NVT and anisotropic NPT; explicit `temp/kk/host` scalar and tensor outputs; an empty temperature group; and actual DPA4C trajectories on carbon (Nano and Plus, 512 atoms), BCN (Nano, 10,296 atoms; Plus, 512 atoms), and native spin (Nano, 512 atoms). The independent reviewer verifies 39 nonempty output files across 19 cases, all byte-identical. CUDA and host template instantiations compile. The source checks and documentation anchor check pass; the remaining documentation checks cannot run because `doxygen` is absent. No dependencies are installed for this check.

Whole-step carbon measurements, six alternating A/B pairs on GPU 0, with the same trained archives and NVT input:

| Grade |   Atoms | Baseline (ms/step) | Scalar accumulator (ms/step) | Time reduction |
| ----- | ------: | -----------------: | ---------------------------: | -------------: |
| Nano  |     512 |           0.132590 |                     0.128535 |          3.06% |
| Nano  |  10,296 |           0.273871 |                     0.270322 |          1.30% |
| Nano  | 100,800 |           1.672788 |                     1.664665 |          0.49% |
| Mini  |     512 |           0.147064 |                     0.139447 |          5.18% |
| Plus  |     512 |           0.257010 |                     0.255069 |          0.76% |
| Plus  | 100,800 |          10.472175 |                    10.450325 |          0.21% |

The small-system benefit is clear; the sub-percent large-system differences establish absence of an observed regression rather than a substantial throughput gain. The per-run values are preserved in `debug/lmp_perf/results/scalar_carbon.json`, with full logs under `/tmp/lmp-perf-20261021/ab_scalar_carbon`. The independent code and correctness review approves the change after the validation above.

## Initial NVT integration

For the existing device-resident NVT path, the chain update computes its scale factors and the initial per-atom kernel applies the ordered composition of scaling, kick and drift. It calls the existing three per-atom operators, so their formulas are defined once. Each chain-loop factor is multiplied separately. The final half-step explicitly applies the chain update and its scaling kernel. The existing qualification requires the standard translational scaling, kick and drift; sphere, SLLOD, spin integration, barostats and temperature computes outside the qualified path retain their own hooks. No additional public setting or state flag is introduced.

All eleven controlled integration configurations and the five real DPA4C cases match the original baseline byte for byte. Nsight Systems confirms five integrator kernels instead of seven, 12 host kernel launches instead of 14, and 24 total kernels instead of 26 per step. The measured integrator device time falls from approximately 19.2 to 18.1 microseconds; the larger expected benefit is the reduction in host submission work. The profile is `debug/cuda_bench/work_nsys/lmp_perf_initial_nano_512`.

The independent reviewer confirms the reuse of the mathematical operators, field synchronization, group and mass handling, and the existing subclass exclusions. Its clarification request is incorporated into the capability comments: directly composing the updates requires all three standard translational operations, not only ordinary velocity scaling.

The repository's existing `unittest/commands/test_nonspin_nh.cpp` is compiled against the existing GoogleTest installation under `source/build_tests`, without configuring a new environment or downloading dependencies. All four tests execute and pass against the original fork, the scalar-reduction build and the fused-integration build. They compare ordinary NVE, NVT, NPT and NPH with the host implementation. The logs are `/tmp/lmp-perf-20261021/gtest_nonspin_{base,scalar,initial}.log`.

Incremental whole-step measurements against the scalar-reduction commit, four alternating A/B pairs per point on GPU 0:

| Grade |   Atoms | Scalar only (ms/step) | Fused initial integration (ms/step) | Time reduction |
| ----- | ------: | --------------------: | ----------------------------------: | -------------: |
| Nano  |     512 |              0.128405 |                            0.125388 |          2.35% |
| Nano  |  10,296 |              0.271298 |                            0.265320 |          2.20% |
| Nano  | 100,800 |              1.665138 |                            1.659095 |          0.36% |
| Mini  |     512 |              0.139812 |                            0.137546 |          1.62% |
| Plus  |     512 |              0.254511 |                            0.253673 |          0.33% |
| Plus  | 100,800 |             10.460375 |                           10.450200 |          0.10% |

The useful gain is in the narrow grades at small and medium sizes. The heavy model and large-system results are effectively unchanged at this measurement's spread. Per-run values are `debug/lmp_perf/results/initial_carbon.json`; full logs are under `/tmp/lmp-perf-20261021/ab_initial_carbon`. The final independent review approves the implementation and verifies 33 byte-identical nonempty output files across the sixteen controlled and real-model cases.

## Neighbor displacement check

The CUDA backend of Kokkos waits when a reduction returns its result to an ordinary host scalar. The outer scalar-return reduction wrapper then waits again. `NeighborKokkos::check_distance_kokkos` needs a host decision, but only one wait is necessary. A persistent scalar View in host-pinned memory lets an execution space that can access this memory write the result directly; the caller waits on that same execution-space instance before reading the value and entering the unchanged MPI collective. Execution spaces without access to the result memory retain the blocking scalar reduction. The selection between checking host and device coordinates is unchanged.

The result belongs to the neighbor object, is allocated once, and is initialized by each reduction, including reductions over zero local atoms. No per-step allocation, transfer, public option or change to the skin criterion is introduced. The independent design review confirms the ownership, explicit completion point, host execution behavior and memory-accessibility boundary.

The focused input suite is `debug/lmp_perf/check_neighbor.py`. Its baseline performs 26 rebuilds in 80 steps for repeated threshold crossings, host integration and an include group, 39 for orthogonal and triclinic box deformation, and 26 in a two-rank configuration with one empty rank. Every case checks at least ten rebuilds, particle states, thermodynamic output, rebuild counts and dangerous-build counts. Raw reference output is under `/tmp/lmp-perf-20261021/neighbor_base` and `neighbor_mpi_base`.

The optimized outputs match all six reference cases byte for byte, including every rebuild count and zero dangerous builds. The five ordinary/spin DPA4C cases also reproduce the original fork's output exactly, and the four existing `NonSpinNoseHoover` tests pass. The reviewer independently verifies all 28 nonempty comparison files and confirms that the host case creates unsuffixed `nve` and the two-rank case has zero local atoms on one rank. CUDA and host paths execute; other accelerator backends are covered by source review only.

The profile `debug/cuda_bench/work_nsys/lmp_perf_neighbor_nano_512` contains one stream synchronization per step instead of two. Kernel count and neighbor-kernel time remain unchanged. Whole-step measurements compare the initial-integration build against the pinned-result build, with six alternating pairs for ordinary NVT and four for Newtonian spin:

| Mode     | Grade |   Atoms | Before (ms/step) | Pinned result (ms/step) | Time reduction |
| -------- | ----- | ------: | ---------------: | ----------------------: | -------------: |
| NVT      | Nano  |     512 |         0.125441 |                0.122232 |          2.56% |
| NVT      | Nano  |  10,296 |         0.266774 |                0.266192 |          0.22% |
| NVT      | Nano  | 100,800 |         1.664760 |                1.660680 |          0.25% |
| NVT      | Plus  |     512 |         0.254762 |                0.250198 |          1.79% |
| Spin NVE | Nano  |     512 |         0.129469 |                0.127922 |          1.20% |
| Spin NVE | Nano  |  10,296 |         0.333087 |                0.331828 |          0.38% |
| Spin NVE | Nano  | 100,800 |         2.322365 |                2.324638 |         -0.10% |

The small ordinary NVT points have a clear benefit. The spin medians and larger ordinary points lie within their run-to-run spread. They show no clear regression in these measurements, but do not establish a definitive speedup. The raw values are `debug/lmp_perf/results/neighbor_carbon.json` and `neighbor_spin.json`; full logs are `/tmp/lmp-perf-20261021/ab_neighbor_carbon` and `ab_neighbor_spin`.

## Combined integration and displacement-check results

Direct comparisons of `6218c78` with the original fork, four alternating pairs per point on GPU 0, keep the total benefit independent of the separate incremental timing campaigns. The final-fork columns in this section refer to this three-optimization snapshot:

| System | Grade |   Atoms | Original fork (ms/step) | Final fork (ms/step) | Time reduction |
| ------ | ----- | ------: | ----------------------: | -------------------: | -------------: |
| Carbon | Nano  |     512 |                0.130146 |             0.123594 |          5.03% |
| Carbon | Nano  |  10,296 |                0.271507 |             0.265287 |          2.29% |
| Carbon | Nano  | 100,800 |                1.670480 |             1.652470 |          1.08% |
| BCN    | Nano  |     512 |                0.138207 |             0.127419 |          7.81% |
| BCN    | Nano  | 100,800 |                2.002263 |             1.978433 |          1.19% |

Every 100,800-atom BCN run rebuilds the neighbor list nine times and reports zero dangerous builds, on both builds. These gains therefore do not arise from changing rebuild frequency. Per-run measurements are `debug/lmp_perf/results/total_carbon.json` and `total_bcn.json`; full logs are the corresponding `ab_total_carbon` and `ab_total_bcn` directories under `/tmp/lmp-perf-20261021`.

A separate direct comparison uses the public `patch_4Jul2026` LAMMPS build and its matching plugin from `deepmd-kit_cpp/lib/deepmd_lmp`, versus the final fork and its matching plugin. Both use the same current DeePMD library, trained Nano archive and carbon NVT input. The public installation uses LAMMPS 4 July 2026 and Kokkos 5.1.99; the fork uses LAMMPS 30 September 2026 and Kokkos 5.2.1. Four alternating pairs give:

|   Atoms | Public LAMMPS (ms/step) | Final fork (ms/step) | Time reduction |
| ------: | ----------------------: | -------------------: | -------------: |
|     512 |                0.144665 |             0.124144 |         14.18% |
|  10,296 |                0.287325 |             0.267036 |          7.06% |
| 100,800 |                1.662848 |             1.649583 |          0.80% |

This is a comparison of two installed runtimes. It includes the fork's pre-existing optimizations, the three commits of this study, and changes in the LAMMPS core and Kokkos versions. The public-to-fork difference cannot be attributed entirely to fork-specific optimization; the original-fork-to-final-fork comparison isolates this study's changes. The measured small and medium ordinary-MD systems run faster on the fork installation, without implying the same benefit for every model, ensemble or size. Per-run values are `debug/lmp_perf/results/public_carbon.json`, with logs under `/tmp/lmp-perf-20261021/ab_public_carbon`.

## Remaining large-system cost

The BCN Nano profile at 100,800 atoms, using the build with scalar reduction and initial integration fusion, contains nine rebuild steps among 200 measured steps. A step between rebuilds has a median of 1.7254 ms, and a rebuild step 6.8945 ms. The measured whole loop is 2.0072 ms per step under Nsight Systems. The LAMMPS neighbor-build kernel consumes about 0.1312 ms per step when amortized over the run, while the DeePMD compact-graph build consumes 0.3093 ms per step. The trace is `debug/cuda_bench/work_nsys/lmp_perf_bcn_rebuild`; CPU compilation was active during this attribution run, so its host timing is not a throughput comparison.

This identifies neighbor rebuilds as a distinct direction for large, mobile, multielement systems. Eliminating a duplicate wait in the displacement check does not eliminate these rebuilds or their kernels. Neighbor search remains dominated by candidate arithmetic on this GPU: Nsight Compute reports approximately 77% double-precision pipeline utilization, with memory throughput below 10% of peak.

## Compact graph across the plugin and backend

The DeePMD Kokkos pair style produces a single canonical graph of source-node indices and explicit fp32 edge vectors. The neighbor-rebuild flag caches candidate counts and atom maps, while cutoff membership, vectors and distance order are recomputed each step. Source words select 32 or 64 bits from the node domain; LAMMPS's rank-local atom limit permits 32 bits. Edge counts, offsets and source-CSR positions always remain int64. External callers with larger node domains use the same frozen archive through its dynamic byte-word interface. The single replay key includes the width, so changing width at the same address invalidates its capture.

The indexed-geometry study is recorded in [section 12.3](dpa4c_cuda.md#123-what-was-tried-and-did-not-help). Its 120-configuration matrices on RTX PRO 6000 and H20 pass numerical checks, but the complete implementation regresses on some workstation workloads. Indexed geometry, its startup selector, extra tensor inputs and second replay pool are removed from production. The original vector kernels retain automatic dispatch from device and kernel capabilities. The independent LAMMPS fork optimizations and their measured results remain unchanged.

## Shared neighbor-bin metadata

The ordinary, ghost-center and radius-dependent GPU team builders stage coordinates and metadata in shared memory. Coordinates and radii retain double precision, while atom types and local indices use `int` and atom tags use `tagint`. Tag storage is present only for the half-list, Newton-on, triclinic case that consumes it. Each bin record is padded to a double boundary so adjacent bins remain aligned at odd capacities. Candidate order, exclusions, distance arithmetic, cutoff decisions and neighbor writes are unchanged.

The baseline stores type numbers as doubles and converts each candidate back to an integer. The typed layout removes these repeated conversions and unused tag slots. On the 100,800-atom BCN input, the profiled ordinary neighbor kernel falls from 2.90 to 2.53 ms, with 59 registers per thread unchanged and reported dynamic shared memory reduced from 5.56 to 3.98 KB. Frequencies are not locked, so this profile establishes attribution rather than the final throughput claim. Reports are `/tmp/lmp-perf-r2-20261021/ncu_{base,typed}`.

The correctness fixture freezes all relevant owned and ghost fields, rebuilds and sorts the actual bins, calls the production neighbor builder, and compares every valid GPU neighbor row. Sorting bins alone is insufficient because ghost indices can differ between independent setup runs. Eight frozen-input cases reproduce the baseline byte for byte, including row order and history bits: full lists, half/Newton/triclinic lists, ghost centers and radius-dependent history lists, each at capacities 17 and 99. Adjacent nonempty bins at capacity 17 exercise the two-bin block layout. The five ordinary/spin model cases also retain byte-identical thermo and final states; all four existing NonSpin tests execute and pass.

Complete CUDA memcheck runs for the capacity-17 ghost and radius/history cases report zero errors. The initial UCX MPI initialization probes produce CUDA context errors before the tested kernels; selecting the existing single-process `ob1`/`self` MPI transport permits the complete checks without suppressing sanitizer diagnostics. The sanitized output rows also match the frozen reference. BIGBIG and non-CUDA accelerator configurations have static review only. The independent `gpt-6.1-sol` high-effort review approves the implementation and verifies the fixture, raw comparisons and timing calculations.

Alternating A/B measurements compare `6218c78` against the typed layout on GPU 0:

| Workload                                 |   Atoms | Pairs | Before (ms/step) | Typed bins (ms/step) | Time reduction |
| ---------------------------------------- | ------: | ----: | ---------------: | -------------------: | -------------: |
| Frozen BCN, rebuild every step           | 100,800 |     6 |         6.626885 |             6.246845 |          5.73% |
| BCN NVT, standard 200-step measurement   | 100,800 |     6 |         1.950948 |             1.934380 |          0.85% |
| BCN NVT, extended 2,000-step measurement | 100,800 |     4 |         1.783365 |             1.772475 |          0.61% |
| BCN NVT                                  |  10,296 |     6 |         0.308187 |             0.301463 |          2.18% |
| BCN NVT                                  |     512 |     6 |         0.129531 |             0.129423 |          0.08% |
| Spin NVE                                 | 100,800 |     4 |         2.332903 |             2.329653 |          0.14% |
| Spin NVE                                 |     512 |     4 |         0.127367 |             0.125360 |          1.58% |

Every forced-rebuild run performs 100 rebuilds in 100 measured steps; all six matched pairs improve. Each extended NVT run performs 52 rebuilds in 2,000 measured steps, with zero dangerous builds. Its warm-up is 200 steps rather than 20, so it samples a later trajectory and a different rebuild frequency from the standard short measurement. The 5.73% result applies only to the forced-rebuild workload. Normal NVT amortizes this cost, and its large-system benefit is modest. The 10,296-atom matched pairs include reversals, so the median alone does not establish a stable gain of that size. Every timed spin loop has zero neighbor rebuilds and therefore does not exercise the optimized builder during measurement. Its differences and the small-system NVT differences lie within the observed spread and do not establish a speedup.

The diagnostic sweep also tests existing parameters without changing production defaults. Smaller bin widths increase runtime on this case, and padding capacity from 99 to 128 provides no stable benefit. The existing flat neighbor builder is faster than the original team builder on the forced-rebuild input, motivating work on the team builder's arithmetic rather than a change to rebuild frequency.

Atom-map sorting is a lower-cost target. The original BCN trace contains 17 comparison-sort kernels per rebuild: one block sort, eight partition kernels and eight merge kernels. Across nine rebuilds, these total 1.080758 ms of GPU time, or 0.120084 ms per rebuild and 0.005404 ms per measured MD step. A packed scalar key for 32-bit atom tags can enable radix sorting while preserving the tag/index order; 64-bit tags require their full-width representation. Eliminating the measured GPU sort cost entirely would save only about 0.27% of the original 2 ms step, before any separately measured host-launch saving.

## Rejected coordinate-bound experiment

The candidate rejects an atom before computing the remaining distance components when its x displacement exceeds a conservative global bound. The bound is derived from the largest encoded cutoff squared, extended upward before and after the square root. This covers rounding at zero, subnormal and ordinary cutoffs; conversion to `KK_FLOAT` before taking the maximum is necessary when the device cutoff rounds upward from its double host source. The final double-precision squared-distance expression remains unchanged. The bound is computed during neighbor initialization without adding a transfer or allocation.

Independent review supports the bound, including 3,684 CPU boundary sum/FMA checks, and all eight complete GPU neighbor-row comparisons remain byte-identical. Performance rejects the candidate: the neighbor kernel rises from 2.53 to 2.69 ms, with register and shared-memory use unchanged. Average active threads per warp fall from 22.48 to 18.34 and barrier wait rises from 20.3 to 23.2 cycles. The inferred cause is extra divergence and coordination, which outweigh the skipped arithmetic.

Six alternating plain-run pairs confirm the regression on the forced-rebuild workload: 6.319360 ms/step for typed bins versus 6.480070 ms/step with the additional bound, a 2.54% increase. Both sides perform 100 rebuilds in 100 measured steps. The candidate is removed from production. Its patch against `f3e7a95` is retained only as private experimental material in `debug/lmp_perf/experiments/axis_rejection.patch`, with results in `results/axis_rejection_rebuild.json` and raw runs under `/tmp/lmp-perf-r2-20261021/ab_axis_rebuild`.

Further neighbor-search work should target candidate organization or a more coherent parallel traversal. The failed experiment shows that fewer scalar operations alone do not establish a faster GPU kernel. A different traversal would need to preserve complete neighbor rows, exclusions, ghost behavior and ordering, and is a larger algorithm change than the accepted typed layout.

## Reproduction

The private drivers retain every run in its own directory and refuse to overwrite an existing result directory. The following command repeats the whole-step comparison of the original fork and the three-optimization snapshot `6218c78`. The script selects the trained archives and the fork's plugin through the existing machine profile. A different output path is required for another repetition.

```bash
env BENCH_GPUS=0 BENCH_CUDA_ARCH=120 \
    OMP_NUM_THREADS=1 DP_INTER_OP_PARALLELISM_THREADS=0 DP_INTRA_OP_PARALLELISM_THREADS=0 \
    /nas/outisli/Software/miniforge3/envs/dpmd/bin/python \
    debug/lmp_perf/benchmark.py \
    --a /tmp/lmp-perf-20261021/base \
    --b /tmp/lmp-perf-20261021/neighbor \
    --output /tmp/lmp-perf-repeat-carbon \
    --gpu GPU-3c221df8-636b-4a96-e8b1-fa90b4145d04 \
    --points nano:512,nano:10296,nano:100800 --repeats 4
```

`--flavor bcn` selects multielement ordinary NVT; `--flavor spin` selects the Newtonian-spin driver. `--plugin-a` and `--plugin-b` select plugins built for different LAMMPS versions when comparing the public release and the fork. Selecting an executable alone is insufficient: the script puts the corresponding build directory first in the LAMMPS library search path.

For the typed-bin comparison, select `/tmp/lmp-perf-r2-20261021/base` and `typed` as the two builds. `--steps 2000 --points nano:100800 --flavor bcn` reproduces the extended trajectory length. `benchmark_rebuild.py` accepts the same `--a`, `--b`, `--output`, `--gpu` and `--repeats` options and uses the frozen 100,800-atom BCN input with a rebuild on every step.

The complete neighbor-row fixture is retained as `debug/lmp_perf/neighbor_rows.cpp`. Its recorded compile command uses the existing CUDA/Kokkos build and MPI installation:

```bash
/nas/outisli/Software/miniforge3/envs/dpmd/bin/python - <<'PY'
import json
import subprocess
from pathlib import Path
command = json.loads(Path('debug/lmp_perf/results/compile_neighbor_rows.json').read_text())
subprocess.run(command, check=True)
PY
/nas/outisli/Software/miniforge3/envs/dpmd/bin/python \
    debug/lmp_perf/check_neighbor_rows.py \
    --probe /tmp/lmp_neighbor_rows \
    --library /tmp/lmp-perf-r2-20261021/typed \
    --output /tmp/lmp-perf-repeat-rows \
    --reference /tmp/lmp-perf-r2-20261021/frozen_base --gpu 1
```

Omit `--reference` when recording a baseline snapshot. The recorded compiler arguments are machine-specific and must match the selected LAMMPS build; snapshots require matching atom-ID widths. Compact timing and verification records are stored beside the existing results under `debug/lmp_perf/results/typed_bins_*.json`.

The neighbor decision check is independent of the model files:

```bash
/nas/outisli/Software/miniforge3/envs/dpmd/bin/python \
    debug/lmp_perf/check_neighbor.py \
    --lmp /tmp/lmp-perf-20261021/neighbor/lmp \
    --library /tmp/lmp-perf-20261021/neighbor \
    --output /tmp/lmp-perf-repeat-neighbor \
    --reference /tmp/lmp-perf-20261021/neighbor_base --gpu 1
```

For the empty-rank case, use `--ranks 2 --cases empty_rank` and `neighbor_mpi_base` as the reference. `check_integrator.py` runs the integration cases and includes a separate host/empty-group observation case. The original eleven-case references are in `controlled_base`; the observation case uses `observed_base`. Compact verification results are preserved in `debug/lmp_perf/results/verification.json`.

## Local delivery

The four accepted code commits are `fee7c14`, `7c28131`, `6218c78` and `f3e7a95` in the LAMMPS fork. The existing installation at `/nas/outisli/Software/lammps/lammps-outisli` is rebuilt and installed from this code. Direct ordinary and Newtonian-spin runs through `bin/lmp`, with `LD_LIBRARY_PATH` unset and only the fork's `LAMMPS_PLUGIN_PATH`, reproduce all 52 reference thermodynamic rows and particle states byte for byte. The installed executable reports `f3e7a95`. The installation checks are `/tmp/lmp-perf-r2-20261021/installed_smoke` and `installed_help.txt`. The rejected coordinate-bound experiment is absent from the installed source.

The final independent review uses `gpt-6.1-sol` with `high` reasoning effort for each optimization and for the combined performance report. It verifies the saved medians against the raw logs and distinguishes the same-version fork comparison from the different-version public-runtime comparison. Source and private-driver lint checks pass. Full documentation checks remain limited by the absent `doxygen`; non-CUDA accelerator backends are not executed. Documentation and private measurement material are maintained separately from the LAMMPS code.

The compact-vector implementation is installed in the existing local `dpmd` environment and H20 `dpmd-new` environment, with the C++ interface in each machine's `Software/deepmd-kit_cpp` prefix. Both public LAMMPS `patch_4Jul2026` and the OutisLi fork use their matching Kokkos plugin. On the workstation, the public runtime uses the C++ prefix's `lib/deepmd_lmp`, and the fork uses its own `lib/deepmd_lmp`. On H20, the fork uses the C++ prefix's plugin, and the public runtime uses its own plugin directory. The H20 public plugin build discovers headers and libraries from the permanent C++ prefix and uses the GCC 11 launcher required by its LAMMPS build.

Installation uses atomic file replacement with retained backups and SHA-256 checks. Per device, 113 related tests, four compact-DPA1 tests, nine guarded width-transition cases and 216 exact C API evaluations pass; one DPA1 case is skipped for its optional build configuration. Each installed Python environment passes 23 focused tests and verifies its existing editable source mapping and the formal `site-packages` operator. C99 and C++ SDK checks pass. Public ordinary, fork ordinary and fork evolving-spin ZBL trajectories reproduce the staged outputs bit for bit, with actual library and plugin paths recorded. The fork executable and `liblammps` match the validated code after normalizing their install-time library search paths, so the core binaries are retained and the matching DeePMD plugins are replaced. Backups remain under each study root's `final_vector/formal_backup_*`; fingerprints and results are retained with the [verification records](../../debug/lmp_perf/graph_geometry/results/final_vector_20261022/).

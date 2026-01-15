# Indexed canonical geometry

Production uses the vector-only canonical ABI with automatic source-index width. The indexed geometry and selector studied below are withdrawn. The final build and verification records live in each study root's `final_vector/`; the remaining instructions reproduce historical source snapshots and do not apply to the current production tree. The historical 120-case results must not be reported as final-ABI validation.

The production implementation uses one frozen archive and one C-compatible geometry view. Vector geometry stores a model-node index and three displacement components per edge; indexed geometry stores one raw atom-image index and reconstructs the displacement from packed coordinates. Both layouts rebuild cutoff membership and canonical distance order every step. LAMMPS' neighbor-rebuild flag caches candidate counts and atom maps, not the compact graph.

All supported DPA4C channel widths and degrees instantiate both layouts. The pair style measures graph construction plus model evaluation on the current device and workload. It retains indexed geometry only when the median paired saving exceeds 1% and three median absolute deviations. Calibration has 46 extra evaluations, followed by the selected production evaluation and one scatter. A substantial workload change or a new pair configuration triggers another calibration. Source CSR, long-range calls and the counted fill require vector geometry. The two replay pools share one model and its constants but consume more memory than a fixed-layout run.

Canonical node and atom-image indices use 32-bit storage when both index domains fit in signed 32 bits, and 64-bit storage otherwise. Edge positions, capacities, row pointers, source order and extents remain signed 64-bit at either width. Thus more than 2^32 edges alone does not widen node indices. The C geometry view carries `index_width`; the exported model accepts zero-copy byte views with a shared word width of four or eight bytes. One archive and one set of weights handle both widths. Packed coordinates retain their low-32-bit wrapping at either width.

LAMMPS bounds its rank-local atom and ghost slots below 2^29, so its producer uses 32-bit node indices and one 16-byte packed record per atom. Global atom tags do not index the local graph. A vector edge carries a four-byte source plus 12 bytes of displacement; an indexed edge carries one four-byte atom-image index. External canonical callers with larger node domains select 64-bit storage, giving 20-byte vector edges, eight-byte indexed edges and 32-byte packed records. Large-index fixtures verify arithmetic and representation without allocating a graph of billions of edges.

DPA4C permits normal floating-point contraction. Each geometry kernel derives its row partition from its own occupancy. Geometry decoding retains the same integer wrapping and vector quantization, while contraction and reduction order can produce normal fp32 differences. Validation compares physical observables with recorded error bounds, checks gradients and the fp64 reference, and retains exact hashes as additional evidence. Translation and replay checks retain their exact contracts where the arithmetic is unchanged. The artificial dense-row virial test uses a per-frame tensor-norm bound, because nearly cancelling off-diagonal components do not provide a useful relative-error scale.

## Matrix and artifacts

The expanded matrix has 120 configurations per device:

| Axis                | Values                                 |
| ------------------- | -------------------------------------- |
| Model grade         | Nano, Mini, Neo, Air, Plus             |
| Potential           | Ordinary energy, native spin           |
| Short-range bridge  | Plain, active ZBL window               |
| Elements            | Carbon, BCN                            |
| Atom count          | 512, 10,296, 100,800                   |
| Geometry comparison | Fixed vector, fixed indexed, automatic |

Ordinary models retain trained checkpoint weights. Spin models use deterministic activated random parameters; there is no trained spin checkpoint in this experiment. ZBL variants retain the corresponding learned weights and add a 0.5–0.8 Å switching window. Their datasets place selected pairs at 0.65 Å, so the bridge is exercised. These datasets differ from the plain datasets; their absolute timing difference is not a measurement of ZBL overhead alone. Input user datasets are not overwritten.

The local study is under `/tmp/dpa4c-matrix-20261022`; the H20 study is under `/aisi-vepfs/outisli/gpu_geometry_20261022`. The `auto_width/` directory contains the candidate, `natural32/` the fixed-32 control with the same wide-prefix scan and floating-point kernel implementation, and `contract_baseline/` the original vector reference. The candidate operator has an additional host-side byte-view decoder; its binary differs from the control. The `int64/` and `int64_native/` directories preserve the earlier fixed-64 measurements. Each device has its own 20 frozen archives. `reexport_matrix_models.py` reads the serialized model from the 32-bit archive, retains the complete learned payload and fitting-route table, and exports a separate automatic-width archive. It verifies both after export and refuses to overwrite an archive. Editing metadata alone cannot change an AOTInductor input dtype. Archives are not copied between architectures. Ordinary MD uses public LAMMPS `patch_4Jul2026`; spin MD uses the OutisLi fork with Newtonian magnetic-moment integration.

The earlier Nano-only experiment remains in `/tmp/dpa4c-graph-20261021`, with compact records in `results/large_md_fresh.json` and related files. Its fixed-moment spin measurements and L2-based selection rule describe that earlier experiment. They do not characterize the current all-profile calibration policy.

## Build and preparation

Use the existing environment and configured builds. Rebuild every consumer of the unpublished geometry signature together. The public and fork runtimes have different Kokkos versions and require their matching plugins. The host compiler must also match: on H20, the LAMMPS builds use GCC 11, while the Python environment places GCC 13 first on `PATH`. `nvcc_wrapper` must use `/usr/bin/g++` for these plugins; otherwise LAMMPS' conditional `fmt`/`std::format` interface can differ at link time.

The local build is:

```bash
export GRAPH_STUDY=/tmp/dpa4c-matrix-20261022
export GRAPH_ROOT="$GRAPH_STUDY/auto_width"
export BENCH_CUDA_ARCH=120
source debug/cuda_bench/machine.sh
TORCH_CUDA_ARCH_LIST=12.0 cmake --build source/build \
    --target deepmd_op_pt deepmd_c deepmd_backend_pt deepmd_backend_ptexpt deepmd_lmp -j4
```

`stage_matrix_runtime.py` takes a CMake-installed prefix through `--deepmd-prefix` and stages its libraries and SDK headers into `integrated/` and `sdk/`. It copies the public runtime from the saved `baseline/public_runtime` and the fork runtime from its current build. All six private plugins must already exist; the script changes their RPATH to `$ORIGIN/../integrated`. Installed core libraries resolve sibling dependencies inside the snapshot. Copying raw build-tree libraries is insufficient because a parent's `DT_RPATH` can override `LD_LIBRARY_PATH` and load an operator from a live build. Staging overwrites generated binaries, so run it only when no process uses that staged directory. `staged_runtime.json` records the source paths and SHA-256 values.

`build_vector_plugin.py` copies the current pair sources and changes only the private geometry choice. It adds no public switch and does not edit production source. Supply the build matching the runtime:

```bash
"$BENCH_PYTHON" debug/lmp_perf/graph_geometry/build_vector_plugin.py \
    --artifacts "$GRAPH_ROOT" --runtime public --geometry vector --build source/build
"$BENCH_PYTHON" debug/lmp_perf/graph_geometry/build_vector_plugin.py \
    --artifacts "$GRAPH_ROOT" --runtime public --geometry indexed --build source/build
"$BENCH_PYTHON" debug/lmp_perf/graph_geometry/build_vector_plugin.py \
    --artifacts "$GRAPH_ROOT" --runtime fork --geometry vector \
    --build /nas/outisli/Software/lammps/lammps-outisli/build_deepmd
"$BENCH_PYTHON" debug/lmp_perf/graph_geometry/build_vector_plugin.py \
    --artifacts "$GRAPH_ROOT" --runtime fork --geometry indexed \
    --build /nas/outisli/Software/lammps/lammps-outisli/build_deepmd
```

On H20, `source/build` matches the fork and `source/build_public_plugin` matches the public runtime. Set `NVCC_WRAPPER_DEFAULT_COMPILER=/usr/bin/g++` before building either reference. The public production-plugin build persists that compiler in its CMake launcher. Before loading the libraries, put the existing `dpmd-new/lib` on `LD_LIBRARY_PATH`, then source `machine.sh`, which places the newer CUDA libraries bundled with PyTorch ahead of the toolkit libraries.

`stage_plugin.py` stages a plugin-only CMake installation, including its discovery symlink. It rebases custom symlink commands that embed the configured prefix and accepts `--backend-libraries` for the permanent DeePMD library directory. `install_staged.py` then installs by atomic replacement, with a backup and fingerprint check for every replaced file. Existing processes retain their mapped library inodes. The H20 public plugin build uses the permanent C++ prefix for `DEEPMD_C_ROOT` and retains the GCC 11 compiler launcher. When changing the SDK root, clear `DEEPMD_INCLUDE_C_DIR` and `deepmd_c` from the CMake cache so discovery follows the requested prefix; verify the generated include and link commands.

Use `check_deployed_runtime.py` for both the public and fork runtime after staging. It records `LD_DEBUG=libs` output and verifies every loaded DeePMD library against the requested prefix, including duplicate loads. The same check accepts `--reference` to compare a formal installation with a validated staged trajectory.

For a new artifact directory, `prepare_matrix_systems.py` writes the synthetic datasets and their manifest, and `freeze_matrix_models.py` writes the model matrix. The freezer refuses to overwrite an archive. It restores the known empty coupling-table shape in the private deserialized model and uses the normal compression/export path. The model JSON records its grade, spin/ZBL status, training status, schema and hash.

Never edit a shared CUDA header during compilation: a compiler can finish an object whose input predates the edit, yet leave it newer than the header. Final validation must use a completed build from one fixed source snapshot.

## Correctness

The production suite covers empty rows, signed coordinate wrapping, periodic images, nonidentity center maps, translation, active spin, radial modes and ZBL windows. The 512-node C32/L2 cases reproduce occupancy-derived partition and spin contraction differences. C64/L3 cases cover the geometric quadratic value and derivative evaluation points.

```bash
CUDA_VISIBLE_DEVICES="${GRAPH_GPU:?Set an authorized idle GPU UUID}" \
    OMP_NUM_THREADS=1 DP_INTER_OP_PARALLELISM_THREADS=0 \
    DP_INTRA_OP_PARALLELISM_THREADS=0 \
    "$BENCH_PYTHON" debug/lmp_perf/graph_geometry/run_memcheck_tests.py \
    source/tests/pt_expt/model/test_dpa4c_graph_lower.py \
    source/tests/pt_expt/utils/test_canonical_graph.py \
    source/tests/pt_expt/utils/test_graph_pt2_metadata.py \
    source/tests/pt_expt/model/test_dpa4c_zbl_bridging.py -q
"$BENCH_PYTHON" debug/lmp_perf/graph_geometry/benchmark_matrix.py \
    --artifacts "$GRAPH_ROOT" --output "$GRAPH_ROOT/verify_repeat" \
    --gpu "$GRAPH_GPU" --cpu 8 --verify
```

The real LAMMPS matrix disables sorting and compares 17-digit dumps before and after an evolving trajectory. The dumps include coordinates, velocities, forces, per-atom energies and, for spin, moments and magnetic forces. It also compares every thermodynamic row, including total energy and all six pressure components. Every number must be finite; spin trajectories must change their moments and have nonzero magnetic force. `benchmark_matrix.py` records maximum absolute, RMS and relative errors for each observable, applies its declared absolute and relative bounds, and reports all failing cases. Exact dump and thermodynamic comparisons remain recorded but do not replace the numerical criteria. The optional `--cases` list partitions correctness work across idle devices; each batch retains its own snapshots, and a merged matrix records the source of every complete case.

`check_runtime_lifecycle.py` checks single-process or two-rank ordinary/spin ZBL runs, repeated run commands, a temporary nine-component atomic-virial request, element-coefficient reassignment and repeated pair settings. It verifies three calibrations for the automatic path and none for either fixed reference. Repeating settings with the same archive exercises selector reset; it does not assert that the underlying evaluator reloads a different model filename.

LAMMPS and MPI subprocesses receive `stdin=DEVNULL`; their input files own the command stream. This prevents a launcher from consuming subsequent commands in an SSH shell script.

The H20 single-node MPI checks set `OMPI_MCA_btl_tcp_if_include=lo`. A separate CPU-only broadcast/allreduce probe establishes that the default TCP interface selection stalls on that node and loopback completes. This is a launcher setting for that environment, with no change to the simulation's MPI implementation.

The C API check uses one instance per model, captures both layouts, changes inputs in place, grows the packed allocation, requests source CSR and atom virial, returns to indexed geometry, and changes charge state. Its halo variant adds a fifth model node to four owned nodes. Outputs are reset to NaN before each call and compared numerically against the corresponding vector/fresh-instance result. Repeated source-CSR replay retains its exact comparison. The width-switch fixture reuses the same backing addresses and logical extents for 32→64→32 transitions; this checks that the replay key includes the index width. A guarded single-artifact check covers both layouts, empty and one-edge graphs, and invalid byte widths and dtypes. Canonical export retains symbolic zero/one extents through tracing and compilation with `backed_size_oblivious`; the scoped configuration is restored afterward.

The independent [builder fixture](builder_regression/README.md) checks 21 stages over K8, K12, K16 and the counted fallback, eight prefix cases across 2^31, 2^32, 2^40 and the signed-64-bit boundary, and five source-CSR sorts with edge positions above 2^40. It exercises indexed source-buffer growth, preserved vector-buffer watermarks, shrinking graphs, source CSR and all guard slots. A separate CUDA reader fixture decodes a packed model-node index above 2^40 and wrapped coordinates. Each fixture records its production-header fingerprint.

`capture_operator_calls.py` derives cases directly from the production test parameter list and records both geometry layouts per case, generic C8/C64 backward and complete source-CSR atomic virial. Its default 64-bit capture contains 27 calls; `--index-dtype uint32` checks the retained operator path separately. `replay_operator_calls.py` checks their recorded outputs under CUDA memcheck without repeating model construction under instrumentation:

```bash
CUDA_VISIBLE_DEVICES="$GRAPH_GPU" "$BENCH_PYTHON" \
    debug/lmp_perf/graph_geometry/capture_operator_calls.py \
    --output "$GRAPH_ROOT/operator_calls_repeat.pt"
CUDA_VISIBLE_DEVICES="$GRAPH_GPU" compute-sanitizer --tool memcheck \
    --launch-timeout 120 --kernel-name kns=deepmd_dpa4c \
    --target-processes application-only --error-exitcode 86 \
    "$BENCH_PYTHON" debug/lmp_perf/graph_geometry/replay_operator_calls.py \
    --calls "$GRAPH_ROOT/operator_calls_repeat.pt"
```

The automatic-width audit also replays the 27 captured calls per width through the final byte-word boundary: canonical source, packed and center tensors become zero-copy uint8 views, and source-order positions use int64. All 54 outputs must equal the retained reference, followed by a zero-error direct-operator memcheck. The evidence archive retains `replay_auto_byte_calls.py`, both input-capture hashes and the complete logs. This check covers the byte decoder without requiring model construction under instrumentation.

Complete C API/AOTI instrumentation is recorded separately. On H20, both the automatic-width and retained fixed-64 runtime report one `cuCtxGetLimit` invalid-argument diagnostic while completing their numerical comparisons; a minimal PyTorch CUDA probe and the direct-operator replay report zero errors. The workstation C API instrumentation is interrupted after prolonged model loading. These runs do not count as zero-error C API sanitizer passes, and no diagnostic is suppressed.

Both scripts accept `--operator-library /absolute/path/libdeepmd_op_pt.so` for an isolated operator build. Capture and replay must use the same library. A build with different floating-point evaluation rules needs a fresh capture; its agreement with the reference model and between geometry layouts is checked before the recorded outputs become the memory-check reference.

## Performance and memory

Run timings only after validation, profiling and builds finish on that machine. Each configuration uses a pilot to choose approximately two seconds of measured MD per sample. The default comparison uses three cyclically rotated repetitions of vector/indexed/automatic. The width comparison uses separately compiled fixed-32 and automatic-width archives with identical model payloads and fitting routes, unchanged floating-point kernel sources, and the same dataset, step count and timestep. Both paths use their normal runtime settings during timing; strict AOTI input guards are enabled separately for candidate correctness checks. Three repetitions balance a three-path comparison; four repetitions balance a two-path fixed-indexed comparison. The pilot and startup calibration are excluded from step time. Every paired repetition must have equal neighbor-rebuild counts and zero dangerous builds. Timing retains normal atom sorting; correctness uses its separate deterministic protocol.

```bash
"$BENCH_PYTHON" debug/lmp_perf/graph_geometry/benchmark_matrix.py \
    --artifacts "$GRAPH_ROOT" --output "$GRAPH_ROOT/timing_repeat" \
    --gpu "$GRAPH_GPU" --cpu 8
"$BENCH_PYTHON" debug/lmp_perf/graph_geometry/measure_memory.py \
    --artifacts "$GRAPH_ROOT" --output "$GRAPH_ROOT/memory_repeat" \
    --gpu "$GRAPH_GPU"
```

Measure automatic index width and geometry against the matched fixed-32 implementation and the original vector implementation:

```bash
"$BENCH_PYTHON" debug/lmp_perf/graph_geometry/benchmark_matrix.py \
    --artifacts "$GRAPH_ROOT" \
    --baseline-artifacts "$GRAPH_STUDY/natural32" --baseline-layout auto \
    --original-artifacts "$GRAPH_STUDY/contract_baseline" \
    --output "$GRAPH_ROOT/timing_width_auto" --gpu "$GRAPH_GPU" --cpu 8 \
    --sides before_contract,auto,original_vector --repeats 3
```

This comparison verifies all three runtime snapshots before and after the run. The width cost is `100 * (auto / before_contract - 1)`; the net time reduction is `100 * (1 - auto / original_vector)`. These have different references and must be reported separately. Fixed-indexed width measurements use `--baseline-layout indexed --sides before_contract,indexed --repeats 4` without `--original-artifacts`. `report_width_matrix.py` retains source batches, recomputes medians, checks cyclic launch order and requires the same validated snapshots and 120 unique configurations. Interrupted batches retain their incomplete status; only complete cases enter the aggregate.

The strict-contraction matrix and rejected shared-decoder experiments retain separate archives under `results/matrix_20261022/`. Their timings and memory records describe those particular binaries. `merge_matrix.py` and `summarize_matrix.py` reproduce the historical four-path strict-contraction report; they do not supply evidence for the automatic-width implementation.

The timestep is at most `2e-6 ps`, and the complete warm-up plus measured trajectory is at most `0.004 ps`, to keep the random spin models in the validated short-time regime. Hardware and grade can choose different step counts and timesteps; comparisons across devices are not identical physical trajectories. Within each configuration, all three layouts use identical inputs and integration settings.

The memory driver checks that its GPU has no compute process before starting, records the driver baseline, and reports the sampled peak and increment above baseline. It pauses 100 ms between queries, in addition to query overhead, so these are sampled peaks rather than allocator maxima. The automatic path must emit its calibration record, proving both layouts were exercised. These runs do not measure throughput.

Each run retains its input, command, library paths and complete screen output. JSON records contain sample times, neighbor counts and automatic choices. A snapshot verifies the staged binaries, all model archives and metadata, and all datasets against their manifests before and after a matrix or memory run. Preserve failed startup and numerical runs separately from accepted timing results. `install_staged.py` delivers validated staged files by atomic replacement, retaining backups and checking every installed fingerprint; this avoids truncating libraries mapped by existing simulations. The SDK, C/C++ libraries, plugins and automatic-width archives form one contract; earlier fixed-width canonical archives require re-export. The metadata declares `canonical_index_dtype="auto"` and `canonical_index_storage="byte_words"`. Changing metadata alone does not change a compiled artifact.

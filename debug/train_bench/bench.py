#!/usr/bin/env python
# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, T201, ANN001, B905
"""
Steady-state training-step benchmark for the SeZM / DPA4 training path.

The harness builds a real trainer from a training script, caches a fixed set of
real batches on the device, and then replays them through
``wrapper -> loss.backward()`` so that the measured wall time reflects the model
graph alone: no sampler jitter, no host-side data staging, and no recompilation
once the warm-up has covered every cached shape.

Examples
--------
Steady-state timing::

    python debug/train_bench.py input.json --steps 60 --warmup 30

Operator breakdown of a single step::

    python debug/train_bench.py input.json --profile --profile-rows 40
"""

from __future__ import (
    annotations,
)

import argparse
import json
import logging
import statistics
from collections import (
    defaultdict,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import torch

from deepmd.pt.utils.env import (
    DEVICE,
)
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)
from deepmd.utils.data_system import (
    process_systems,
)
from deepmd.utils.path import (
    DPPath,
)


def override_inductor_options(spec: str) -> dict:
    """
    Layer experimental Inductor options over the production set.

    The shipped option set in ``build_inductor_compile_options`` is deliberately
    conservative: it disables the Inductor and Triton features that have
    misbehaved on the combination of data-dependent edge counts and a
    second-order autograd graph. Those choices were made for the inference
    graph, so measuring what they cost the training graph needs a way to vary
    them without touching the default.

    Parameters
    ----------
    spec : str
        JSON object of option overrides, e.g. ``'{"max_fusion_size": 64}'``.

    Returns
    -------
    dict
        The parsed overrides.
    """
    from deepmd.pt.model.model.sezm_model import (
        SeZMModel,
    )

    overrides = json.loads(spec)
    original = SeZMModel._inductor_compile_options

    def patched(self, *, inference: bool = False) -> dict:
        options = original(self, inference=inference)
        options.update(overrides)
        return options

    SeZMModel._inductor_compile_options = patched
    return overrides


def drop_native_bmm_override() -> None:
    """
    Remove PyTorch's Triton override of ``aten::bmm``.

    PyTorch 2.13 registers an override that replaces ``bmm`` with a Triton
    kernel for the true outer product (``K == 1``). None of this model's
    batched matmuls have that shape, but the predicate guarding the override
    calls ``torch._C._is_cow_tensor``, which Dynamo cannot trace: the graph
    breaks at every ``bmm`` it reaches before the shape condition is even
    evaluated. Dropping the override restores a single compiled backward.
    """
    import torch._native.registry as native_registry

    native_registry.deregister_op_overrides(disable_op_symbols="bmm")


def build_trainer(
    input_file: str,
    backend: str = "pt",
    batch_size: str | None = None,
) -> Any:
    """
    Build a trainer from a training script without running it.

    Parameters
    ----------
    input_file : str
        Path to the DeePMD-kit training script.
    backend : str
        ``"pt"`` or ``"pt_expt"``. Both expose the same training dispatch and
        the same accelerated operators, so a variant sweep is comparable
        across them.
    batch_size : str or None
        Training batch-size override, such as ``"mix:3200"``. ``None`` keeps
        the value from the training script.

    Returns
    -------
    Any
        A fully initialized trainer of the selected backend.
    """
    with Path(input_file).open() as fp:
        config = json.load(fp)
    if batch_size is not None:
        config.setdefault("training", {}).setdefault("training_data", {})[
            "batch_size"
        ] = batch_size
    if backend == "pt_expt":
        # The two backends spell the compile switch differently: pt reads
        # ``model.use_compile``, pt_expt ``training.enable_compile``. A
        # cross-backend comparison must put both in the same regime, so the
        # pt spelling of the script is carried over.
        config.setdefault("training", {})["enable_compile"] = bool(
            config.get("model", {}).get("use_compile", False)
        )
    config = update_deepmd_input(config, warning=False, dump=None)
    config = normalize(config)
    if backend == "pt_expt":
        from deepmd.pt_expt.entrypoints.main import (
            get_trainer,
        )
    else:
        from deepmd.pt.entrypoints.main import (
            get_trainer,
        )
    return get_trainer(config)


def collect_batches(
    trainer: Any, count: int, cache: str | None = None
) -> list[tuple[dict, dict]]:
    """
    Draw and cache a fixed number of training batches.

    Opening the LMDB reader and drawing a batch dominates the start-up of a
    short benchmark, and a sweep that varies only a compile flag replays the
    identical batch every time. Persisting the drawn batches makes the runs both
    faster and exactly comparable.

    Parameters
    ----------
    trainer : Any
        Initialized trainer.
    count : int
        Number of distinct batches to cache.
    cache : str or None
        Path of a file holding previously drawn batches. Read when it exists,
        written otherwise.

    Returns
    -------
    list[tuple[dict, dict]]
        Pairs of model inputs and labels, resident on the compute device.
    """
    if cache is not None and Path(cache).exists():
        loaded = torch.load(cache, map_location=DEVICE, weights_only=False)
        if len(loaded) >= count:
            return loaded[:count]
    batches = []
    for _ in range(count):
        # The pt trainer returns a trailing log entry the pt_expt one does not.
        drawn = trainer.get_data(is_train=True)
        batches.append((drawn[0], drawn[1]))
    if cache is not None:
        torch.save(batches, cache)
    return batches


_COMPILED_AUTOGRAD_COMPILER = None


def enable_compiled_autograd() -> None:
    """Route ``loss.backward()`` through Compiled Autograd.

    The force-loss backward re-executes the compiled forward's autograd graph
    node by node in eager mode (a compiled backward is opaque to a second
    differentiation), which dominates the training step. Compiled Autograd
    dynamo-compiles that replay instead. Only the outer backward is wrapped:
    the in-forward force ``autograd.grad`` contains view nodes Compiled
    Autograd rejects.
    """
    global _COMPILED_AUTOGRAD_COMPILER
    _COMPILED_AUTOGRAD_COMPILER = torch.compile(dynamic=True)


def run_step(trainer: Any, batch: tuple[dict, dict], lr: float, mode: str) -> None:
    """
    Execute one training step, or a prefix of one.

    Parameters
    ----------
    trainer : Any
        Initialized trainer.
    batch : tuple[dict, dict]
        Cached model inputs and labels.
    lr : float
        Learning-rate value handed to the loss for preference scheduling.
    mode : str
        ``"full"`` runs the forward, the force, and the force-loss backward, as
        the real training loop does. ``"forward"`` stops before the force-loss
        backward: the model forward and its inner ``autograd.grad`` for the
        force still run, but nothing is differentiated a second time. The
        difference between the two isolates the second backward.
    """
    input_dict, label_dict = batch
    trainer.optimizer.zero_grad(set_to_none=True)
    _, loss, _ = trainer.wrapper(
        **input_dict, cur_lr=lr, label=label_dict, task_key="Default"
    )
    if mode == "full":
        if _COMPILED_AUTOGRAD_COMPILER is not None:
            with torch._dynamo.compiled_autograd._enable(_COMPILED_AUTOGRAD_COMPILER):
                loss.backward()
        else:
            loss.backward()


def summarize_profile(
    prof: torch.profiler.profile,
    rows: int,
    by_shape: bool = False,
    steps: int = 1,
) -> str:
    """
    Aggregate device time from a profiler trace.

    A trace records each kernel twice: once as the launching host-side operator
    and once as the device-side kernel. The two views are reported separately so
    that neither total double-counts, and so that a kernel can be traced back to
    the operator (and tensor shapes) that issued it.

    Parameters
    ----------
    prof : torch.profiler.profile
        A completed profiler session.
    rows : int
        Number of leading rows to report per table.
    by_shape : bool, default=False
        Key the host-side table by operator name and input shapes.
    steps : int, default=1
        Number of training steps covered by the trace; times are divided by it.

    Returns
    -------
    str
        Two formatted tables sorted by descending device time.
    """
    cuda = torch.autograd.DeviceType.CUDA
    tables = []
    for title, want_kernel in (("device kernels", True), ("host-side ops", False)):
        device_us: dict[str, float] = defaultdict(float)
        launches: dict[str, int] = defaultdict(int)
        for evt in prof.key_averages(group_by_input_shape=by_shape and not want_kernel):
            if evt.self_device_time_total <= 0:
                continue
            if (evt.device_type == cuda) != want_kernel:
                continue
            key = evt.key
            if by_shape and not want_kernel:
                key = f"{key} {evt.input_shapes}"
            device_us[key] += evt.self_device_time_total
            launches[key] += evt.count
        total = sum(device_us.values())
        width = 96 if by_shape and not want_kernel else 68
        lines = [
            "",
            f"=== {title}: {total / 1e3 / steps:.2f} ms/step ===",
            f"{'name':<{width}}{'ms/step':>10}{'share':>8}{'calls/step':>12}",
            "-" * (width + 30),
        ]
        for name, value in sorted(device_us.items(), key=lambda kv: -kv[1])[:rows]:
            lines.append(
                f"{name[: width - 1]:<{width}}{value / 1e3 / steps:>10.3f}"
                f"{value / total * 100:>7.1f}%{launches[name] / steps:>12.0f}"
            )
        tables.append("\n".join(lines))
    return "\n".join(tables)


def kernel_family(name: str) -> str:
    """
    Classify a device kernel by the machinery that produced it.

    The step launches thousands of kernels and no single one dominates, so the
    actionable view is which *category* of work the time falls into: compiler-
    generated pointwise and reduction kernels, library GEMMs, ATen fallbacks, or
    the hand-written fused operators.

    Parameters
    ----------
    name : str
        Device kernel name as recorded by the profiler.

    Returns
    -------
    str
        Category label.
    """
    if name.startswith("triton_poi_fused"):
        return "inductor pointwise"
    if name.startswith("triton_red_fused") or name.startswith("triton_per_fused"):
        return "inductor reduction"
    if name.startswith("triton_tem_fused") or name.startswith("triton_mm"):
        return "inductor gemm"
    if "cutlass" in name or "gemm" in name.lower() or "gemv" in name.lower():
        return "library gemm"
    if name.startswith("_") and name.endswith("kernel"):
        return "sezm fused triton"
    if "indexFunc" in name or "index_elementwise" in name:
        return "aten scatter/gather"
    if "elementwise_kernel" in name or "vectorized_elementwise" in name:
        return "aten elementwise"
    if "reduce_kernel" in name or "Reduce" in name:
        return "aten reduction"
    if "Memcpy" in name or "Memset" in name:
        return "memcpy/memset"
    if name.startswith("void at::native"):
        return "aten other"
    return "other"


LAYOUT_OPS = (
    "clone",
    "copy",
    "permute",
    "transpose",
    "contiguous",
    "view",
    "slice",
    "cat",
    "unsqueeze",
    "squeeze",
    "expand",
    "select",
)

MATH_OPS = (
    "add",
    "mul",
    "div",
    "sub",
    "neg",
    "sigmoid",
    "silu",
    "tanh",
    "exp",
    "softmax",
    "sum",
    "sqrt",
    "rsqrt",
    "pow",
    "mean",
    "erf",
    "gelu",
)


def summarize_fusion_content(prof: torch.profiler.profile, steps: int) -> str:
    """
    Split compiler-generated kernel time by what the fused body actually does.

    Inductor names a generated kernel after every operation it fused, so the
    name is a faithful inventory of the body. Splitting the time by whether a
    kernel carries any arithmetic at all, and how much of its op list is pure
    data movement, separates work that is intrinsic to the model from traffic
    that only exists to satisfy a layout mismatch.

    Parameters
    ----------
    prof : torch.profiler.profile
        A completed profiler session.
    steps : int
        Number of training steps covered by the trace.

    Returns
    -------
    str
        A formatted table of the generated-kernel time split.
    """
    cuda = torch.autograd.DeviceType.CUDA
    buckets: dict[str, float] = defaultdict(float)
    launches: dict[str, int] = defaultdict(int)
    for evt in prof.key_averages():
        if evt.device_type != cuda or evt.self_device_time_total <= 0:
            continue
        if not evt.key.startswith("triton_"):
            continue
        tokens = set(evt.key.split("_"))
        layout = len(tokens & set(LAYOUT_OPS))
        math = len(tokens & set(MATH_OPS))
        if math == 0:
            bucket = "pure layout (no arithmetic)"
        elif layout > math:
            bucket = "layout-dominated"
        elif layout > 0:
            bucket = "mixed"
        else:
            bucket = "arithmetic only"
        buckets[bucket] += evt.self_device_time_total
        launches[bucket] += evt.count
    total = sum(buckets.values())
    lines = [
        "",
        f"=== generated-kernel content: {total / 1e3 / steps:.2f} ms/step ===",
        f"{'content':<30}{'ms/step':>10}{'share':>8}{'launches':>11}",
        "-" * 60,
    ]
    for name, value in sorted(buckets.items(), key=lambda kv: -kv[1]):
        lines.append(
            f"{name:<30}{value / 1e3 / steps:>10.3f}"
            f"{value / total * 100:>7.1f}%{launches[name] / steps:>11.0f}"
        )
    return "\n".join(lines)


def summarize_host_time(prof: torch.profiler.profile, steps: int, rows: int) -> str:
    """
    Report where host time goes, and how much of the step it accounts for.

    A small configuration can be host-bound: the device finishes each launch
    long before Python has issued the next one. Comparing the summed host self
    time against the device total tells which regime a shape is in, and the
    per-entry breakdown identifies the launches that are paying an unusual
    dispatch cost.

    Parameters
    ----------
    prof : torch.profiler.profile
        A completed profiler session.
    steps : int
        Number of training steps covered by the trace.
    rows : int
        Number of leading rows to report.

    Returns
    -------
    str
        A formatted table sorted by descending host self time.
    """
    cuda = torch.autograd.DeviceType.CUDA
    host_us: dict[str, float] = defaultdict(float)
    launches: dict[str, int] = defaultdict(int)
    device_total = 0.0
    for evt in prof.key_averages():
        if evt.device_type == cuda:
            if not evt.key.startswith("## Call CompiledFxGraph"):
                device_total += evt.self_device_time_total
            continue
        if evt.self_cpu_time_total <= 0:
            continue
        host_us[evt.key] += evt.self_cpu_time_total
        launches[evt.key] += evt.count
    total = sum(host_us.values())
    lines = [
        "",
        f"=== host self time: {total / 1e3 / steps:.2f} ms/step "
        f"(device {device_total / 1e3 / steps:.2f} ms/step) ===",
        f"{'name':<62}{'ms/step':>10}{'calls':>9}{'us/call':>10}",
        "-" * 92,
    ]
    for name, value in sorted(host_us.items(), key=lambda kv: -kv[1])[:rows]:
        count = launches[name]
        lines.append(
            f"{name[:61]:<62}{value / 1e3 / steps:>10.3f}"
            f"{count / steps:>9.0f}{value / count:>10.1f}"
        )
    return "\n".join(lines)


def summarize_families(prof: torch.profiler.profile, steps: int) -> str:
    """
    Aggregate device time and launch count per kernel category.

    Parameters
    ----------
    prof : torch.profiler.profile
        A completed profiler session.
    steps : int
        Number of training steps covered by the trace.

    Returns
    -------
    str
        A formatted table sorted by descending device time.
    """
    cuda = torch.autograd.DeviceType.CUDA
    device_us: dict[str, float] = defaultdict(float)
    launches: dict[str, int] = defaultdict(int)
    for evt in prof.key_averages():
        if evt.device_type != cuda or evt.self_device_time_total <= 0:
            continue
        # The compiled-graph wrappers cover every kernel inside them and would
        # double count the whole step.
        if evt.key.startswith("## Call CompiledFxGraph"):
            continue
        family = kernel_family(evt.key)
        device_us[family] += evt.self_device_time_total
        launches[family] += evt.count
    total = sum(device_us.values())
    lines = [
        "",
        f"=== kernel families: {total / 1e3 / steps:.2f} ms/step ===",
        f"{'family':<24}{'ms/step':>10}{'share':>8}{'launches':>11}{'us/launch':>11}",
        "-" * 64,
    ]
    for name, value in sorted(device_us.items(), key=lambda kv: -kv[1]):
        count = launches[name]
        lines.append(
            f"{name:<24}{value / 1e3 / steps:>10.3f}{value / total * 100:>7.1f}%"
            f"{count / steps:>11.0f}{value / count:>11.1f}"
        )
    return "\n".join(lines)


def count_kernels(prof: torch.profiler.profile) -> int:
    """
    Count device kernel launches recorded in a profiler trace.

    Parameters
    ----------
    prof : torch.profiler.profile
        A completed profiler session.

    Returns
    -------
    int
        Number of kernel events.
    """
    return sum(
        1
        for evt in prof.events()
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.device_time > 0
    )


def main() -> None:
    """Parse the command line and run the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="training script")
    parser.add_argument(
        "--backend",
        choices=("pt", "pt_expt"),
        default="pt",
        help="trainer backend; both share the training dispatch and operators",
    )
    parser.add_argument("--batches", type=int, default=1, help="distinct batches")
    parser.add_argument(
        "--batch-size",
        default=None,
        help='training batch-size override, for example "mix:3200"',
    )
    parser.add_argument(
        "--batch-cache", default="", help="path for persisting the drawn batches"
    )
    parser.add_argument("--warmup", type=int, default=30, help="warm-up steps")
    parser.add_argument("--steps", type=int, default=60, help="measured steps")
    parser.add_argument("--profile", action="store_true", help="operator breakdown")
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument("--profile-rows", type=int, default=35)
    parser.add_argument("--by-shape", action="store_true", help="group by shapes")
    parser.add_argument("--tag", default="", help="label printed with the result")
    parser.add_argument(
        "--drop-bmm-override",
        action="store_true",
        help="remove PyTorch's untraceable aten::bmm override (see docstring)",
    )
    parser.add_argument(
        "--inductor",
        default="",
        help="JSON overrides for the Inductor options, e.g. '{\"max_fusion_size\": 64}'",
    )
    parser.add_argument("--mode", default="full", choices=["full", "forward"])
    parser.add_argument(
        "--compiled-autograd",
        action="store_true",
        help="dynamo-compile the force-loss backward replay",
    )
    parser.add_argument(
        "--ncu",
        action="store_true",
        help="mark one post-warm-up step for Nsight Compute and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    # ``process_systems`` and ``DPPath`` are imported for their side effects on
    # the LMDB reader registry; reference them so linters see the dependency.
    assert process_systems is not None and DPPath is not None

    if args.drop_bmm_override:
        drop_native_bmm_override()
        print("dropped the native aten::bmm override")
    if args.inductor:
        print(f"inductor overrides: {override_inductor_options(args.inductor)}")
    if args.compiled_autograd:
        enable_compiled_autograd()
        print("compiled autograd enabled")
    trainer = build_trainer(args.input, args.backend, args.batch_size)
    # The two backends draw incompatible batches: pt pads frames to a common
    # atom count and pt_expt concatenates them behind an ``n_node`` vector,
    # which selects a different model entry (``forward_ragged``). Sharing one
    # cache would make one of them run the other's layout, so the cache is
    # keyed by backend.
    cache_key = args.batch_cache
    if cache_key and args.batch_size:
        cache_key = f"{cache_key}-{args.batch_size.replace(':', '-')}"
    batch_cache = f"{cache_key}-{args.backend}" if cache_key else None
    batches = collect_batches(trainer, args.batches, batch_cache)
    lr = float(trainer.lr_schedule.value(0))

    for i in range(args.warmup):
        run_step(trainer, batches[i % len(batches)], lr, args.mode)
    torch.cuda.synchronize()

    if args.ncu:
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        run_step(trainer, batches[0], lr, args.mode)
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        return

    # Peak VRAM is measured over the timed steps only, excluding whatever the
    # warmup and compilation phases allocated and released.
    torch.cuda.reset_peak_memory_stats()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.steps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.steps)]
    for i in range(args.steps):
        starts[i].record()
        run_step(trainer, batches[i % len(batches)], lr, args.mode)
        ends[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    times.sort()
    trimmed = times[: max(1, int(len(times) * 0.9))]
    inputs = batches[0][0]
    natoms = int(inputs["atype"].numel())
    # The frame count is part of the workload, and the two backends do not
    # batch alike: pt pads a fixed number of frames to a common atom count
    # while pt_expt concatenates however many fit an atom budget.
    n_node = inputs.get("n_node")
    nframes = (
        int(n_node.shape[0]) if n_node is not None else int(inputs["atype"].shape[0])
    )
    print(
        f"[{args.tag or 'bench'}] natoms={natoms} nframes={nframes} "
        f"batches={args.batches} steps={args.steps}"
    )
    print(
        f"  median {statistics.median(times):8.3f} ms | "
        f"p10-trimmed mean {statistics.mean(trimmed):8.3f} ms | "
        f"min {times[0]:8.3f} ms | max {times[-1]:8.3f} ms"
    )
    peak_bytes = torch.cuda.max_memory_allocated()
    print(f"  peak VRAM {peak_bytes / 2**30:.3f} GiB ({peak_bytes / 2**20:.1f} MiB)")

    if args.profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=args.by_shape,
        ) as prof:
            for i in range(args.profile_steps):
                run_step(trainer, batches[i % len(batches)], lr, args.mode)
            torch.cuda.synchronize()
        print()
        print(
            f"kernels per step: {count_kernels(prof) / args.profile_steps:.0f}"
            f"  (over {args.profile_steps} steps)"
        )
        print(summarize_families(prof, args.profile_steps))
        print(summarize_host_time(prof, args.profile_steps, 14))
        print(summarize_fusion_content(prof, args.profile_steps))
        print(
            summarize_profile(
                prof,
                args.profile_rows,
                by_shape=args.by_shape,
                steps=args.profile_steps,
            )
        )


if __name__ == "__main__":
    main()

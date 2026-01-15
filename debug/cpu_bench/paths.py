# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253
"""Executable inference paths compared by the DPA4C CPU benchmarks.

Every path exposes the same callable interface: it takes the positional
arguments of ``forward_common_lower_graph`` and returns the model output
dictionary. The differences are which code generates the arithmetic --
eager PyTorch, an Inductor-compiled graph, an AOTInductor package, or the
hand-written fused CPU operator -- and whether the descriptor is compressed.
"""

from __future__ import (
    annotations,
)

import os
from typing import (
    TYPE_CHECKING,
    Any,
)

import torch

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

from harness import (
    build_lower_inputs,
    load_model,
)


def eager(model: Any) -> Callable[..., dict]:
    """Return the eager graph lower of a model."""
    return model.forward_common_lower_graph


def inductor(model: Any, sample: Any) -> Callable[..., dict]:
    """Return the Inductor-compiled graph lower of a model.

    The trace mirrors the deployment freeze rather than the trainer's
    evaluation slot: it keeps CPU SIMD enabled, because the trainer disables
    it only to work around a scatter-vectorization assertion on the per-frame
    virial, which no deployed artifact pays.

    Parameters
    ----------
    model
        Eval-mode pt_expt model.
    sample
        :class:`~harness.LowerInputs` used to trace the graph.

    Returns
    -------
    Callable
        A callable with the ``forward_common_lower_graph`` signature.
    """
    from torch._decomp import (
        get_decompositions,
    )
    from torch.fx.experimental.proxy_tensor import (
        make_fx,
    )

    from deepmd.pt.utils.compile_compat import (
        apply_global_compile_patches,
        build_inductor_compile_options,
    )

    args = sample.args

    def lower(
        atype: torch.Tensor,
        n_node: torch.Tensor,
        n_local: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_mask: torch.Tensor,
        destination_order: torch.Tensor,
        destination_row_ptr: torch.Tensor,
        source_order: torch.Tensor,
        source_row_ptr: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return model.forward_common_lower_graph(
            atype,
            n_node,
            n_local,
            edge_index,
            edge_vec,
            edge_mask,
            destination_order,
            destination_row_ptr,
            source_order,
            source_row_ptr,
            destination_sorted=args[10],
            do_atomic_virial=args[11],
        )

    traced = make_fx(
        lower,
        tracing_mode="symbolic",
        _allow_non_fake_inputs=True,
        decomposition_table=get_decompositions([torch.ops.aten.silu_backward.default]),
    )(*args[:10])
    apply_global_compile_patches()
    options = build_inductor_compile_options(inference=True)
    options["assert_indirect_indexing"] = False
    compiled = torch.compile(traced, backend="inductor", dynamic=True, options=options)

    def run(*call_args: Any, **_: Any) -> dict:
        return compiled(*call_args[:10])

    return run


def package(path: str) -> Callable[..., dict]:
    """Return the callable of a frozen AOTInductor graph-lower package.

    Parameters
    ----------
    path
        Path of a ``.pt2`` archive frozen with ``lower_kind="graph"``.

    Returns
    -------
    Callable
        A callable with the ``forward_common_lower_graph`` signature; the
        trailing scalar flags of that signature are baked into the artifact
        and therefore ignored.
    """
    from torch._inductor import (
        aoti_load_package,
    )

    runner = aoti_load_package(path)

    def run(*call_args: Any, **_: Any) -> dict:
        return runner(*call_args[:10])

    return run


def build(
    grade: str,
    variant: str,
    atoms: int,
    stride: float = 0.002,
) -> tuple[Any, Any, Callable[..., dict]]:
    """Build a model, its lower inputs and an executable path.

    Parameters
    ----------
    grade
        Released DPA4C grade name.
    variant
        ``"eager"``, ``"inductor"``, ``"eager-compress"``,
        ``"inductor-compress"`` or ``"kernel"``.
    atoms
        Requested atom count of the diamond supercell.
    stride
        Radial table spacing in Å used by the compressed variants.

    Returns
    -------
    tuple
        The model, its :class:`~harness.LowerInputs`, and the callable path.
    """
    from harness import (
        GRADES,
    )

    compress = variant.endswith("compress") or variant == "kernel"
    if variant == "kernel":
        os.environ["DP_CPU_INFER"] = os.environ.get("DP_CPU_INFER", "2")
    model = load_model(GRADES[grade], compress=False)
    if compress:
        model.get_descriptor().enable_compression(0.0, table_stride_1=stride)
    sample = build_lower_inputs(model, atoms)
    if variant.startswith("inductor"):
        return model, sample, inductor(model, sample)
    return model, sample, eager(model)

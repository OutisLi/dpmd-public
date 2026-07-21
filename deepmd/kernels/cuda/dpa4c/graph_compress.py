# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Bindings and table construction for the compressed DPA4C mega kernel.

The CUDA operator evaluates the complete DPA4C descriptor directly from
destination-CSR edges. Its forward performs the radial spline lookup, both
packed moment reductions, factorized angular feedback, and invariant readout
inside one center-local kernel. It retains the two center moments and smooth
normalizer; the analytical backward consumes this state and writes every
edge-vector gradient in one kernel.

The table stores the composed scalar-distance map

.. math::

   r\mapsto \chi(r)\operatorname{RadialMLP}(\operatorname{RBF}(r)),

where ``RBF`` already contains the DPA4 C³ envelope. Type features and the
second envelope factor remain analytical kernel operations.
"""

from __future__ import (
    annotations,
)

import copy
import math
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.descriptor.dpa4c import (
    _build_l2_basis,
    _packed_l2_to_stf,
)

__all__ = [
    "build_radial_table",
    "dpa4c_graph_compress",
    "dpa4c_graph_compress_energy_force",
    "ef_op_available",
    "ensure_registered",
    "mega_eligible",
    "op_available",
]

_CUDA_WIDTHS = (4, 8, 16, 32, 64, 128)


def op_available() -> bool:
    """Return whether the compiled DPA4C compressed operator is loaded."""
    op = getattr(torch.ops.deepmd, "dpa4c_graph_compress", None)
    return isinstance(op, torch._ops.OpOverloadPacket)


def ef_op_available() -> bool:
    """Return whether the descriptor, fitting, and force operators are loaded."""
    return (
        op_available()
        and isinstance(
            getattr(torch.ops.deepmd, "graph_fitting", None),
            torch._ops.OpOverloadPacket,
        )
        and isinstance(
            getattr(torch.ops.deepmd, "edge_force_virial", None),
            torch._ops.OpOverloadPacket,
        )
    )


def mega_eligible(descriptor: Any) -> bool:
    """Return whether the descriptor width has a compiled CUDA specialization."""
    return int(descriptor.channels) in _CUDA_WIDTHS


def _quintic_coefficients(
    values: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    stride: float,
) -> torch.Tensor:
    """Build C²-matching quintic Hermite coefficients.

    Parameters
    ----------
    values
        Function values with shape ``(S + 1, C)``.
    first
        First derivatives with shape ``(S + 1, C)``.
    second
        Second derivatives with shape ``(S + 1, C)``.
    stride
        Uniform interval width in Å.

    Returns
    -------
    torch.Tensor
        Coefficients with shape ``(S, 6 * C)`` in channel-major polynomial
        order ``[c0, ..., c5]``.
    """
    left_value, right_value = values[:-1], values[1:]
    left_first, right_first = first[:-1], first[1:]
    left_second, right_second = second[:-1], second[1:]
    delta = right_value - left_value
    h = float(stride)
    c0 = left_value
    c1 = left_first
    c2 = 0.5 * left_second
    c3 = (
        20.0 * delta
        - (8.0 * right_first + 12.0 * left_first) * h
        - (3.0 * left_second - right_second) * h * h
    ) / (2.0 * h**3)
    c4 = (
        -30.0 * delta
        + (14.0 * right_first + 16.0 * left_first) * h
        + (3.0 * left_second - 2.0 * right_second) * h * h
    ) / (2.0 * h**4)
    c5 = (
        12.0 * delta
        - 6.0 * (right_first + left_first) * h
        + (right_second - left_second) * h * h
    ) / (2.0 * h**5)
    return torch.stack([c0, c1, c2, c3, c4, c5], dim=-1).reshape(
        values.shape[0] - 1,
        -1,
    )


def build_radial_table(
    descriptor: Any,
    stride: float = 0.002,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tabulate the composed DPA4C radial branch.

    Parameters
    ----------
    descriptor
        pt_expt DPA4C descriptor whose DPA4 radial modules define the table.
    stride
        Uniform distance spacing in Å.

    Returns
    -------
    table
        Quintic coefficients with shape ``(S, 6 * channels)``, fp32.
    info
        CPU metadata ``[stride, table_max, rcut, eps, degree_floor]``, fp64.

    Raises
    ------
    ValueError
        If ``stride`` is not positive.
    """
    if stride <= 0.0:
        raise ValueError(f"`stride` must be positive, got {stride}")
    sample_parameter = next(descriptor.radial_embedding.parameters())
    device = sample_parameter.device
    radial_basis = copy.deepcopy(descriptor.radial_basis).to(
        device=device,
        dtype=torch.float64,
    )
    radial_embedding = copy.deepcopy(descriptor.radial_embedding).to(
        device=device,
        dtype=torch.float64,
    )
    edge_envelope = copy.deepcopy(descriptor.edge_envelope).to(
        device=device,
        dtype=torch.float64,
    )
    interval_count = math.ceil(float(descriptor.rcut) / float(stride))
    table_max = interval_count * float(stride)
    distance = torch.arange(
        interval_count + 1,
        dtype=torch.float64,
        device=device,
    )
    distance = (distance * float(stride)).requires_grad_(True)

    # === Step 1. Evaluate the complete radial map at every table knot ===
    basis = radial_basis(distance[:, None])
    envelope = edge_envelope(distance[:, None])
    values = radial_embedding(basis) * envelope

    # === Step 2. Differentiate each output channel independently ===
    first_columns = []
    second_columns = []
    for channel in range(int(descriptor.channels)):
        (first_channel,) = torch.autograd.grad(
            values[:, channel].sum(),
            distance,
            create_graph=True,
            retain_graph=True,
        )
        (second_channel,) = torch.autograd.grad(
            first_channel.sum(),
            distance,
            retain_graph=True,
        )
        first_columns.append(first_channel)
        second_columns.append(second_channel)
    first = torch.stack(first_columns, dim=-1)
    second = torch.stack(second_columns, dim=-1)

    # === Step 3. Convert knot data to the runtime spline layout ===
    table = _quintic_coefficients(values, first, second, float(stride))
    table = table.detach().to(torch.float32).contiguous()
    info = torch.tensor(
        [
            float(stride),
            table_max,
            float(descriptor.rcut),
            float(descriptor._EPS),
            float(descriptor._DEGREE_NORM_FLOOR),
        ],
        dtype=torch.float64,
        device="cpu",
    )
    return table, info


def _table_lookup(
    table: torch.Tensor,
    radius: torch.Tensor,
    stride: float,
    table_max: float,
    channels: int,
) -> torch.Tensor:
    """Evaluate a uniform quintic table with a clamped high-distance tail."""
    coordinate = radius.clamp(min=0.0, max=table_max)
    index = torch.floor(coordinate / stride).to(torch.int64)
    index = index.clamp(max=table.shape[0] - 1)
    dx = coordinate - index.to(coordinate.dtype) * stride
    coeff = table[index].reshape(-1, channels, 6)
    dx = dx[:, None]
    return (
        coeff[..., 0]
        + (
            coeff[..., 1]
            + (
                coeff[..., 2]
                + (coeff[..., 3] + (coeff[..., 4] + coeff[..., 5] * dx) * dx) * dx
            )
            * dx
        )
        * dx
    )


def _c3_envelope(radius: torch.Tensor, rcut: float) -> torch.Tensor:
    """Evaluate the fixed exponent-five DPA4 C³ envelope."""
    u = ((float(rcut) - radius) / float(rcut)).clamp(0.0, 1.0)
    x = 1.0 - u
    series = 1.0 + x * (4.0 + x * (10.0 + x * (20.0 + 35.0 * x)))
    return u**4 * series


def _cpu_descriptor(
    edge_vec: torch.Tensor,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    destination_order: torch.Tensor,
    destination_row_ptr: torch.Tensor,
    atype: torch.Tensor,
    table: torch.Tensor,
    type_embedding: torch.Tensor,
    feedback_weight: torch.Tensor,
    scalar_weight: torch.Tensor,
    vector_weight: torch.Tensor,
    tensor_weight: torch.Tensor,
    canonical: bool,
    table_stride: float,
    table_max: float,
    rcut: float,
    eps: float,
    degree_floor: float,
) -> torch.Tensor:
    """Reference implementation of the compressed DPA4C descriptor."""
    del destination_order, destination_row_ptr, canonical
    compute = edge_vec.to(torch.float32)
    src, dst = edge_index[0].to(torch.long), edge_index[1].to(torch.long)
    n_node = atype.shape[0]
    channels = type_embedding.shape[1]
    ntypes = type_embedding.shape[0] - 1

    radius = torch.sqrt((compute * compute).sum(dim=-1) + float(eps) ** 2)
    direction = compute / radius[:, None]
    real_type = (atype[src] < ntypes) & (atype[dst] < ntypes)
    mask = edge_mask & real_type
    maskf = mask.to(compute.dtype)
    envelope = _c3_envelope(radius, float(rcut)) * maskf
    radial = _table_lookup(
        table.to(compute.device),
        radius,
        float(table_stride),
        float(table_max),
        channels,
    )
    edge_type = type_embedding[atype[src]] + type_embedding[atype[dst]]
    amplitude = (radial + edge_type * envelope[:, None]) * maskf[:, None]
    basis = _build_l2_basis(direction) * maskf[:, None]

    degree = torch.zeros(n_node, dtype=compute.dtype, device=compute.device)
    degree.index_add_(0, dst, envelope * envelope)
    normalizer = torch.rsqrt(degree + float(degree_floor))

    first = torch.zeros(
        n_node,
        channels,
        9,
        dtype=compute.dtype,
        device=compute.device,
    )
    first.index_add_(0, dst, amplitude[:, :, None] * basis[:, None, :])
    first = first * normalizer[:, None, None]
    first_edge = first[dst]
    normalizer_edge = normalizer[dst]
    feedback_parts = []
    for begin, end in ((0, 1), (1, 4), (4, 9)):
        projected = (first_edge[:, :, begin:end] * basis[:, None, begin:end]).sum(
            dim=-1
        )
        norm = (basis[:, begin:end] ** 2).sum(dim=-1)
        feedback_parts.append(
            projected - normalizer_edge[:, None] * amplitude * norm[:, None]
        )
    feedback_input = torch.stack(feedback_parts, dim=-1)
    feedback = torch.tanh(
        (feedback_input * feedback_weight.reshape(1, channels, 3)).sum(dim=-1)
    )
    modulated = amplitude * (1.0 + feedback)

    second = torch.zeros_like(first)
    second.index_add_(0, dst, modulated[:, :, None] * basis[:, None, :])
    second = second * normalizer[:, None, None]

    scalar = second[:, :, 0] @ scalar_weight + type_embedding[atype]
    vector = second[:, :, 1:4].permute(0, 2, 1) @ vector_weight
    vector = vector.permute(0, 2, 1)
    tensor = second[:, :, 4:9].permute(0, 2, 1) @ tensor_weight
    tensor = tensor.permute(0, 2, 1)
    tensor_matrix = _packed_l2_to_stf(tensor)
    tensor_vector = (tensor_matrix @ vector.unsqueeze(-1)).squeeze(-1)
    tensor_squared = tensor_matrix @ tensor_matrix
    invariants = torch.stack(
        [
            scalar,
            (vector * vector).sum(dim=-1),
            (tensor_matrix * tensor_matrix).sum(dim=(-2, -1)),
            (vector * tensor_vector).sum(dim=-1),
            torch.linalg.diagonal(
                tensor_squared @ tensor_matrix,
                dim1=-2,
                dim2=-1,
            ).sum(dim=-1),
            (tensor_vector * tensor_vector).sum(dim=-1),
        ],
        dim=-1,
    )
    return invariants.reshape(n_node, 6 * channels)


def _cpu_forward(*args: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU custom-op implementation returning descriptor and opaque state."""
    descriptor = _cpu_descriptor(*args)
    channels = descriptor.shape[1]
    state = torch.zeros(
        descriptor.shape[0],
        19,
        channels,
        dtype=descriptor.dtype,
        device=descriptor.device,
    )
    return descriptor, state


def _forward_fake(
    edge_vec: torch.Tensor,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    destination_order: torch.Tensor,
    destination_row_ptr: torch.Tensor,
    atype: torch.Tensor,
    table: torch.Tensor,
    type_embedding: torch.Tensor,
    feedback_weight: torch.Tensor,
    scalar_weight: torch.Tensor,
    vector_weight: torch.Tensor,
    tensor_weight: torch.Tensor,
    canonical: bool,
    table_stride: float,
    table_max: float,
    rcut: float,
    eps: float,
    degree_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del (
        edge_index,
        edge_mask,
        destination_order,
        destination_row_ptr,
        table,
        feedback_weight,
        scalar_weight,
        vector_weight,
        tensor_weight,
        canonical,
        table_stride,
        table_max,
        rcut,
        eps,
        degree_floor,
    )
    descriptor = torch.empty(
        atype.shape[0],
        6 * type_embedding.shape[1],
        dtype=torch.float32,
        device=edge_vec.device,
    )
    state = torch.empty(
        atype.shape[0],
        19,
        type_embedding.shape[1],
        dtype=torch.float32,
        device=edge_vec.device,
    )
    return descriptor, state


def _backward_fake(
    descriptor_gradient: torch.Tensor,
    state: torch.Tensor,
    edge_vec: torch.Tensor,
    *args: Any,
) -> torch.Tensor:
    del descriptor_gradient, state, args
    return torch.empty_like(edge_vec)


def _cpu_backward(
    descriptor_gradient: torch.Tensor,
    state: torch.Tensor,
    edge_vec: torch.Tensor,
    *args: Any,
) -> torch.Tensor:
    del state
    if edge_vec.shape[0] == 0:
        return torch.zeros_like(edge_vec)
    value = edge_vec.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        descriptor = _cpu_descriptor(value, *args)
        (gradient,) = torch.autograd.grad(
            (descriptor * descriptor_gradient.to(descriptor.dtype)).sum(),
            value,
        )
    return gradient.to(edge_vec.dtype)


def _setup_context(ctx: Any, inputs: tuple, output: tuple) -> None:
    ctx.save_for_backward(output[1], *inputs[:12])
    ctx.scalars = inputs[12:]
    ctx.mark_non_differentiable(output[1])
    ctx.set_materialize_grads(False)


def _backward(
    ctx: Any,
    descriptor_gradient: torch.Tensor,
    state_gradient: torch.Tensor | None,
) -> tuple:
    del state_gradient
    tensors = ctx.saved_tensors
    scalars = ctx.scalars
    edge_gradient = torch.ops.deepmd.dpa4c_graph_compress_backward(
        descriptor_gradient,
        tensors[0],
        tensors[1],
        *tensors[2:],
        *scalars,
    )
    return (edge_gradient,) + (None,) * 17


_cpu_library: torch.library.Library | None = None


def ensure_registered() -> None:
    """Register fake, CPU, and autograd implementations once."""
    global _cpu_library
    if _cpu_library is not None or not op_available():
        return
    torch.library.register_fake("deepmd::dpa4c_graph_compress")(_forward_fake)
    torch.library.register_fake("deepmd::dpa4c_graph_compress_backward")(_backward_fake)
    torch.library.register_autograd(
        "deepmd::dpa4c_graph_compress",
        _backward,
        setup_context=_setup_context,
    )
    _cpu_library = torch.library.Library("deepmd", "IMPL")
    _cpu_library.impl("dpa4c_graph_compress", _cpu_forward, "CPU")
    _cpu_library.impl(
        "dpa4c_graph_compress_backward",
        _cpu_backward,
        "CPU",
    )


def dpa4c_graph_compress(
    descriptor: Any,
    graph: Any,
    atype: torch.Tensor,
    type_embedding: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the compressed DPA4C graph descriptor.

    Parameters
    ----------
    descriptor
        Compressed pt_expt DPA4C descriptor.
    graph
        NeighborGraph with destination CSR topology.
    atype
        Flat node types with shape ``(N,)``.
    type_embedding
        Complete DPA4 type table with shape
        ``(ntypes + 1, channels)``.

    Returns
    -------
    torch.Tensor
        Invariant descriptor with shape ``(N, 6 * channels)``, fp32.
    """
    ensure_registered()
    if graph.destination_order is None or graph.destination_row_ptr is None:
        raise ValueError("DPA4C compressed CUDA requires destination CSR topology")
    channels = int(descriptor.channels)
    if channels not in _CUDA_WIDTHS:
        raise ValueError(
            f"DPA4C compressed CUDA supports channels {_CUDA_WIDTHS}, got {channels}"
        )
    info = descriptor.compress_info
    from torch.fx.experimental.proxy_tensor import (
        disable_proxy_modes_tracing,
    )

    with disable_proxy_modes_tracing():
        stride, table_max, rcut, eps, degree_floor = (
            float(value) for value in info.tolist()
        )
    descriptor_output, _state = torch.ops.deepmd.dpa4c_graph_compress(
        graph.edge_vec.contiguous(),
        graph.edge_index.contiguous(),
        graph.edge_mask.contiguous(),
        graph.destination_order.contiguous(),
        graph.destination_row_ptr.contiguous(),
        atype.contiguous(),
        descriptor.compress_data.contiguous(),
        type_embedding.to(torch.float32).contiguous(),
        descriptor.adam_degree_weight.to(torch.float32).contiguous(),
        descriptor.scalar_projection.w.to(torch.float32).contiguous(),
        descriptor.vector_projection.w.to(torch.float32).contiguous(),
        descriptor.tensor_projection.w.to(torch.float32).contiguous(),
        bool(graph.destination_sorted),
        stride,
        table_max,
        rcut,
        eps,
        degree_floor,
    )
    return descriptor_output


def dpa4c_graph_compress_energy_force(
    descriptor: Any,
    fitting: Any,
    graph: Any,
    atype: torch.Tensor,
    type_embedding: torch.Tensor,
    ownership: torch.Tensor,
    atom_bias: torch.Tensor,
    node_capacity: int,
    do_atomic_virial: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate compressed DPA4C energy, force, and virial without an autograd tape.

    Parameters
    ----------
    descriptor
        Compressed pt_expt DPA4C descriptor.
    fitting
        Eligible pt_expt energy fitting network.
    graph
        NeighborGraph with destination and source CSR topology.
    atype
        Flat node atom types with shape ``(N,)``.
    type_embedding
        Complete DPA4 type table with shape
        ``(ntypes + 1, channels)``.
    ownership
        Boolean mask selecting energy-contributing nodes with shape ``(N,)``.
    atom_bias
        Combined atomic energy bias with shape ``(ntypes,)``.
    node_capacity
        Force-scatter node capacity.
    do_atomic_virial
        Whether to return per-node virials.

    Returns
    -------
    energy
        Per-frame energy with shape ``(F, 1)``, fp64.
    atom_energy
        Per-node energy with shape ``(N, 1)``, fp64.
    force
        Per-node force with shape ``(N, 3)``, fp32.
    virial
        Per-frame virial with shape ``(F, 3, 3)``, fp32.
    atom_virial
        Per-node virial with shape ``(N, 3, 3)`` or an empty tensor.
    """
    from deepmd.kernels.cuda.edge_force_virial import (
        edge_force_virial,
    )
    from deepmd.kernels.cuda.edge_force_virial import (
        ensure_registered as ensure_force_registered,
    )
    from deepmd.kernels.cuda.graph_fitting import (
        ensure_registered as ensure_fitting_registered,
    )

    ensure_registered()
    ensure_fitting_registered()
    ensure_force_registered()
    if (
        graph.destination_order is None
        or graph.destination_row_ptr is None
        or graph.source_order is None
        or graph.source_row_ptr is None
    ):
        raise ValueError(
            "DPA4C compressed energy-force inference requires destination "
            "and source CSR topology"
        )

    radial_table = descriptor.compress_data.contiguous()
    type_table = type_embedding.to(torch.float32).contiguous()
    info = descriptor.compress_info
    from torch.fx.experimental.proxy_tensor import (
        disable_proxy_modes_tracing,
    )

    with disable_proxy_modes_tracing():
        stride, table_max, rcut, eps, degree_floor = (
            float(value) for value in info.tolist()
        )
    operator_args = (
        graph.edge_index.contiguous(),
        graph.edge_mask.contiguous(),
        graph.destination_order.contiguous(),
        graph.destination_row_ptr.contiguous(),
        atype.contiguous(),
        radial_table,
        type_table,
        descriptor.adam_degree_weight.to(torch.float32).contiguous(),
        descriptor.scalar_projection.w.to(torch.float32).contiguous(),
        descriptor.vector_projection.w.to(torch.float32).contiguous(),
        descriptor.tensor_projection.w.to(torch.float32).contiguous(),
        bool(graph.destination_sorted),
        stride,
        table_max,
        rcut,
        eps,
        degree_floor,
    )
    edge_vec = graph.edge_vec.to(torch.float32).contiguous()
    node_descriptor, state = torch.ops.deepmd.dpa4c_graph_compress(
        edge_vec,
        *operator_args,
    )

    *hidden, head = fitting.nets[0].layers
    empty = hidden[0].w.new_empty(0)
    weights = [layer.w.contiguous() for layer in hidden]
    biases = [
        layer.b.contiguous() if layer.b is not None else empty for layer in hidden
    ]
    timesteps = [
        layer.idt.contiguous() if layer.idt is not None else empty for layer in hidden
    ]
    residuals = [1 if layer.resnet else 0 for layer in hidden]
    head_weight = head.w.reshape(-1).contiguous()
    head_bias = (
        head.b.reshape(-1).to(torch.float32).contiguous()
        if head.b is not None
        else empty
    )
    from deepmd.kernels.triton.dpa1.activation import (
        ACT_CODES,
    )

    activation = ACT_CODES[str(hidden[0].activation_function).lower()]
    atom_energy_raw, fitting_saved = torch.ops.deepmd.graph_fitting(
        node_descriptor,
        atype,
        weights,
        biases,
        timesteps,
        residuals,
        head_weight,
        head_bias,
        atom_bias.to(torch.float64).contiguous(),
        activation,
    )
    owned = ownership[:, None].to(atom_energy_raw.dtype)
    atom_energy = atom_energy_raw * owned
    from deepmd.dpmodel.utils.neighbor_graph import (
        frame_id_from_n_node,
    )

    frame_index = frame_id_from_n_node(
        graph.n_node,
        n_total=atom_energy.shape[0],
    )
    energy = torch.zeros(
        graph.n_node.shape[0],
        1,
        dtype=atom_energy.dtype,
        device=atom_energy.device,
    ).index_add_(0, frame_index, atom_energy)
    del node_descriptor

    descriptor_gradient = torch.ops.deepmd.graph_fitting_backward(
        owned,
        fitting_saved,
        weights,
        residuals,
        head_weight,
    )
    del fitting_saved
    edge_gradient = torch.ops.deepmd.dpa4c_graph_compress_backward(
        descriptor_gradient,
        state,
        edge_vec,
        *operator_args,
    )
    force, atom_virial, virial = edge_force_virial(
        edge_gradient,
        edge_vec,
        graph.edge_index,
        graph.edge_mask,
        graph.destination_order,
        graph.destination_row_ptr,
        graph.source_order,
        graph.source_row_ptr,
        graph.n_node,
        node_capacity,
        do_atomic_virial,
    )
    return energy, atom_energy, force, virial, atom_virial

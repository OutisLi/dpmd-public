# SPDX-License-Identifier: LGPL-3.0-or-later
"""Numerical contract of the compressed DPA4C CUDA mega kernel."""

import dataclasses

import pytest
import torch

from deepmd.dpmodel.utils.neighbor_graph import (
    NeighborGraph,
    attach_edge_csr,
    graph_from_dense_quartet,
)
from deepmd.kernels.cuda.dpa4c.graph_compress import (
    _cpu_descriptor,
    _table_lookup,
    build_radial_table,
    dpa4c_graph_compress_energy_force,
    ensure_registered,
    op_available,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)
from deepmd.pt_expt.descriptor.dpa4c import (
    DescrptDPA4C,
)

_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or not op_available(),
    reason="CUDA and the compiled DPA4C operator are required",
)


def _build_descriptor(channels: int) -> DescrptDPA4C:
    return (
        DescrptDPA4C(
            rcut=3.0,
            ntypes=2,
            channels=channels,
            n_radial=8,
            radial_mlp=[channels],
            precision="float32",
            seed=17,
        )
        .cuda()
        .eval()
    )


def _build_graph(descriptor: DescrptDPA4C, canonical: bool):
    generator = torch.Generator(device="cuda").manual_seed(23)
    coordinate = torch.rand(
        1,
        24,
        3,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    coordinate = coordinate * 5.0
    atype = torch.arange(24, device="cuda").reshape(1, -1) % 2
    coord_ext, atype_ext, mapping, nlist = extend_input_and_build_neighbor_list(
        coordinate,
        atype,
        descriptor.rcut,
        [48],
        mixed_types=True,
        box=None,
    )
    graph, flat_type = graph_from_dense_quartet(
        coord_ext,
        atype_ext,
        nlist,
        mapping,
    )
    graph = attach_edge_csr(
        graph,
        flat_type.shape[0],
        canonicalize=canonical,
    )
    return graph, flat_type


def _arguments(
    descriptor: DescrptDPA4C,
    graph,
    atype: torch.Tensor,
):
    ensure_registered()
    table, info = build_radial_table(descriptor)
    return (
        graph.edge_index,
        graph.edge_mask,
        graph.destination_order,
        graph.destination_row_ptr,
        atype,
        table,
        descriptor.type_embedding.call(),
        descriptor.adam_degree_weight,
        descriptor.scalar_projection.w,
        descriptor.vector_projection.w,
        descriptor.tensor_projection.w,
        bool(graph.destination_sorted),
        *(float(value) for value in info),
    )


@_GPU
@pytest.mark.parametrize("channels", [4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("canonical", [False, True])
def test_forward_backward_parity(channels: int, canonical: bool) -> None:
    descriptor = _build_descriptor(channels)
    graph, atype = _build_graph(descriptor, canonical)
    arguments = _arguments(descriptor, graph, atype)
    ensure_registered()

    edge_vec = graph.edge_vec.detach().clone().requires_grad_(True)
    output, _state = torch.ops.deepmd.dpa4c_graph_compress(
        edge_vec,
        *arguments,
    )
    cotangent = torch.linspace(
        -0.7,
        1.3,
        output.numel(),
        dtype=output.dtype,
        device=output.device,
    ).reshape_as(output)
    (gradient,) = torch.autograd.grad((output * cotangent).sum(), edge_vec)

    reference_edge = graph.edge_vec.detach().clone().requires_grad_(True)
    reference = _cpu_descriptor(reference_edge, *arguments)
    (reference_gradient,) = torch.autograd.grad(
        (reference * cotangent).sum(),
        reference_edge,
    )
    torch.testing.assert_close(output, reference, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        gradient,
        reference_gradient,
        atol=3e-6,
        rtol=3e-5,
    )


@_GPU
def test_compressed_matches_uncompressed_descriptor() -> None:
    descriptor = _build_descriptor(8)
    graph, atype = _build_graph(descriptor, canonical=False)
    arguments = _arguments(descriptor, graph, atype)
    compressed, _state = torch.ops.deepmd.dpa4c_graph_compress(
        graph.edge_vec,
        *arguments,
    )
    reference, _ = descriptor.call_graph(
        graph,
        atype,
        type_embedding=descriptor.type_embedding.call(),
    )
    torch.testing.assert_close(compressed, reference, atol=3e-5, rtol=3e-5)

    cotangent = torch.linspace(
        -0.7,
        1.3,
        compressed.numel(),
        dtype=compressed.dtype,
        device=compressed.device,
    ).reshape_as(compressed)
    compressed_edge = graph.edge_vec.detach().clone().requires_grad_(True)
    compressed_value, _state = torch.ops.deepmd.dpa4c_graph_compress(
        compressed_edge,
        *arguments,
    )
    (compressed_gradient,) = torch.autograd.grad(
        (compressed_value * cotangent).sum(),
        compressed_edge,
    )
    reference_edge = graph.edge_vec.detach().clone().requires_grad_(True)
    reference_graph = dataclasses.replace(graph, edge_vec=reference_edge)
    reference_value, _ = descriptor.call_graph(
        reference_graph,
        atype,
        type_embedding=descriptor.type_embedding.call(),
    )
    (reference_gradient,) = torch.autograd.grad(
        (reference_value * cotangent).sum(),
        reference_edge,
    )
    torch.testing.assert_close(
        compressed_gradient,
        reference_gradient,
        atol=1e-4,
        rtol=5e-4,
    )


@_GPU
@pytest.mark.parametrize("channels", [8, 64, 128])
def test_descriptor_compression_routing_and_serialization(
    monkeypatch: pytest.MonkeyPatch,
    channels: int,
) -> None:
    descriptor = _build_descriptor(channels)
    graph, atype = _build_graph(descriptor, canonical=False)
    reference, _ = descriptor.call_graph(graph, atype)
    descriptor.enable_compression(min_nbor_dist=0.5)
    monkeypatch.setenv("DP_CUDA_INFER", "1")
    actual, _ = descriptor.call_graph(graph, atype)
    restored = DescrptDPA4C.deserialize(descriptor.serialize()).cuda().eval()
    restored_output, _ = restored.call_graph(graph, atype)
    torch.testing.assert_close(actual, reference, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(restored_output, actual)


@_GPU
def test_radial_table_accuracy() -> None:
    descriptor = _build_descriptor(8)
    table, info = build_radial_table(descriptor)
    radius = torch.linspace(0.0, descriptor.rcut, 2001, device="cuda")
    reference = descriptor.radial_embedding(descriptor.radial_basis(radius[:, None]))
    reference = reference * descriptor.edge_envelope(radius[:, None])
    actual = _table_lookup(
        table,
        radius,
        float(info[0]),
        float(info[1]),
        descriptor.channels,
    )
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-6)


@_GPU
def test_radial_table_is_c2_at_internal_knots() -> None:
    descriptor = _build_descriptor(8)
    table, info = build_radial_table(descriptor)
    stride = float(info[0])
    knot = 617 * stride
    cotangent = torch.linspace(
        -0.7,
        1.3,
        descriptor.channels,
        device="cuda",
    )

    def derivatives(radius_value: float) -> list[torch.Tensor]:
        radius = torch.tensor(
            [radius_value],
            dtype=torch.float32,
            device="cuda",
            requires_grad=True,
        )
        value = (
            _table_lookup(
                table,
                radius,
                stride,
                float(info[1]),
                descriptor.channels,
            )[0]
            * cotangent
        ).sum()
        first = torch.autograd.grad(value, radius, create_graph=True)[0]
        second = torch.autograd.grad(first, radius, create_graph=True)[0]
        return [value, first[0], second[0]]

    left = derivatives(knot - 1e-6)
    right = derivatives(knot + 1e-6)
    tolerances = (3e-6, 1e-4, 2e-3)
    for lhs, rhs, atol in zip(left, right, tolerances, strict=True):
        torch.testing.assert_close(lhs, rhs, atol=atol, rtol=0.0)


@_GPU
@pytest.mark.parametrize("channels", [4, 64, 128])
def test_compressed_cutoff_matches_removed_topology(channels: int) -> None:
    descriptor = _build_descriptor(channels)
    edge_index = torch.tensor(
        [[1, 0], [0, 1]],
        dtype=torch.long,
        device="cuda",
    )
    radius = torch.tensor(descriptor.rcut, device="cuda")
    zero = torch.zeros_like(radius)
    edge_vec = torch.stack(
        [
            torch.stack([radius, zero, zero]),
            torch.stack([-radius, zero, zero]),
        ]
    ).requires_grad_(True)
    graph = NeighborGraph(
        n_node=torch.tensor([2], dtype=torch.long, device="cuda"),
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_mask=torch.ones(2, dtype=torch.bool, device="cuda"),
    )
    graph = attach_edge_csr(graph, 2, canonicalize=False)
    atype = torch.zeros(2, dtype=torch.long, device="cuda")
    arguments = _arguments(descriptor, graph, atype)
    retained, _state = torch.ops.deepmd.dpa4c_graph_compress(
        edge_vec,
        *arguments,
    )
    (gradient,) = torch.autograd.grad(retained.sum(), edge_vec)

    removed_graph = dataclasses.replace(
        graph,
        edge_mask=torch.zeros_like(graph.edge_mask),
    )
    removed_edge = edge_vec.detach().clone().requires_grad_(True)
    removed, _removed_state = torch.ops.deepmd.dpa4c_graph_compress(
        removed_edge,
        *_arguments(descriptor, removed_graph, atype),
    )
    (removed_gradient,) = torch.autograd.grad(removed.sum(), removed_edge)
    torch.testing.assert_close(retained, removed, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        gradient,
        torch.zeros_like(gradient),
        atol=2e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        removed_gradient,
        torch.zeros_like(removed_gradient),
        atol=0.0,
        rtol=0.0,
    )


@_GPU
@pytest.mark.parametrize("channels", [4, 64, 128])
def test_in_row_mask_matches_removed_edge(channels: int) -> None:
    descriptor = _build_descriptor(channels)
    edge_index = torch.tensor(
        [[1, 0], [0, 1]],
        dtype=torch.long,
        device="cuda",
    )
    edge_vec = torch.tensor(
        [[1.0, 0.2, -0.1], [-0.7, 0.3, 0.4]],
        dtype=torch.float32,
        device="cuda",
    )
    graph = NeighborGraph(
        n_node=torch.tensor([2], dtype=torch.long, device="cuda"),
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_mask=torch.ones(2, dtype=torch.bool, device="cuda"),
    )
    graph = attach_edge_csr(graph, 2, canonicalize=False)
    atype = torch.zeros(2, dtype=torch.long, device="cuda")

    full, _full_state = torch.ops.deepmd.dpa4c_graph_compress(
        edge_vec,
        *_arguments(descriptor, graph, atype),
    )
    masked_graph = dataclasses.replace(
        graph,
        edge_mask=torch.tensor([True, False], dtype=torch.bool, device="cuda"),
    )
    masked_edge = edge_vec.detach().clone().requires_grad_(True)
    masked, _masked_state = torch.ops.deepmd.dpa4c_graph_compress(
        masked_edge,
        *_arguments(descriptor, masked_graph, atype),
    )
    cotangent = torch.linspace(
        -0.7,
        1.3,
        masked.numel(),
        dtype=masked.dtype,
        device=masked.device,
    ).reshape_as(masked)
    (masked_gradient,) = torch.autograd.grad(
        (masked * cotangent).sum(),
        masked_edge,
    )

    removed_edge = edge_vec[:1].detach().clone().requires_grad_(True)
    removed_graph = NeighborGraph(
        n_node=graph.n_node,
        edge_index=edge_index[:, :1].contiguous(),
        edge_vec=removed_edge,
        edge_mask=torch.ones(1, dtype=torch.bool, device="cuda"),
    )
    removed_graph = attach_edge_csr(removed_graph, 2, canonicalize=False)
    removed, _removed_state = torch.ops.deepmd.dpa4c_graph_compress(
        removed_edge,
        *_arguments(descriptor, removed_graph, atype),
    )
    (removed_gradient,) = torch.autograd.grad(
        (removed * cotangent).sum(),
        removed_edge,
    )

    assert not torch.allclose(full, masked)
    torch.testing.assert_close(masked, removed, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        masked_gradient[:1],
        removed_gradient,
        atol=3e-6,
        rtol=3e-5,
    )
    torch.testing.assert_close(
        masked_gradient[1],
        torch.zeros_like(masked_gradient[1]),
        atol=0.0,
        rtol=0.0,
    )
    assert torch.count_nonzero(masked_gradient[0]).item() > 0


@_GPU
def test_int32_edge_indices() -> None:
    descriptor = _build_descriptor(8)
    graph, atype = _build_graph(descriptor, canonical=False)
    graph32 = dataclasses.replace(
        graph,
        edge_index=graph.edge_index.to(torch.int32),
        destination_order=graph.destination_order.to(torch.int32),
    )
    output64, _state64 = torch.ops.deepmd.dpa4c_graph_compress(
        graph.edge_vec,
        *_arguments(descriptor, graph, atype),
    )
    output32, _state32 = torch.ops.deepmd.dpa4c_graph_compress(
        graph32.edge_vec,
        *_arguments(descriptor, graph32, atype),
    )
    torch.testing.assert_close(output32, output64)


@_GPU
@pytest.mark.parametrize("channels", [8, 64, 128])
def test_compact_canonical_parity(channels: int) -> None:
    from deepmd.kernels.cuda.dpa4c.canonical import (
        ensure_registered as ensure_canonical_registered,
    )

    descriptor = _build_descriptor(channels)
    graph, atype = _build_graph(descriptor, canonical=True)
    arguments = _arguments(descriptor, graph, atype)
    ensure_canonical_registered()
    generic_output, generic_state = torch.ops.deepmd.dpa4c_graph_compress(
        graph.edge_vec,
        *arguments,
    )
    canonical_arguments = (
        graph.edge_index[0],
        graph.destination_row_ptr,
        atype,
        *arguments[5:11],
        *arguments[12:],
    )
    compact_output, compact_state = torch.ops.deepmd.dpa4c_canonical_compress(
        graph.edge_vec,
        *canonical_arguments,
    )
    torch.testing.assert_close(compact_output, generic_output)
    torch.testing.assert_close(compact_state, generic_state)

    cotangent = torch.randn_like(generic_output)
    generic_gradient = torch.ops.deepmd.dpa4c_graph_compress_backward(
        cotangent,
        generic_state,
        graph.edge_vec,
        *arguments,
    )
    compact_gradient = torch.ops.deepmd.dpa4c_canonical_compress_backward(
        cotangent,
        compact_state,
        graph.edge_vec,
        *canonical_arguments,
    )
    torch.testing.assert_close(
        compact_gradient,
        generic_gradient,
        atol=2e-6,
        rtol=2e-6,
    )


@_GPU
@pytest.mark.parametrize("channels", [8, 64, 128])
def test_fused_energy_force_parity(channels: int) -> None:
    from deepmd.kernels.cuda.edge_force_virial import (
        edge_force_virial,
    )
    from deepmd.pt_expt.fitting.ener_fitting import (
        EnergyFittingNet,
    )

    descriptor = _build_descriptor(channels)
    graph, atype = _build_graph(descriptor, canonical=False)
    table, info = build_radial_table(descriptor)
    descriptor.compress_data = table
    descriptor.compress_info = info
    fitting = (
        EnergyFittingNet(
            ntypes=2,
            dim_descrpt=descriptor.get_dim_out(),
            neuron=[32, 32],
            activation_function="silu",
            precision="float32",
            mixed_types=True,
            seed=29,
        )
        .cuda()
        .eval()
    )
    fitting.bias_atom_e = torch.tensor(
        [[0.3], [-0.2]],
        dtype=torch.float64,
        device="cuda",
    )
    ownership = torch.ones(atype.shape[0], dtype=torch.bool, device="cuda")
    fused = dpa4c_graph_compress_energy_force(
        descriptor,
        fitting,
        graph,
        atype,
        descriptor.type_embedding.call(),
        ownership,
        fitting.bias_atom_e[:, 0],
        atype.shape[0],
        True,
    )

    edge_vec = graph.edge_vec.detach().clone().requires_grad_(True)
    arguments = _arguments(descriptor, graph, atype)
    node_descriptor, _state = torch.ops.deepmd.dpa4c_graph_compress(
        edge_vec,
        *arguments,
    )
    atom_energy = fitting.call_graph(node_descriptor, atype)[fitting.var_name]
    (edge_gradient,) = torch.autograd.grad(atom_energy.sum(), edge_vec)
    force, atom_virial, virial = edge_force_virial(
        edge_gradient,
        edge_vec.detach(),
        graph.edge_index,
        graph.edge_mask,
        graph.destination_order,
        graph.destination_row_ptr,
        graph.source_order,
        graph.source_row_ptr,
        graph.n_node,
        atype.shape[0],
        True,
    )
    torch.testing.assert_close(
        fused[1],
        atom_energy.to(torch.float64),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(fused[2], force, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(fused[3], virial, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(fused[4], atom_virial, atol=1e-6, rtol=1e-5)


class _ExportModule(torch.nn.Module):
    def forward(self, edge_vec: torch.Tensor, *arguments):
        return torch.ops.deepmd.dpa4c_graph_compress(edge_vec, *arguments)


@_GPU
def test_torch_export() -> None:
    descriptor = _build_descriptor(8)
    graph, atype = _build_graph(descriptor, canonical=False)
    arguments = _arguments(descriptor, graph, atype)
    module = _ExportModule().cuda().eval()
    exported = torch.export.export(
        module,
        (graph.edge_vec, *arguments),
        strict=False,
    )
    actual = exported.module()(graph.edge_vec, *arguments)
    reference = module(graph.edge_vec, *arguments)
    torch.testing.assert_close(actual, reference)

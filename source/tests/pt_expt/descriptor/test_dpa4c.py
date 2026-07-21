# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import torch

from deepmd.dpmodel.descriptor.dpa4c import DescrptDPA4C as DPDescrptDPA4C
from deepmd.dpmodel.descriptor.dpa4c import (
    _packed_l2_contractions,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    NeighborGraph,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)
from deepmd.pt_expt.descriptor.dpa4c import (
    DescrptDPA4C,
)
from deepmd.pt_expt.utils import (
    env,
)


class TestDPA4C:
    def setup_method(self) -> None:
        self.descriptor = DescrptDPA4C(
            rcut=3.0,
            ntypes=2,
            channels=4,
            n_radial=4,
            radial_mlp=[4],
            precision="float64",
            seed=17,
        ).to(env.DEVICE)
        self.coord = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.1, 0.2, -0.1],
                    [-0.4, 0.9, 0.3],
                    [0.2, -0.5, 1.2],
                    [-0.7, -0.3, -0.8],
                ]
            ],
            dtype=torch.float64,
            device=env.DEVICE,
        )
        self.atype = torch.tensor(
            [[0, 1, 0, 1, 0]],
            dtype=torch.long,
            device=env.DEVICE,
        )

    def _dimer_probe(
        self,
        descriptor: DescrptDPA4C,
        distance: torch.Tensor,
        *,
        active: bool = True,
    ) -> torch.Tensor:
        """Evaluate a scalar descriptor probe for one undirected dimer."""
        zero = torch.zeros_like(distance)
        edge_vec = torch.stack(
            [
                torch.stack([distance, zero, zero]),
                torch.stack([-distance, zero, zero]),
            ]
        )
        graph = NeighborGraph(
            n_node=torch.tensor([2], dtype=torch.long, device=env.DEVICE),
            edge_index=torch.tensor(
                [[1, 0], [0, 1]],
                dtype=torch.long,
                device=env.DEVICE,
            ),
            edge_vec=edge_vec,
            edge_mask=torch.full(
                (2,),
                active,
                dtype=torch.bool,
                device=env.DEVICE,
            ),
        )
        atype = torch.zeros(2, dtype=torch.long, device=env.DEVICE)
        output, _ = descriptor.call_graph(graph, atype)
        cotangent = torch.linspace(
            -0.7,
            1.3,
            output.numel(),
            dtype=output.dtype,
            device=output.device,
        ).reshape_as(output)
        return (output * cotangent).sum()

    def _dimer_derivatives(
        self,
        descriptor: DescrptDPA4C,
        distance: float,
    ) -> list[torch.Tensor]:
        """Return value and first three radial derivatives of a dimer probe."""
        radius = torch.tensor(
            distance,
            dtype=torch.float64,
            device=env.DEVICE,
            requires_grad=True,
        )
        derivatives = [self._dimer_probe(descriptor, radius)]
        for _ in range(3):
            derivatives.append(
                torch.autograd.grad(
                    derivatives[-1],
                    radius,
                    create_graph=True,
                )[0]
            )
        return derivatives

    def _inputs(
        self,
        coord: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return extend_input_and_build_neighbor_list(
            coord,
            self.atype,
            self.descriptor.get_rcut(),
            [8],
            mixed_types=True,
            box=None,
        )

    def test_dpmodel_and_pt_expt_match(self) -> None:
        coord_ext, atype_ext, mapping, nlist = self._inputs(self.coord)
        result = self.descriptor(
            coord_ext,
            atype_ext,
            nlist,
            mapping=mapping,
        )[0]
        dpmodel = DPDescrptDPA4C.deserialize(self.descriptor.serialize())
        reference = dpmodel(
            coord_ext.cpu().numpy(),
            atype_ext.cpu().numpy(),
            nlist.cpu().numpy(),
            mapping=mapping.cpu().numpy(),
        )[0]
        np.testing.assert_allclose(
            result.detach().cpu().numpy(),
            reference,
            atol=1e-12,
            rtol=1e-12,
        )

    def test_coordinate_gradient_matches_finite_difference(self) -> None:
        coord = self.coord.detach().clone().requires_grad_(True)
        coord_ext, atype_ext, mapping, nlist = self._inputs(coord)
        output = self.descriptor(
            coord_ext,
            atype_ext,
            nlist,
            mapping=mapping,
        )[0]
        cotangent = torch.linspace(
            -0.8,
            1.1,
            output.numel(),
            dtype=output.dtype,
            device=output.device,
        ).reshape_as(output)
        (gradient,) = torch.autograd.grad((output * cotangent).sum(), coord)

        epsilon = 1e-6
        finite_difference = torch.empty_like(coord)
        flat = coord.detach().reshape(-1)
        for index in range(flat.numel()):
            positive = flat.clone()
            negative = flat.clone()
            positive[index] += epsilon
            negative[index] -= epsilon
            positive_inputs = self._inputs(positive.reshape_as(coord))
            negative_inputs = self._inputs(negative.reshape_as(coord))
            positive_output = self.descriptor(
                positive_inputs[0],
                positive_inputs[1],
                positive_inputs[3],
                mapping=positive_inputs[2],
            )[0]
            negative_output = self.descriptor(
                negative_inputs[0],
                negative_inputs[1],
                negative_inputs[3],
                mapping=negative_inputs[2],
            )[0]
            finite_difference.reshape(-1)[index] = (
                (positive_output * cotangent).sum()
                - (negative_output * cotangent).sum()
            ) / (2.0 * epsilon)
        torch.testing.assert_close(
            gradient,
            finite_difference,
            atol=2e-8,
            rtol=2e-8,
        )

    def test_all_parameters_receive_finite_gradients(self) -> None:
        coord_ext, atype_ext, mapping, nlist = self._inputs(self.coord)
        output = self.descriptor(
            coord_ext,
            atype_ext,
            nlist,
            mapping=mapping,
        )[0]
        output.square().mean().backward()
        for name, parameter in self.descriptor.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name

    def test_channelwise_tanh_feedback_range(self) -> None:
        with torch.no_grad():
            self.descriptor.adam_degree_weight.zero_()
            self.descriptor.adam_degree_weight[:, 0].copy_(
                torch.linspace(
                    -10.0,
                    10.0,
                    self.descriptor.channels,
                    dtype=torch.float64,
                    device=env.DEVICE,
                )
            )
        scalar = torch.ones(
            2,
            self.descriptor.channels,
            dtype=torch.float64,
            device=env.DEVICE,
        )
        zero = torch.zeros_like(scalar)
        feedback = self.descriptor._mix_angular_feedback((scalar, zero, zero))
        expected = torch.tanh(self.descriptor.adam_degree_weight[:, 0]).expand_as(
            feedback
        )
        torch.testing.assert_close(feedback, expected)
        assert torch.all(feedback > -1.0)
        assert torch.all(feedback < 1.0)
        assert torch.unique(feedback[0]).numel() == self.descriptor.channels

    def test_degree_normalization_has_continuous_derivative(self) -> None:
        dst = torch.zeros(1, dtype=torch.long, device=env.DEVICE)
        derivatives = []
        for value in (0.5 - 1e-7, 0.5 + 1e-7):
            envelope = torch.tensor(
                [value],
                dtype=torch.float64,
                device=env.DEVICE,
                requires_grad=True,
            )
            normalizer = self.descriptor._degree_normalizer(
                envelope,
                dst,
                1,
            )
            (derivative,) = torch.autograd.grad(normalizer.sum(), envelope)
            derivatives.append(derivative)
        torch.testing.assert_close(
            derivatives[0],
            derivatives[1],
            atol=2e-6,
            rtol=2e-6,
        )

    def test_cutoff_is_c3_continuous(self) -> None:
        descriptor = (
            DescrptDPA4C(
                rcut=3.0,
                ntypes=1,
                channels=4,
                n_radial=6,
                radial_mlp=[4],
                precision="float64",
                seed=31,
            )
            .to(env.DEVICE)
            .eval()
        )
        inside = self._dimer_derivatives(
            descriptor,
            descriptor.rcut - 1.0e-5,
        )
        boundary = self._dimer_derivatives(descriptor, descriptor.rcut)
        outside = self._dimer_derivatives(
            descriptor,
            descriptor.rcut + 1.0e-5,
        )
        torch.testing.assert_close(boundary[0], outside[0], atol=1e-14, rtol=0.0)
        for order in range(1, 4):
            torch.testing.assert_close(
                boundary[order],
                outside[order],
                atol=1e-14,
                rtol=0.0,
            )
        assert inside[1].abs() < 1e-12
        assert inside[2].abs() < 2e-8
        assert inside[3].abs() < 5e-4

    def test_cutoff_edge_matches_removed_topology(self) -> None:
        descriptor = (
            DescrptDPA4C(
                rcut=3.0,
                ntypes=1,
                channels=4,
                n_radial=6,
                radial_mlp=[4],
                precision="float64",
                seed=31,
            )
            .to(env.DEVICE)
            .eval()
        )
        radius = torch.tensor(
            descriptor.rcut,
            dtype=torch.float64,
            device=env.DEVICE,
        )
        retained = self._dimer_probe(descriptor, radius, active=True)
        removed = self._dimer_probe(descriptor, radius, active=False)
        torch.testing.assert_close(retained, removed, atol=1e-14, rtol=0.0)

    def test_float32_nextafter_cutoff_is_stable(self) -> None:
        descriptor = (
            DescrptDPA4C(
                rcut=3.0,
                ntypes=1,
                channels=4,
                n_radial=6,
                radial_mlp=[4],
                precision="float32",
                seed=31,
            )
            .to(env.DEVICE)
            .eval()
        )
        cutoff = torch.tensor(
            descriptor.rcut,
            dtype=torch.float32,
            device=env.DEVICE,
        )
        radius = torch.nextafter(cutoff, torch.zeros_like(cutoff))
        radius.requires_grad_(True)
        near_value = self._dimer_probe(descriptor, radius)
        (near_gradient,) = torch.autograd.grad(near_value, radius)
        cutoff_value = self._dimer_probe(descriptor, cutoff)
        assert torch.isfinite(near_value)
        assert torch.isfinite(near_gradient)
        torch.testing.assert_close(
            near_value,
            cutoff_value,
            atol=2e-6,
            rtol=2e-6,
        )

    def test_coincident_edge_has_finite_third_derivative(self) -> None:
        descriptor = (
            DescrptDPA4C(
                rcut=3.0,
                ntypes=1,
                channels=4,
                n_radial=6,
                radial_mlp=[4],
                precision="float64",
                seed=31,
            )
            .to(env.DEVICE)
            .eval()
        )
        derivatives = self._dimer_derivatives(descriptor, 0.0)
        for derivative in derivatives:
            assert torch.isfinite(derivative)
        torch.testing.assert_close(
            derivatives[1],
            torch.zeros_like(derivatives[1]),
            atol=1e-14,
            rtol=0.0,
        )
        torch.testing.assert_close(
            derivatives[3],
            torch.zeros_like(derivatives[3]),
            atol=1e-12,
            rtol=0.0,
        )

    def test_native_packed_contractions_match_dpmodel(self) -> None:
        generator = torch.Generator(device=env.DEVICE)
        generator.manual_seed(43)
        tensor = torch.randn(
            (11, 7, 5),
            dtype=torch.float64,
            device=env.DEVICE,
            generator=generator,
        )
        vector = torch.randn(
            (11, 7, 3),
            dtype=torch.float64,
            device=env.DEVICE,
            generator=generator,
        )
        result = self.descriptor._contract_l2_invariants(tensor, vector)
        reference = _packed_l2_contractions(tensor, vector)
        for actual, expected in zip(result, reference, strict=True):
            torch.testing.assert_close(
                actual,
                expected,
                atol=1e-13,
                rtol=1e-13,
            )

    def test_force_loss_double_backward(self) -> None:
        coord = self.coord.detach().clone().requires_grad_(True)
        coord_ext, atype_ext, mapping, nlist = self._inputs(coord)
        output = self.descriptor(
            coord_ext,
            atype_ext,
            nlist,
            mapping=mapping,
        )[0]
        (force_gradient,) = torch.autograd.grad(
            output.square().sum(),
            coord,
            create_graph=True,
        )
        parameter_gradients = torch.autograd.grad(
            force_gradient.square().mean(),
            tuple(self.descriptor.parameters()),
            allow_unused=True,
        )
        assert any(gradient is not None for gradient in parameter_gradients)
        for gradient in parameter_gradients:
            if gradient is not None:
                assert torch.isfinite(gradient).all()

    def test_serialization_preserves_parameters(self) -> None:
        restored = DescrptDPA4C.deserialize(self.descriptor.serialize()).to(env.DEVICE)
        original_parameters = dict(self.descriptor.named_parameters())
        restored_parameters = dict(restored.named_parameters())
        assert original_parameters.keys() == restored_parameters.keys()
        for name in original_parameters:
            torch.testing.assert_close(
                restored_parameters[name],
                original_parameters[name],
            )

    def test_trainable_false_freezes_dpa4_components(self) -> None:
        descriptor = DescrptDPA4C(
            rcut=3.0,
            ntypes=2,
            channels=4,
            n_radial=4,
            radial_mlp=[4],
            precision="float64",
            trainable=False,
            seed=17,
        )
        parameters = dict(descriptor.named_parameters())
        assert not any(parameter.requires_grad for parameter in parameters.values())
        state = descriptor.state_dict()
        assert "adam_degree_weight" in state
        assert "type_embedding.adam_type_embedding" in state
        assert "radial_basis.adam_freqs" in state

    def test_dense_adapter_torch_export_for_bounded_test_lists(self) -> None:
        coord_ext, atype_ext, mapping, nlist = self._inputs(self.coord)
        exported = torch.export.export(
            self.descriptor.eval(),
            (coord_ext, atype_ext, nlist),
            kwargs={"mapping": mapping},
            strict=False,
        )
        result = exported.module()(
            coord_ext,
            atype_ext,
            nlist,
            mapping=mapping,
        )[0]
        reference = self.descriptor(
            coord_ext,
            atype_ext,
            nlist,
            mapping=mapping,
        )[0]
        torch.testing.assert_close(result, reference)

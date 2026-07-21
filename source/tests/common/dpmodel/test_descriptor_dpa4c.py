# SPDX-License-Identifier: LGPL-3.0-or-later

import dataclasses
import inspect
import unittest

import numpy as np
import pytest

from deepmd.dpmodel.descriptor.dpa4_nn.radial import (
    RadialBasis,
)
from deepmd.dpmodel.descriptor.dpa4c import (
    DescrptDPA4C,
    _build_l2_basis,
    _packed_l2_contractions,
    _packed_l2_to_stf,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    graph_from_dense_quartet,
)
from deepmd.dpmodel.utils.nlist import (
    extend_input_and_build_neighbor_list,
)


class TestDPA4C(unittest.TestCase):
    def setUp(self) -> None:
        self.descriptor = DescrptDPA4C(
            rcut=3.0,
            ntypes=2,
            channels=4,
            n_radial=4,
            radial_mlp=[4],
            precision="float64",
            seed=17,
        )
        self.coord = np.array(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.1, 0.2, -0.1],
                    [-0.4, 0.9, 0.3],
                    [0.2, -0.5, 1.2],
                    [-0.7, -0.3, -0.8],
                ]
            ],
            dtype=np.float64,
        )
        self.atype = np.array([[0, 1, 0, 1, 0]], dtype=np.int64)

    def test_compact_public_contract(self) -> None:
        parameters = inspect.signature(DescrptDPA4C.__init__).parameters
        for removed in (
            "sel",
            "rcut_smth",
            "rank",
            "readout_rank",
            "out_dim",
            "tebd_dim",
            "use_tebd_bias",
        ):
            self.assertNotIn(removed, parameters)
        self.assertEqual(
            self.descriptor.get_dim_out(),
            6 * self.descriptor.channels,
        )
        self.assertEqual(self.descriptor.radial_basis.exponent, 5)
        self.assertFalse(self.descriptor.radial_basis.apply_envelope)
        self.assertEqual(self.descriptor.edge_envelope.p, 5)
        self.assertEqual(
            self.descriptor.adam_degree_weight.shape,
            (self.descriptor.channels, 3),
        )
        for layer in (
            self.descriptor.scalar_projection,
            self.descriptor.vector_projection,
            self.descriptor.tensor_projection,
        ):
            self.assertIsNone(layer.b)
        for layer in self.descriptor.radial_embedding.net:
            if hasattr(layer, "b"):
                self.assertIsNone(layer.b)
        table = self.descriptor.type_embedding.call()
        gathered = self.descriptor.type_embedding.call(np.array([1, 0]))
        np.testing.assert_array_equal(gathered, table[[1, 0]])

    def test_radial_basis_envelope_is_optional(self) -> None:
        radius = np.array([[1.7]], dtype=np.float64)
        default = RadialBasis(
            rcut=3.0,
            basis_type="gaussian",
            n_radial=4,
            precision="float64",
            exponent=5,
        )
        raw = RadialBasis(
            rcut=3.0,
            basis_type="gaussian",
            n_radial=4,
            precision="float64",
            exponent=5,
            apply_envelope=False,
        )
        np.testing.assert_allclose(
            default(radius),
            raw(radius) * default.envelope(radius),
            atol=1e-15,
            rtol=1e-15,
        )
        restored = RadialBasis.deserialize(raw.serialize())
        self.assertFalse(restored.apply_envelope)

    def _dense_inputs(
        self,
        coord: np.ndarray,
        atype: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return extend_input_and_build_neighbor_list(
            coord,
            atype,
            self.descriptor.get_rcut(),
            [8],
            mixed_types=True,
            box=None,
        )

    def _evaluate(
        self,
        coord: np.ndarray,
        atype: np.ndarray,
    ) -> np.ndarray:
        coord_ext, atype_ext, mapping, nlist = self._dense_inputs(coord, atype)
        return self.descriptor(
            coord_ext,
            atype_ext,
            nlist,
            mapping=mapping,
        )[0]

    def test_dense_adapter_matches_graph(self) -> None:
        coord_ext, atype_ext, mapping, nlist = self._dense_inputs(
            self.coord,
            self.atype,
        )
        dense = self.descriptor(
            coord_ext,
            atype_ext,
            nlist,
            mapping=mapping,
        )[0]
        graph, atype_local = graph_from_dense_quartet(
            coord_ext,
            atype_ext,
            nlist,
            mapping,
        )
        graph_output, rotation = self.descriptor.call_graph(
            graph,
            atype_local,
            type_embedding=self.descriptor.type_embedding.call(),
        )
        self.assertIsNone(rotation)
        np.testing.assert_allclose(
            graph_output.reshape(dense.shape),
            dense,
            atol=1e-12,
            rtol=1e-12,
        )

    def test_rotation_translation_and_permutation_invariance(self) -> None:
        reference = self._evaluate(self.coord, self.atype)
        rotation = np.array(
            [
                [-2.0 / 3.0, 2.0 / 15.0, 11.0 / 15.0],
                [2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0],
                [1.0 / 3.0, 14.0 / 15.0, 2.0 / 15.0],
            ],
            dtype=np.float64,
        )
        rotated = self._evaluate(self.coord @ rotation.T, self.atype)
        translated = self._evaluate(
            self.coord + np.array([1.7, -0.8, 2.1]),
            self.atype,
        )
        permutation = np.array([2, 4, 0, 3, 1])
        permuted = self._evaluate(
            self.coord[:, permutation],
            self.atype[:, permutation],
        )
        np.testing.assert_allclose(rotated, reference, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(translated, reference, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(
            permuted,
            reference[:, permutation],
            atol=1e-12,
            rtol=1e-12,
        )

    def test_serialization_round_trip(self) -> None:
        reference = self._evaluate(self.coord, self.atype)
        restored = DescrptDPA4C.deserialize(self.descriptor.serialize())
        coord_ext, atype_ext, mapping, nlist = self._dense_inputs(
            self.coord,
            self.atype,
        )
        result = restored(
            coord_ext,
            atype_ext,
            nlist,
            mapping=mapping,
        )[0]
        np.testing.assert_array_equal(result, reference)

    def test_empty_environment_preserves_center_type_signal(self) -> None:
        descriptor = DescrptDPA4C(
            rcut=3.0,
            ntypes=1,
            channels=3,
            n_radial=4,
            radial_mlp=[4],
            precision="float64",
            seed=23,
        )
        coord = np.zeros((1, 1, 3), dtype=np.float64)
        atype = np.zeros((1, 1), dtype=np.int64)
        nlist = np.full((1, 1, 4), -1, dtype=np.int64)
        output = descriptor(coord, atype, nlist)[0]
        self.assertGreater(np.max(np.abs(output)), 0.0)
        descriptor_two_types = DescrptDPA4C(
            rcut=3.0,
            ntypes=2,
            channels=3,
            n_radial=4,
            radial_mlp=[4],
            precision="float64",
            seed=23,
        )
        type_zero = descriptor_two_types(coord, atype, nlist)[0]
        type_one = descriptor_two_types(
            coord,
            np.ones_like(atype),
            nlist,
        )[0]
        self.assertGreater(np.max(np.abs(type_zero - type_one)), 0.0)

    def test_single_neighbor_signal_survives_self_removal(self) -> None:
        descriptor = DescrptDPA4C(
            rcut=3.0,
            ntypes=1,
            channels=3,
            n_radial=4,
            radial_mlp=[4],
            precision="float64",
            seed=31,
        )
        coord = np.array([[[0.0, 0.0, 0.0], [1.1, 0.2, -0.1]]])
        atype = np.zeros((1, 2), dtype=np.int64)
        nlist = np.array([[[1, -1], [0, -1]]], dtype=np.int64)
        output = descriptor(coord, atype, nlist)[0]
        self.assertGreater(np.max(np.abs(output)), 0.0)

    def test_legacy_capacity_dense_neighbor_list_fails(self) -> None:
        coord = np.array([[[0.0, 0.0, 0.0], [1.1, 0.2, -0.1]]])
        atype = np.zeros((1, 2), dtype=np.int64)
        nlist = np.full((1, 2, 256), -1, dtype=np.int64)
        nlist[0, 0, 0] = 1
        nlist[0, 1, 0] = 0
        with self.assertRaisesRegex(ValueError, "carry-all graph-native path"):
            self.descriptor(coord, atype, nlist)

    def test_neighbor_statistics_reject_excess_capacity(self) -> None:
        class FakeUpdateSel:
            def update_one_sel(
                self,
                train_data: object,
                type_map: object,
                rcut: float,
                sel: str,
                mixed_type: bool,
            ) -> tuple[float, list[int]]:
                del train_data, type_map, rcut, mixed_type
                assert sel == "auto:1.0"
                return 0.7, [260]

        original = DescrptDPA4C._update_sel_cls
        DescrptDPA4C._update_sel_cls = FakeUpdateSel
        try:
            with self.assertRaisesRegex(ValueError, "more than 256 neighbors"):
                DescrptDPA4C.update_sel(
                    None,
                    None,
                    {"rcut": 3.0},
                )
        finally:
            DescrptDPA4C._update_sel_cls = original

    def test_addition_theorem_normalization(self) -> None:
        rng = np.random.default_rng(29)
        left = rng.normal(size=(32, 3))
        right = rng.normal(size=(32, 3))
        left /= np.linalg.norm(left, axis=-1, keepdims=True)
        right /= np.linalg.norm(right, axis=-1, keepdims=True)
        left_basis = _build_l2_basis(left)
        right_basis = _build_l2_basis(right)
        cosine = np.sum(left * right, axis=-1)
        np.testing.assert_allclose(
            np.sum(left_basis[:, 1:4] * right_basis[:, 1:4], axis=-1),
            cosine,
            atol=1e-15,
            rtol=1e-15,
        )
        np.testing.assert_allclose(
            np.sum(left_basis[:, 4:9] * right_basis[:, 4:9], axis=-1),
            0.5 * (3.0 * cosine**2 - 1.0),
            atol=1e-15,
            rtol=1e-15,
        )

    def test_packed_contractions_match_matrix_reference(self) -> None:
        rng = np.random.default_rng(41)
        packed = rng.normal(size=(9, 7, 5))
        vector = rng.normal(size=(9, 7, 3))
        tensor = _packed_l2_to_stf(packed)
        tensor_vector = np.matmul(tensor, vector[..., None])[..., 0]
        tensor_cubed = np.matmul(np.matmul(tensor, tensor), tensor)
        reference = (
            np.sum(tensor * tensor, axis=(-2, -1)),
            np.sum(vector * tensor_vector, axis=-1),
            np.sum(
                np.diagonal(tensor_cubed, axis1=-2, axis2=-1),
                axis=-1,
            ),
            np.sum(tensor_vector * tensor_vector, axis=-1),
        )
        result = _packed_l2_contractions(packed, vector)
        for actual, expected in zip(result, reference, strict=False):
            np.testing.assert_allclose(
                actual,
                expected,
                atol=1e-13,
                rtol=1e-13,
            )

    def test_jax_graph_execution_matches_numpy(self) -> None:
        pytest.importorskip("jax")
        from deepmd.jax.env import (
            jax,
            jnp,
        )

        descriptor = DescrptDPA4C(
            rcut=3.0,
            ntypes=2,
            channels=4,
            n_radial=4,
            radial_mlp=[4],
            precision="float64",
            seed=17,
        )
        coord_ext, atype_ext, mapping, nlist = self._dense_inputs(
            self.coord,
            self.atype,
        )
        graph, atype_local = graph_from_dense_quartet(
            coord_ext,
            atype_ext,
            nlist,
            mapping,
        )
        reference, _ = descriptor.call_graph(graph, atype_local)
        graph_jax = dataclasses.replace(
            graph,
            n_node=jnp.asarray(graph.n_node),
            edge_index=jnp.asarray(graph.edge_index),
            edge_vec=jnp.asarray(graph.edge_vec),
            edge_mask=jnp.asarray(graph.edge_mask),
        )
        atype_jax = jnp.asarray(atype_local)

        def evaluate(edge_vec: object) -> object:
            current_graph = dataclasses.replace(graph_jax, edge_vec=edge_vec)
            return descriptor.call_graph(current_graph, atype_jax)[0]

        result = jax.jit(evaluate)(graph_jax.edge_vec)
        np.testing.assert_allclose(
            np.asarray(result),
            reference,
            atol=2e-10,
            rtol=2e-10,
        )

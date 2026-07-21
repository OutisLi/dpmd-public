# SPDX-License-Identifier: LGPL-3.0-or-later
"""PyTorch-exportable execution backend for DPA4C.

``deepmd.dpmodel.descriptor.dpa4c`` defines the graph algorithm and tensor
contracts. This module implements its performance-critical primitives with
native PyTorch operations, promotes DPA4 trainable arrays to parameters, and
registers the descriptor with the pt_expt backend.
"""

import math
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.descriptor.dpa4c import DescrptDPA4C as DescrptDPA4CDP
from deepmd.kernels.utils import (
    cuda_infer_level,
)
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt_expt.descriptor.dpa4 import (
    _promote_trainable_tree,
)
from deepmd.pt_expt.utils.update_sel import (
    UpdateSel,
)


@BaseDescriptor.register("dpa4c")
@torch_module
class DescrptDPA4C(DescrptDPA4CDP):
    """Execute the backend-neutral DPA4C equations with PyTorch tensors.

    Notes
    -----
    DPA4 components such as ``SeZMTypeEmbedding`` and ``RadialBasis`` store
    trainable arrays in the dpmodel representation. The wrapper promotes those
    arrays after construction and deserialization so they remain visible to
    PyTorch optimizers and force-loss double backward. Graph gathers,
    reductions, radial evaluation, angular projection, and invariant
    contractions use native PyTorch primitives to avoid array-API dispatch in
    the eager hot path.
    """

    _update_sel_cls = UpdateSel

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Construct and parameterize a PyTorch DPA4C descriptor.

        Parameters
        ----------
        *args
            Positional arguments forwarded to the dpmodel DPA4C constructor.
        **kwargs
            Keyword arguments forwarded to the dpmodel DPA4C constructor.
        """
        super().__init__(*args, **kwargs)
        _promote_trainable_tree(self)
        self._promote_degree_weight()
        self.compress = False

    def _promote_degree_weight(self) -> None:
        """Promote channel-wise degree weights from a buffer to a parameter."""
        weight = self._buffers.get("adam_degree_weight")
        if weight is None or not self.trainable:
            return
        del self._buffers["adam_degree_weight"]
        self.adam_degree_weight = torch.nn.Parameter(
            weight,
            requires_grad=True,
        )

    def call_graph(
        self,
        graph: Any,
        atype: torch.Tensor,
        type_embedding: torch.Tensor | None = None,
        comm_dict: dict | None = None,
    ) -> tuple[torch.Tensor, None]:
        """Evaluate the graph descriptor with compressed CUDA dispatch.

        Parameters
        ----------
        graph
            NeighborGraph over the flat node axis.
        atype
            Flat atom types with shape ``(N,)``.
        type_embedding
            Optional complete DPA4 type table.
        comm_dict
            Communication metadata accepted by the common graph ABI; unused.

        Returns
        -------
        descriptor
            Invariant descriptor with shape ``(N, 6 * channels)``.
        rot_mat
            ``None``.
        """
        if type_embedding is None:
            type_embedding = self.type_embedding.call()
        if (
            self.compress
            and not self.training
            and not self.exclude_types
            and cuda_infer_level() >= 1
            and graph.destination_order is not None
            and graph.destination_row_ptr is not None
        ):
            from deepmd.kernels.cuda.dpa4c.graph_compress import (
                dpa4c_graph_compress,
                mega_eligible,
                op_available,
            )

            if op_available() and mega_eligible(self):
                return dpa4c_graph_compress(
                    self,
                    graph,
                    atype,
                    type_embedding,
                ), None
        return super().call_graph(
            graph,
            atype,
            type_embedding=type_embedding,
            comm_dict=comm_dict,
        )

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptDPA4C":
        """Deserialize DPA4C and restore trainable PyTorch parameters.

        Parameters
        ----------
        data
            Versioned dpmodel descriptor dictionary.

        Returns
        -------
        DescrptDPA4C
            Reconstructed PyTorch descriptor.
        """
        data = data.copy()
        compression = data.pop("compress", None)
        obj = super().deserialize(data)
        obj = _promote_trainable_tree(obj)
        obj._promote_degree_weight()
        obj.compress = False
        if compression is not None:
            variables = compression["@variables"]
            obj._set_compression(
                torch.as_tensor(variables["data"]),
                torch.as_tensor(variables["info"]),
            )
        return obj

    def _set_compression(
        self,
        table: torch.Tensor,
        info: torch.Tensor,
    ) -> None:
        """Store immutable radial spline data as module buffers."""
        device = next(self.parameters()).device
        table = table.to(device=device, dtype=torch.float32).contiguous()
        info = info.to(device=device, dtype=torch.float64).contiguous()
        if "compress_data" in self._buffers:
            self._buffers["compress_data"] = table
            self._buffers["compress_info"] = info
        else:
            self.register_buffer("compress_data", table)
            self.register_buffer("compress_info", info)
        self.compress = True

    def enable_compression(
        self,
        min_nbor_dist: float,
        table_extrapolate: float = 1.0,
        table_stride_1: float = 0.002,
        table_stride_2: float = 0.002,
        check_frequency: int = -1,
    ) -> None:
        """Build the DPA4C radial table used only inside the mega kernel.

        Parameters
        ----------
        min_nbor_dist
            Minimum neighbor distance accepted by the common compression ABI.
            DPA4C tabulates the finite DPA4 radial basis from zero to ``rcut``
            and therefore does not use this value.
        table_extrapolate
            Common compression parameter; unused because the C³ radial map is
            exactly zero beyond ``rcut``.
        table_stride_1
            Uniform radial spline spacing in Å.
        table_stride_2
            Common two-region table spacing; unused by the uniform DPA4C table.
        check_frequency
            Common overflow-check setting; unused because the radial domain is
            bounded analytically.

        Raises
        ------
        ValueError
            If compression is already enabled or ``channels`` has no compiled
            CUDA specialization.
        """
        del min_nbor_dist, table_extrapolate, table_stride_2, check_frequency
        if self.compress:
            raise ValueError("Compression is already enabled.")
        from deepmd.kernels.cuda.dpa4c.graph_compress import (
            build_radial_table,
            mega_eligible,
        )

        if not mega_eligible(self):
            raise ValueError(
                "DPA4C compressed CUDA supports channels 4, 8, 16, 32, 64, "
                f"or 128, got {self.channels}"
            )
        table, info = build_radial_table(self, table_stride_1)
        self._set_compression(table, info)

    def serialize(self) -> dict:
        """Serialize DPA4C and optional immutable compression buffers."""
        from deepmd.dpmodel.common import (
            to_numpy_array,
        )

        data = super().serialize()
        if self.compress:
            data["compress"] = {
                "@variables": {
                    "data": to_numpy_array(self.compress_data),
                    "info": to_numpy_array(self.compress_info),
                }
            }
        return data

    def fused_energy_force_graph(
        self,
        fitting: Any,
        graph: Any,
        atype: torch.Tensor,
        ownership: torch.Tensor,
        atom_bias: torch.Tensor,
        do_atomic_virial: bool,
    ) -> (
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
        | None
    ):
        """Evaluate the inference-only compressed energy-force composition.

        Returns ``None`` when the model or graph cannot use the level-two CUDA
        path, allowing the caller to retain the generic autograd lower.
        """
        if (
            self.training
            or not self.compress
            or bool(self.exclude_types)
            or cuda_infer_level() < 2
            or graph.destination_order is None
            or graph.destination_row_ptr is None
            or graph.source_order is None
            or graph.source_row_ptr is None
        ):
            return None
        from deepmd.kernels.cuda.dpa4c.graph_compress import (
            dpa4c_graph_compress_energy_force,
            ef_op_available,
            mega_eligible,
        )
        from deepmd.kernels.cuda.graph_fitting import (
            fitting_eligible,
        )

        if (
            not ef_op_available()
            or not mega_eligible(self)
            or not fitting_eligible(fitting)
        ):
            return None
        return dpa4c_graph_compress_energy_force(
            self,
            fitting,
            graph,
            atype,
            self.type_embedding.call(),
            ownership,
            atom_bias,
            atype.shape[0],
            do_atomic_virial,
        )

    def _gather(
        self,
        values: torch.Tensor,
        index: torch.Tensor,
        xp: Any | None = None,
    ) -> torch.Tensor:
        """Gather rows with the native PyTorch indexing primitive.

        Parameters
        ----------
        values
            Source tensor with indexed leading axis.
        index
            Integer row indices.
        xp
            Array namespace accepted for dpmodel signature parity; unused.

        Returns
        -------
        torch.Tensor
            Gathered rows with shape ``index.shape + values.shape[1:]``.
        """
        del xp
        return torch.index_select(values, 0, index)

    def _cutoff_envelope(self, distance: torch.Tensor) -> torch.Tensor:
        """Evaluate the fixed p=5 C³ envelope with native PyTorch ops.

        Parameters
        ----------
        distance
            Edge distances with shape ``(E, 1)`` in Å.

        Returns
        -------
        torch.Tensor
            Envelope values with shape ``(E, 1)``.
        """
        u = torch.clamp(
            (self.rcut - distance) / self.rcut,
            min=0.0,
            max=1.0,
        )
        x = 1.0 - u
        series = 1.0 + x * (4.0 + x * (10.0 + x * (20.0 + x * 35.0)))
        return u**4 * series

    def _angular_basis(self, direction: torch.Tensor) -> torch.Tensor:
        """Build the packed l=0,1,2 Cartesian basis with one stack.

        Parameters
        ----------
        direction
            Edge directions with shape ``(E, 3)``.

        Returns
        -------
        torch.Tensor
            Packed angular basis with shape ``(E, 9)``.
        """
        x, y, z = direction[:, 0], direction[:, 1], direction[:, 2]
        squared_norm = x * x + y * y + z * z
        sqrt_three = math.sqrt(3.0)
        return torch.stack(
            (
                torch.ones_like(x),
                x,
                y,
                z,
                sqrt_three * x * y,
                sqrt_three * y * z,
                0.5 * (3.0 * z * z - squared_norm),
                sqrt_three * x * z,
                0.5 * sqrt_three * (x * x - y * y),
            ),
            dim=-1,
        )

    def _degree_normalizer(
        self,
        envelope: torch.Tensor,
        dst: torch.Tensor,
        n_total: int,
    ) -> torch.Tensor:
        """Compute smooth destination degrees with native ``index_add``.

        Parameters
        ----------
        envelope
            Masked edge envelopes with shape ``(E,)``.
        dst
            Destination node indices with shape ``(E,)``.
        n_total
            Number of output nodes.

        Returns
        -------
        torch.Tensor
            Node normalization factors with shape ``(N, 1)``.
        """
        degree = torch.zeros(
            (n_total, 1),
            dtype=envelope.dtype,
            device=envelope.device,
        )
        degree = torch.index_add(
            degree,
            0,
            dst,
            (envelope * envelope)[:, None],
        )
        return torch.rsqrt(degree + self._DEGREE_NORM_FLOOR)

    def _moments(
        self,
        edge_amplitude: torch.Tensor,
        basis: torch.Tensor,
        dst: torch.Tensor,
        n_total: int,
        normalizer: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate normalized packed moments with native ``index_add``.

        Parameters
        ----------
        edge_amplitude
            Edge amplitudes with shape ``(E, channels)``.
        basis
            Packed angular basis with shape ``(E, 9)``.
        dst
            Destination node indices with shape ``(E,)``.
        n_total
            Number of output nodes.
        normalizer
            Node normalization factors with shape ``(N, 1)``.

        Returns
        -------
        torch.Tensor
            Normalized moments with shape ``(N, channels, 9)``.
        """
        outer = edge_amplitude[:, :, None] * basis[:, None, :]
        moments = torch.zeros(
            (n_total, self.channels, self._ANGULAR_DIM),
            dtype=edge_amplitude.dtype,
            device=edge_amplitude.device,
        )
        moments = torch.index_add(moments, 0, dst, outer)
        return moments * normalizer[:, :, None]

    def _angular_feedback(
        self,
        first_moments: torch.Tensor,
        amplitude: torch.Tensor,
        basis: torch.Tensor,
        dst: torch.Tensor,
        normalizer_on_edge: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project center moments onto edges without a width-nine gather.

        Parameters
        ----------
        first_moments
            First normalized moments with shape ``(N, channels, 9)``.
        amplitude
            Edge amplitudes with shape ``(E, channels)``.
        basis
            Packed angular basis with shape ``(E, 9)``.
        dst
            Destination node indices with shape ``(E,)``.
        normalizer_on_edge
            Destination normalizers with shape ``(E, 1)``.

        Returns
        -------
        scalar_feedback, vector_feedback, tensor_feedback
            Degree-specific edge responses with shape ``(E, channels)``.
        """
        normalized_amplitude = amplitude * normalizer_on_edge

        scalar_basis = basis[:, 0]
        scalar_feedback = (
            torch.index_select(first_moments[:, :, 0], 0, dst) * scalar_basis[:, None]
            - normalized_amplitude * (scalar_basis * scalar_basis)[:, None]
        )

        vector_basis = basis[:, 1:4]
        vector_feedback = torch.sum(
            torch.index_select(first_moments[:, :, 1:4], 0, dst)
            * vector_basis[:, None, :],
            dim=-1,
        )
        vector_feedback = (
            vector_feedback
            - normalized_amplitude
            * torch.sum(
                vector_basis * vector_basis,
                dim=-1,
            )[:, None]
        )

        tensor_basis = basis[:, 4:9]
        tensor_feedback = torch.sum(
            torch.index_select(first_moments[:, :, 4:9], 0, dst)
            * tensor_basis[:, None, :],
            dim=-1,
        )
        tensor_feedback = (
            tensor_feedback
            - normalized_amplitude
            * torch.sum(
                tensor_basis * tensor_basis,
                dim=-1,
            )[:, None]
        )
        return scalar_feedback, vector_feedback, tensor_feedback

    def _mix_angular_feedback(
        self,
        degree_feedback: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Mix angular degrees with fused PyTorch multiply-add operations.

        Parameters
        ----------
        degree_feedback
            Scalar, vector, and tensor edge responses, each with shape
            ``(E, channels)``.

        Returns
        -------
        torch.Tensor
            Tanh-activated feedback with shape ``(E, channels)``.
        """
        weight = self.adam_degree_weight
        argument = degree_feedback[0] * weight[None, :, 0]
        argument = torch.addcmul(
            argument,
            degree_feedback[1],
            weight[None, :, 1],
        )
        argument = torch.addcmul(
            argument,
            degree_feedback[2],
            weight[None, :, 2],
        )
        return torch.tanh(argument)

    def _contract_l2_invariants(
        self,
        tensor: torch.Tensor,
        vector: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Contract packed node states with fused multiply-add operations.

        Parameters
        ----------
        tensor
            Packed degree-two coefficients with shape ``(N, channels, 5)``.
        vector
            Degree-one coefficients with shape ``(N, channels, 3)``.

        Returns
        -------
        tensor_norm, vector_tensor_vector, tensor_trace3, vector_tensor2_vector
            Four parity-even invariants with shape ``(N, channels)``.
        """
        inv_sqrt_two = 1.0 / math.sqrt(2.0)
        inv_sqrt_six = 1.0 / math.sqrt(6.0)
        q0, q1, q2, q3, q4 = tensor.unbind(dim=-1)
        qxy = q0 * inv_sqrt_two
        qyz = q1 * inv_sqrt_two
        qxz = q3 * inv_sqrt_two
        q2_scaled = q2 * inv_sqrt_six
        q4_scaled = q4 * inv_sqrt_two
        qxx = -q2_scaled + q4_scaled
        qyy = -q2_scaled - q4_scaled
        qzz = 2.0 * q2_scaled
        vx, vy, vz = vector.unbind(dim=-1)

        qv_x = torch.addcmul(qxx * vx, qxy, vy)
        qv_x = torch.addcmul(qv_x, qxz, vz)
        qv_y = torch.addcmul(qxy * vx, qyy, vy)
        qv_y = torch.addcmul(qv_y, qyz, vz)
        qv_z = torch.addcmul(qxz * vx, qyz, vy)
        qv_z = torch.addcmul(qv_z, qzz, vz)

        tensor_norm = torch.sum(tensor * tensor, dim=-1)
        vector_tensor_vector = torch.addcmul(vx * qv_x, vy, qv_y)
        vector_tensor_vector = torch.addcmul(
            vector_tensor_vector,
            vz,
            qv_z,
        )
        vector_tensor2_vector = torch.addcmul(qv_x * qv_x, qv_y, qv_y)
        vector_tensor2_vector = torch.addcmul(
            vector_tensor2_vector,
            qv_z,
            qv_z,
        )

        determinant = qxx * qyy * qzz
        determinant = torch.addcmul(
            determinant,
            qxy * qxz,
            qyz,
            value=2.0,
        )
        determinant = torch.addcmul(
            determinant,
            qxx * qyz,
            qyz,
            value=-1.0,
        )
        determinant = torch.addcmul(
            determinant,
            qyy * qxz,
            qxz,
            value=-1.0,
        )
        determinant = torch.addcmul(
            determinant,
            qzz * qxy,
            qxy,
            value=-1.0,
        )
        tensor_trace3 = 3.0 * determinant
        return (
            tensor_norm,
            vector_tensor_vector,
            tensor_trace3,
            vector_tensor2_vector,
        )

    def share_params(
        self,
        base_class: Any,
        shared_level: int,
        model_prob: float = 1.0,
        resume: bool = False,
    ) -> None:
        """Share all DPA4C modules and statistic buffers.

        Parameters
        ----------
        base_class
            DPA4C descriptor that owns the shared module tree.
        shared_level
            Sharing level. Only complete sharing at level ``0`` is supported.
        model_prob
            Model sampling probability accepted by the multitask ABI; unused.
        resume
            Whether the operation occurs while restoring a checkpoint; unused.

        Raises
        ------
        TypeError
            If ``base_class`` is not the same descriptor class.
        NotImplementedError
            If ``shared_level`` is not zero.
        """
        del model_prob, resume
        if self.__class__ != base_class.__class__:
            raise TypeError("Only DPA4C descriptors can share parameters.")
        if shared_level != 0:
            raise NotImplementedError("DPA4C supports only shared_level=0.")
        for name in self._modules:
            self._modules[name] = base_class._modules[name]
        for name in self._buffers:
            self._buffers[name] = base_class._buffers[name]
        if "adam_degree_weight" in self._parameters:
            self._parameters["adam_degree_weight"] = base_class._parameters[
                "adam_degree_weight"
            ]

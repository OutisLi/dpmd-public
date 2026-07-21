# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Compact and Compressible DPA4 descriptor.

DPA4C is a graph-native local descriptor for lightweight training and
distillation from DPA4. It reuses the DPA4 radial basis, bias-free radial
network, type embedding, and C³ cutoff envelope, while replacing equivariant
message passing with two center-local moment reductions.

For every edge :math:`j\to i`, a learned radial/type amplitude
:math:`a_{ijc}` is coupled to a normalized packed Cartesian basis
:math:`B^{(\ell)}(\hat{\mathbf r}_{ij})` for :math:`\ell=0,1,2`. The first
moment reduction is projected back onto each edge, yielding an exact
factorization of a low-rank three-body Legendre contraction. A residual tanh
gate modulates the edge amplitudes before the second moment reduction. The
resulting scalar, vector, and symmetric-traceless tensor channels are converted
to six parity-even node invariants.

The forward equations have :math:`O(E C + N C^2)` complexity for ``E`` graph
edges, ``N`` nodes, and ``C`` channels. They do not construct neighbor pairs,
edge tensor products, Wigner rotations, attention matrices, or intermediate
source-node features.
"""

from __future__ import (
    annotations,
)

import dataclasses
import math
from typing import (
    TYPE_CHECKING,
    Any,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    xp_asarray_nodetach,
)
from deepmd.dpmodel.common import (
    cast_precision,
    get_xp_precision,
    to_numpy_array,
)
from deepmd.dpmodel.utils import (
    NativeLayer,
    PairExcludeMask,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.dpmodel.utils.update_sel import (
    UpdateSel,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_descriptor import (
    BaseDescriptor,
)
from .dpa4_nn import (
    C3CutoffEnvelope,
    RadialBasis,
    RadialMLP,
    SeZMTypeEmbedding,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

    from deepmd.dpmodel.array_api import (
        Array,
    )
    from deepmd.utils.data_system import (
        DeepmdDataSystem,
    )
    from deepmd.utils.path import (
        DPPath,
    )


def _build_l2_basis(direction: Array) -> Array:
    r"""Build the normalized packed Cartesian basis through degree two.

    For a direction :math:`\mathbf u=(x,y,z)`, the returned rows are

    .. math::

       B^{(0)}(\mathbf u) &= [1],\\
       B^{(1)}(\mathbf u) &= [x,y,z],\\
       B^{(2)}(\mathbf u) &=
       [\sqrt3xy,\sqrt3yz,\tfrac12(3z^2-q),
        \sqrt3xz,\tfrac{\sqrt3}{2}(x^2-y^2)],

    where :math:`q=x^2+y^2+z^2`. For unit vectors, each degree block satisfies

    .. math::

       B^{(\ell)}(\mathbf u)\cdot B^{(\ell)}(\mathbf v)
       =P_\ell(\mathbf u\cdot\mathbf v).

    Parameters
    ----------
    direction
        Edge directions with shape ``(E, 3)``. Valid edges are regularized
        unit vectors; masked edges are zero vectors.

    Returns
    -------
    Array
        Packed basis with shape ``(E, 9)`` and degree slices ``[0:1]``,
        ``[1:4]``, and ``[4:9]``.
    """
    xp = array_api_compat.array_namespace(direction)
    x, y, z = direction[:, 0], direction[:, 1], direction[:, 2]
    q = x * x + y * y + z * z
    sqrt_three = math.sqrt(3.0)
    return xp.stack(
        [
            xp.ones_like(x),
            x,
            y,
            z,
            sqrt_three * x * y,
            sqrt_three * y * z,
            0.5 * (3.0 * z * z - q),
            sqrt_three * x * z,
            0.5 * sqrt_three * (x * x - y * y),
        ],
        axis=-1,
    )


def _packed_l2_components(
    packed: Array,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Convert packed degree-two coefficients to independent STF entries.

    Parameters
    ----------
    packed
        Degree-two coefficients with shape ``(..., 5)``.

    Returns
    -------
    qxx, qyy, qzz, qxy, qxz, qyz
        Independent entries of the corresponding symmetric-traceless matrix,
        each with shape ``packed.shape[:-1]``.
    """
    inv_sqrt_two = 1.0 / math.sqrt(2.0)
    inv_sqrt_six = 1.0 / math.sqrt(6.0)
    q0, q1, q2, q3, q4 = (packed[..., ii] for ii in range(5))
    qxy = q0 * inv_sqrt_two
    qyz = q1 * inv_sqrt_two
    qxz = q3 * inv_sqrt_two
    qxx = -q2 * inv_sqrt_six + q4 * inv_sqrt_two
    qyy = -q2 * inv_sqrt_six - q4 * inv_sqrt_two
    qzz = 2.0 * q2 * inv_sqrt_six
    return qxx, qyy, qzz, qxy, qxz, qyz


def _packed_l2_to_stf(packed: Array) -> Array:
    r"""Convert packed degree-two coefficients to STF matrices.

    The five packed coefficients are components in an orthonormal basis of
    symmetric-traceless ``3 x 3`` matrices. The conversion preserves the
    invariant inner product:

    .. math::

       \mathbf q\cdot\mathbf p
       = Q(\mathbf q):Q(\mathbf p).

    Parameters
    ----------
    packed
        Degree-two coefficients with shape ``(..., 5)``.

    Returns
    -------
    Array
        Symmetric-traceless matrices with shape ``(..., 3, 3)``.
    """
    xp = array_api_compat.array_namespace(packed)
    qxx, qyy, qzz, qxy, qxz, qyz = _packed_l2_components(packed)
    return xp.stack(
        [
            xp.stack([qxx, qxy, qxz], axis=-1),
            xp.stack([qxy, qyy, qyz], axis=-1),
            xp.stack([qxz, qyz, qzz], axis=-1),
        ],
        axis=-2,
    )


def _packed_l2_contractions(
    packed: Array,
    vector: Array,
) -> tuple[Array, Array, Array, Array]:
    r"""Contract degree-two coefficients without materializing STF matrices.

    The packed basis is orthonormal, so ``Q:Q`` is the packed Euclidean norm.
    The remaining contractions are evaluated from the six independent matrix
    entries. For a symmetric matrix, the cubic trace is

    .. math::

       \operatorname{tr}(Q^3)
       = \sum_i Q_{ii}^3
       + 3\sum_{i<j}(Q_{ii}+Q_{jj})Q_{ij}^2
       + 6Q_{xy}Q_{xz}Q_{yz}.

    Parameters
    ----------
    packed
        Degree-two coefficients with shape ``(..., 5)``.
    vector
        Degree-one coefficients with shape ``(..., 3)``.

    Returns
    -------
    tensor_norm
        ``Q:Q`` with shape ``packed.shape[:-1]``.
    vector_tensor_vector
        ``v^T Q v`` with shape ``packed.shape[:-1]``.
    tensor_trace3
        ``tr(Q^3)`` with shape ``packed.shape[:-1]``.
    vector_tensor2_vector
        ``v^T Q^2 v`` with shape ``packed.shape[:-1]``.
    """
    xp = array_api_compat.array_namespace(packed, vector)
    qxx, qyy, qzz, qxy, qxz, qyz = _packed_l2_components(packed)
    vx, vy, vz = vector[..., 0], vector[..., 1], vector[..., 2]

    qv_x = qxx * vx + qxy * vy + qxz * vz
    qv_y = qxy * vx + qyy * vy + qyz * vz
    qv_z = qxz * vx + qyz * vy + qzz * vz

    tensor_norm = xp.sum(packed * packed, axis=-1)
    vector_tensor_vector = vx * qv_x + vy * qv_y + vz * qv_z
    vector_tensor2_vector = qv_x * qv_x + qv_y * qv_y + qv_z * qv_z
    tensor_trace3 = (
        qxx * qxx * qxx
        + qyy * qyy * qyy
        + qzz * qzz * qzz
        + 3.0
        * ((qxx + qyy) * qxy * qxy + (qxx + qzz) * qxz * qxz + (qyy + qzz) * qyz * qyz)
        + 6.0 * qxy * qxz * qyz
    )
    return (
        tensor_norm,
        vector_tensor_vector,
        tensor_trace3,
        vector_tensor2_vector,
    )


@BaseDescriptor.register("dpa4c")
class DescrptDPA4C(NativeOP, BaseDescriptor):
    r"""Construct the Compact and Compressible DPA4 descriptor.

    Let :math:`\chi(r)` denote the fixed DPA4 C³ envelope, :math:`R(r)` the
    DPA4 radial basis, and :math:`T_z` the type embedding. The edge amplitude
    is

    .. math::

       a_{ijc}=\chi(r_{ij})
       \left[\operatorname{RadialMLP}(R(r_{ij}))_c
       +T_{z_i,c}+T_{z_j,c}\right].

    The smooth destination degree and its DPA4 normalization are

    .. math::

       d_i=\sum_j\chi(r_{ij})^2,\qquad
       n_i=(d_i+0.25)^{-1/2}.

    The first packed moments are

    .. math::

       X_{ick}=n_i\sum_j a_{ijc}B_k(\hat{\mathbf r}_{ij}).

    Projecting each degree block back onto edge ``ij`` and removing its own
    diagonal contribution gives

    .. math::

       c_{ijc}^{(\ell)}
       &=B^{(\ell)}(\hat{\mathbf r}_{ij})
         \cdot X_{ic}^{(\ell)}
         -n_i a_{ijc}\lVert B^{(\ell)}(\hat{\mathbf r}_{ij})\rVert^2\\
       &=n_i\sum_{k\ne j}a_{ikc}
         P_\ell(\hat{\mathbf r}_{ij}\cdot\hat{\mathbf r}_{ik}).

    Every channel has three independent degree weights
    :math:`w_{c0},w_{c1},w_{c2}`. The residual tanh modulation preserves the
    two-body signal in a one-neighbor environment and bounds each edge gate to
    ``(0, 2)``:

    .. math::

       b_{ijc}=a_{ijc}\left[1+
       \tanh\left(
       \sum_{\ell=0}^2w_{c\ell}c_{ijc}^{(\ell)}\right)\right].

    The second moments :math:`Z_{ick}=n_i\sum_jb_{ijc}B_k` remain equivariant
    until the node-only readout constructs six parity-even invariants:
    :math:`s`, :math:`v\cdot v`, :math:`Q:Q`, :math:`v^TQv`,
    :math:`\operatorname{tr}(Q^3)`, and :math:`v^TQ^2v`.

    Parameters
    ----------
    rcut
        Outer cutoff radius in Å.
    ntypes
        Number of atom types.
    channels
        Channels per packed angular coefficient. The descriptor exposes six
        invariant families and therefore has width ``6 * channels``.
    basis_type
        DPA4 radial basis type: ``"bessel"`` or ``"gaussian"``.
    n_radial
        Number of DPA4 radial basis functions.
    radial_mlp
        Hidden widths of the bias-free radial network. A value of ``0`` is
        replaced by ``channels``. The output layer is appended automatically.
    activation_function
        Hidden activation of the radial network.
    exclude_types
        Ordered atom-type pairs excluded from the descriptor.
    precision
        Floating-point precision of descriptor parameters.
    trainable
        Whether descriptor parameters are trainable.
    type_map
        Atom-type names.
    seed
        Random seed.
    spin
        Reserved for descriptor API compatibility; only ``None`` is supported.

    Raises
    ------
    NotImplementedError
        If ``spin`` is not ``None``.
    ValueError
        If ``rcut``, ``channels``, or ``n_radial`` is not positive.
    """

    _update_sel_cls = UpdateSel
    _ANGULAR_DIM = 9
    _ENVELOPE_EXPONENT = 5
    _DEGREE_NORM_FLOOR = 0.25
    _EPS = 1.0e-7
    _LEGACY_NLIST_CAPACITY = 256

    def __init__(
        self,
        rcut: float,
        ntypes: int,
        channels: int = 8,
        basis_type: str = "bessel",
        n_radial: int = 16,
        radial_mlp: list[int] | None = None,
        activation_function: str = "silu",
        exclude_types: list[tuple[int, int]] = [],
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
        type_map: list[str] | None = None,
        seed: int | list[int] | None = None,
        spin: None = None,
    ) -> None:
        # === Step 1. Validate the public architecture contract ===
        if spin is not None:
            raise NotImplementedError("DPA4C does not support spin inputs.")
        if rcut <= 0.0:
            raise ValueError(f"`rcut` must be positive, got {rcut}")
        if channels <= 0:
            raise ValueError(f"`channels` must be positive, got {channels}")
        if n_radial <= 0:
            raise ValueError(f"`n_radial` must be positive, got {n_radial}")

        # === Step 2. Resolve scalar configuration ===
        self.rcut = float(rcut)
        self.ntypes = int(ntypes)
        self.channels = int(channels)
        self.basis_type = str(basis_type).lower()
        self.n_radial = int(n_radial)
        if radial_mlp is None:
            radial_mlp = [0]
        self.radial_mlp = [
            self.channels if width == 0 else int(width) for width in radial_mlp
        ]
        self.activation_function = str(activation_function)
        self.precision = precision
        self.trainable = bool(trainable)
        self.type_map = type_map
        self.seed = seed

        # === Step 3. Build the shared DPA4 edge representation ===
        # The type embedding and every learned linear map are bias-free. The
        # radial basis is raw; one separate p=5 DPA4 envelope gates the complete
        # radial-plus-type edge amplitude.
        self.type_embedding = SeZMTypeEmbedding(
            ntypes=self.ntypes,
            embed_dim=self.channels,
            precision=self.precision,
            seed=child_seed(seed, 0),
            trainable=self.trainable,
            padding=True,
        )
        self.radial_basis = RadialBasis(
            rcut=self.rcut,
            basis_type=self.basis_type,
            n_radial=self.n_radial,
            precision=self.precision,
            exponent=self._ENVELOPE_EXPONENT,
            apply_envelope=False,
        )
        self.radial_embedding = RadialMLP(
            [self.n_radial, *self.radial_mlp, self.channels],
            activation_function=self.activation_function,
            precision=self.precision,
            trainable=self.trainable,
            radial_norm=False,
            seed=child_seed(seed, 1),
        )
        self.edge_envelope = C3CutoffEnvelope(
            rcut=self.rcut,
            exponent=self._ENVELOPE_EXPONENT,
            precision=self.precision,
        )

        # === Step 4. Build the channel-wise angular-feedback gate ===
        # Each radial channel learns an independent mixture over l=0,1,2.
        # Small initial weights keep 1+tanh(h) close to its identity value.
        degree_rng = np.random.default_rng(child_seed(seed, 2))
        self.adam_degree_weight = degree_rng.normal(
            scale=0.1,
            size=(self.channels, 3),
        ).astype(PRECISION_DICT[self.precision])

        # === Step 5. Build the node-only invariant projections ===
        # Degree-specific channel projections preserve the SO(3) coefficient
        # axes. The six invariant families remain explicit for the fitting net.
        self.scalar_projection = NativeLayer(
            self.channels,
            self.channels,
            bias=False,
            precision=self.precision,
            seed=child_seed(seed, 4),
            trainable=self.trainable,
        )
        self.vector_projection = NativeLayer(
            self.channels,
            self.channels,
            bias=False,
            precision=self.precision,
            seed=child_seed(seed, 5),
            trainable=self.trainable,
        )
        self.tensor_projection = NativeLayer(
            self.channels,
            self.channels,
            bias=False,
            precision=self.precision,
            seed=child_seed(seed, 6),
            trainable=self.trainable,
        )
        # === Step 6. Initialize interface state ===
        # DPA4C uses analytic radial normalization, so these empty arrays exist
        # only to satisfy the common descriptor statistics contract.
        self.mean = np.zeros(0, dtype=PRECISION_DICT[self.precision])
        self.stddev = np.ones(0, dtype=PRECISION_DICT[self.precision])
        self.compress = False
        self.reinit_exclude(exclude_types)

    def call_graph(
        self,
        graph: Any,
        atype: Array,
        type_embedding: Array | None = None,
        comm_dict: dict | None = None,
    ) -> tuple[Array, None]:
        """Evaluate DPA4C on a flat neighbor graph.

        Parameters
        ----------
        graph
            Neighbor graph containing ``edge_index`` with shape ``(2, E)``,
            ``edge_vec`` with shape ``(E, 3)`` in Å, and ``edge_mask`` with
            shape ``(E,)``.
        atype
            Flat node types with shape ``(N,)``. Padding nodes use type index
            ``ntypes`` and therefore gather the zero type-embedding row.
        type_embedding
            Optional precomputed DPA4 type table with shape
            ``(ntypes + 1, channels)``. If omitted, the descriptor evaluates
            its type-embedding module.
        comm_dict
            Communication metadata accepted by the common graph ABI. DPA4C
            does not read source-node features, so no halo-feature exchange is
            required and this argument is unused.

        Returns
        -------
        descriptor
            Rotation- and permutation-invariant node features with shape
            ``(N, 6 * channels)`` and the same floating dtype as ``edge_vec``.
        rot_mat
            ``None``. DPA4C does not expose an equivariant fitting input.
        """
        del comm_dict
        # === Step 1. Resolve type features and compute precision ===
        if type_embedding is None:
            type_embedding = self.type_embedding.call()
        xp = array_api_compat.array_namespace(graph.edge_vec)
        in_dtype = graph.edge_vec.dtype
        compute_dtype = get_xp_precision(xp, self.precision)
        if in_dtype != compute_dtype:
            graph = dataclasses.replace(
                graph,
                edge_vec=xp.astype(graph.edge_vec, compute_dtype),
            )

        # === Step 2. Evaluate the graph-native equations ===
        descriptor, _ = self._call_graph(graph, atype, type_embedding)

        # === Step 3. Restore the graph input dtype ===
        if descriptor.dtype != in_dtype:
            descriptor = xp.astype(descriptor, in_dtype)
        return descriptor, None

    @cast_precision
    def call(
        self,
        coord_ext: Array,
        atype_ext: Array,
        nlist: Array,
        mapping: Array | None = None,
        fparam: Array | None = None,
        comm_dict: dict | None = None,
        charge_spin: Array | None = None,
    ) -> tuple[Array, None, None, None, Array]:
        """Adapt a bounded dense neighbor list to the graph-native equations.

        This method exists for the common descriptor ABI and numerical
        reference tests. Production DPA4C execution uses :meth:`call_graph`
        with a carry-all graph. A rectangular list at the internal compatibility
        capacity is rejected because its completeness cannot be established.

        Parameters
        ----------
        coord_ext
            Extended coordinates with shape ``(F, N_all, 3)`` or
            ``(F, 3 * N_all)`` in Å.
        atype_ext
            Extended atom types with shape ``(F, N_all)``.
        nlist
            Bounded neighbor list with shape ``(F, N_local, N_slot)``. Negative
            indices denote padding.
        mapping
            Extended-to-local owner mapping with shape ``(F, N_all)``. ``None``
            denotes the identity mapping.
        fparam
            Frame parameters accepted by the common descriptor ABI; unused.
        comm_dict
            Communication metadata accepted by the common descriptor ABI;
            unused.
        charge_spin
            Charge/spin conditioning accepted by the common descriptor ABI;
            unsupported and unused.

        Returns
        -------
        descriptor
            Invariant features with shape ``(F, N_local, 6 * channels)``.
        rot_mat
            ``None``.
        g2
            ``None``.
        h2
            ``None``.
        envelope
            Per-slot C³ envelope with shape
            ``(F, N_local, N_slot, 1)``.

        Raises
        ------
        ValueError
            If ``N_slot`` reaches the fixed legacy capacity, where the adapter
            cannot prove that no physical neighbor was truncated.
        """
        from deepmd.dpmodel.utils.neighbor_graph import (
            graph_from_dense_quartet,
        )

        del fparam, comm_dict, charge_spin
        xp = array_api_compat.array_namespace(coord_ext, atype_ext, nlist)
        nf, nloc, nnei = nlist.shape

        # === Step 1. Reject an ambiguous fixed-capacity environment ===
        if nnei >= self._LEGACY_NLIST_CAPACITY:
            raise ValueError(
                "DPA4C does not execute through the legacy fixed-capacity "
                "neighbor list. Use the carry-all graph-native path."
            )

        # === Step 2. Convert the dense quartet without compacting its edge axis ===
        graph, atype_local = graph_from_dense_quartet(
            coord_ext,
            atype_ext,
            nlist,
            mapping,
        )

        # === Step 3. Evaluate the same graph-native equations ===
        descriptor, envelope = self._call_graph(
            graph,
            atype_local,
            self.type_embedding.call(),
        )

        # === Step 4. Restore the common dense descriptor ABI ===
        descriptor = xp.reshape(
            descriptor,
            (nf, nloc, descriptor.shape[-1]),
        )
        envelope = xp.reshape(envelope, (nf, nloc, nnei, 1))
        return descriptor, None, None, None, envelope

    def _call_graph(
        self,
        graph: Any,
        atype: Array,
        type_embedding: Array,
    ) -> tuple[Array, Array]:
        r"""Evaluate the factorized moment-feedback equations.

        The method implements

        .. math::

           a_{ec} &\xrightarrow{\operatorname{segment\_sum}}
           X_{ick}\\
           &\xrightarrow{B_e^{(\ell)}\cdot X_i^{(\ell)}
             -\text{self}}
           c_{ec\ell}\\
           &\xrightarrow{\tanh}
           b_{ec}\\
           &\xrightarrow{\operatorname{segment\_sum}}
           Z_{ick}
           \xrightarrow{\text{node invariants}}D_{ic}.

        Parameters
        ----------
        graph
            Neighbor graph in descriptor compute precision.
        atype
            Flat node types with shape ``(N,)``.
        type_embedding
            Complete type table with shape ``(ntypes + 1, channels)``.

        Returns
        -------
        descriptor
            Invariant node features with shape ``(N, 6 * channels)``.
        envelope
            Masked per-edge C³ envelope with shape ``(E, 1)``.
        """
        xp = array_api_compat.array_namespace(graph.edge_vec)

        # === Step 1. Place the precomputed type table in the graph namespace ===
        # A converted dpmodel may already store the table in the active
        # namespace. Conversion is required only for a direct NumPy-defined
        # descriptor evaluated with JAX or another array backend.
        type_namespace = array_api_compat.array_namespace(type_embedding)
        if type_namespace is not xp:
            type_embedding = xp.asarray(
                type_embedding,
                dtype=graph.edge_vec.dtype,
                device=array_api_compat.device(graph.edge_vec),
            )
        dst = graph.edge_index[1]
        n_total = atype.shape[0]
        node_type = self._gather(type_embedding, atype, xp)
        edge_type_table = xp.reshape(
            type_embedding[:, None, :] + type_embedding[None, :, :],
            (-1, self.channels),
        )

        # === Step 2. Build edge amplitudes and packed angular coefficients ===
        amplitude, basis, envelope = self._edge_features(
            graph,
            atype,
            edge_type_table,
        )

        # === Step 3. Compute the shared DPA4 smooth-degree normalization ===
        normalizer = self._degree_normalizer(
            envelope,
            dst,
            n_total,
        )

        # === Step 4. Aggregate the first center moments X ===
        first_moments = self._moments(
            amplitude,
            basis,
            dst,
            n_total,
            normalizer,
        )
        normalizer_on_edge = self._gather(normalizer, dst, xp)

        # === Step 5. Project X back onto each edge and remove e=f ===
        # For degree l, the contraction equals the weighted sum of
        # P_l(cos(theta_ef)) over edges f sharing the same destination. The
        # explicit self term removes f=e without constructing an edge-pair list.
        degree_feedback = self._angular_feedback(
            first_moments,
            amplitude,
            basis,
            dst,
            normalizer_on_edge,
        )

        # === Step 6. Apply the residual edge-wise angular modulation ===
        # The channel-wise weights act only on the three degree responses and
        # therefore preserve the compact diagonal channel structure.
        feedback = self._mix_angular_feedback(degree_feedback)
        modulated = amplitude * (1.0 + feedback)

        # === Step 7. Aggregate the feedback-conditioned moments Z ===
        second_moments = self._moments(
            modulated,
            basis,
            dst,
            n_total,
            normalizer,
        )

        # === Step 8. Add the on-site DPA4 type state and invariantize ===
        return (
            self._invariant_readout(second_moments, node_type),
            envelope[:, None],
        )

    def _edge_features(
        self,
        graph: Any,
        atype: Array,
        edge_type_table: Array,
    ) -> tuple[Array, Array, Array]:
        """Build DPA4 radial/type amplitudes and packed angular coefficients.

        Parameters
        ----------
        graph
            Neighbor graph in descriptor compute precision.
        atype
            Flat node types with shape ``(N,)``.
        edge_type_table
            Pairwise sums of type embeddings with shape
            ``((ntypes + 1) ** 2, channels)``.

        Returns
        -------
        amplitude
            Masked edge amplitudes with shape ``(E, channels)``.
        basis
            Masked packed Cartesian basis with shape ``(E, 9)``.
        envelope
            Masked C³ envelope with shape ``(E,)``.
        """
        from deepmd.dpmodel.utils.neighbor_graph import (
            apply_pair_exclusion,
        )

        # === Step 1. Merge graph and descriptor-level exclusion masks ===
        graph = apply_pair_exclusion(graph, atype, self.emask)
        xp = array_api_compat.array_namespace(graph.edge_vec)
        src, dst = graph.edge_index[0], graph.edge_index[1]
        center_type = self._gather(atype, dst, xp)
        neighbor_type = self._gather(atype, src, xp)

        # === Step 2. Build regularized distances and directions ===
        # sqrt(r^2 + eps^2) keeps the direction finite for coincident or guard
        # edges. Valid physical edges are unaffected above the 1e-7 Å scale.
        distance_squared = xp.sum(
            graph.edge_vec * graph.edge_vec,
            axis=-1,
            keepdims=True,
        )
        distance = xp.sqrt(distance_squared + self._EPS * self._EPS)
        direction = graph.edge_vec / distance
        real_type = (center_type < self.ntypes) & (neighbor_type < self.ntypes)
        edge_mask = graph.edge_mask & real_type
        mask = xp.astype(edge_mask[:, None], graph.edge_vec.dtype)

        # === Step 3. Evaluate the shared DPA4 radial representation ===
        # DPA4C requests the raw radial basis. One explicit C³ envelope gates
        # the combined radial and type feature, so the edge amplitude contains
        # exactly one cutoff factor and vanishes smoothly at rcut.
        envelope = self._cutoff_envelope(distance) * mask
        radial_basis = self.radial_basis.call(distance)
        radial = self.radial_embedding.call(radial_basis)
        pair_index = center_type * (self.ntypes + 1) + neighbor_type
        edge_type = self._gather(
            edge_type_table,
            pair_index,
            xp,
        )
        amplitude = (radial + edge_type) * envelope

        # === Step 4. Build the masked l=0,1,2 Cartesian basis ===
        basis = self._angular_basis(direction) * mask
        return amplitude, basis, envelope[:, 0]

    def _gather(
        self,
        values: Array,
        index: Array,
        xp: Any | None = None,
    ) -> Array:
        """Gather rows from a node- or type-indexed array.

        Parameters
        ----------
        values
            Source array with indexed leading axis.
        index
            Integer row indices.
        xp
            Active array namespace. If omitted, it is inferred from ``values``.

        Returns
        -------
        Array
            Gathered rows with shape ``index.shape + values.shape[1:]``.
        """
        if xp is None:
            xp = array_api_compat.array_namespace(values)
        return xp.take(values, index, axis=0)

    def _cutoff_envelope(self, distance: Array) -> Array:
        """Evaluate the fixed C³ edge envelope.

        Parameters
        ----------
        distance
            Edge distances with shape ``(E, 1)`` in Å.

        Returns
        -------
        Array
            Envelope values with shape ``(E, 1)``.
        """
        return self.edge_envelope.call(distance)

    def _angular_basis(self, direction: Array) -> Array:
        """Build the packed Cartesian basis for edge directions.

        Parameters
        ----------
        direction
            Edge directions with shape ``(E, 3)``.

        Returns
        -------
        Array
            Packed basis with shape ``(E, 9)``.
        """
        return _build_l2_basis(direction)

    def _degree_normalizer(
        self,
        envelope: Array,
        dst: Array,
        n_total: int,
    ) -> Array:
        r"""Compute the DPA4 smooth inverse-square-root degree.

        The normalization is

        .. math::

           n_i=\left(\sum_{e:\operatorname{dst}(e)=i}\chi_e^2+0.25\right)^{-1/2}.

        The additive floor, rather than a hard maximum, preserves smooth first
        and higher derivatives as the local degree crosses ``0.25``.

        Parameters
        ----------
        envelope
            Masked edge envelopes with shape ``(E,)``.
        dst
            Destination node indices with shape ``(E,)``.
        n_total
            Number of output nodes ``N``.

        Returns
        -------
        Array
            Node normalization factors with shape ``(N, 1)``.
        """
        from deepmd.dpmodel.utils.neighbor_graph import (
            segment_sum,
        )

        xp = array_api_compat.array_namespace(envelope)
        degree = segment_sum(
            (envelope * envelope)[:, None],
            dst,
            n_total,
        )
        return 1.0 / xp.sqrt(degree + self._DEGREE_NORM_FLOOR)

    def _moments(
        self,
        edge_amplitude: Array,
        basis: Array,
        dst: Array,
        n_total: int,
        normalizer: Array,
    ) -> Array:
        r"""Aggregate normalized packed moments on destination nodes.

        Parameters
        ----------
        edge_amplitude
            Per-edge channel amplitudes with shape ``(E, channels)``.
        basis
            Packed angular basis with shape ``(E, 9)``.
        dst
            Destination node indices with shape ``(E,)``.
        n_total
            Number of output nodes ``N``.
        normalizer
            Smooth inverse-square-root degree with shape ``(N, 1)``.

        Returns
        -------
        Array
            Normalized moments with shape ``(N, channels, 9)``.
        """
        from deepmd.dpmodel.utils.neighbor_graph import (
            segment_sum,
        )

        # The outer product is edge-local and does not couple two neighbors.
        outer = edge_amplitude[:, :, None] * basis[:, None, :]
        moments = segment_sum(outer, dst, n_total)
        return moments * normalizer[:, :, None]

    def _angular_feedback(
        self,
        first_moments: Array,
        amplitude: Array,
        basis: Array,
        dst: Array,
        normalizer_on_edge: Array,
    ) -> tuple[Array, Array, Array]:
        """Project packed moments onto edges degree by degree.

        The degree slices are gathered independently so the forward path never
        materializes the full first-moment tensor with shape
        ``(E, channels, 9)``.

        Parameters
        ----------
        first_moments
            First normalized center moments with shape ``(N, channels, 9)``.
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
            Self-free edge responses for degrees zero, one, and two, each with
            shape ``(E, channels)``.
        """
        xp = array_api_compat.array_namespace(first_moments)
        normalized_amplitude = amplitude * normalizer_on_edge

        scalar_basis = basis[:, 0]
        scalar_feedback = (
            self._gather(first_moments[:, :, 0], dst, xp) * scalar_basis[:, None]
            - normalized_amplitude * (scalar_basis * scalar_basis)[:, None]
        )

        vector_basis = basis[:, 1:4]
        vector_feedback = xp.sum(
            self._gather(first_moments[:, :, 1:4], dst, xp) * vector_basis[:, None, :],
            axis=-1,
        )
        vector_feedback = (
            vector_feedback
            - normalized_amplitude
            * xp.sum(
                vector_basis * vector_basis,
                axis=-1,
            )[:, None]
        )

        tensor_basis = basis[:, 4:9]
        tensor_feedback = xp.sum(
            self._gather(first_moments[:, :, 4:9], dst, xp) * tensor_basis[:, None, :],
            axis=-1,
        )
        tensor_feedback = (
            tensor_feedback
            - normalized_amplitude
            * xp.sum(
                tensor_basis * tensor_basis,
                axis=-1,
            )[:, None]
        )
        return scalar_feedback, vector_feedback, tensor_feedback

    def _mix_angular_feedback(
        self,
        degree_feedback: tuple[Array, Array, Array],
    ) -> Array:
        """Mix the three angular degrees without an ``(E, C, 3)`` tensor.

        Parameters
        ----------
        degree_feedback
            Scalar, vector, and tensor edge responses, each with shape
            ``(E, channels)``.

        Returns
        -------
        Array
            Tanh-activated feedback with shape ``(E, channels)``.
        """
        xp = array_api_compat.array_namespace(degree_feedback[0])
        weight = xp_asarray_nodetach(
            xp,
            self.adam_degree_weight,
            device=array_api_compat.device(degree_feedback[0]),
        )
        argument = (
            degree_feedback[0] * weight[None, :, 0]
            + degree_feedback[1] * weight[None, :, 1]
            + degree_feedback[2] * weight[None, :, 2]
        )
        return xp.tanh(argument)

    def _contract_l2_invariants(
        self,
        tensor: Array,
        vector: Array,
    ) -> tuple[Array, Array, Array, Array]:
        """Contract degree-two and degree-one node states.

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
        return _packed_l2_contractions(tensor, vector)

    def _invariant_readout(
        self,
        moments: Array,
        center_type: Array,
    ) -> Array:
        r"""Construct the six parity-even DPA4C node invariants.

        After degree-specific channel projections, let ``s`` be the scalar
        state, ``v`` the vector state, and ``Q`` the symmetric-traceless
        matrix state. The readout contains

        .. math::

           [s,\;v\cdot v,\;Q:Q,\;v^TQv,\;\operatorname{tr}(Q^3),\;v^TQ^2v].

        The final invariant is independent of the preceding five and resolves
        the orientation of ``v`` relative to the squared eigenspaces of ``Q``.

        Parameters
        ----------
        moments
            Feedback-conditioned packed moments with shape
            ``(N, channels, 9)``.
        center_type
            On-site DPA4 type embeddings with shape ``(N, channels)``.

        Returns
        -------
        Array
            Invariant descriptor with shape ``(N, 6 * channels)``.
        """
        xp = array_api_compat.array_namespace(moments)

        # === Step 1. Mix channels independently within each degree ===
        scalar = self.scalar_projection.call(moments[:, :, 0]) + center_type
        vector = self.vector_projection.call(
            xp.permute_dims(moments[:, :, 1:4], (0, 2, 1))
        )
        vector = xp.permute_dims(vector, (0, 2, 1))
        tensor = self.tensor_projection.call(
            xp.permute_dims(moments[:, :, 4:9], (0, 2, 1))
        )
        tensor = xp.permute_dims(tensor, (0, 2, 1))

        # === Step 2. Evaluate the integrity basis of parity-even invariants ===
        vector_norm = xp.sum(vector * vector, axis=-1)
        (
            tensor_norm,
            vector_tensor_vector,
            tensor_trace3,
            vector_tensor2_vector,
        ) = self._contract_l2_invariants(tensor, vector)

        # === Step 3. Project the six invariant families to descriptor channels ===
        invariants = xp.stack(
            [
                scalar,
                vector_norm,
                tensor_norm,
                vector_tensor_vector,
                tensor_trace3,
                vector_tensor2_vector,
            ],
            axis=-1,
        )
        return xp.reshape(
            invariants,
            (moments.shape[0], 6 * self.channels),
        )

    @property
    def dim_out(self) -> int:
        """Return the descriptor output width.

        Returns
        -------
        int
            Number of invariant descriptor channels.
        """
        return self.get_dim_out()

    def get_rcut(self) -> float:
        """Return the outer cutoff radius.

        Returns
        -------
        float
            Cutoff radius in Å.
        """
        return self.rcut

    def get_rcut_smth(self) -> float:
        """Return the smoothing-onset radius required by the common ABI.

        DPA4C uses one C³ envelope over the complete interval ``[0, rcut]``;
        there is no independent smoothing-onset parameter.

        Returns
        -------
        float
            The outer cutoff radius in Å.
        """
        return self.rcut

    def get_sel(self) -> list[int]:
        """Return the internal capacity of the bounded dense adapter.

        DPA4C graph equations do not use this value for truncation or
        normalization. Production NeighborGraph builders carry every edge
        inside ``rcut``.

        Returns
        -------
        list[int]
            A one-element compatibility capacity. Lists at this exact capacity
            are rejected by :meth:`call`; it is never a physical model
            parameter.
        """
        return [self._LEGACY_NLIST_CAPACITY]

    def get_ntypes(self) -> int:
        """Return the number of atom types.

        Returns
        -------
        int
            Number of real atom types, excluding the padding type.
        """
        return self.ntypes

    def get_type_map(self) -> list[str] | None:
        """Return atom-type names.

        Returns
        -------
        list[str] or None
            Ordered atom-type names, or ``None`` when no map was supplied.
        """
        return self.type_map

    def get_dim_out(self) -> int:
        """Return the DPA4C scalar output width.

        Returns
        -------
        int
            ``6 * channels``, preserving all six invariant families.
        """
        return 6 * self.channels

    def get_dim_emb(self) -> int:
        """Return the equivariant fitting-input width.

        Returns
        -------
        int
            Zero because DPA4C exposes only invariant descriptor channels.
        """
        return 0

    def mixed_types(self) -> bool:
        """Return whether a mixed-type neighbor list is required.

        Returns
        -------
        bool
            Always ``True``.
        """
        return True

    def has_message_passing(self) -> bool:
        """Return whether the descriptor exchanges source-node features.

        Returns
        -------
        bool
            ``False`` because both reductions use edge geometry and static
            atom types only.
        """
        return False

    def has_message_passing_across_ranks(self) -> bool:
        """Return whether intermediate halo features require communication.

        Returns
        -------
        bool
            Always ``False``.
        """
        return False

    def need_sorted_nlist_for_lower(self) -> bool:
        """Return whether graph edges must be destination-sorted.

        Returns
        -------
        bool
            ``False`` because all reductions use destination-indexed
            ``segment_sum`` operations.
        """
        return False

    def get_env_protection(self) -> float:
        """Return the fixed geometric regularization scale.

        Returns
        -------
        float
            Direction regularization in Å.
        """
        return self._EPS

    def uses_graph_lower(self) -> bool:
        """Return whether the graph-native lower is available.

        Returns
        -------
        bool
            Always ``True``.
        """
        return True

    def reinit_exclude(
        self,
        exclude_types: list[tuple[int, int]] | None = None,
    ) -> None:
        """Rebuild the descriptor-level pair exclusion mask.

        Parameters
        ----------
        exclude_types
            Ordered atom-type pairs to remove. ``None`` denotes no exclusions.
        """
        if exclude_types is None:
            exclude_types = []
        self.exclude_types = list(exclude_types)
        self.emask = PairExcludeMask(
            self.ntypes,
            exclude_types=self.exclude_types,
        )

    def share_params(
        self,
        base_class: Any,
        shared_level: int,
        model_prob: float = 1.0,
        resume: bool = False,
    ) -> None:
        """Share all descriptor parameters for multitask training.

        Parameters
        ----------
        base_class
            DPA4C descriptor that owns the shared parameters.
        shared_level
            Sharing level. DPA4C supports only level ``0``, which shares the
            complete descriptor.
        model_prob
            Model sampling probability accepted by the common multitask ABI;
            unused because DPA4C has no mergeable input statistics.
        resume
            Whether sharing occurs during checkpoint restoration; unused.

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
        for name in (
            "type_embedding",
            "radial_basis",
            "radial_embedding",
            "scalar_projection",
            "vector_projection",
            "tensor_projection",
        ):
            setattr(self, name, getattr(base_class, name))
        self.adam_degree_weight = base_class.adam_degree_weight
        self.mean = base_class.mean
        self.stddev = base_class.stddev

    def change_type_map(
        self,
        type_map: list[str],
        model_with_new_type_stat: Any | None = None,
    ) -> None:
        """Reject unsupported atom-type remapping.

        Parameters
        ----------
        type_map
            Requested atom-type map.
        model_with_new_type_stat
            Optional descriptor carrying statistics for newly introduced
            types. DPA4C does not use descriptor input statistics.

        Raises
        ------
        NotImplementedError
            Always, because DPA4-family type-table remapping is not defined.
        """
        del type_map, model_with_new_type_stat
        raise NotImplementedError("DPA4C does not support changing `type_map`.")

    def set_stat_mean_and_stddev(self, mean: Array, stddev: Array) -> None:
        """Store interface-compatible descriptor statistics.

        Parameters
        ----------
        mean
            Mean array. DPA4C does not consume it in the forward equations.
        stddev
            Standard-deviation array. DPA4C does not consume it in the forward
            equations.
        """
        self.mean = mean
        self.stddev = stddev

    def get_stat_mean_and_stddev(self) -> tuple[Array, Array]:
        """Return interface-compatible descriptor statistics.

        Returns
        -------
        mean
            Stored mean array.
        stddev
            Stored standard-deviation array.
        """
        return self.mean, self.stddev

    def compute_input_stats(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        path: DPPath | None = None,
    ) -> None:
        """Preserve the analytic DPA4 radial normalization.

        DPA4C has no data-dependent descriptor normalization. The method is a
        deliberate no-op required by the common descriptor interface.

        Parameters
        ----------
        merged
            Sampled training systems or a callable returning them.
        path
            Optional statistics path; unused.
        """
        del merged, path

    def serialize(self) -> dict:
        """Serialize the descriptor.

        Returns
        -------
        dict
            Versioned descriptor configuration, nested DPA4 components, and
            interface statistics.
        """
        data = {
            "@class": "Descriptor",
            "type": "dpa4c",
            "@version": 1,
            "rcut": self.rcut,
            "ntypes": self.ntypes,
            "channels": self.channels,
            "basis_type": self.basis_type,
            "n_radial": self.n_radial,
            "radial_mlp": self.radial_mlp,
            "activation_function": self.activation_function,
            "exclude_types": self.exclude_types,
            "precision": np.dtype(PRECISION_DICT[self.precision]).name,
            "trainable": self.trainable,
            "type_map": self.type_map,
            "seed": self.seed,
            "spin": None,
            "type_embedding": self.type_embedding.serialize(),
            "radial_basis": self.radial_basis.serialize(),
            "radial_embedding": self.radial_embedding.serialize(),
            "scalar_projection": self.scalar_projection.serialize(),
            "vector_projection": self.vector_projection.serialize(),
            "tensor_projection": self.tensor_projection.serialize(),
            "@variables": {
                "mean": to_numpy_array(self.mean),
                "stddev": to_numpy_array(self.stddev),
                "adam_degree_weight": to_numpy_array(self.adam_degree_weight),
            },
        }
        if self.compress:
            data["compress"] = {
                "@variables": {
                    "data": to_numpy_array(self.compress_data),
                    "info": to_numpy_array(self.compress_info),
                }
            }
        return data

    @classmethod
    def deserialize(cls, data: dict) -> DescrptDPA4C:
        """Deserialize a DPA4C descriptor.

        Parameters
        ----------
        data
            Versioned descriptor dictionary produced by :meth:`serialize`.

        Returns
        -------
        DescrptDPA4C
            Reconstructed descriptor with restored trainable components.

        Raises
        ------
        ValueError
            If the serialized version is unsupported or a nested component is
            malformed.
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        data.pop("type")
        compression = data.pop("compress", None)
        variables = data.pop("@variables")
        type_embedding = data.pop("type_embedding")
        radial_basis = data.pop("radial_basis")
        radial_embedding = data.pop("radial_embedding")
        scalar_projection = data.pop("scalar_projection")
        vector_projection = data.pop("vector_projection")
        tensor_projection = data.pop("tensor_projection")

        obj = cls(**data)
        obj.type_embedding = SeZMTypeEmbedding.deserialize(type_embedding)
        obj.radial_basis = RadialBasis.deserialize(radial_basis)
        obj.radial_embedding = RadialMLP.deserialize(radial_embedding)
        obj.scalar_projection = NativeLayer.deserialize(scalar_projection)
        obj.vector_projection = NativeLayer.deserialize(vector_projection)
        obj.tensor_projection = NativeLayer.deserialize(tensor_projection)
        obj.mean = variables["mean"]
        obj.stddev = variables["stddev"]
        obj.adam_degree_weight = variables["adam_degree_weight"]
        if compression is not None:
            obj.compress_data = compression["@variables"]["data"]
            obj.compress_info = compression["@variables"]["info"]
            obj.compress = True
        return obj

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: list[str] | None,
        local_jdata: dict,
    ) -> tuple[dict, float]:
        """Compute neighbor statistics without introducing a public ``sel``.

        The statistics pass requests the exact ``auto:1.0`` mixed-type
        capacity solely to verify that the bounded dense compatibility path is
        not required beyond its safety limit. The returned descriptor
        configuration remains graph-native and contains no ``sel`` field.

        Parameters
        ----------
        train_data
            Training dataset used for neighbor statistics.
        type_map
            Ordered atom-type names.
        local_jdata
            DPA4C descriptor configuration.

        Returns
        -------
        local_jdata
            Unmodified descriptor configuration.
        min_nbor_dist
            Minimum observed neighbor distance in Å.

        Raises
        ------
        ValueError
            If the observed neighbor count exceeds the bounded dense
            compatibility capacity.
        """
        local_jdata_cpy = local_jdata.copy()
        min_nbor_dist, sel = cls._update_sel_cls().update_one_sel(
            train_data,
            type_map,
            local_jdata_cpy["rcut"],
            "auto:1.0",
            True,
        )
        if sum(sel) > cls._LEGACY_NLIST_CAPACITY:
            raise ValueError(
                "DPA4C requires a carry-all graph because the training data "
                f"contains more than {cls._LEGACY_NLIST_CAPACITY} neighbors."
            )
        return local_jdata_cpy, min_nbor_dist

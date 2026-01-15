# SPDX-License-Identifier: LGPL-3.0-or-later
"""Gate the radial channels beyond a boundary by the mass of other neighbours.

For an edge ``j -> i`` with envelope mass ``w_ij = s_ij**2`` and total
destination mass ``S_i = sum_k w_ik``, every radial basis value whose Gaussian
centre lies beyond the boundary is multiplied by ``g_ij = (S_i - w_ij) / S_i``
before any consumer of the edge cache reads it: the radial network, the
environment seed and the geometric initial embedding. An isolated pair has
``g_ij = 0`` exactly, so its outer channels vanish together with their
derivatives and the pair's function beyond the inner support reduces to the
isolated-atom limit; a neighbour of a coordinated atom keeps its full radial
resolution to the cutoff. The inner channels are unchanged.

The boundary is recorded on every radial basis at construction and read by the
edge-cache builders of the PT and array-API descriptors, so training,
evaluation and export share one equation. Changing the option for a later
model does not alter an existing model's function.
"""

from __future__ import (
    annotations,
)

import dataclasses
from functools import (
    wraps,
)
from typing import (
    Any,
)

import array_api_compat
from edge_type_neighbors import (
    other_neighbor_weight,
)

from deepmd.dpmodel.array_api import (
    xp_asarray_nodetach,
)

_boundary = 0.0
_installed = False


def gate_cache(cache: Any, basis: Any, n_nodes: int) -> Any:
    """Return the cache with the outer radial channels weighted by ``g_ij``.

    Parameters
    ----------
    cache : EdgeFeatureCache or EdgeCache
        Edge cache whose ``edge_rbf`` has shape (E, n_radial).
    basis : RadialBasis
        Radial basis holding the Gaussian centres in ``adam_freqs`` and the
        recorded boundary in Angstrom.
    n_nodes : int
        Number of destination nodes.
    """
    boundary = getattr(basis, "_experiment_outer_gate_boundary", 0.0)
    if not boundary:
        return cache
    rbf = cache.edge_rbf
    xp = array_api_compat.array_namespace(rbf)
    device = array_api_compat.device(rbf)
    centres = xp.reshape(
        xp_asarray_nodetach(xp, basis.adam_freqs[...], device=device), (-1,)
    )
    outer = xp.astype(centres > boundary, rbf.dtype)  # (n_radial,)
    weight = xp.astype(other_neighbor_weight(cache, n_nodes), rbf.dtype)  # (E,)
    factor = 1.0 - outer[None, :] * (1.0 - weight[:, None])  # one inside, g_ij outside
    gated = rbf * factor
    if dataclasses.is_dataclass(cache):
        return dataclasses.replace(cache, edge_rbf=gated)
    return cache._replace(edge_rbf=gated)


def configure(descriptor: Any, boundary: float = 0.0) -> None:
    """Record the boundary on an existing descriptor's radial basis."""
    descriptor.radial_basis._experiment_outer_gate_boundary = float(boundary)


def install(boundary: float = 0.0) -> None:
    """Set the boundary for subsequently constructed PT and array-API models.

    Parameters
    ----------
    boundary : float
        Radial channels whose centre lies beyond this radius, in Angstrom, are
        gated. Zero disables the gate.
    """
    global _boundary, _installed
    if boundary < 0.0:
        raise ValueError(
            f"The outer-channel boundary must be nonnegative, got {boundary}"
        )
    _boundary = float(boundary)
    if _installed:
        return
    from deepmd.dpmodel.descriptor import dpa4 as native_dpa4
    from deepmd.dpmodel.descriptor.dpa4_nn.radial import RadialBasis as NativeBasis
    from deepmd.pt.model.descriptor import sezm as pt_sezm
    from deepmd.pt.model.descriptor.sezm_nn.radial import (
        RadialBasis,
    )

    for basis in (RadialBasis, NativeBasis):
        _install_constructor(basis)
    pt_sezm.build_edge_cache = _wrap(pt_sezm.build_edge_cache)
    pt_sezm.build_edge_cache_from_edges = _wrap(pt_sezm.build_edge_cache_from_edges)
    native_dpa4._edge_cache_from_arrays = _wrap(native_dpa4._edge_cache_from_arrays)
    _installed = True


def _install_constructor(basis: type) -> None:
    original = basis.__init__

    @wraps(original)
    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        self._experiment_outer_gate_boundary = _boundary

    basis.__init__ = init


def _wrap(builder: Any) -> Any:
    """Gate the outer channels of every cache the builder returns."""

    @wraps(builder)
    def build(*args: Any, **kwargs: Any) -> Any:
        cache = builder(*args, **kwargs)
        basis = kwargs.get("radial_basis")
        if basis is None or not getattr(basis, "_experiment_outer_gate_boundary", 0.0):
            return cache
        return gate_cache(cache, basis, kwargs["type_ebed"].shape[0])

    return build

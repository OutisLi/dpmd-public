# SPDX-License-Identifier: LGPL-3.0-or-later
"""Fade the whole edge input of a nearly isolated pair smoothly with its distance.

For an edge ``j -> i`` with other-neighbour fraction ``g_ij`` (the share of the
destination's envelope mass carried by edges other than ``j``, zero for an
isolated pair and close to one inside a crystal) and length ``r_ij``, every
radial basis value and the additive type feature of the edge are multiplied by

    f_ij = g_ij + (1 - g_ij) * s(r_ij),

where ``s`` is the quintic smoothstep that equals one up to ``r_lo``, zero from
``r_hi`` and is twice continuously differentiable in between. A crystal edge is
unchanged; an isolated pair keeps its full input inside ``r_lo``, fades over
``[r_lo, r_hi]`` and presents no input beyond ``r_hi``, so its function reaches
the isolated-atom limit continuously instead of at a channel boundary. The
factor is applied to the edge cache before any consumer reads it, so training,
evaluation and export share one equation.
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

_window: tuple[float, float] = (0.0, 0.0)
_installed = False


def smoothstep(x: Any, xp: Any) -> Any:
    """Quintic smoothstep from one at ``x <= 0`` to zero at ``x >= 1``."""
    x = xp.clip(x, 0.0, 1.0)
    return 1.0 - x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)


def fade_cache(cache: Any, n_nodes: int, r_lo: float, r_hi: float) -> Any:
    """Return the cache with ``edge_rbf`` and ``edge_type_feat`` multiplied by ``f_ij``."""
    rbf = cache.edge_rbf
    xp = array_api_compat.array_namespace(rbf)
    vec = cache.edge_vec
    length = xp.sqrt(xp.sum(vec * vec, axis=-1))  # (E,)
    s = smoothstep((length - r_lo) / (r_hi - r_lo), xp)
    g = other_neighbor_weight(cache, n_nodes)
    factor = g + (1.0 - g) * s  # (E,)
    rbf = rbf * xp.astype(factor, rbf.dtype)[:, None]
    type_feat = (
        cache.edge_type_feat * xp.astype(factor, cache.edge_type_feat.dtype)[:, None]
    )
    if dataclasses.is_dataclass(cache):
        return dataclasses.replace(cache, edge_rbf=rbf, edge_type_feat=type_feat)
    return cache._replace(edge_rbf=rbf, edge_type_feat=type_feat)


def install(r_lo: float = 0.0, r_hi: float = 0.0) -> None:
    """Set the fade window, in Angstrom, for every edge cache built afterwards.

    A window with ``r_hi <= r_lo`` disables the fade.
    """
    global _window, _installed
    if r_lo < 0.0 or r_hi < 0.0:
        raise ValueError(f"The fade window must be nonnegative, got {r_lo}, {r_hi}")
    _window = (float(r_lo), float(r_hi))
    if _installed:
        return
    from deepmd.dpmodel.descriptor import dpa4 as native_dpa4
    from deepmd.pt.model.descriptor import sezm as pt_sezm

    pt_sezm.build_edge_cache = _wrap(pt_sezm.build_edge_cache)
    pt_sezm.build_edge_cache_from_edges = _wrap(pt_sezm.build_edge_cache_from_edges)
    native_dpa4._edge_cache_from_arrays = _wrap(native_dpa4._edge_cache_from_arrays)
    _installed = True


def window() -> tuple[float, float]:
    return _window


def _wrap(builder: Any) -> Any:
    """Fade every cache the builder returns when a window is set."""

    @wraps(builder)
    def build(*args: Any, **kwargs: Any) -> Any:
        cache = builder(*args, **kwargs)
        r_lo, r_hi = _window
        if r_hi <= r_lo:
            return cache
        return fade_cache(cache, kwargs["type_ebed"].shape[0], r_lo, r_hi)

    return build

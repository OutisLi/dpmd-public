# SPDX-License-Identifier: LGPL-3.0-or-later
"""Restrict the additive edge-type channel to contributions from other neighbors.

For an edge ``j -> i`` with envelope mass ``w_ij = s_ij**2``, the type-channel
factor is ``(sum_k w_ik - w_ij) / sum_k w_ik``. It is zero for an isolated pair
and approaches one for an edge whose mass is small relative to the remaining
neighbors. The radial features and node type embeddings remain unchanged.
"""

from __future__ import (
    annotations,
)

from contextlib import (
    nullcontext,
)
from functools import (
    wraps,
)
from types import (
    MethodType,
)
from typing import (
    Any,
)

import array_api_compat

from deepmd.dpmodel.array_api import (
    xp_add_at,
)
from deepmd.dpmodel.common import (
    get_xp_precision,
)

_enabled = False
_installed = False


def other_neighbor_weight(edge_cache: Any, n_nodes: int) -> Any:
    """Return the fraction of each destination's mass carried by other edges.

    Parameters
    ----------
    edge_cache : EdgeCache
        Edge envelopes, destination indices, and optional source/validity masks.
    n_nodes : int
        Number of destination nodes.

    Returns
    -------
    Array
        Dimensionless factors with shape (E,). Empty and fully masked
        neighborhoods have zero weight. Source gates enter the total mass.
    """
    xp = array_api_compat.array_namespace(edge_cache.edge_env)
    envelope = xp.reshape(edge_cache.edge_env, (-1,))
    mass = envelope * envelope
    if edge_cache.edge_src_gate is not None:
        mass = mass * xp.reshape(edge_cache.edge_src_gate, (-1,))
    mask = getattr(edge_cache, "edge_mask", None)
    if mask is not None:
        mass = mass * xp.astype(xp.reshape(mask, (-1,)), mass.dtype)
    total = xp_add_at(
        xp.zeros((n_nodes,), dtype=mass.dtype, device=array_api_compat.device(mass)),
        edge_cache.dst,
        mass,
    )
    edge_total = xp.take(total, edge_cache.dst, axis=0)
    denominator = xp.where(edge_total > 0, edge_total, 1)
    return (edge_total - mass) / denominator


def _radial_features(descriptor: Any, edge_cache: Any, n_nodes: int) -> Any:
    """Construct radial and type features before the interaction-block AMP region."""
    xp = array_api_compat.array_namespace(edge_cache.edge_rbf)
    context = nullcontext()
    if array_api_compat.is_torch_array(edge_cache.edge_rbf):
        import torch

        context = torch.autocast(
            device_type=edge_cache.edge_rbf.device.type, enabled=False
        )
    with context:
        # Re-evaluation preserves small radial values that cannot be recovered
        # by subtracting a large type feature after floating-point addition.
        radial = descriptor.radial_embedding(edge_cache.edge_rbf)
        radial = xp.reshape(
            radial,
            (
                edge_cache.src.shape[0],
                descriptor.node_init_lmax + 1,
                descriptor.channels,
            ),
        )
        if descriptor.version >= 1.1:
            radial = radial * xp.reshape(edge_cache.edge_env, (-1, 1, 1))
        dtype = get_xp_precision(xp, descriptor.precision)
        radial = xp.astype(radial, dtype)
        weight = xp.astype(other_neighbor_weight(edge_cache, n_nodes), dtype)
        edge_type = xp.astype(edge_cache.edge_type_feat, dtype) * weight[:, None]
        return radial + edge_type[:, None, :]


def _forward_blocks(
    self: Any,
    x: Any,
    edge_cache: Any,
    radial_feat_per_block: list[Any],
    comm_dict: dict[str, Any] | None = None,
) -> Any:
    radial = _radial_features(self, edge_cache, x.shape[0])
    features = [radial[:, :length, :] for length in self.rad_sizes_per_block]
    return self._experiment_native_forward_blocks(
        x, edge_cache, features, comm_dict=comm_dict
    )


def configure(descriptor: Any, enabled: bool = True) -> None:
    """Configure the edge-type rule on an existing descriptor."""
    if enabled:
        if not getattr(descriptor, "_experiment_edge_type_neighbors", False):
            descriptor._experiment_native_forward_blocks = descriptor._forward_blocks
        descriptor._forward_blocks = MethodType(_forward_blocks, descriptor)
    elif getattr(descriptor, "_experiment_edge_type_neighbors", False):
        descriptor._forward_blocks = descriptor._experiment_native_forward_blocks
        del descriptor._experiment_native_forward_blocks
    descriptor._experiment_edge_type_neighbors = enabled


def install(enabled: bool = False) -> None:
    """Set the option for subsequently constructed PT and PT-expt descriptors."""
    global _enabled, _installed
    _enabled = bool(enabled)
    if _installed:
        return
    from deepmd.dpmodel.descriptor.dpa4 import (
        DescrptDPA4,
    )
    from deepmd.pt.model.descriptor.sezm import (
        DescrptSeZM,
    )

    for descriptor in (DescrptSeZM, DescrptDPA4):
        _install_constructor(descriptor)
    _installed = True


def _install_constructor(descriptor: type) -> None:
    original = descriptor.__init__

    @wraps(original)
    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        configure(self, _enabled)

    descriptor.__init__ = init

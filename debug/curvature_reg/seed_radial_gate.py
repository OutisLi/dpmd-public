# SPDX-License-Identifier: LGPL-3.0-or-later
"""Condition the environment seed on radial signals without a type-only output.

The first G-network matrix is split into its existing radial and type rows.
Instead of ``SiLU(radial + types)``, it evaluates
``SiLU(radial) * sigmoid(types)``. Both the projected radial input and the
resulting seed therefore vanish when the bias-free radial projection is zero.
No parameter tensor is added or resized.
"""

from __future__ import (
    annotations,
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
    xp_asarray_nodetach,
    xp_sigmoid,
)

_enabled = False
_installed = False


def _pt_forward(self: Any, value: Any) -> Any:
    import torch
    import torch.nn.functional as functional

    from deepmd.pt.utils import (
        env,
    )

    original_dtype = value.dtype
    if not env.DP_DTYPE_PROMOTION_STRICT:
        value = value.to(self.prec)
    split = self._experiment_radial_columns
    radial = functional.linear(value[..., :split], self.matrix[:split].t())
    types = functional.linear(value[..., split:], self.matrix[split:].t())
    output = functional.silu(radial) * torch.sigmoid(types)
    return output.to(original_dtype) if not env.DP_DTYPE_PROMOTION_STRICT else output


def _native_call(self: Any, value: Any) -> Any:
    xp = array_api_compat.array_namespace(value)
    matrix = xp_asarray_nodetach(xp, self.w[...], device=array_api_compat.device(value))
    split = self._experiment_radial_columns
    radial = xp.matmul(value[..., :split], matrix[:split])
    types = xp.matmul(value[..., split:], matrix[split:])
    radial = xp.astype(radial, value.dtype)
    types = xp.astype(types, value.dtype)
    return radial * xp_sigmoid(radial) * xp_sigmoid(types)


def _export_call(self: Any, value: Any) -> Any:
    import torch
    import torch.nn.functional as functional

    split = self._experiment_radial_columns
    radial = torch.matmul(value[..., :split], self.w[:split])
    types = torch.matmul(value[..., split:], self.w[split:])
    if not self.autocast_output:
        radial = radial.to(value.dtype)
        types = types.to(value.dtype)
    return functional.silu(radial) * torch.sigmoid(types)


def configure(descriptor: Any, enabled: bool = True) -> None:
    """Configure radial/type gating on the descriptor's environment seed.

    The experiment requires an enabled, bias-free SiLU environment seed. The
    remaining embedding, normalization, and output projections are preserved.
    """
    seed = descriptor.env_seed_embedding
    if enabled and seed is None:
        raise ValueError(
            "Seed radial gating requires a bias-free SiLU environment seed"
        )
    if seed is not None:
        _configure_seed(seed, enabled)


def _configure_seed(seed: Any, enabled: bool) -> None:
    """Configure the owning seed after its first G layer has been constructed."""
    from deepmd.pt.model.network.mlp import (
        MLPLayer,
    )
    from deepmd.pt_expt.utils.network import NativeLayer as ExportLayer

    if enabled and (seed.mlp_bias or seed.activation_function != "silu"):
        raise ValueError(
            "Seed radial gating requires a bias-free SiLU environment seed"
        )
    seed._experiment_seed_radial_gate = enabled
    layer = seed.g_layer1
    pt_layer = isinstance(layer, MLPLayer)
    attribute = "forward" if pt_layer else "call"
    if enabled:
        if not getattr(layer, "_experiment_seed_radial_gate", False):
            layer._experiment_seed_original = getattr(layer, attribute)
        layer._experiment_radial_columns = seed.rbf_out_dim
        method = (
            _pt_forward
            if pt_layer
            else _export_call
            if isinstance(layer, ExportLayer)
            else _native_call
        )
        setattr(layer, attribute, MethodType(method, layer))
    elif getattr(layer, "_experiment_seed_radial_gate", False):
        setattr(layer, attribute, layer._experiment_seed_original)
        del layer._experiment_seed_original
        del layer._experiment_radial_columns
    layer._experiment_seed_radial_gate = enabled


def install(enabled: bool = False) -> None:
    """Set the option for subsequently constructed environment seed modules."""
    global _enabled, _installed
    _enabled = bool(enabled)
    if _installed:
        return
    from deepmd.dpmodel.descriptor.dpa4_nn.embedding import (
        EnvironmentInitialEmbedding as NativeSeed,
    )
    from deepmd.pt.model.descriptor.sezm_nn.embedding import (
        EnvironmentInitialEmbedding,
    )

    for embedding in (EnvironmentInitialEmbedding, NativeSeed):
        _install_constructor(embedding)
    _installed = True


def _install_constructor(embedding: type) -> None:
    original = embedding.__init__

    @wraps(original)
    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        _configure_seed(self, _enabled)

    embedding.__init__ = init

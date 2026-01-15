# SPDX-License-Identifier: LGPL-3.0-or-later
"""Separate attention's neighbor selection from its envelope mass.

For edge envelopes ``s`` and logits ``l``, the normalized selection is
``p = softmax(l + 2 log(s))`` over each destination. The total edge weight is
``S / (S + z)``, where ``S = sum(s**2)`` and ``z`` is the native learned null
mass. A common logit offset therefore changes neither the total weight nor the
relative neighbor selection. Source gates and padded-edge masks enter both
normalizations. An existing output envelope remains on the final edge weight.
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
    xp_add_at,
    xp_asarray_nodetach,
    xp_einsum,
    xp_maximum_at,
)
from deepmd.dpmodel.descriptor.dpa4_nn.attention import (
    _stop_gradient,
)
from deepmd.dpmodel.utils.network import (
    softplus_t,
)

_enabled = False
_installed = False


def normalize(
    logits: Any,
    edge_env: Any,
    dst: Any,
    n_nodes: int,
    z_bias_raw: Any,
    eps: float,
    src_weight: Any = None,
    edge_mask: Any = None,
) -> Any:
    """Normalize edge selection and total envelope mass independently.

    Parameters
    ----------
    logits : Array
        Attention scores with shape (E, F, H).
    edge_env : Array
        Nonnegative cutoff envelopes with shape (E,) or (E, 1).
    dst : Array
        Destination indices with shape (E,).
    n_nodes : int
        Number of destination nodes.
    z_bias_raw : Array
        Native unconstrained null-mass parameters with shape (F, H).
    eps : float
        Native numerical floor on the positive null mass.
    src_weight : Array, optional
        Nonnegative source gates with shape (E,) or (E, 1).
    edge_mask : Array, optional
        Binary padded-edge validity mask with shape (E,) or (E, 1).

    Returns
    -------
    Array
        Edge weights with shape (E, F, H). Empty and fully masked segments
        return zero; the weights in each active segment sum to ``S/(S+z)``.
    """
    xp = array_api_compat.array_namespace(logits)
    device = array_api_compat.device(logits)
    n_edge, n_focus, n_head = logits.shape
    n_channel = n_focus * n_head
    input_dtype = logits.dtype
    compute_dtype = xp.float32 if "float16" in str(input_dtype) else input_dtype
    scores = xp.astype(xp.reshape(logits, (n_edge, n_channel)), compute_dtype)
    envelope = xp.astype(xp.reshape(edge_env, (n_edge,)), compute_dtype)
    ones = xp.ones((n_edge,), dtype=compute_dtype, device=device)
    active = envelope > 0
    log_weight = 2 * xp.log(xp.where(active, envelope, ones))
    physical_mass = envelope * envelope
    source_ratio = None
    if src_weight is not None:
        source = xp.astype(xp.reshape(src_weight, (n_edge,)), compute_dtype)
        positive = source > 0
        safe_source = xp.where(positive, source, ones)
        source_scale = _stop_gradient(safe_source)
        log_weight = log_weight + xp.log(source_scale)
        source_ratio = xp.where(positive, source / source_scale, 0)
        physical_mass = physical_mass * source
        active = active & positive
    if edge_mask is not None:
        active = active & (xp.reshape(edge_mask, (n_edge,)) > 0)
    physical_mass = xp.where(active, physical_mass, 0)
    effective = xp.where(active[:, None], scores + log_weight[:, None], -xp.inf)

    # === Step 1. Relative neighbor selection ===
    # The shared numerical shift cancels from the normalized weights. Empty
    # segments use a finite shift and a unit divisor on their zero numerator.
    maximum = xp_maximum_at(
        xp.full((n_nodes, n_channel), -xp.inf, dtype=compute_dtype, device=device),
        dst,
        effective,
    )
    maximum = _stop_gradient(xp.where(xp.isfinite(maximum), maximum, 0))
    mass = xp.exp(effective - xp.take(maximum, dst, axis=0))
    if source_ratio is not None:
        mass = mass * source_ratio[:, None]
    total = xp_add_at(
        xp.zeros((n_nodes, n_channel), dtype=compute_dtype, device=device), dst, mass
    )
    denominator = xp.where(total > 0, total, 1)
    selection = mass / xp.take(denominator, dst, axis=0)

    # === Step 2. Total envelope mass against the native null mass ===
    envelope_mass = xp_add_at(
        xp.zeros((n_nodes,), dtype=compute_dtype, device=device), dst, physical_mass
    )[:, None]
    null_mass = xp.reshape(
        softplus_t(xp.astype(z_bias_raw, compute_dtype)) + float(eps), (1, n_channel)
    )
    occupancy = envelope_mass / (envelope_mass + null_mass)
    alpha = selection * xp.take(occupancy, dst, axis=0)
    return xp.reshape(xp.astype(alpha, input_dtype), (n_edge, n_focus, n_head))


def _attention_weights(
    self: Any, x_l0_node: Any, edge_cache: Any, rad_feat: Any
) -> Any:
    """Retain the native query, key, radial score, and output-envelope paths."""
    xp = array_api_compat.array_namespace(x_l0_node)
    device = array_api_compat.device(x_l0_node)
    n_edge = edge_cache.src.shape[0]
    q_node, k_node = self.attention_qk(x_l0_node)
    shape = (n_edge, self.attn_n_focus, self.n_atten_head, self.head_dim)
    q_edge = xp.reshape(xp.take(q_node, edge_cache.dst, axis=0), shape)
    k_edge = xp.reshape(xp.take(k_node, edge_cache.src, axis=0), shape)
    logits = xp.sum(q_edge * k_edge, axis=-1) * self.head_dim**-0.5
    radial = xp.reshape(
        rad_feat[:, 0, :], (n_edge, self.attn_n_focus, self.attn_focus_dim)
    )
    matrix = xp_asarray_nodetach(xp, self.adamw_attn_logit_w[...], device=device)
    logits = logits + xp_einsum("efi,ifo->efo", xp.astype(radial, matrix.dtype), matrix)
    null = xp_asarray_nodetach(xp, self.adamw_attn_z_bias_raw[...], device=device)
    alpha = normalize(
        logits,
        edge_cache.edge_env,
        edge_cache.dst,
        x_l0_node.shape[0],
        null,
        self.eps,
        src_weight=edge_cache.edge_src_gate,
        edge_mask=getattr(edge_cache, "edge_mask", None),
    )
    if getattr(self, "_experiment_output_envelope", False):
        alpha = alpha * xp.astype(
            xp.reshape(edge_cache.edge_env, (n_edge, 1, 1)), alpha.dtype
        )
    return alpha


def configure(descriptor: Any, enabled: bool = True) -> None:
    """Configure the normalization on an existing SeZM descriptor."""
    for block in descriptor.blocks:
        _configure_convolution(block.so2_conv, enabled)


def _configure_convolution(convolution: Any, enabled: bool) -> None:
    if enabled and convolution.n_atten_head == 0:
        raise ValueError("Separate envelope mass requires an attention convolution")
    if enabled:
        convolution.attention_weights = MethodType(_attention_weights, convolution)
        # The complete convolution absorbs the native attention equation. The
        # fused value and flash-aggregation operators retain this weight seam.
        convolution._cuda_conv_fn = None
    elif getattr(convolution, "_experiment_separate_attention_mass", False):
        del convolution.attention_weights
    convolution._experiment_separate_attention_mass = enabled


def install(enabled: bool = False) -> None:
    """Set the option for subsequently constructed PT and PT-expt models."""
    global _enabled, _installed
    _enabled = bool(enabled)
    if _installed:
        return
    from deepmd.dpmodel.descriptor.dpa4_nn.so2 import (
        SO2Convolution as NativeConvolution,
    )
    from deepmd.pt.model.descriptor.sezm_nn.so2 import (
        SO2Convolution,
    )
    from deepmd.pt_expt.descriptor.dpa4_nn.so2 import (
        SO2Convolution as ExportConvolution,
    )

    for convolution in (SO2Convolution, NativeConvolution, ExportConvolution):
        _install_constructor(convolution)
    _installed = True


def _install_constructor(convolution: type) -> None:
    original = convolution.__init__

    @wraps(original)
    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        _configure_convolution(self, _enabled)

    convolution.__init__ = init

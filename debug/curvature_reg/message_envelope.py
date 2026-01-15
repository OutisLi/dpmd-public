# SPDX-License-Identifier: LGPL-3.0-or-later
"""Apply the physical cutoff after normalized attention's value transformation.

For an isolated edge, the existing attention weight is
``a = s**2 * exp(l) / (s**2 * exp(l) + z)``, with envelope ``s`` and null
mass ``z``. A large constant logit ``l`` moves a sharp transition toward the
cutoff: ``da/dr = 2 * a * (1 - a) * s'/s``. Multiplying the value by ``s``
instead makes the effective weight ``s*a`` and its derivative, for constant
``l``, ``s' * a * (3 - 2*a)``, bounded by ``9/8 * abs(s')`` independently
of the logit. The learned attention distribution is otherwise unchanged.
"""

from __future__ import (
    annotations,
)

from functools import (
    wraps,
)
from typing import (
    Any,
)

import array_api_compat

_enabled = False
_installed = False


def configure(descriptor: Any, enabled: bool = True) -> None:
    """Configure the final per-edge envelope on an existing SeZM descriptor."""
    for block in descriptor.blocks:
        _configure_convolution(block.so2_conv, enabled)


def _configure_convolution(convolution: Any, enabled: bool) -> None:
    """Configure the value seam on the final convolution instance."""
    if enabled and (convolution.n_atten_head == 0 or not convolution.needs_local_frame):
        raise ValueError("The output-envelope experiment requires SO(2) attention")
    convolution._experiment_output_envelope = enabled
    if enabled:
        # The full inference operator absorbs the value transformation. The separate
        # fused value and flash-aggregation operators retain the observable seam.
        convolution._cuda_conv_fn = None


def install(enabled: bool = False) -> None:
    """Set the output-envelope option for subsequently constructed models."""
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

    for convolution in (SO2Convolution, NativeConvolution):
        _install_weights(convolution)
    for convolution in (SO2Convolution, NativeConvolution, ExportConvolution):
        _install_constructor(convolution)
    _installed = True


def _install_weights(convolution: type) -> None:
    """Apply the envelope to scalar weights after their normalization."""
    original = convolution.attention_weights

    @wraps(original)
    def attention_weights(
        self: Any, x_l0_node: Any, edge_cache: Any, rad_feat: Any
    ) -> Any:
        alpha = original(self, x_l0_node, edge_cache, rad_feat)
        if getattr(self, "_experiment_output_envelope", False):
            xp = array_api_compat.array_namespace(alpha)
            envelope = xp.astype(
                xp.reshape(edge_cache.edge_env, (-1, 1, 1)), alpha.dtype, copy=False
            )
            alpha = alpha * envelope
        return alpha

    convolution.attention_weights = attention_weights


def _install_constructor(convolution: type) -> None:
    """Preserve the option when deserialization reconstructs a convolution."""
    original = convolution.__init__

    @wraps(original)
    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        _configure_convolution(self, _enabled)

    convolution.__init__ = init

# SPDX-License-Identifier: LGPL-3.0-or-later
"""A parameter-free RMS normalization of the descriptor entering the energy head.

The normalization acts after the descriptor's readout, before the fitting
network. Its epsilon is a numerical floor, not an amplitude knee. The head
receives approximately unit RMS when the descriptor variance exceeds that floor.
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


def configure(fitting: Any, enabled: bool = True) -> None:
    """Set descriptor normalization on an existing SeZM energy fitting network."""
    fitting._experiment_fitting_rmsnorm = bool(enabled)


def normalize(descriptor: Any) -> Any:
    """Normalize the last channel axis at the descriptor's compute precision."""
    xp = array_api_compat.array_namespace(descriptor)
    mean_square = xp.mean(descriptor * descriptor, axis=-1, keepdims=True)
    return descriptor / xp.sqrt(mean_square + 1e-5)


def install(enabled: bool = False) -> None:
    """Set the normalization option for subsequently constructed fitting networks."""
    global _enabled, _installed
    _enabled = bool(enabled)
    if _installed:
        return
    from deepmd.dpmodel.fitting.dpa4_ener import SeZMEnergyFittingNet as NativeFitting
    from deepmd.pt.model.task.sezm_ener import (
        SeZMEnergyFittingNet,
    )
    from deepmd.pt_expt.fitting.dpa4_ener import SeZMEnergyFittingNet as ExportFitting

    _install_call(SeZMEnergyFittingNet, "forward")
    _install_call(NativeFitting, "call")
    for fitting in (SeZMEnergyFittingNet, NativeFitting, ExportFitting):
        _install_constructor(fitting)
    _installed = True


def _install_call(fitting: type, method: str) -> None:
    """Normalize only the descriptor argument, before frame or atom parameters are appended."""
    original = getattr(fitting, method)

    @wraps(original)
    def call(self: Any, descriptor: Any, *args: Any, **kwargs: Any) -> Any:
        if getattr(self, "_experiment_fitting_rmsnorm", False):
            descriptor = normalize(descriptor)
        return original(self, descriptor, *args, **kwargs)

    setattr(fitting, method, call)


def _install_constructor(fitting: type) -> None:
    """Restore the option when a serialized fitting network is reconstructed."""
    original = fitting.__init__

    @wraps(original)
    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        configure(self, _enabled)

    fitting.__init__ = init

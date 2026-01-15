# SPDX-License-Identifier: LGPL-3.0-or-later
"""Keep a pretrained teacher's original function when its student adds head constraints."""

from __future__ import (
    annotations,
)

from functools import (
    wraps,
)
from typing import (
    Any,
)

from fitting_normalization import configure as configure_fitting
from message_envelope import configure as configure_envelope


def _restore_function(reference: Any) -> None:
    """Restore the teacher's attention and fitting equations without changing its weights."""
    if reference is None:
        return
    for model in reference.model.values():
        configure_envelope(model.atomic_model.descriptor, False)
        configure_fitting(model.atomic_model.fitting_net, False)


def install() -> None:
    """Preserve the unmodified teacher at initial freezing and checkpoint restoration."""
    from deepmd.pt_expt.train.safeguard import (
        TrainingSafeguard,
    )

    freeze = TrainingSafeguard.maybe_freeze_reference
    restore = TrainingSafeguard.load_state_dict

    @wraps(freeze)
    def maybe_freeze_reference(self: Any, wrapper: Any, step: int) -> None:
        previous = self.reference
        freeze(self, wrapper, step)
        if self.reference is not previous:
            _restore_function(self.reference)

    @wraps(restore)
    def load_state_dict(self: Any, state: dict[str, Any], wrapper: Any) -> None:
        restore(self, state, wrapper)
        _restore_function(self.reference)

    TrainingSafeguard.maybe_freeze_reference = maybe_freeze_reference
    TrainingSafeguard.load_state_dict = load_state_dict

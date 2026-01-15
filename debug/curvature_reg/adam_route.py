# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Route selected parameter tensors of the hybrid Muon optimizer to its Adam path.

The hybrid optimizer routes every matrix to Muon unless the parameter's leaf
name begins with ``adam_`` or ``adamw_``, which the model uses for the type
embedding, the radial frequencies and the norm scales. Muon's orthogonalized
momentum moves every singular direction of a matrix at unit rate, including the
directions no training frame constrains, whereas Adam leaves a direction without
gradient where it is. This patch extends the model's own naming convention to
further tensors chosen by name: any parameter whose full name contains one of
the configured substrings takes the ``"adamw"`` route — Adam with the same
decoupled weight decay the Muon path applies to matrices, so that the update
rule is the only thing that changes — and every other parameter keeps the
route the optimizer would have chosen.
"""

from __future__ import (
    annotations,
)

import logging

log = logging.getLogger("deepmd.adam_route")

_patterns: tuple[str, ...] = ()
_installed = False


def install(patterns: list[str] | tuple[str, ...] | None) -> None:
    """Route parameters whose names contain any of ``patterns`` to the AdamW path."""
    global _patterns, _installed
    _patterns = tuple(p.lower() for p in (patterns or ()) if p)
    if _installed:
        return
    # The package now declares routed tensors itself (every fitting network and
    # the DPA4 descriptor's radial-reading layers and readout). A research run
    # must route exactly the tensors named on its command line, so the
    # package declaration is switched off and the flag's list is authoritative;
    # an empty list means no routing at all.
    from deepmd.pt.model.model.model import (
        BaseModel,
    )

    BaseModel.adam_route_patterns = lambda self: []
    if not _patterns:
        _installed = True
        log.info("Adam routing: package declaration disabled, no tensor routed")
        return
    import deepmd.pt.optimizer.hybrid_muon as hybrid_muon

    original = hybrid_muon.get_adam_route
    routed: set[str] = set()

    def get_adam_route(param_name: str | None) -> str:
        if param_name is not None:
            lowered = param_name.lower()
            if any(pattern in lowered for pattern in _patterns):
                if param_name not in routed:
                    routed.add(param_name)
                    log.info("Adam route: %s", param_name)
                return "adamw"
        return original(param_name)

    hybrid_muon.get_adam_route = get_adam_route
    _installed = True
    log.info("Adam routing installed for name patterns %s", list(_patterns))


def patterns() -> tuple[str, ...]:
    return _patterns

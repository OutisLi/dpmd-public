# SPDX-License-Identifier: LGPL-3.0-or-later
"""Squared force consistency on transformed training configurations."""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    import torch


def squared_reference_error(
    pred: torch.Tensor,
    ref: torch.Tensor,
    atype: torch.Tensor,
    tol: float,
    rel_tol: float,
    fcap: float,
) -> torch.Tensor:
    """Average squared vector differences over the reference's trusted atoms.

    Force arrays have shape (..., 3), in eV/A; atom types have the leading
    shape, with -1 marking phantom atoms. The reference is detached by the
    training driver. Its force norm must not exceed ``fcap``. With no trusted
    atom, the loss and prediction gradient are zero. The driver supplies
    ``tol`` and ``rel_tol`` through its penalty interface; squared consistency
    does not use either tolerance.
    """
    import torch

    trusted = (atype >= 0) & (torch.linalg.vector_norm(ref, dim=-1) <= fcap)
    difference = torch.where(trusted[..., None], pred - ref, torch.zeros_like(pred))
    return difference.square().sum() / torch.clamp(trusted.sum(), min=1)


def install() -> None:
    """Select squared consistency in the shared PT and PT-expt training driver."""
    from deepmd.pt_expt.train import (
        safeguard,
    )

    safeguard.reference_hinge = squared_reference_error

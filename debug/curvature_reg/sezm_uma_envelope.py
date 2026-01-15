# SPDX-License-Identifier: LGPL-3.0-or-later
"""The DimeNet polynomial cutoff envelope, as UMA and eSEN use it.

SeZM's ``C3CutoffEnvelope`` is ``E(x) = (1 - x)^4 * sum(comb(k+3, 3) x^k, k < p)``, which
vanishes at the cutoff together with its first three derivatives. UMA uses the DimeNet
form

    E(x) = 1 - (p+1)(p+2)/2 * x^p + p(p+2) * x^(p+1) - p(p+1)/2 * x^(p+2),   p = 5,

whose value and first two derivatives vanish at the cutoff while the third does not, and
which keeps three to ten times more amplitude over the outer third of the range
(0.096 against 0.031 at ``r = 0.83 rcut``, 0.026 against 0.005 at ``0.9 rcut``). The two
therefore differ both in smoothness class at the cutoff and in how far the message
reaches; installing this one turns the envelope into a single experimental factor.

The scaled distance is ``x = r / rcut`` in both, so only the polynomial is replaced.
"""

from typing import (
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    import torch


def _forward(self, dst: "torch.Tensor") -> "torch.Tensor":  # noqa: ANN001
    """Evaluate the DimeNet envelope on the distances ``dst`` in Å, zero beyond the cutoff."""
    import torch

    x = (dst / self.rcut_tensor).clamp(min=0.0, max=1.0)
    p = float(self.p)
    a = -(p + 1.0) * (p + 2.0) / 2.0
    b = p * (p + 2.0)
    c = -p * (p + 1.0) / 2.0
    value = 1.0 + x.pow(p) * (a + x * (b + c * x))
    return torch.where(x < 1.0, value, torch.zeros_like(value))


def install() -> None:
    """Replace the C³ envelope of every SeZM descriptor built from now on."""
    from deepmd.pt.model.descriptor.sezm_nn.radial import (
        C3CutoffEnvelope,
    )

    C3CutoffEnvelope.forward = _forward

# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shape-agnostic forward for the SeZM depth-attention residual.

``DepthAttnRes.forward`` reads its batch dimension with ``int(source0.shape[0])`` and
rebuilds the attention weights from that integer. Under ``torch.compile`` with dynamic
shapes the node count is a symbol, so the integer is a stale guard and the graph fails
with ``shape '[13, 64]' is invalid for input of size 64*s67``. The replacement below
computes the same tensor without materializing the batch dimension as a Python integer:
the pseudo-query is expanded against the source's own shape, and the weighted sum is
accumulated source by source instead of through a stacked tensor, which also removes the
memory peak that exhausts a 96 GiB card on the OMat24 batch.

The forward is numerically identical to the original; installing it only removes the
guard, so it is applied unconditionally by the launcher and no record is kept in
``patches.json``.
"""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

    import torch

    from deepmd.pt.model.descriptor.sezm_nn.attn_res import (
        DepthAttnRes,
    )


def _forward(
    self: DepthAttnRes,
    *,
    sources: list[torch.Tensor],
    scalar_extractor: Callable[[torch.Tensor], torch.Tensor],
    current_x: torch.Tensor | None = None,
) -> torch.Tensor:
    """Aggregate same-shape sources with depth attention (see the module docstring)."""
    import torch

    source0 = sources[0]
    if len(sources) == 1:
        return source0
    value_dtype = source0.dtype

    # === Step 1. Query: the current state's scalars, or the learned pseudo-query ===
    if self.input_dependent:
        query = self.query_proj(scalar_extractor(current_x).to(dtype=self.dtype))
    else:
        query = self.adamw_pseudo_query.unsqueeze(0).expand(source0.shape[0], -1)

    # === Step 2. Keys from every source's scalars, normalized ===
    keys = self.key_norm(
        torch.stack(
            [scalar_extractor(source).to(dtype=self.dtype) for source in sources],
            dim=1,
        )
    )  # (B, S, C)
    alpha = torch.softmax(torch.einsum("bc,bsc->bs", query, keys), dim=1)  # (B, S)

    # === Step 3. Weighted sum over the sources ===
    # The sources are accumulated one at a time instead of stacked: the stack and its product
    # hold ``2 * len(sources)`` copies of the node tensor at once, which exhausts a 96 GiB card
    # on the OMat24 batch, while the running sum holds two.
    aggregated = None
    for index, source in enumerate(sources):
        weight = alpha[:, index].reshape((-1,) + (1,) * (source.ndim - 1))
        term = source.to(dtype=self.dtype) * weight
        aggregated = term if aggregated is None else aggregated + term
    return aggregated.to(dtype=value_dtype)


def install() -> None:
    """Replace ``DepthAttnRes.forward`` with the shape-agnostic version."""
    from deepmd.pt.model.descriptor.sezm_nn.attn_res import (
        DepthAttnRes,
    )

    DepthAttnRes.forward = _forward

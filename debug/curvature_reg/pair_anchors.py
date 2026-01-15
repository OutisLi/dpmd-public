# SPDX-License-Identifier: LGPL-3.0-or-later
"""Isolated-pair anchor source for the reference term.

In a drawn fraction of the anchor frames, two real atoms of the frame are kept
with their types and placed alone in a cubic cell whose side exceeds twice the
cutoff, at a separation drawn log-uniformly from ``[r_lo, r_hi]``; every other
atom of the frame becomes a phantom (type ``-1``). The reference model then
supplies the pair's forces as pseudo-labels exactly as it does for thinned,
shelled, collided and dilated cells, so the training model is held to the
reference's own pair curves, including the compressed range no data frame
visits. The geometric generators act on the remaining frames unchanged; the
pair frames are written after them, so their separation is the drawn one.

The pair draw consumes one integer from the checkpointed anchor RNG per event
and seeds its own generator with it, as the cluster source does, so the anchor
stream stays reproducible from the training seed.
"""

from __future__ import (
    annotations,
)

import logging
from functools import (
    wraps,
)
from typing import (
    Any,
)

import array_api_compat
import numpy as np

log = logging.getLogger("deepmd.pair_anchors")


def pair_source(
    inputs: dict[str, Any],
    rng: np.random.Generator,
    rcut: float,
    fraction: float,
    r_lo: float,
    r_hi: float,
) -> dict[str, Any]:
    """Replace a drawn fraction of the frames by isolated pairs of their own atoms."""
    coord, atype = inputs["coord"], inputs["atype"]
    xp = array_api_compat.array_namespace(coord, atype)
    device = array_api_compat.device(coord)
    positions = (
        np.asarray(array_api_compat.to_device(coord, "cpu")).reshape(-1, 3).copy()
    )
    types = np.asarray(array_api_compat.to_device(atype, "cpu")).reshape(-1).copy()
    if inputs.get("n_node") is None:
        counts = np.full(atype.shape[0], atype.shape[1], dtype=int)
    else:
        counts = np.asarray(array_api_compat.to_device(inputs["n_node"], "cpu"))
    offsets = np.concatenate(([0], np.cumsum(counts)))
    box = inputs.get("box")
    if box is None:
        raise ValueError("The pair-anchor source requires periodic input frames")
    cells = np.asarray(array_api_compat.to_device(box, "cpu")).reshape(-1, 3, 3).copy()
    selected_frames = rng.random(len(counts)) < fraction
    for frame in np.flatnonzero(selected_frames):
        slots = np.arange(offsets[frame], offsets[frame + 1])
        real = slots[types[slots] >= 0]
        if len(real) < 2:
            continue
        pair = rng.choice(real, size=2, replace=False)
        separation = float(np.exp(rng.uniform(np.log(r_lo), np.log(r_hi))))
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction)
        side = 2.0 * rcut + separation + 1.0
        origin = np.full(3, rcut + 0.5)
        positions[pair[0]] = origin
        positions[pair[1]] = origin + separation * direction
        types[np.setdiff1d(real, pair)] = -1
        cells[frame] = np.eye(3) * side
    result = dict(inputs)
    result["coord"] = xp.reshape(
        xp.asarray(positions, dtype=coord.dtype, device=device), coord.shape
    )
    result["atype"] = xp.reshape(
        xp.asarray(types, dtype=atype.dtype, device=device), atype.shape
    )
    result["box"] = xp.reshape(
        xp.asarray(cells, dtype=box.dtype, device=array_api_compat.device(box)),
        box.shape,
    )
    return result


def install(rcut: float, fraction: float, r_lo: float, r_hi: float) -> None:
    """Write isolated pairs into a fraction of every anchor batch after the geometric generators."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"pair-anchor fraction must lie in (0, 1], got {fraction}")
    if not 0.0 < r_lo < r_hi:
        raise ValueError(
            f"pair-anchor range must satisfy 0 < r_lo < r_hi, got {r_lo}, {r_hi}"
        )
    import deepmd.pt_expt.train.safeguard as driver

    original = driver.transform_batch
    announced = False

    @wraps(original)
    def transform_batch(
        inputs: dict[str, Any],
        params: dict[str, Any],
        rng: np.random.Generator,
        force: Any | None = None,
    ) -> dict[str, Any]:
        nonlocal announced
        source_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
        transformed = original(inputs, params, rng, force=force)
        if not announced:
            log.info(
                "Pair anchors active: fraction %.2f of the anchor frames, separation %.2f-%.2f A, cell side 2*%.1f + r + 1 A",
                fraction,
                r_lo,
                r_hi,
                rcut,
            )
            announced = True
        return pair_source(transformed, source_rng, rcut, fraction, r_lo, r_hi)

    driver.transform_batch = transform_batch

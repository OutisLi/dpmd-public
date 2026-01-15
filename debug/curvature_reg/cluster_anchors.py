# SPDX-License-Identifier: LGPL-3.0-or-later
"""Finite-cluster anchor sources with the original four geometric transforms.

The periodic control and cluster arm consume one identical draw from the
checkpointed anchor RNG per event. Cluster selection uses an independent
generator initialized by that draw, so no additional RNG state is required.
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
from ase.geometry import (
    find_mic,
)

log = logging.getLogger("deepmd.cluster_anchors")


def cluster_source(
    inputs: dict[str, Any],
    rng: np.random.Generator,
    rcut: float,
    selection: str = "random",
) -> dict[str, Any]:
    """Draw finite subclusters in half the frames without modifying the original batch."""
    if inputs.get("box") is None:
        raise ValueError("The cluster-source experiment requires periodic input frames")
    if selection not in ("random", "nearest"):
        raise ValueError(f"Unknown cluster selection {selection!r}")
    coord, atype = inputs["coord"], inputs["atype"]
    xp = array_api_compat.array_namespace(coord, atype)
    device = array_api_compat.device(coord)
    positions = (
        np.asarray(array_api_compat.to_device(coord, "cpu")).reshape(-1, 3).copy()
    )
    types = np.asarray(array_api_compat.to_device(atype, "cpu")).reshape(-1).copy()
    cells = (
        np.asarray(array_api_compat.to_device(inputs["box"], "cpu"))
        .reshape(-1, 3, 3)
        .copy()
    )
    if inputs.get("n_node") is None:
        counts = np.full(atype.shape[0], atype.shape[1], dtype=int)
    else:
        counts = np.asarray(array_api_compat.to_device(inputs["n_node"], "cpu"))
    offsets = np.concatenate(([0], np.cumsum(counts)))
    selected_frames = rng.random(len(counts)) < 0.5
    for frame in np.flatnonzero(selected_frames):
        slots = np.arange(offsets[frame], offsets[frame + 1])
        real = slots[types[slots] >= 0]
        if not len(real):
            continue
        count = (
            len(real)
            if len(real) <= 2
            else int(np.exp(rng.uniform(np.log(2), np.log(len(real) + 1))))
        )
        selected = rng.choice(real, size=count, replace=False)
        if selection == "nearest":
            # Reusing the first sampled atom preserves the size and RNG stream
            # of the random-subset control while changing only cluster locality.
            _, distances = find_mic(
                positions[real] - positions[selected[0]], cells[frame], pbc=True
            )
            selected = real[np.argsort(distances, kind="stable")[:count]]
        displacement = positions[selected] - positions[selected[0]]
        displacement, _ = find_mic(displacement, cells[frame], pbc=True)
        side = float(np.ptp(displacement, axis=0).max()) + 2.0 * rcut + 1.0
        positions[selected] = displacement - displacement.min(axis=0) + rcut + 0.5
        types[np.setdiff1d(real, selected)] = -1
        cells[frame] = np.eye(3) * side
    result = dict(inputs)
    result["coord"] = xp.reshape(
        xp.asarray(positions, dtype=coord.dtype, device=device), coord.shape
    )
    result["atype"] = xp.reshape(
        xp.asarray(types, dtype=atype.dtype, device=device), atype.shape
    )
    result["box"] = xp.reshape(
        xp.asarray(
            cells,
            dtype=inputs["box"].dtype,
            device=array_api_compat.device(inputs["box"]),
        ),
        inputs["box"].shape,
    )
    return result


def install(
    source: str, rcut: float, selection: str = "random", start_step: int = 0
) -> None:
    """Select anchor sources from an absolute optimizer step, with paired RNG consumption."""
    if source not in ("periodic", "clusters"):
        raise ValueError(f"Unknown anchor source {source!r}")
    if start_step < 0:
        raise ValueError(
            f"Anchor source start step must be non-negative, got {start_step}"
        )
    import deepmd.pt_expt.train.safeguard as driver

    original = driver.transform_batch
    current_step = 0
    reported_phase = None

    original_step = driver.TrainingSafeguard.anchor_step

    @wraps(original_step)
    def anchor_step(
        self: Any,
        wrapper: Any,
        task_key: str,
        input_dict: dict[str, Any],
        label_dict: dict[str, Any],
        step: int,
        sync_module: Any | None = None,
    ) -> None:
        nonlocal current_step
        current_step = step
        original_step(
            self, wrapper, task_key, input_dict, label_dict, step, sync_module
        )

    driver.TrainingSafeguard.anchor_step = anchor_step

    @wraps(original)
    def transform_batch(
        inputs: dict[str, Any],
        params: dict[str, Any],
        rng: np.random.Generator,
        force: Any | None = None,
    ) -> dict[str, Any]:
        nonlocal reported_phase
        source_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
        active = current_step >= start_step
        if active != reported_phase:
            log.info(
                "Anchor source %s: active=%s at optimizer step %d, configured start=%d",
                source,
                active,
                current_step,
                start_step,
            )
            reported_phase = active
        prepared = (
            cluster_source(inputs, source_rng, rcut, selection)
            if source == "clusters" and active
            else inputs
        )
        return original(prepared, params, rng, force=force)

    driver.transform_batch = transform_batch

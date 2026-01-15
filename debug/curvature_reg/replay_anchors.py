# SPDX-License-Identifier: LGPL-3.0-or-later
"""Mix pretraining geometries into a fine-tuning model's reference anchors."""

from __future__ import (
    annotations,
)

import logging
from functools import (
    wraps,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

import array_api_compat
import numpy as np

if TYPE_CHECKING:
    from deepmd.dpmodel.utils.lmdb_data import (
        LmdbDataReader,
    )

log = logging.getLogger("deepmd.replay_anchors")


def replay_source(
    inputs: dict[str, Any],
    force: Any | None,
    reader: LmdbDataReader,
    rng: np.random.Generator,
    fraction: float,
) -> tuple[dict[str, Any], Any | None, list[int]]:
    """Replace selected rectangular frames with complete LMDB configurations.

    Frames must have periodic cells and a fixed number of atom slots. Replay
    configurations are sampled uniformly conditional on fitting those slots;
    surplus slots receive the existing phantom type, -1. No atom is truncated.
    Source force labels replace collision directions only. The frozen reference
    supplies the actual supervision on the subsequently transformed geometry.
    Coordinates and cells are in A; force directions are in eV/A.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"Replay fraction must be in [0, 1], got {fraction}")
    if inputs.get("n_node") is not None or len(inputs["atype"].shape) != 2:
        raise ValueError("Replay anchors require rectangular atom slots")
    if inputs.get("box") is None:
        raise ValueError("Replay anchors require periodic input frames")
    extra = [
        name
        for name, value in inputs.items()
        if name not in ("coord", "atype", "box") and value is not None
    ]
    if extra:
        raise ValueError(f"Replay anchors do not define frame-specific inputs {extra}")
    coord, atype, box = inputs["coord"], inputs["atype"], inputs["box"]
    frames, slots = atype.shape
    selected = np.flatnonzero(rng.random(frames) < fraction)
    if not len(selected):
        return inputs, force, []
    if not any(0 < count <= slots for count in reader.frame_nlocs):
        raise ValueError(
            f"Replay dataset has no complete configuration fitting {slots} atom slots"
        )
    xp = array_api_compat.array_namespace(coord, atype, box)
    positions = (
        np.asarray(array_api_compat.to_device(coord, "cpu"))
        .reshape(frames, slots, 3)
        .copy()
    )
    types = np.asarray(array_api_compat.to_device(atype, "cpu")).copy()
    cells = np.asarray(array_api_compat.to_device(box, "cpu")).reshape(frames, 9).copy()
    directions = (
        None
        if force is None
        else np.asarray(array_api_compat.to_device(force, "cpu"))
        .reshape(frames, slots, 3)
        .copy()
    )
    indices = []
    for frame in selected:
        index = int(rng.integers(len(reader.frame_nlocs)))
        while not 0 < reader.frame_nlocs[index] <= slots:
            index = int(rng.integers(len(reader.frame_nlocs)))
        datum = reader[index]
        count = int(reader.frame_nlocs[index])
        positions[frame] = 0
        positions[frame, :count] = datum["coord"].reshape(count, 3)
        types[frame] = -1
        types[frame, :count] = datum["atype"].reshape(count)
        cells[frame] = datum["box"].reshape(9)
        if directions is not None:
            directions[frame] = 0
            directions[frame, :count] = datum["force"].reshape(count, 3)
        indices.append(index)
    result = dict(inputs)
    for name, host, original in (
        ("coord", positions, coord),
        ("atype", types, atype),
        ("box", cells, box),
    ):
        result[name] = xp.reshape(
            xp.asarray(
                host, dtype=original.dtype, device=array_api_compat.device(original)
            ),
            original.shape,
        )
    if directions is not None:
        force = xp.reshape(
            xp.asarray(
                directions, dtype=force.dtype, device=array_api_compat.device(force)
            ),
            force.shape,
        )
    return result, force, indices


def install(path: str, type_map: list[str], fraction: float) -> None:
    """Keep the anchor loss and transforms while selecting their source distribution."""
    import deepmd.pt_expt.train.safeguard as driver
    from deepmd.dpmodel.utils.lmdb_data import (
        LmdbDataReader,
    )

    reader = LmdbDataReader(path, type_map, batch_size=1)
    original = driver.transform_batch
    events = 0

    @wraps(original)
    def transform_batch(
        inputs: dict[str, Any],
        params: dict[str, Any],
        rng: np.random.Generator,
        force: Any | None = None,
    ) -> dict[str, Any]:
        nonlocal events
        source_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
        prepared, directions, indices = replay_source(
            inputs, force, reader, source_rng, fraction
        )
        events += 1
        if events == 1 or events % 100 == 0:
            log.info(
                "Replay anchor event %d: %d/%d source frames from %s; indices=%s",
                events,
                len(indices),
                inputs["atype"].shape[0],
                path,
                indices,
            )
        return original(prepared, params, rng, force=directions)

    driver.transform_batch = transform_batch

# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201, TID253
"""Measure reference consistency by coordination on fixed training-frame anchors."""

from __future__ import (
    annotations,
)

import argparse
import json
import sys
from pathlib import (
    Path,
)

import numpy as np
import torch
from ase import (
    Atoms,
)
from ase.neighborlist import (
    neighbor_list,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cluster_anchors import (
    cluster_source,
)
from repro_spike import (
    load_model,
)

from deepmd.dpmodel.utils.lmdb_data import (
    LmdbDataReader,
)
from deepmd.dpmodel.utils.safeguard import (
    resolve_safeguard_params,
    transform_batch,
)


def predictions(
    model: torch.nn.Module, inputs: dict[str, np.ndarray], batch: int
) -> np.ndarray:
    """Pack equal-size frames for the PT API and evaluate coordinate gradients, in eV/A."""
    model.requires_grad_(False)
    counts = inputs.get("n_node")
    if counts is None:
        counts = np.full(len(inputs["atype"]), inputs["atype"].shape[1])
    offsets = np.r_[0, counts.cumsum()]
    positions = inputs["coord"].reshape(-1, 3)
    types = inputs["atype"].reshape(-1)
    result = np.empty_like(positions)
    for count in np.unique(counts):
        frames = np.flatnonzero(counts == count)
        for start in range(0, len(frames), batch):
            selected = frames[start : start + batch]
            slots = offsets[selected, None] + np.arange(count)
            out = model(
                torch.as_tensor(positions[slots], device="cuda"),
                torch.as_tensor(types[slots], device="cuda"),
                box=torch.as_tensor(inputs["box"][selected], device="cuda"),
            )
            result[slots] = out["force"].detach().cpu().numpy().reshape(*slots.shape, 3)
    result = result.reshape((*inputs["atype"].shape, 3))
    if not np.isfinite(result).all():
        raise ValueError("Non-finite anchor prediction")
    return result


def coordination(inputs: dict[str, np.ndarray], cutoff: float) -> np.ndarray:
    """Count all periodic neighbours of real atoms within the model cutoff."""
    counts = inputs.get("n_node")
    if counts is None:
        counts = np.full(len(inputs["atype"]), inputs["atype"].shape[1])
    offsets = np.r_[0, counts.cumsum()]
    result = np.full(inputs["atype"].size, -1, dtype=int)
    positions = inputs["coord"].reshape(-1, 3)
    all_types = inputs["atype"].reshape(-1)
    for frame in range(len(counts)):
        start, end = offsets[frame : frame + 2]
        types = all_types[start:end]
        real = types >= 0
        if not real.any():
            continue
        atoms = Atoms(
            "H" * int(real.sum()),
            positions=positions[start:end][real],
            cell=inputs["box"][frame].reshape(3, 3),
            pbc=True,
        )
        src = neighbor_list("i", atoms, cutoff, self_interaction=False)
        result[start:end][real] = np.bincount(src, minlength=int(real.sum()))
    return result.reshape(inputs["atype"].shape)


def training_sample(
    config: dict, frames: int, type_map: list[str]
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Read fixed training frames in the dataset's rectangular or ragged layout."""
    path = Path(config["training"]["training_data"]["systems"])
    rng = np.random.default_rng(6129)
    if path.suffix == ".lmdb":
        reader = LmdbDataReader(str(path), type_map, batch_size=1)
        draw = rng.choice(len(reader.frame_nlocs), frames, replace=False)
        rows = [reader[int(index)] for index in draw]
        inputs = {
            "coord": np.concatenate([row["coord"].reshape(-1, 3) for row in rows]),
            "atype": np.concatenate([row["atype"].reshape(-1) for row in rows]),
            "box": np.stack([row["box"].reshape(9) for row in rows]),
            "n_node": np.asarray(reader.frame_nlocs[draw], dtype=np.int64),
        }
        force = np.concatenate([row["force"].reshape(-1, 3) for row in rows])
        return inputs, force, draw
    data = path / "set.000"
    coords = np.load(data / "coord.npy")
    atoms = coords[0].size // 3
    draw = rng.choice(len(coords), frames, replace=False)
    inputs = {
        "coord": coords.reshape(-1, atoms, 3)[draw],
        "box": np.load(data / "box.npy").reshape(-1, 9)[draw],
        "atype": np.full((frames, atoms), type_map.index("H"), dtype=np.int64),
    }
    force = np.load(data / "force.npy").reshape(-1, atoms, 3)[draw]
    return inputs, force, draw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("student", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--selection", choices=["random", "nearest"], default="random")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["periodic", "clusters"],
        default=["periodic", "clusters"],
    )
    parser.add_argument(
        "--stored-reference",
        action="store_true",
        help="Use the exact frozen reference stored in the teacher checkpoint",
    )
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(".npz").exists():
        raise FileExistsError(args.out)
    root = args.student.parent
    if root.name == "ckpt":
        root = root.parent
    config = json.loads((root / "input.json").read_text())
    patch_record = root / "patches.json"
    patches = json.loads(patch_record.read_text()) if patch_record.exists() else {}
    squared_loss = patches.get("squared_reference_loss", False)
    student, type_map = load_model(args.student, "cuda")
    teacher, teacher_types = load_model(args.teacher, "cuda")
    if type_map != teacher_types:
        raise ValueError(
            "The diagnostic requires identical teacher and student type maps"
        )
    finetune = True
    if args.stored_reference:
        state = torch.load(args.teacher, map_location="cpu", weights_only=False)
        finetune = bool(state["safeguard"]["finetune"])
        reference = state["safeguard"]["reference"]
        teacher.load_state_dict(
            {
                key.removeprefix("model.Default."): value
                for key, value in reference.items()
                if key.startswith("model.Default.")
            },
            strict=True,
        )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inputs, forces, draw = training_sample(config, args.frames, type_map)
    cutoff = float(config["model"]["descriptor"]["rcut"])
    params = resolve_safeguard_params(config["loss"]["safeguard"], finetune)
    summaries = []
    arrays = {"source_frame_indices": draw}
    for source in args.sources:
        rng = np.random.default_rng(9325)
        source_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
        prepared = (
            inputs
            if source == "periodic"
            else cluster_source(inputs, source_rng, cutoff, args.selection)
        )
        anchor = transform_batch(prepared, params, rng, force=forces)
        degree = coordination(anchor, cutoff)
        reference = predictions(teacher, anchor, args.batch)
        predicted = predictions(student, anchor, args.batch)
        ref_norm = np.linalg.norm(reference, axis=-1)
        difference = np.linalg.norm(predicted - reference, axis=-1)
        eligible = (anchor["atype"] >= 0) & (ref_norm <= params["fcap"])
        tolerance = (
            0.0 if squared_loss else params["tol"] + params["rel_tol"] * ref_norm
        )
        excess = np.maximum(difference - tolerance, 0)
        penalty = np.where(eligible, excess**2, 0)
        records = []
        for lo, hi in ((0, 0), (1, 1), (2, 3), (4, 15), (16, 1000000)):
            selected = eligible & (degree >= lo) & (degree <= hi)
            count = int(selected.sum())
            if not count:
                continue
            record = {
                "degree_range": [lo, hi],
                "eligible_atoms": count,
                "eligible_atom_share": float(count / eligible.sum()),
                "active_fraction": float((excess[selected] > 0).mean()),
                "mean_force_difference_eV_A": float(difference[selected].mean()),
                "maximum_force_difference_eV_A": float(difference[selected].max()),
                "squared_excess_sum": float(penalty[selected].sum()),
            }
            records.append(record)
            print(source, json.dumps(record), flush=True)
        summaries.append(
            {
                "source": source,
                "real_atoms": int((degree >= 0).sum()),
                "eligible_atoms": int(eligible.sum()),
                "groups": records,
            }
        )
        arrays.update(
            {
                f"{source}_degree": degree,
                f"{source}_difference": difference,
                f"{source}_reference_norm": ref_norm,
                f"{source}_eligible": eligible,
                f"{source}_excess": excess,
            }
        )
    args.out.write_text(
        json.dumps(
            {
                "student": str(args.student.resolve()),
                "teacher": str(args.teacher.resolve()),
                "frames": args.frames,
                "stored_reference": args.stored_reference,
                "reference_loss": "squared_error" if squared_loss else "hinge",
                "anchor_phase": "dense_and_sparse" if finetune else "sparse",
                "parameters": params,
                "selection": args.selection,
                "samples": summaries,
            },
            indent=2,
        )
        + "\n"
    )
    np.savez_compressed(args.out.with_suffix(".npz"), **arrays)


if __name__ == "__main__":
    main()

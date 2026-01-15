# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201
"""
Run ``dp --pt train`` with the safeguard's reference refreshed periodically.

The package freezes the reference once, at ``ref_step``: for a fine-tune the
pretrained model, for a pretraining the random initialization.  A random
initialization is smooth and bounded but carries no physics, and on the
sparse anchors it contradicts what the model has already generalized from
the data (the repulsion of a close pair, the depth of a bond), which is the
cause of the environment-dependent contact forces of the production run.
This launcher replaces the reference every ``refresh`` steps by a fresh
frozen copy of the model as it then is, so that the reference always carries
the physics the data have taught, and the term only forbids the fast
off-data drift of the interval.  The first interval starts from the random
initialization as before.  Everything else is the package's own safeguard;
the change is applied by monkey patch on the driver so that the installed
package stays untouched.
"""

from __future__ import (
    annotations,
)

import argparse
import copy
import json
import logging
import sys
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    import torch

    from deepmd.pt_expt.train.safeguard import (
        TrainingSafeguard,
    )

# A child of the package logger, so the report lands in the training log.
log = logging.getLogger("deepmd.train_refresh")


def install(
    refresh: int,
    max_refreshes: int,
    dense_after_refresh: bool,
    collision_after_refresh: bool,
    pin_sparse_after_refresh: bool,
    near_after_refresh: bool,
    label_tube_after_refresh: bool,
    rel_after_refresh: float | None,
    compose_generators: bool,
    dilate_after_refresh: float | None,
    label_tube_near_only: bool = False,
    pair_term: bool = False,
    pair_tol: float = 5.0,
    pair_rel: float = 1.0,
    anchor_descent: int = 0,
    anchor_descent_step: float = 0.05,
    ensemble_refs: list[str] | None = None,
    pair_term_anchors: bool = False,
    refresh_steps: set[int] | None = None,
    repulsion_weight: float = 0.0,
    repulsion_ratio: float = 0.55,
    anchor_draws: int = 1,
    pair_contact_ratio: float = 0.0,
    pair_contact_tol: float = 1.0,
    pair_contact_rel: float = 0.5,
    fstep_after_refresh: float | None = None,
    pair_squeeze: tuple[float, float] | None = None,
    pair_squeeze_ratio: tuple[float, float] | None = None,
    dimer_weight: float = 0.0,
    dimer_ratio: tuple[float, float] = (0.3, 0.5),
    dimer_count: int = 32,
    dimer_contacts: str | None = None,
    data_sign_weight: float = 0.0,
    data_sign_margin: float = 5.0,
    data_sign_count: int = 32,
    tail_weight: float = 0.0,
    tail_range: tuple[float, float] = (4.0, 6.0),
    tail_count: int = 24,
    type_substitution: tuple[float, float] | None = None,
) -> None:
    """Patch the driver so the reference is re-frozen every ``refresh`` steps."""
    import contextlib
    import inspect

    import numpy as np
    import torch

    import deepmd.pt_expt.train.safeguard as driver

    original = driver.TrainingSafeguard.anchor_step
    stage_applied = False
    # After a pinning hand-over the anchor events alternate between two
    # families: the far anchors (dilation, deletion) with a closed tube, which
    # pin the model to the reference where the data do not reach, and the near
    # anchors (Gaussian shell, collision) with the ordinary tube, inside which
    # the data keep shaping the model.
    families: dict[str, list[dict]] = {}
    # With the label-widened tube, every anchor atom's tube is widened by the
    # reference's own error on that atom in the frame the anchor was made
    # from: the reference is trusted, atom by atom, only to the extent it
    # reproduces the atom's label. An atom keeps its identity under every
    # generator (a deleted atom is a phantom and outside the hinge anyway), so
    # one rule covers the near and the far anchors. ``event["ref_error"]``
    # holds |F_ref - F_label| per atom of the current event's source batch,
    # with shape (nf, nloc), from one reference forward on the source frames;
    # the patched hinge subtracts it from the excess.
    event: dict = {"ref_error": None, "far_atoms": None}
    original_hinge = driver.reference_hinge

    def hinge_with_label(pred, ref, atype, tol, rel_tol, fcap):  # noqa: ANN001, ANN202
        extra = event["ref_error"]
        if extra is None:
            return original_hinge(pred, ref, atype, tol, rel_tol, fcap)
        extra = extra.reshape(atype.shape)
        far = event["far_atoms"]
        if far is not None:
            # Near/far separation: the reference's label error widens the
            # tube only on the frames that kept the data's density (shell,
            # collision); the thinned and stretched frames keep the fixed
            # tube, so the far anchors hold the isolated pairs as before.
            extra = torch.where(
                far.reshape(atype.shape), torch.zeros_like(extra), extra
            )
        diff = torch.sqrt(((pred - ref) ** 2).sum(-1) + 1e-30)
        ref_norm = torch.sqrt((ref**2).sum(-1) + 1e-30)
        trusted = torch.logical_and(atype >= 0, ref_norm <= fcap)
        excess = diff - tol - rel_tol * ref_norm - extra
        excess = torch.where(trusted, excess.clamp(min=0.0), torch.zeros_like(excess))
        count = trusted.to(pred.dtype).sum()
        return (excess * excess).sum() / torch.clamp(count, min=1.0)

    if label_tube_after_refresh:
        driver.reference_hinge = hinge_with_label

    # The package draws one generator per anchor frame. With composition every
    # enabled generator is applied in sequence to every frame — deletion, then
    # the Gaussian shell, then the collision, then the dilation — so that one
    # anchor frame combines a thinned environment, a displaced and compressed
    # pair, and a stretched box, which is the geometry of the isolated pairs
    # in which the collapse forms. Each stage runs the package's own transform
    # with the other generators switched off, so the amplitudes are the ones
    # of the resolved parameters.
    original_transform = driver.transform_batch
    off = {"sigma": [], "fstep": 0.0, "dilate": [1.0, 1.0], "drop": [0.0, 0.0]}

    def compose_transform(input_dict, params, rng, force=None):  # noqa: ANN001, ANN202
        enabled = {
            "drop": params["drop"][1] > 0.0,
            "sigma": bool(params["sigma"]),
            "fstep": params["fstep"] > 0.0 and force is not None,
            "dilate": params["dilate"][1] > 1.0,
        }
        out = input_dict
        for key in ("drop", "sigma", "fstep", "dilate"):
            if enabled[key]:
                single = {**params, **off, key: params[key]}
                out = original_transform(
                    out, single, rng, force=force if key == "fstep" else None
                )
        return out

    if compose_generators:
        driver.transform_batch = compose_transform

    # The geometric generators hold the model to the reference on displaced,
    # thinned and stretched copies of the data. The elements the data do not
    # contain receive no such hold: their type embeddings never see a gradient,
    # the shared network moves on without them, and the model's response at those
    # embeddings grows without bound. Type substitution extends the same hinge to
    # the type axis: after the geometric transform, on a drawn fraction of the
    # anchor frames a drawn fraction of the real atoms takes a type drawn
    # uniformly from the whole type map, so that every type embedding, present in
    # the data or not, is held to the reference's response in the data's
    # geometries. The draw uses the anchor generator's stream, so the anchors
    # remain reproducible from the training seed.
    substitution = {"ntypes": None, "events": 0, "atoms": 0, "frames": 0}
    geometric_transform = driver.transform_batch

    def substitute_transform(input_dict, params, rng, force=None):  # noqa: ANN001, ANN202
        out = geometric_transform(input_dict, params, rng, force=force)
        atom_fraction, frame_fraction = type_substitution
        atype = out["atype"]
        flat = atype.reshape(-1)
        counts = frame_counts(out)
        frames = np.repeat(np.arange(len(counts)), counts)
        real = flat.detach().cpu().numpy() >= 0
        chosen_frames = rng.random(len(counts)) < frame_fraction
        chosen = (
            chosen_frames[frames] & real & (rng.random(len(frames)) < atom_fraction)
        )
        n_chosen = int(chosen.sum())
        if n_chosen:
            drawn = rng.integers(substitution["ntypes"], size=n_chosen)
            new_flat = flat.clone()
            new_flat[torch.as_tensor(chosen, device=flat.device)] = torch.as_tensor(
                drawn, dtype=flat.dtype, device=flat.device
            )
            out["atype"] = new_flat.reshape(atype.shape)
        substitution["events"] += 1
        substitution["atoms"] += n_chosen
        substitution["frames"] += int(chosen_frames.sum())
        if substitution["events"] <= 3 or substitution["events"] % 1000 == 0:
            log.info(
                "Type substitution event %d: %d of %d real atoms in %d of %d frames "
                "took a type drawn from %d (cumulative %d atoms in %d frames).",
                substitution["events"],
                n_chosen,
                int(real.sum()),
                int(chosen_frames.sum()),
                len(counts),
                substitution["ntypes"],
                substitution["atoms"],
                substitution["frames"],
            )
        return out

    def frame_counts(input_dict):  # noqa: ANN001, ANN202
        """Atoms per frame for either batch layout (one host copy of ``n_node``)."""
        n_node = input_dict.get("n_node")
        if n_node is not None:
            return [int(n) for n in n_node.tolist()]
        nf, nloc = input_dict["atype"].shape
        return [nloc] * nf

    if type_substitution is not None:
        driver.transform_batch = substitute_transform

    # A frame is far when the transform changed its box (dilation) or turned
    # atoms into phantoms (deletion); the classification reads the effect, so
    # a drawn generator that left a frame unchanged counts as near.
    current_transform = driver.transform_batch

    def transform_recording_family(input_dict, params, rng, force=None):  # noqa: ANN001, ANN202
        out = current_transform(input_dict, params, rng, force=force)
        counts = frame_counts(input_dict)
        atype_in = input_dict["atype"].reshape(-1)
        atype_out = out["atype"].reshape(-1)
        device = atype_in.device
        frame_of = torch.repeat_interleave(
            torch.arange(len(counts), device=device),
            torch.tensor(counts, device=device),
        )
        far_frame = torch.zeros(len(counts), dtype=torch.bool, device=device)
        lost = torch.logical_and(atype_in >= 0, atype_out < 0)
        far_frame[frame_of[lost]] = True
        box_in, box_out = input_dict.get("box"), out.get("box")
        if box_in is not None and box_out is not None:
            changed = (
                box_in.reshape(len(counts), -1) != box_out.reshape(len(counts), -1)
            ).any(-1)
            far_frame |= changed.to(device)
        event["far_atoms"] = far_frame[frame_of]
        return out

    if label_tube_near_only:
        driver.transform_batch = transform_recording_family

    # Model-driven anchors: after the generators, every anchor frame follows
    # the model's own forces for a few steepest-descent steps (each real atom
    # moves along its force by at most ``anchor_descent_step`` Å per step),
    # so that the anchors land where the model's dynamics would carry the
    # configuration — a forming hole attracts them — instead of where a
    # random generator left them. The reference is consulted at the moved
    # positions. The descent needs no parameter gradients: forces are read
    # from a forward whose graph is dropped after each step.
    descent_state = {"wrapper": None, "task_key": None}
    descent_base = driver.transform_batch

    def transform_with_descent(input_dict, params, rng, force=None):  # noqa: ANN001, ANN202
        out = descent_base(input_dict, params, rng, force=force)
        wrapper, task_key = descent_state["wrapper"], descent_state["task_key"]
        if wrapper is None:
            return out
        coord = out["coord"]
        # Both layouts meet in the flat (n_atoms, 3) view; phantom atoms stay put.
        real = (out["atype"].reshape(-1) >= 0).to(coord.dtype).reshape(-1, 1)
        for _ in range(anchor_descent):
            inputs = {k: v for k, v in out.items() if v is not None}
            f = (
                driver.TrainingSafeguard._predict(wrapper, inputs, task_key)["force"]
                .detach()
                .reshape(-1, 3)
            )
            norm = torch.linalg.vector_norm(f, dim=-1, keepdim=True)
            step = (
                f
                / torch.clamp(norm, min=1e-12)
                * torch.clamp(norm, max=anchor_descent_step)
            )
            coord = (coord.reshape(-1, 3) + step * real).reshape(coord.shape)
            out = dict(out, coord=coord)
        return out

    if anchor_descent > 0:
        driver.transform_batch = transform_with_descent

    # Pair squeeze (ledger E55): after the drawn generator, one pair per anchor
    # frame is closed to a distance drawn uniformly from ``pair_squeeze`` — a
    # random real atom and its nearest neighbour (minimum image), moved toward
    # each other symmetrically along their separation — so that every anchor
    # event labels the region below the data's shortest contacts, which the
    # generators alone reach once in a few hundred frames. Rectangular batches
    # only; a pair already inside the drawn distance is left where it is.
    squeeze_base = driver.transform_batch
    squeeze_log = {"calls": 0}
    # The squeeze waits for the hand-over: before it the reference is the random
    # initialization, whose function below the data's contacts labels nothing.
    squeeze_state = {"active": False, "radii": None}
    # With ``pair_squeeze_ratio`` the drawn distance is a fraction of the pair's
    # covalent contact (the sum of the two covalent radii) instead of an absolute
    # distance, so that every pair type is closed to the same physical regime —
    # the one in which the frozen reference is trusted (0.30-1.00 contacts) and in
    # which the tearing three-body states of §11.39 live — rather than the heavy
    # pairs to a fraction far below it (E55's 0.30-0.55 Å were 0.11-0.35 contacts
    # for Cu-Cu). The per-type radii are resolved from the model's type map at the
    # first anchor step.

    def transform_with_squeeze(input_dict, params, rng, force=None):  # noqa: ANN001, ANN202
        out = squeeze_base(input_dict, params, rng, force=force)
        if not squeeze_state["active"]:
            return out
        if out.get("n_node") is not None:
            raise NotImplementedError(
                "pair squeeze expects a rectangular batch (nf, nloc, 3)"
            )
        coord = out["coord"]
        nf, nloc = out["atype"].shape
        x = coord.reshape(nf, nloc, 3)
        real = (out["atype"] >= 0).to(x.device)
        n_real = real.sum(dim=1)
        # === Step 1. One random real atom per frame, and its nearest real neighbour ===
        pick_np = np.array(
            [rng.integers(int(n)) if n > 1 else 0 for n in n_real.tolist()]
        )
        order = torch.argsort(
            (~real).to(torch.int64), dim=1, stable=True
        )  # real atoms first
        i_idx = order[
            torch.arange(nf, device=x.device), torch.as_tensor(pick_np, device=x.device)
        ]
        xi = x[torch.arange(nf, device=x.device), i_idx]  # (nf, 3)
        d = x - xi[:, None, :]  # (nf, nloc, 3), from atom i to every atom
        box = out.get("box")
        if box is not None:
            h = box.reshape(nf, 3, 3).to(device=x.device, dtype=x.dtype)
            s = torch.linalg.solve(h.transpose(1, 2), d.transpose(1, 2)).transpose(
                1, 2
            )  # fractional
            s = s - torch.round(s)
            d = s @ h
        r = torch.linalg.vector_norm(d, dim=-1)  # (nf, nloc)
        r = torch.where(real, r, torch.full_like(r, float("inf")))
        r[torch.arange(nf, device=x.device), i_idx] = float("inf")
        j_idx = torch.argmin(r, dim=1)
        rij = r[torch.arange(nf, device=x.device), j_idx]
        # === Step 2. Close the pair to the drawn distance, symmetrically ===
        if pair_squeeze_ratio is not None:
            radii = squeeze_state["radii"].to(device=x.device, dtype=x.dtype)
            types_i = out["atype"][torch.arange(nf, device=x.device), i_idx].clamp(
                min=0
            )
            types_j = out["atype"][torch.arange(nf, device=x.device), j_idx].clamp(
                min=0
            )
            contact = radii[types_i.to(radii.device)] + radii[types_j.to(radii.device)]
            ratio = torch.as_tensor(
                rng.uniform(pair_squeeze_ratio[0], pair_squeeze_ratio[1], size=nf),
                device=x.device,
                dtype=x.dtype,
            )
            target = ratio * contact.to(device=x.device)
        else:
            target = torch.as_tensor(
                rng.uniform(pair_squeeze[0], pair_squeeze[1], size=nf),
                device=x.device,
                dtype=x.dtype,
            )
        valid = (n_real > 1) & torch.isfinite(rij) & (rij > target)
        shift = torch.where(valid, (rij - target) / 2.0, torch.zeros_like(rij))
        unit = (
            d[torch.arange(nf, device=x.device), j_idx]
            / torch.clamp(rij, min=1e-12)[:, None]
        )
        x = x.clone()
        x[torch.arange(nf, device=x.device), i_idx] += shift[:, None] * unit
        x[torch.arange(nf, device=x.device), j_idx] -= shift[:, None] * unit
        if squeeze_log["calls"] < 3:
            squeeze_log["calls"] += 1
            log.info(
                "Pair squeeze: %d frames, nearest-neighbour distance of the chosen atoms %.2f-%.2f A before, "
                "%.2f-%.2f A after (%d frames closed).",
                nf,
                float(rij[valid].min()) if valid.any() else float("nan"),
                float(rij[valid].max()) if valid.any() else float("nan"),
                float(target[valid].min()) if valid.any() else float("nan"),
                float(target[valid].max()) if valid.any() else float("nan"),
                int(valid.sum()),
            )
        return dict(out, coord=x.reshape(coord.shape))

    if pair_squeeze is not None or pair_squeeze_ratio is not None:
        driver.transform_batch = transform_with_squeeze
        if pair_squeeze_ratio is not None:
            log.info(
                "Pair squeeze installed: one pair per anchor frame closed to [%.2f, %.2f] covalent contacts.",
                *pair_squeeze_ratio,
            )
        else:
            log.info(
                "Pair squeeze installed: one pair per anchor frame closed to [%.2f, %.2f] A.",
                *pair_squeeze,
            )

    # Worst-of-K anchor selection (ledger E4): every anchor frame is drawn
    # ``anchor_draws`` times from the active transform and the draw on which
    # the model violates the reference tube most is the one trained on, so
    # the restoring force at a rare region no longer depends on how often a
    # random draw happens to visit it. Each candidate is scored without
    # parameter gradients by the frame's share of the driver's hinge (the
    # summed squared excess over the trusted atoms, with the plain tube); the
    # driver then runs its gradient forward on the selected batch. Every draw
    # keeps the source batch's atom layout (deleted atoms become phantoms), so
    # the selection is a per-frame gather over the draw axis in the flat
    # per-atom view shared by the rectangular and the ragged layout.
    worst_state: dict = {"driver": None, "wrapper": None, "task_key": None}
    worst_stats = {"events": 0, "selected": 0.0, "mean": 0.0}
    worst_base = driver.transform_batch

    def frame_excess(safeguard, wrapper, task_key, draw, params, frames, nf):  # noqa: ANN001, ANN202
        """Summed squared hinge excess of every frame of one draw, with shape (nf,)."""
        inputs = {k: v for k, v in draw.items() if v is not None}
        atype = inputs["atype"].reshape(-1)
        ref = (
            safeguard._predict(safeguard.reference, inputs, task_key)["force"]
            .detach()
            .reshape(-1, 3)
        )
        pred = (
            safeguard._predict(wrapper, inputs, task_key)["force"]
            .detach()
            .reshape(-1, 3)
        )
        diff = torch.linalg.vector_norm(pred - ref, dim=-1)
        ref_norm = torch.linalg.vector_norm(ref, dim=-1)
        trusted = torch.logical_and(atype >= 0, ref_norm <= params["fcap"])
        excess = (diff - params["tol"] - params["rel_tol"] * ref_norm).clamp(min=0.0)
        excess = torch.where(trusted, excess, torch.zeros_like(excess))
        return torch.zeros(nf, dtype=excess.dtype, device=excess.device).index_add_(
            0, frames, excess.square()
        )

    def transform_worst_of_k(input_dict, params, rng, force=None):  # noqa: ANN001, ANN202
        draws = [
            worst_base(input_dict, params, rng, force=force)
            for _ in range(anchor_draws)
        ]
        safeguard, wrapper, task_key = (
            worst_state["driver"],
            worst_state["wrapper"],
            worst_state["task_key"],
        )
        if wrapper is None or safeguard.reference is None:
            return draws[0]
        counts = frame_counts(input_dict)
        nf, n_atoms = len(counts), sum(counts)
        device = draws[0]["coord"].device
        frames = torch.repeat_interleave(
            torch.arange(nf, device=device), torch.tensor(counts, device=device)
        )
        scores = torch.stack(
            [
                frame_excess(safeguard, wrapper, task_key, d, params, frames, nf)
                for d in draws
            ]
        )
        choice = scores.argmax(dim=0)  # (nf,): the draw every frame takes
        out = dict(draws[0])
        for key, per_atom in (("coord", True), ("atype", True), ("box", False)):
            if out.get(key) is None:
                continue
            stack = torch.stack([d[key] for d in draws])
            if per_atom:
                flat = stack.reshape(anchor_draws, n_atoms, -1)
                picked = flat[
                    choice[frames].to(flat.device),
                    torch.arange(n_atoms, device=flat.device),
                ]
            else:
                flat = stack.reshape(anchor_draws, nf, -1)
                picked = flat[
                    choice.to(flat.device), torch.arange(nf, device=flat.device)
                ]
            out[key] = picked.reshape(draws[0][key].shape)
        worst_stats["events"] += 1
        worst_stats["selected"] += float(scores.max(dim=0).values.sum()) / nf
        worst_stats["mean"] += float(scores.mean(dim=0).sum()) / nf
        return out

    if anchor_draws > 1:
        driver.transform_batch = transform_worst_of_k

    class EnsembleReference(torch.nn.Module):
        """A reference whose prediction is the mean of several frozen models.

        The members are the model's own copy at the hand-over and independently
        trained checkpoints of the same architecture at the same step, loaded
        into further copies. The forward mirrors a model wrapper's: it returns
        ``(prediction, None, None)`` with every tensor output averaged over the
        members, so the driver's ``_predict`` reads it like a single reference.
        """

        def __init__(self, members) -> None:  # noqa: ANN001
            super().__init__()
            self.members = torch.nn.ModuleList(members)

        def forward(self, *args, skip_loss=False, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
            preds = []
            for m in self.members:
                out = (
                    m(*args, skip_loss=skip_loss, **kwargs)
                    if "skip_loss" in inspect.signature(m.forward).parameters
                    else m(*args, **kwargs)
                )
                preds.append(out[0])
            mean = dict(preds[0])
            for key, value in preds[0].items():
                if (
                    isinstance(value, torch.Tensor)
                    and value.is_floating_point()
                    and all(
                        isinstance(q.get(key), torch.Tensor)
                        and q[key].shape == value.shape
                        for q in preds[1:]
                    )
                ):
                    mean[key] = torch.stack([q[key] for q in preds]).mean(0)
            return mean, None, None

    def ensemble_reference(wrapper):  # noqa: ANN001, ANN202
        members = [copy.deepcopy(wrapper)]
        for path in ensemble_refs:
            member = copy.deepcopy(wrapper)
            state = torch.load(path, map_location="cpu", weights_only=False)
            member.load_state_dict(state["model"] if "model" in state else state)
            members.append(member)
        log.info(
            "Safeguard reference is the mean of %d models (the model's own copy and %s).",
            len(members),
            ", ".join(ensemble_refs),
        )
        return EnsembleReference(members)

    pair_stats = {"events": 0, "violations": 0.0, "excess": 0.0}

    def closest_pairs(input_dict, force):  # noqa: ANN001, ANN202
        """Closest real pair of every frame (minimum image): the two-atom copies in the frame's cell,
        the unit vector from the first atom to the second, and the radial force ``(F_j - F_i) . u / 2``
        that ``force`` gives the pair, positive when repulsive.
        """
        coord, atype, box = (
            input_dict["coord"],
            input_dict["atype"],
            input_dict.get("box"),
        )
        counts = frame_counts(input_dict)
        xyz = coord.reshape(-1, 3)
        types = atype.reshape(-1)
        values = force.reshape(-1, 3).to(xyz.dtype)
        boxes = (
            box.reshape(len(counts), 3, 3).to(xyz.device, xyz.dtype)
            if box is not None
            else None
        )
        pair_xyz, pair_type, pair_box, units, centers, pair_idx = [], [], [], [], [], []
        start = 0
        for f, n in enumerate(counts):
            sl = slice(start, start + n)
            start += n
            x, t = xyz[sl].detach(), types[sl]
            real = t >= 0
            if int(real.sum()) < 2:
                continue
            d = x[None, :, :] - x[:, None, :]
            if boxes is not None:
                frac = d @ torch.linalg.inv(boxes[f])
                d = (frac - torch.round(frac)) @ boxes[f]
            dist = torch.linalg.vector_norm(d, dim=-1)
            valid = torch.logical_and(real[:, None], real[None, :])
            valid = torch.logical_and(
                valid, ~torch.eye(n, dtype=torch.bool, device=x.device)
            )
            dist = torch.where(valid, dist, torch.full_like(dist, float("inf")))
            k = int(torch.argmin(dist))
            i, j = divmod(k, n)
            u = d[i, j] / dist[i, j]
            pair_xyz.append(torch.stack([x[i], x[i] + d[i, j]]))
            pair_type.append(t[[i, j]])
            units.append(u)
            centers.append(0.5 * torch.dot(values[sl][j] - values[sl][i], u))
            pair_idx.append(torch.tensor([sl.start + i, sl.start + j], device=x.device))
            if boxes is not None:
                pair_box.append(boxes[f].reshape(-1))
        if not pair_xyz:
            return None
        nf = len(pair_xyz)
        if input_dict.get("n_node") is not None:
            pair_inputs = {
                "coord": torch.cat(pair_xyz),
                "atype": torch.cat(pair_type),
                "n_node": torch.full(
                    (nf,),
                    2,
                    dtype=input_dict["n_node"].dtype,
                    device=input_dict["n_node"].device,
                ),
            }
        else:
            pair_inputs = {
                "coord": torch.stack(pair_xyz),
                "atype": torch.stack(pair_type),
            }
        if boxes is not None:
            pair_inputs["box"] = torch.stack(pair_box).to(box.device, box.dtype)
        return (
            pair_inputs,
            torch.stack(units),
            torch.stack(centers).detach(),
            torch.stack(pair_idx),
        )

    def compressed_pairs(input_dict, force, wrapper):  # noqa: ANN001, ANN202
        """Most compressed real pair of every frame, as a two-atom frame in vacuum.

        The pair is the one with the smallest distance over its covalent contact under the exact
        minimum image (the rounded image and its 26 neighbours, so that skewed cells are read
        correctly), so that a heavy pair compressed far below its own contact is not hidden behind
        an ordinary H-H or H-O bond. Returns the vacuum two-atom inputs (no cell), the unit vector
        from the first atom to the second, the label's radial force on the pair ``(F_j - F_i) . u / 2``
        (positive when repulsive) and the distance over the contact; ``None`` when no frame holds
        two real atoms. Frames whose most compressed pair lies at or beyond one contact are
        returned too and are left to the caller's cut.
        """
        coord, atype = input_dict["coord"], input_dict["atype"]
        counts = frame_counts(input_dict)
        xyz = coord.reshape(-1, 3)
        types = atype.reshape(-1)
        values = force.reshape(-1, 3).to(xyz.dtype)
        box = input_dict.get("box")
        boxes = (
            box.reshape(len(counts), 3, 3).to(xyz.device, xyz.dtype)
            if box is not None
            else None
        )
        image_shifts = torch.tensor(
            [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
            device=xyz.device,
        )
        radii = covalent_radii_for(wrapper, xyz)
        pair_xyz, pair_type, units, centers, ratios = [], [], [], [], []
        start = 0
        for f, n in enumerate(counts):
            sl = slice(start, start + n)
            start += n
            x, tt = xyz[sl].detach(), types[sl]
            real = tt >= 0
            if int(real.sum()) < 2:
                continue
            d = x[None, :, :] - x[:, None, :]
            if boxes is not None:
                frac = d @ torch.linalg.inv(boxes[f])
                frac = frac - torch.round(frac)
                # the exact minimum image: the rounded image and its 26 neighbours
                cand = (
                    frac[None] + image_shifts.to(frac.dtype)[:, None, None, :]
                ) @ boxes[f]
                dist_c = torch.linalg.vector_norm(cand, dim=-1)
                best = torch.argmin(dist_c, dim=0, keepdim=True)
                d = torch.gather(
                    cand, 0, best[..., None].expand(-1, -1, -1, 3)
                ).squeeze(0)
            dist = torch.linalg.vector_norm(d, dim=-1)
            contact = radii[tt.clamp(min=0)]
            rel = dist / (contact[:, None] + contact[None, :])
            valid = torch.logical_and(real[:, None], real[None, :])
            valid = torch.logical_and(
                valid, ~torch.eye(n, dtype=torch.bool, device=x.device)
            )
            rel = torch.where(valid, rel, torch.full_like(rel, float("inf")))
            k = int(torch.argmin(rel))
            i, j = divmod(k, n)
            u = d[i, j] / dist[i, j]
            pair_xyz.append(torch.stack([torch.zeros_like(x[i]), d[i, j]]))
            pair_type.append(tt[[i, j]])
            units.append(u)
            centers.append(0.5 * torch.dot(values[sl][j] - values[sl][i], u))
            ratios.append(rel[i, j])
        if not pair_xyz:
            return None
        nf = len(pair_xyz)
        if input_dict.get("n_node") is not None:
            pair_inputs = {
                "coord": torch.cat(pair_xyz),
                "atype": torch.cat(pair_type),
                "n_node": torch.full(
                    (nf,),
                    2,
                    dtype=input_dict["n_node"].dtype,
                    device=input_dict["n_node"].device,
                ),
            }
        else:
            pair_inputs = {
                "coord": torch.stack(pair_xyz),
                "atype": torch.stack(pair_type),
            }
        return (
            pair_inputs,
            torch.stack(units),
            torch.stack(centers).detach(),
            torch.stack(ratios).detach(),
        )

    contact_cache: dict = {}

    def contact_table_for(wrapper, like):  # noqa: ANN001, ANN202
        """The data's typical closest distance of every type pair, as an (ntypes, ntypes) tensor like ``like``.

        Read from the JSON written by ``pair_contact_stats.py`` (the 5th percentile of the per-frame
        closest distance of each element pair, keyed ``A-B``); zero for pairs the table lacks.
        """
        import json

        key = (like.dtype, like.device)
        if key not in contact_cache:
            type_map = (
                wrapper.model["Default"].get_type_map()
                if hasattr(wrapper, "model")
                else wrapper.get_type_map()
            )
            index = {e: i for i, e in enumerate(type_map)}
            table = torch.zeros(
                (len(type_map), len(type_map)), dtype=like.dtype, device=like.device
            )
            for name, entry in json.load(open(dimer_contacts))["pairs"].items():
                a, b = name.split("-")
                if a in index and b in index:
                    table[index[a], index[b]] = table[index[b], index[a]] = float(
                        entry["p5"]
                    )
            contact_cache[key] = table
            log.info(
                "Dimer inequality reference distances from %s: %d of %d type pairs from the data, the rest from the covalent contact.",
                dimer_contacts,
                int((table > 0).sum()) // 2
                + int(torch.count_nonzero(torch.diagonal(table) > 0)) // 2,
                len(type_map) * (len(type_map) + 1) // 2,
            )
        return contact_cache[key]

    radii_cache: dict = {}

    def covalent_radii_for(wrapper, like):  # noqa: ANN001, ANN202
        """Covalent radius of every type of the model's type map, as a tensor like ``like``."""
        from ase.data import (
            atomic_numbers,
            covalent_radii,
        )

        key = (like.dtype, like.device)
        if key not in radii_cache:
            type_map = (
                wrapper.model["Default"].get_type_map()
                if hasattr(wrapper, "model")
                else wrapper.get_type_map()
            )
            radii_cache[key] = torch.tensor(
                [covalent_radii[atomic_numbers[e]] for e in type_map],
                dtype=like.dtype,
                device=like.device,
            )
        return radii_cache[key]

    def pair_hinge(self, wrapper, task_key, pairs):  # noqa: ANN001, ANN202
        pair_inputs, u, center, _ = pairs
        nf = u.shape[0]
        pred = self._predict(wrapper, pair_inputs, task_key)["force"].reshape(nf, 2, 3)
        f_pair = 0.5 * ((pred[:, 1] - pred[:, 0]) * u).sum(-1)
        tol, rel = pair_tol, pair_rel
        if pair_contact_ratio > 0.0:
            # The tight tube on contacts only (ledger E20): a closest pair
            # below ``pair_contact_ratio`` of the covalent contact distance is
            # a contact, on which the pair repulsion dominates the radial
            # force and the isolated pair must follow the frame's within the
            # tight tube; a bonded pair keeps the wide tube, so the isolated
            # dimer wells are left to the data.
            xyz = pair_inputs["coord"].reshape(nf, 2, 3)
            radii = covalent_radii_for(wrapper, xyz)[
                pair_inputs["atype"].reshape(nf, 2).clamp(min=0)
            ]
            contact = (xyz[:, 1] - xyz[:, 0]).norm(
                dim=-1
            ) < pair_contact_ratio * radii.sum(dim=-1)
            tol = torch.where(
                contact,
                torch.full_like(center, pair_contact_tol),
                torch.full_like(center, pair_tol),
            )
            rel = torch.where(
                contact,
                torch.full_like(center, pair_contact_rel),
                torch.full_like(center, pair_rel),
            )
        excess = (f_pair - center).abs() - tol - rel * center.abs()
        return excess.clamp(min=0.0)

    def pair_consistency(
        self: TrainingSafeguard,
        wrapper: torch.nn.Module,
        task_key: str,
        input_dict: dict[str, Any],
        label_dict: dict[str, Any],
        step: int,
        sync_module: torch.nn.Module | None,
    ) -> None:
        """One pair, one force, as a training term.

        For every frame of the batch the closest real pair (minimum image) is
        evaluated alone in the frame's own cell, and the model's radial force
        on that isolated pair is held to the label's radial force on the same
        pair inside the frame: ``relu(|f_pair - f_label| - tol - rel |f_label|)^2``
        averaged over the frames. The tube is wide by default (``tol`` 5 eV/A,
        ``rel`` 1) so that the physical environment dependence of a pair force
        passes and only a sign flip or an order-of-magnitude error on an
        isolated pair is penalized. Gradients flow through the isolated pair
        alone; the label needs no reference model.

        With ``pair_term_anchors`` the same statement is made inside the wall:
        the batch is compressed by the collision generator (every atom moved
        against its force), the model's own radial force on the compressed
        closest pair inside the frame is the centre (detached), and the pair
        alone at the compressed distance is held to it — one pair, one force,
        at distances the data's closest pairs never reach.
        """
        force = label_dict.get("force")
        if force is None or not bool(label_dict.get("find_force", True)):
            return
        params = self.params[task_key]
        stages = [closest_pairs(input_dict, force)]
        if pair_term_anchors:
            compress = {
                **params,
                "sigma": [],
                "dilate": [1.0, 1.0],
                "drop": [0.0, 0.0],
                "fstep": params["fstep"] or 0.3,
            }
            squeezed = driver.transform_batch(
                input_dict, compress, self.rng, force=force
            )
            squeezed = {k: v for k, v in squeezed.items() if v is not None}
            in_frame = self._predict(wrapper, squeezed, task_key)["force"].detach()
            stages.append(closest_pairs(squeezed, in_frame))
        stages = [st for st in stages if st is not None]
        if not stages:
            return
        sync_ctx = (
            sync_module.no_sync()
            if sync_module is not None
            else contextlib.nullcontext()
        )
        with sync_ctx:
            excess = torch.cat(
                [pair_hinge(self, wrapper, task_key, st) for st in stages]
            )
            (params["strength"] * (excess * excess).mean()).backward()
        self._discard_if_nonfinite(wrapper)
        pair_stats["events"] += 1
        pair_stats["violations"] += float((excess > 0).float().mean())
        pair_stats["excess"] += float(excess.detach().mean())
        if step % 2000 == 0 and pair_stats["events"]:
            log.info(
                "Pair term at step %d: %.1f%% of the closest pairs outside the tube, mean excess %.3f eV/A over the last %d events.",
                step,
                100.0 * pair_stats["violations"] / pair_stats["events"],
                pair_stats["excess"] / pair_stats["events"],
                pair_stats["events"],
            )
            pair_stats.update(events=0, violations=0.0, excess=0.0)

    def reference_error(self, task_key, input_dict, label_dict):  # noqa: ANN001, ANN202
        """|F_ref - F_label| per atom of the source batch, or None without a force label."""
        force = label_dict.get("force")
        if force is None or not bool(label_dict.get("find_force", True)):
            return None
        source = {k: v for k, v in input_dict.items() if v is not None}
        ref_force = self._predict(self.reference, source, task_key)["force"].detach()
        shape = (*input_dict["atype"].shape, 3)
        return torch.sqrt(
            ((ref_force.reshape(shape) - force.reshape(shape)) ** 2).sum(-1)
        )

    repulsion_stats = {"events": 0, "pairs": 0, "violations": 0.0, "excess": 0.0}

    def repulsion_hinge(
        self: TrainingSafeguard,
        wrapper: torch.nn.Module,
        task_key: str,
        input_dict: dict[str, Any],
        label_dict: dict[str, Any],
        step: int,
        sync_module: torch.nn.Module | None,
    ) -> None:
        """Monotone repulsion as an inequality (ledger E11).

        Every batch frame is copied with its atoms pushed 0-0.3 Å against the label force (the
        collision generator alone), and on each copy whose closest pair lies below 0.55 of the
        covalent contact distance the model's radial force on that pair inside the frame is
        required to be repulsive: relu(-f_r)^2 summed over the copies, weighted by
        ``repulsion_weight``. No reference and no label enter; the constraint is the sign of the
        force on a pair closer than any bond, where physics leaves no doubt.
        """
        force = label_dict.get("force")
        if force is None or not bool(label_dict.get("find_force", True)):
            return
        params = self.params[task_key]
        compress = {
            **params,
            "sigma": [],
            "dilate": [1.0, 1.0],
            "drop": [0.0, 0.0],
            "fstep": params["fstep"] or 0.3,
        }
        squeezed = driver.transform_batch(input_dict, compress, self.rng, force=force)
        squeezed = {k: v for k, v in squeezed.items() if v is not None}
        sync_ctx = (
            sync_module.no_sync()
            if sync_module is not None
            else contextlib.nullcontext()
        )
        with sync_ctx:
            forces = self._predict(wrapper, squeezed, task_key)["force"]
            stage = closest_pairs(squeezed, forces)
            if stage is None:
                return
            pair_inputs, units, _, idx = stage
            pair_xyz, pair_type = (
                pair_inputs["coord"].reshape(-1, 2, 3),
                pair_inputs["atype"].reshape(-1, 2),
            )
            radii = covalent_radii_for(wrapper, pair_xyz)[
                pair_type.clamp(min=0)
            ]  # (nf, 2)
            # The second copy already carries the minimum-image displacement from the first.
            close = (pair_xyz[:, 1] - pair_xyz[:, 0]).norm(
                dim=-1
            ) < repulsion_ratio * radii.sum(dim=-1)
            flat = forces.reshape(-1, 3)
            radial = 0.5 * ((flat[idx[:, 1]] - flat[idx[:, 0]]) * units).sum(
                dim=-1
            )  # positive = repulsive
            if not bool(close.any()):
                return
            excess = torch.relu(-radial[close])
            loss = repulsion_weight * excess.square().sum()
            loss.backward()
            self._discard_if_nonfinite(wrapper)
        repulsion_stats["events"] += 1
        repulsion_stats["pairs"] += int(close.sum())
        repulsion_stats["violations"] += float((excess > 0).sum())
        repulsion_stats["excess"] += float(excess.detach().sum())
        if step % 2000 == 0 and repulsion_stats["pairs"] > 0:
            log.info(
                "Repulsion inequality at step %d: %d compressed pairs below the contact over %d events, %.1f%% attractive, mean excess %.3f eV/A.",
                step,
                repulsion_stats["pairs"],
                repulsion_stats["events"],
                100.0
                * repulsion_stats["violations"]
                / max(repulsion_stats["pairs"], 1),
                repulsion_stats["excess"] / max(repulsion_stats["violations"], 1.0),
            )
            for k in ("events", "pairs", "violations", "excess"):
                repulsion_stats[k] = 0 if k != "excess" else 0.0

    dimer_stats = {
        "events": 0,
        "pairs": 0,
        "violations": 0.0,
        "excess": 0.0,
        "announced": False,
    }

    def dimer_hinge(self, wrapper, task_key, input_dict, step, sync_module):  # noqa: ANN001, ANN202
        """Monotone repulsion on the isolated pair (ledger E58).

        At every anchor event ``dimer_count`` two-atom frames are built in vacuum: the two
        element types drawn uniformly from the types present in the batch (same-type pairs
        included), the separation drawn uniformly from ``dimer_ratio`` times the covalent
        contact distance (the sum of the two covalent radii), the orientation uniform on the
        sphere. The model's radial force on the pair, ``(F_1 - F_0) . u / 2`` with ``u`` the
        unit vector from atom 0 to atom 1, is positive when repulsive and is required to be so:
        relu(-f_r)^2 summed over the frames, weighted by ``dimer_weight``. No reference, no label
        and no potential value enter — only the sign of the force on a pair closer than any
        bond, where physics leaves no doubt. Ledger E11 evaluated the same inequality on the
        compressed closest pair inside every collision copy, where it was almost never violated
        while the isolated H2 holed below it: the hole lives in the sparse limit, so the
        inequality is evaluated there. The upper bound of ``dimer_ratio`` must stay below the
        shortest dimer bond of the elements in play, measured in covalent contacts (0.58-0.68 for
        the multiply bonded transition-metal dimers, 0.92 for O2, 1.19 for H2).

        With ``dimer_contacts`` (ledger E59) the reference distance of a pair is not its covalent
        contact but the distance at which the training data typically holds that pair at its
        closest — the 5th percentile of the per-frame closest A-B distance from
        ``pair_contact_stats.py`` — and the covalent contact only for pairs the table lacks. The
        holes of the surveyed checkpoints sit at the data's shortest contacts (section 10.13.12),
        which a fraction of the covalent contact cannot reach without crossing the bonds of the
        multiply bonded dimers; a fraction of the data's own contact reaches them, at the price of
        asking repulsion of the few isolated metal dimers that bind inside it.
        """
        atype = input_dict["atype"].reshape(-1)
        types = torch.unique(atype[atype >= 0]).tolist()
        if not types:
            return
        coord = input_dict["coord"]
        nd = dimer_count
        pair_type = torch.as_tensor(
            np.stack(
                [self.rng.choice(types, size=nd), self.rng.choice(types, size=nd)],
                axis=1,
            ),
            dtype=atype.dtype,
            device=atype.device,
        )  # (nd, 2)
        radii = covalent_radii_for(wrapper, coord)[pair_type]  # (nd, 2)
        reference = radii.sum(dim=-1)  # (nd,) the covalent contact
        if dimer_contacts is not None:
            table = contact_table_for(wrapper, coord)
            reference = torch.where(
                table[pair_type[:, 0], pair_type[:, 1]] > 0,
                table[pair_type[:, 0], pair_type[:, 1]],
                reference,
            )
        ratio = torch.as_tensor(
            self.rng.uniform(dimer_ratio[0], dimer_ratio[1], size=nd),
            dtype=coord.dtype,
            device=coord.device,
        )
        r = ratio * reference  # (nd,)
        u = torch.as_tensor(
            self.rng.normal(size=(nd, 3)), dtype=coord.dtype, device=coord.device
        )
        u = u / torch.linalg.vector_norm(u, dim=-1, keepdim=True)
        xyz = torch.stack(
            [torch.zeros_like(u), r[:, None] * u], dim=1
        )  # (nd, 2, 3), non-periodic
        if input_dict.get("n_node") is not None:
            n_node = input_dict["n_node"]
            inputs = {
                "coord": xyz.reshape(-1, 3),
                "atype": pair_type.reshape(-1),
                "n_node": torch.full(
                    (nd,), 2, dtype=n_node.dtype, device=n_node.device
                ),
            }
        else:
            inputs = {"coord": xyz, "atype": pair_type}
        sync_ctx = (
            sync_module.no_sync()
            if sync_module is not None
            else contextlib.nullcontext()
        )
        with sync_ctx:
            pred = self._predict(wrapper, inputs, task_key)["force"].reshape(nd, 2, 3)
            radial = 0.5 * ((pred[:, 1] - pred[:, 0]) * u).sum(
                dim=-1
            )  # positive = repulsive
            excess = torch.relu(-radial)
            loss = dimer_weight * excess.square().sum()
            loss.backward()
            self._discard_if_nonfinite(wrapper)
        if not dimer_stats["announced"]:
            dimer_stats["announced"] = True
            log.info(
                "Dimer inequality active at step %d: %d isolated pairs per event from %d element types, separations %.2f-%.2f A, %d attractive at this event.",
                step,
                nd,
                len(types),
                float(r.min()),
                float(r.max()),
                int((excess > 0).sum()),
            )
        dimer_stats["events"] += 1
        dimer_stats["pairs"] += nd
        dimer_stats["violations"] += float((excess > 0).sum())
        dimer_stats["excess"] += float(excess.detach().sum())
        if step % 2000 == 0 and dimer_stats["pairs"] > 0:
            log.info(
                "Dimer inequality at step %d: %d isolated pairs over %d events, %.1f%% attractive, mean excess %.3f eV/A.",
                step,
                dimer_stats["pairs"],
                dimer_stats["events"],
                100.0 * dimer_stats["violations"] / max(dimer_stats["pairs"], 1),
                dimer_stats["excess"] / max(dimer_stats["violations"], 1.0),
            )
            for k in ("events", "pairs", "violations", "excess"):
                dimer_stats[k] = 0 if k != "excess" else 0.0

    tail_stats = {
        "events": 0,
        "pairs": 0,
        "abs_force": 0.0,
        "max_force": 0.0,
        "announced": False,
    }

    def tail_penalty(self, wrapper, task_key, input_dict, step, sync_module):  # noqa: ANN001, ANN202
        """Zero force on the isolated pair's tail (ledger E67).

        At every anchor event ``tail_count`` two-atom frames are built in vacuum: the two
        element types drawn uniformly from the types present in the batch (same-type pairs
        included), the separation drawn uniformly from ``tail_range`` in A — the lower end
        raised to 1.6 covalent contacts for a pair whose contact reaches it — and the
        orientation uniform on the sphere. Beyond 1.6 contacts and 4 A an isolated pair has
        no covalent interaction left, so the model's radial force ``(F_1 - F_0) . u / 2`` is
        required to vanish there: ``f_r^2`` summed over the frames, weighted by ``tail_weight``.
        The safeguard's reference tube (1 eV/A plus a tenth of the reference force) cannot
        bind a tail force of a few tenths of an eV/A against a zero reference; this term is
        the absolute criterion A1 itself, imposed on a deterministic cover of the interval
        instead of on whatever anchors happen to fall in it. No reference and no label enter.
        """
        atype = input_dict["atype"].reshape(-1)
        types = torch.unique(atype[atype >= 0]).tolist()
        if not types:
            return
        coord = input_dict["coord"]
        nd = tail_count
        pair_type = torch.as_tensor(
            np.stack(
                [self.rng.choice(types, size=nd), self.rng.choice(types, size=nd)],
                axis=1,
            ),
            dtype=atype.dtype,
            device=atype.device,
        )  # (nd, 2)
        contact = covalent_radii_for(wrapper, coord)[pair_type].sum(dim=-1)  # (nd,)
        lo = torch.clamp(1.6 * contact, min=tail_range[0])
        hi = torch.full_like(lo, tail_range[1])
        frac = torch.as_tensor(
            self.rng.uniform(0.0, 1.0, size=nd), dtype=coord.dtype, device=coord.device
        )
        r = lo + frac * torch.clamp(hi - lo, min=0.0)  # (nd,)
        u = torch.as_tensor(
            self.rng.normal(size=(nd, 3)), dtype=coord.dtype, device=coord.device
        )
        u = u / torch.linalg.vector_norm(u, dim=-1, keepdim=True)
        xyz = torch.stack(
            [torch.zeros_like(u), r[:, None] * u], dim=1
        )  # (nd, 2, 3), non-periodic
        if input_dict.get("n_node") is not None:
            n_node = input_dict["n_node"]
            inputs = {
                "coord": xyz.reshape(-1, 3),
                "atype": pair_type.reshape(-1),
                "n_node": torch.full(
                    (nd,), 2, dtype=n_node.dtype, device=n_node.device
                ),
            }
        else:
            inputs = {"coord": xyz, "atype": pair_type}
        sync_ctx = (
            sync_module.no_sync()
            if sync_module is not None
            else contextlib.nullcontext()
        )
        with sync_ctx:
            pred = self._predict(wrapper, inputs, task_key)["force"].reshape(nd, 2, 3)
            radial = 0.5 * ((pred[:, 1] - pred[:, 0]) * u).sum(
                dim=-1
            )  # eV/A, zero in physics
            loss = tail_weight * radial.square().sum()
            loss.backward()
            self._discard_if_nonfinite(wrapper)
        abs_force = radial.detach().abs()
        if not tail_stats["announced"]:
            tail_stats["announced"] = True
            log.info(
                "Tail term active at step %d: %d isolated pairs per event from %d element types, separations %.2f-%.2f A, mean |f_r| %.3f eV/A at this event.",
                step,
                nd,
                len(types),
                float(r.min()),
                float(r.max()),
                float(abs_force.mean()),
            )
        tail_stats["events"] += 1
        tail_stats["pairs"] += nd
        tail_stats["abs_force"] += float(abs_force.sum())
        tail_stats["max_force"] = max(tail_stats["max_force"], float(abs_force.max()))
        if step % 2000 == 0 and tail_stats["pairs"] > 0:
            log.info(
                "Tail term at step %d: %d isolated pairs over %d events, mean |f_r| %.4f eV/A, largest %.3f eV/A.",
                step,
                tail_stats["pairs"],
                tail_stats["events"],
                tail_stats["abs_force"] / tail_stats["pairs"],
                tail_stats["max_force"],
            )
            tail_stats.update(events=0, pairs=0, abs_force=0.0, max_force=0.0)

    data_sign_stats = {
        "events": 0,
        "frames": 0,
        "kept": 0,
        "violations": 0.0,
        "excess": 0.0,
        "ratio": 0.0,
        "announced": False,
    }

    def data_sign_hinge(
        self: TrainingSafeguard,
        wrapper: torch.nn.Module,
        task_key: str,
        input_dict: dict[str, Any],
        label_dict: dict[str, Any],
        step: int,
        sync_module: torch.nn.Module | None,
    ) -> None:
        """The data's repulsion carried into the sparse limit (ledger E59).

        For every batch frame the most compressed real pair (the smallest distance over the
        covalent contact) is read with the label's radial force on it. Where the label pushes the
        pair apart by more than ``data_sign_margin`` eV/A — the data's own compressed contacts,
        whose labels are the innermost physics the training set carries — the same two atoms alone
        in vacuum, at the same separation, are required to repel as well: relu(-f_r)^2 on the
        isolated pair's radial force, summed over those frames and weighted by
        ``data_sign_weight``. Only the sign of a genuine label enters, no value, no reference and no
        potential; where the label is attractive or small (a bonded pair) nothing is asked. The
        pair-consistency term of E2/E20 held the isolated pair's force to the in-frame label's
        value inside a tube and deformed the isolated curve where the two legitimately differ;
        the sign is what the isolated pair shares with the frame at a compressed contact. The
        vacuum batch always holds ``data_sign_count`` frames — the most compressed qualifying
        pairs, repeated to fill the batch with the repeats masked out of the loss — so that the
        compiled forward sees one shape.
        """
        force = label_dict.get("force")
        if force is None or not bool(label_dict.get("find_force", True)):
            return
        stage = compressed_pairs(input_dict, force, wrapper)
        if stage is None:
            return
        pair_inputs, units, centers, ratios = stage
        # Only pairs inside their covalent contact are carried: a far pair in a small cell can carry a
        # large radial label force that has nothing to do with the pair itself.
        keep = (centers > data_sign_margin) & (ratios < 1.0)
        nk = int(keep.sum())
        data_sign_stats["events"] += 1
        data_sign_stats["frames"] += int(centers.shape[0])
        if nk == 0:
            return
        # The nk qualifying pairs, most compressed first, cut or repeated to the fixed count.
        order = torch.argsort(
            torch.where(keep, ratios, torch.full_like(ratios, float("inf")))
        )[: min(nk, data_sign_count)]
        nb = data_sign_count
        idx = order[torch.arange(nb, device=order.device) % order.shape[0]]
        weight = (torch.arange(nb, device=order.device) < order.shape[0]).to(
            units.dtype
        )
        if pair_inputs.get("n_node") is not None:
            atom_idx = (
                2 * idx[:, None] + torch.arange(2, device=idx.device)[None, :]
            ).reshape(-1)
            inputs = {
                "coord": pair_inputs["coord"][atom_idx],
                "atype": pair_inputs["atype"][atom_idx],
                "n_node": pair_inputs["n_node"][idx],
            }
        else:
            inputs = {
                "coord": pair_inputs["coord"][idx],
                "atype": pair_inputs["atype"][idx],
            }
        u = units[idx]
        sync_ctx = (
            sync_module.no_sync()
            if sync_module is not None
            else contextlib.nullcontext()
        )
        with sync_ctx:
            pred = self._predict(wrapper, inputs, task_key)["force"].reshape(nb, 2, 3)
            radial = 0.5 * ((pred[:, 1] - pred[:, 0]) * u).sum(
                dim=-1
            )  # positive = repulsive
            excess = torch.relu(-radial) * weight
            loss = data_sign_weight * excess.square().sum()
            loss.backward()
            self._discard_if_nonfinite(wrapper)
        nk = int(order.shape[0])
        excess = excess[:nk]
        if not data_sign_stats["announced"]:
            data_sign_stats["announced"] = True
            log.info(
                "Data-sign inequality active at step %d: %d of %d frames carry a label-repulsive compressed pair (above %.1f eV/A), at 0.%02d-0.%02d of the contact, %d of them attractive alone.",
                step,
                nk,
                int(centers.shape[0]),
                data_sign_margin,
                int(100 * float(ratios[keep].min())),
                int(100 * float(ratios[keep].max())),
                int((excess > 0).sum()),
            )
        data_sign_stats["kept"] += nk
        data_sign_stats["violations"] += float((excess > 0).sum())
        data_sign_stats["excess"] += float(excess.detach().sum())
        data_sign_stats["ratio"] += float(ratios[keep].sum())
        if step % 2000 == 0 and data_sign_stats["kept"] > 0:
            log.info(
                "Data-sign inequality at step %d: %d label-repulsive compressed pairs in %d frames over %d events (mean %.2f of the contact), %.1f%% attractive alone, mean excess %.3f eV/A.",
                step,
                data_sign_stats["kept"],
                data_sign_stats["frames"],
                data_sign_stats["events"],
                data_sign_stats["ratio"] / max(data_sign_stats["kept"], 1),
                100.0 * data_sign_stats["violations"] / max(data_sign_stats["kept"], 1),
                data_sign_stats["excess"] / max(data_sign_stats["violations"], 1.0),
            )
            for k in ("events", "frames", "kept"):
                data_sign_stats[k] = 0
            for k in ("violations", "excess", "ratio"):
                data_sign_stats[k] = 0.0

    def anchor_step(
        self: TrainingSafeguard,
        wrapper: torch.nn.Module,
        task_key: str,
        input_dict: dict[str, Any],
        label_dict: dict[str, Any],
        step: int,
        sync_module: torch.nn.Module | None = None,
    ) -> None:
        nonlocal stage_applied
        event["ref_error"] = None
        event["far_atoms"] = None
        if pair_squeeze_ratio is not None and squeeze_state["radii"] is None:
            type_map = (
                wrapper.model["Default"].get_type_map()
                if hasattr(wrapper, "model")
                else wrapper.get_type_map()
            )
            from ase.data import (
                atomic_numbers,
                covalent_radii,
            )

            squeeze_state["radii"] = torch.tensor(
                [float(covalent_radii[atomic_numbers[e]]) for e in type_map],
                dtype=torch.float64,
            )
            log.info("Pair squeeze radii resolved for %d types.", len(type_map))
        if type_substitution is not None and substitution["ntypes"] is None:
            type_map = (
                wrapper.model["Default"].get_type_map()
                if hasattr(wrapper, "model")
                else wrapper.get_type_map()
            )
            substitution["ntypes"] = len(type_map)
            log.info(
                "Type substitution active: %.2f of the real atoms in %.2f of the anchor "
                "frames take a type drawn uniformly from %d types.",
                type_substitution[0],
                type_substitution[1],
                substitution["ntypes"],
            )
        if families:
            freq = self.params[task_key]["freq"]
            self.params[task_key] = families[task_key][(step // freq) % 2]
        if refresh_steps is not None:
            # An explicit schedule of hand-over steps (ledger E3) replaces the periodic rule.
            due = self.reference is not None and step in refresh_steps
        else:
            due = (
                self.reference is not None
                and step > self.ref_step
                and (step - self.ref_step) % refresh == 0
                and (
                    max_refreshes == 0
                    or (step - self.ref_step) // refresh <= max_refreshes
                )
            )
        if due:
            self.reference = (
                ensemble_reference(wrapper) if ensemble_refs else copy.deepcopy(wrapper)
            )
            self.reference.eval()
            self.reference.requires_grad_(False)
            log.info("Safeguard reference refreshed at training step %d.", step)
        first_refresh = min(refresh_steps) if refresh_steps else self.ref_step + refresh
        if self.reference is not None and step >= first_refresh and not stage_applied:
            # The generator phase follows the absolute schedule. A restart restores the
            # reference from its checkpoint and reconstructs this phase without replacing it.
            stage_applied = True
            if (
                pair_squeeze is not None or pair_squeeze_ratio is not None
            ) and not squeeze_state["active"]:
                squeeze_state["active"] = True
                log.info("Pair squeeze active from training step %d.", step)
            if dense_after_refresh and not self.finetune:
                # A reference that carries the data's physics admits the dense
                # generators of the fine-tune recipe: from here on the run is
                # resolved as a fine-tune of itself (``generators: auto`` then
                # yields the dense and sparse set).
                self.finetune = True
                self._resolve_params()
                log.info("Safeguard generators switched to the dense and sparse set.")
            elif collision_after_refresh and not self.finetune:
                # Only the generator that reaches contacts is added to the
                # sparse set: atoms pushed against their force, without the
                # Gaussian shell that sits on every data frame.
                self.finetune = True
                for cfg in self._raw_params.values():
                    if cfg is not None:
                        cfg.update(
                            {
                                "sigma": [],
                                "fstep": 0.3,
                                "dilate": [1.3, 2.0],
                                "drop": [0.5, 0.9],
                            }
                        )
                self._resolve_params()
                log.info(
                    "Safeguard generators switched to the sparse set with collisions."
                )
            elif near_after_refresh and not self.finetune:
                # Only the near generators from here on: the Gaussian shell and
                # the collision, which reach the wall of the contacts; no
                # dilation, no deletion.
                self.finetune = True
                for cfg in self._raw_params.values():
                    if cfg is not None:
                        cfg.update(
                            {
                                "sigma": [0.1, 0.2],
                                "fstep": 0.3,
                                "dilate": [1.0, 1.0],
                                "drop": [0.0, 0.0],
                            }
                        )
                self._resolve_params()
                log.info(
                    "Safeguard generators switched to the near set (shell, collision)."
                )
            elif pin_sparse_after_refresh and not families:
                for key, params in self.params.items():
                    far = dict(
                        params,
                        sigma=[],
                        fstep=0.0,
                        dilate=[1.3, 2.0],
                        drop=[0.5, 0.9],
                        tol=0.0,
                        rel_tol=0.0,
                    )
                    near = dict(
                        params,
                        sigma=[0.1, 0.2],
                        fstep=0.3,
                        dilate=[1.0, 1.0],
                        drop=[0.0, 0.0],
                    )
                    families[key] = [far, near]
                log.info(
                    "Safeguard anchors alternate between the far family (dilation, deletion; "
                    "closed tube) and the near family (shell, collision; tube %.2f + %.2f |F|).",
                    params["tol"],
                    params["rel_tol"],
                )
            elif label_tube_after_refresh and not self.finetune:
                # The dense and sparse set of the fine-tune, with the tube of
                # every anchor frame that stayed near its source frame widened
                # by the reference's error against the frame's label.
                self.finetune = True
                self._resolve_params()
                log.info(
                    "Safeguard generators switched to the dense and sparse set; every anchor "
                    "atom's tube is widened by the reference's error against its label%s.",
                    " on the near anchors only (shell, collision)"
                    if label_tube_near_only
                    else "",
                )
            if rel_after_refresh is not None:
                for cfg in self.params.values():
                    if cfg is not None:
                        cfg["rel_tol"] = rel_after_refresh
                log.info("Safeguard relative tolerance set to %.2f.", rel_after_refresh)
            if dilate_after_refresh is not None:
                for cfg in self.params.values():
                    if cfg is not None:
                        cfg["dilate"] = [1.0, dilate_after_refresh]
                log.info(
                    "Safeguard dilation range set to [1.0, %.2f].", dilate_after_refresh
                )
            if fstep_after_refresh is not None:
                # The collision generator's reach: every atom moves against its
                # force by up to this distance, so a pair can close by twice it.
                for cfg in self.params.values():
                    if cfg is not None:
                        cfg["fstep"] = fstep_after_refresh
                log.info("Safeguard collision step set to %.2f A.", fstep_after_refresh)
            log.info(
                "Safeguard post-handover generators at step %d: %s", step, self.params
            )
        params = self.params.get(task_key)
        event_due = (
            params is not None
            and params["strength"] > 0.0
            and self.reference is not None
            and step % params["freq"] == 0
        )
        if label_tube_after_refresh and self.finetune and event_due:
            event["ref_error"] = reference_error(self, task_key, input_dict, label_dict)
        if pair_term and event_due:
            pair_consistency(
                self, wrapper, task_key, input_dict, label_dict, step, sync_module
            )
        if repulsion_weight > 0.0 and event_due:
            repulsion_hinge(
                self, wrapper, task_key, input_dict, label_dict, step, sync_module
            )
        if dimer_weight > 0.0 and event_due:
            dimer_hinge(self, wrapper, task_key, input_dict, step, sync_module)
        if data_sign_weight > 0.0 and event_due:
            data_sign_hinge(
                self, wrapper, task_key, input_dict, label_dict, step, sync_module
            )
        if tail_weight > 0.0 and event_due:
            tail_penalty(self, wrapper, task_key, input_dict, step, sync_module)
        descent_state.update(wrapper=wrapper, task_key=task_key)
        worst_state.update(driver=self, wrapper=wrapper, task_key=task_key)
        result = original(
            self, wrapper, task_key, input_dict, label_dict, step, sync_module
        )
        if anchor_draws > 1 and step % 2000 == 0 and worst_stats["events"] > 0:
            log.info(
                "Worst-of-%d anchors at step %d: the selected draw's hinge excess is %.3f per frame against the draws' mean %.3f over %d events.",
                anchor_draws,
                step,
                worst_stats["selected"] / worst_stats["events"],
                worst_stats["mean"] / worst_stats["events"],
                worst_stats["events"],
            )
            worst_stats.update(events=0, selected=0.0, mean=0.0)
        return result

    driver.TrainingSafeguard.anchor_step = anchor_step


def install_batch_error_log_dpmodel(
    dump_above: float = 500.0, max_dumps: int = 20
) -> None:
    """The batch-error log for the ``pt-expt`` backend, whose trainer evaluates the array-API
    ``EnergyLoss`` of ``deepmd.dpmodel.loss.ener`` on the model's predictions and the labels.

    ``EnergyLoss.call`` is patched so that every training call (a call whose predicted forces carry
    a gradient; validation runs without one) appends one line to ``batch_max_error_rank<r>.log``:
    the training-call index, the largest per-atom force error of the batch in eV/A and the number of
    atoms above 20 eV/A. The loss does not see the coordinates, so the geometry column of the ``pt``
    log is left empty and a needle dump holds the predicted and labelled forces only.
    """
    import numpy as np
    import torch

    from deepmd.dpmodel.loss.ener import (
        EnergyLoss,
    )

    original = EnergyLoss.call
    state: dict = {"step": 0, "file": None, "dumps": 0}

    def call(self, learning_rate, natoms, model_dict, label_dict, mae=False):  # noqa: ANN001, ANN202
        out = original(self, learning_rate, natoms, model_dict, label_dict, mae)
        force = model_dict.get("force")
        if (
            isinstance(force, torch.Tensor)
            and force.requires_grad
            and "force" in label_dict
        ):
            err = torch.linalg.vector_norm(
                force.detach().reshape(-1, 3)
                - label_dict["force"].reshape(-1, 3).to(force.dtype),
                dim=-1,
            )
            peak, above = float(err.max()), int((err > 20.0).sum())
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )
            if state["file"] is None:
                state["file"] = open(f"batch_max_error_rank{rank}.log", "a")
                state["file"].write(
                    "# step  largest per-atom force error (eV/A)  atoms above 20 eV/A  lr  nearest-neighbour distance (not available on the pt-expt loss)\n"
                )
            state["file"].write(
                f"{state['step']:8d} {peak:10.3f} {above:5d} {float(learning_rate):.3e} -\n"
            )
            state["file"].flush()
            if peak > dump_above and state["dumps"] < max_dumps:
                arrays = {
                    "step": np.array(state["step"]),
                    "per_atom_error": err.cpu().numpy(),
                    "model_force": force.detach().cpu().numpy(),
                    "label_force": label_dict["force"].detach().cpu().numpy(),
                }
                np.savez(f"needle_rank{rank}_step{state['step']}.npz", **arrays)
                state["dumps"] += 1
                log.warning(
                    f"Needle at training call {state['step']}: largest per-atom force error {peak:.1f} eV/A; forces written to needle_rank{rank}_step{state['step']}.npz"
                )
            state["step"] += 1
        return out

    EnergyLoss.call = call
    log.info(
        "Batch-error log installed on the pt-expt loss: batch_max_error_rank<r>.log"
    )


def install_hard_replay(
    replay: int,
    ratio: float,
    width: int,
    size: int,
    every: int = 1,
    weight: float = 1.0,
) -> None:
    """Hard-contact replay (ledger E6): the rare close contacts of the data revisited every few steps.

    Every training batch is scanned for frames whose closest pair, under the minimum image, lies
    below ``ratio`` times the sum of the two atoms' covalent radii; such frames (padded to ``width``
    atoms, larger frames skipped) enter a per-rank buffer of ``size`` frames. Every ``every`` training
    calls, ``replay`` frames drawn from the buffer are evaluated by the same loss against their own
    labels and that loss, scaled by ``weight``, is added to the batch's, so that the contacts on
    which the plain run's function wanders and needles are visited every few hundred steps instead
    of once per epoch. With ``weight`` equal to the inverse of the number of times a buffered frame
    is replayed before it is replaced, the replay is importance sampling: the expected gradient of
    the data is unchanged and only the variance on the hard frames falls. Independent of the
    safeguard and of any model patch: evaluation needs nothing.
    """
    import numpy as np
    import torch
    from ase.data import (
        atomic_numbers,
        covalent_radii,
    )

    from deepmd.pt.loss.ener import (
        EnergyStdLoss,
    )

    original = EnergyStdLoss.forward
    rng = np.random.default_rng(0)
    state: dict = {
        "calls": 0,
        "buffer": [],
        "radii": None,
        "replayed": 0,
        "err": 0.0,
        "kept": 0,
    }

    def radii_for(model, device):  # noqa: ANN001, ANN202
        if state["radii"] is None:
            type_map = model.get_type_map()
            state["radii"] = torch.tensor(
                [covalent_radii[atomic_numbers[e]] for e in type_map],
                dtype=torch.float32,
                device=device,
            )
        return state["radii"]

    def closest_ratio(input_dict, model):  # noqa: ANN001, ANN202
        """Closest-pair distance over the sum of covalent radii, per frame, under the minimum image."""
        coord = input_dict["coord"]
        atype = input_dict["atype"]
        nf = atype.shape[0]
        xyz = coord.reshape(nf, -1, 3).to(torch.float32)
        real = atype >= 0
        radii = radii_for(model, xyz.device)[atype.clamp(min=0)]
        d = xyz[:, :, None, :] - xyz[:, None, :, :]
        box = input_dict.get("box")
        if box is not None:
            cell = box.reshape(nf, 3, 3).to(xyz.device, torch.float32)
            frac = torch.einsum("nijk,nkl->nijl", d, torch.linalg.inv(cell))
            frac = frac - torch.round(frac)
            d = torch.einsum("nijl,nlk->nijk", frac, cell)
        dist = d.norm(dim=-1)
        valid = (
            real[:, :, None]
            & real[:, None, :]
            & ~torch.eye(atype.shape[1], dtype=torch.bool, device=xyz.device)
        )
        rel = torch.where(
            valid,
            dist / (radii[:, :, None] + radii[:, None, :]),
            torch.full_like(dist, float("inf")),
        )
        return rel.flatten(1).min(dim=1).values

    def store(input_dict, label, hard):  # noqa: ANN001, ANN202
        coord, atype, box = input_dict["coord"], input_dict["atype"], input_dict["box"]
        nf, nloc = atype.shape
        for k in torch.nonzero(hard).flatten().tolist():
            n_real = int((atype[k] >= 0).sum())
            if n_real > width:
                continue
            c = torch.zeros(width, 3)
            a = torch.full((width,), -1, dtype=torch.long)
            f = torch.zeros(width, 3)
            keep = (atype[k] >= 0).cpu()
            c[:n_real] = coord[k].reshape(nloc, 3)[keep].detach().float().cpu()
            a[:n_real] = atype[k][keep].cpu()
            f[:n_real] = label["force"][k].reshape(nloc, 3)[keep].detach().float().cpu()
            frame = {
                "coord": c,
                "atype": a,
                "box": box[k].reshape(9).detach().float().cpu(),
                "energy": label["energy"][k].reshape(1).detach().float().cpu(),
                "force": f,
            }
            if len(state["buffer"]) < size:
                state["buffer"].append(frame)
            else:
                state["buffer"][int(rng.integers(size))] = frame
            state["kept"] += 1

    def replay_batch(input_dict, label):  # noqa: ANN001, ANN202
        picks = [
            state["buffer"][i]
            for i in rng.choice(len(state["buffer"]), size=replay, replace=False)
        ]
        dev, dt = input_dict["coord"].device, input_dict["coord"].dtype
        r_input = {
            "coord": torch.stack([p["coord"] for p in picks]).to(dev, dt),
            "atype": torch.stack([p["atype"] for p in picks]).to(dev),
            "box": torch.stack([p["box"] for p in picks]).to(
                input_dict["box"].device, dt
            ),
        }
        r_label = {k: v for k, v in label.items() if k.startswith("find_")}
        r_label["energy"] = torch.stack([p["energy"] for p in picks]).to(dev, dt)
        r_label["force"] = torch.stack([p["force"] for p in picks]).to(dev, dt)
        if "find_virial" in r_label:
            r_label["find_virial"] = torch.zeros_like(
                torch.as_tensor(r_label["find_virial"])
            )
        return r_input, r_label

    def forward(self, input_dict, model, label, natoms, learning_rate, mae=False):  # noqa: ANN001, ANN202
        out = original(self, input_dict, model, label, natoms, learning_rate, mae)
        if not model.training or "force" not in label or "box" not in input_dict:
            return out
        state["calls"] += 1
        with torch.no_grad():
            hard = closest_ratio(input_dict, model) < ratio
        if bool(hard.any()):
            store(input_dict, label, hard)
        if len(state["buffer"]) >= replay and state["calls"] % every == 0:
            r_input, r_label = replay_batch(input_dict, label)
            r_pred, r_loss, r_more = original(
                self, r_input, model, r_label, width, learning_rate, mae
            )
            model_pred, loss, more = out
            err = (
                r_pred["force"].detach().reshape(replay, -1, 3)
                - r_label["force"].reshape(replay, -1, 3)
            ).norm(dim=-1)
            err = err[r_input["atype"] >= 0]
            state["replayed"] += 1
            state["err"] += float(err.mean())
            out = (model_pred, loss + weight * r_loss, more)
        if state["calls"] % 1000 == 0 or state["calls"] == 10:
            n = max(state["replayed"], 1)
            message = (
                f"Hard-contact replay at call {state['calls']}: {state['kept']} frames kept ({len(state['buffer'])} in the buffer), "
                f"{state['replayed']} replays, mean replay force error {state['err'] / n:.3f} eV/A."
            )
            log.info(message)
            print(f"[hard_replay] {message}", file=sys.stderr, flush=True)
            state["replayed"], state["err"] = 0, 0.0
        return out

    EnergyStdLoss.forward = forward
    message = (
        f"Hard-contact replay installed: closest pair below {ratio:.2f} of the covalent contact, buffer {size} frames of up to "
        f"{width} atoms, {replay} frames replayed every {every} calls with weight {weight:g}."
    )
    log.info(message)
    print(f"[hard_replay] {message}", file=sys.stderr, flush=True)


def weighted_predictions(
    pred: dict, w_frame: torch.Tensor, w_virial: torch.Tensor
) -> dict:
    """Return the predictions whose parameter sensitivity is scaled per frame by the fractional weights.

    Every prediction p(θ) of a frame is replaced by p̂ + w (p(θ) - p̂) with p̂ the detached value: the
    loss sees the same value, so the residual and the reported errors are unchanged, while every
    derivative with respect to the parameters is multiplied by w — the frame's contribution to the
    gradient is scaled by w exactly, whatever the functional form of the loss (MAE, MSE, Huber, the
    per-atom force norm), and w = 0 removes it. Scaling the label instead would leave an L1 gradient
    unchanged, since its magnitude does not depend on the size of the residual. ``w_frame`` with shape
    (nf,) weights the energy and force terms, ``w_virial`` with shape (nf,) the virial term.
    """
    out = dict(pred)
    for key, w in (
        ("energy", w_frame),
        ("atom_energy", w_frame),
        ("force", w_frame),
        ("virial", w_virial),
        ("atom_virial", w_virial),
    ):
        if key in pred and pred[key].requires_grad:
            p_ = pred[key]
            frozen = p_.detach()
            out[key] = frozen + w.to(p_.dtype).reshape(-1, *([1] * (p_.ndim - 1))) * (
                p_ - frozen
            )
    return out


class _WeightedModel:
    """Model proxy handing the loss predictions whose gradient contributions carry the fractional weights."""

    def __init__(self, model, w_frame: torch.Tensor, w_virial: torch.Tensor) -> None:  # noqa: ANN001
        self._model, self._w_frame, self._w_virial = model, w_frame, w_virial

    def __call__(self, **kwargs):  # noqa: ANN003, ANN204
        return weighted_predictions(
            self._model(**kwargs), self._w_frame, self._w_virial
        )

    def __getattr__(self, name: str):  # noqa: ANN204
        return getattr(self._model, name)


def install_stratified_replay(
    replay: int, ratio: float, width: int, size: int, count: int, stream_weight: float
) -> None:
    """Objective-preserving replay of the hard contacts (ledger E23).

    A hard frame — closest pair, under the minimum image, below ``ratio`` times the sum of the two
    atoms' covalent radii — is trained with weight ``stream_weight`` when the data stream delivers
    it and with weight ``(1 - stream_weight) / count`` on each of the ``count`` later replays drawn
    from a per-rank buffer of ``size`` frames (of up to ``width`` atoms; larger frames are not
    kept). Its total weight is therefore exactly one, the same as every other frame's: the expected
    gradient of the data is unchanged and only the intermittency of the rare frames' updates falls,
    each of their gradients arriving as ``count + 1`` smaller kicks instead of one. Up to ``replay``
    buffered frames that fit the batch's frame width are appended to every training batch so that
    they share the batch's normalization (a mean over frames for the energy and virial terms, a mean over real atoms for
    the force term); the fractional weights scale each frame's gradient contribution through the
    predictions (see :func:`weighted_predictions`), the replayed frames carrying no virial label and
    taking weight zero on the virial term. The model is evaluated once per batch. The
    predictions handed back to the trainer are those of the batch's own frames, so the batch-error
    log keeps reading true residuals. The batch grows by up to ``replay`` frames, which lowers the
    loss's normalization by that share — immaterial to the Muon path, whose update magnitude does
    not depend on the gradient's scale.
    """
    import numpy as np
    import torch
    from ase.data import (
        atomic_numbers,
        covalent_radii,
    )

    from deepmd.pt.loss.ener import (
        EnergyStdLoss,
    )

    original = EnergyStdLoss.forward
    rng = np.random.default_rng(0)
    state: dict = {
        "calls": 0,
        "buffer": [],
        "radii": None,
        "kept": 0,
        "replayed": 0,
        "evicted": 0,
        "batches_with_hard": 0,
    }
    replay_weight = (1.0 - stream_weight) / count

    def radii_for(model, device):  # noqa: ANN001, ANN202
        if state["radii"] is None:
            type_map = model.get_type_map()
            state["radii"] = torch.tensor(
                [covalent_radii[atomic_numbers[e]] for e in type_map],
                dtype=torch.float32,
                device=device,
            )
        return state["radii"]

    def closest_ratio(input_dict, model):  # noqa: ANN001, ANN202
        """Closest-pair distance over the sum of covalent radii, per frame, under the minimum image."""
        coord, atype = input_dict["coord"], input_dict["atype"]
        nf = atype.shape[0]
        xyz = coord.reshape(nf, -1, 3).to(torch.float32)
        real = atype >= 0
        radii = radii_for(model, xyz.device)[atype.clamp(min=0)]
        d = xyz[:, :, None, :] - xyz[:, None, :, :]
        cell = input_dict["box"].reshape(nf, 3, 3).to(xyz.device, torch.float32)
        frac = torch.einsum("nijk,nkl->nijl", d, torch.linalg.inv(cell))
        frac = frac - torch.round(frac)
        d = torch.einsum("nijl,nlk->nijk", frac, cell)
        dist = d.norm(dim=-1)
        valid = (
            real[:, :, None]
            & real[:, None, :]
            & ~torch.eye(atype.shape[1], dtype=torch.bool, device=xyz.device)
        )
        rel = torch.where(
            valid,
            dist / (radii[:, :, None] + radii[:, None, :]),
            torch.full_like(dist, float("inf")),
        )
        return rel.flatten(1).min(dim=1).values

    def store(input_dict, label, hard):  # noqa: ANN001, ANN202
        coord, atype, box = input_dict["coord"], input_dict["atype"], input_dict["box"]
        nf, nloc = atype.shape
        for k in torch.nonzero(hard).flatten().tolist():
            keep = (atype[k] >= 0).cpu()
            n_real = int(keep.sum())
            if n_real > width:
                continue
            frame = {
                "n": n_real,
                "coord": coord[k].reshape(nloc, 3)[keep].detach().float().cpu(),
                "atype": atype[k][keep].cpu(),
                "box": box[k].reshape(9).detach().float().cpu(),
                "energy": label["energy"][k].reshape(1).detach().float().cpu(),
                "force": label["force"][k]
                .reshape(nloc, 3)[keep]
                .detach()
                .float()
                .cpu(),
                "extra": {
                    key: (
                        value[k][keep]
                        if value.ndim >= 2 and value.shape[1] == nloc
                        else value[k]
                    )
                    .detach()
                    .cpu()
                    for key, value in input_dict.items()
                    if key not in ("coord", "atype", "box")
                    and isinstance(value, torch.Tensor)
                },
                "replays": 0,
            }
            if len(state["buffer"]) < size:
                state["buffer"].append(frame)
            else:
                state["buffer"][int(rng.integers(size))] = frame
            state["kept"] += 1

    def pick(nloc):  # noqa: ANN001, ANN202
        """Up to ``replay`` buffered frames that fit the batch's frame width, fewest replays first.

        A frame wider than the batch's frames waits for a wider batch: padding the whole batch up to
        the replayed frame's width would multiply its memory. Frames that reach ``count`` replays
        leave the buffer.
        """
        buffer = state["buffer"]
        eligible = [i for i in range(len(buffer)) if buffer[i]["n"] <= nloc]
        order = sorted(eligible, key=lambda i: (buffer[i]["replays"], rng.random()))
        picks = [buffer[i] for i in order[:replay]]
        for frame in picks:
            frame["replays"] += 1
        state["buffer"] = [f for f in buffer if f["replays"] < count]
        state["evicted"] += len(buffer) - len(state["buffer"])
        return picks

    def merged(input_dict, label, picks):  # noqa: ANN001, ANN202
        """The batch with the replayed frames appended, the replayed frames padded to the batch's frame width."""
        coord, atype, box = input_dict["coord"], input_dict["atype"], input_dict["box"]
        nf, nloc = atype.shape
        m = nloc
        dev, dt = coord.device, coord.dtype

        def pad(t, n_to, fill):  # noqa: ANN001, ANN202
            if t.shape[1] == n_to:
                return t
            shape = (t.shape[0], n_to - t.shape[1], *t.shape[2:])
            return torch.cat(
                [t, torch.full(shape, fill, dtype=t.dtype, device=t.device)], dim=1
            )

        def stack(key, fill):  # noqa: ANN001, ANN202
            return torch.stack([pad(f[key].unsqueeze(0), m, fill)[0] for f in picks])

        r = len(picks)

        def like(t, layout):  # noqa: ANN001, ANN202
            """Return the (nf + r, m, 3) tensor ``t`` in the layout of the batch tensor ``layout``: flat (nf, nloc * 3) or (nf, nloc, 3)."""
            return t.reshape(nf + r, m * 3) if layout.ndim == 2 else t

        merged_input = dict(input_dict)
        merged_input["coord"] = like(
            torch.cat(
                [
                    pad(coord.reshape(nf, nloc, 3), m, 0.0),
                    stack("coord", 0.0).to(dev, dt),
                ]
            ),
            coord,
        )
        merged_input["atype"] = torch.cat(
            [pad(atype, m, -1), stack("atype", -1).to(dev)]
        )
        merged_input["box"] = torch.cat(
            [
                box.reshape(nf, 9),
                torch.stack([f["box"] for f in picks]).to(box.device, box.dtype),
            ]
        ).reshape(nf + r, *box.shape[1:])
        # Frame-level fields (fparam, charge_spin) are stacked; atom-level fields (aparam) are padded like the coordinates.
        for key, value in input_dict.items():
            if key in ("coord", "atype", "box") or not isinstance(value, torch.Tensor):
                continue
            rows = [f["extra"][key].to(value.device, value.dtype) for f in picks]
            if value.ndim >= 2 and value.shape[1] == nloc:
                merged_input[key] = torch.cat(
                    [
                        pad(value, m, 0),
                        torch.stack([pad(row.unsqueeze(0), m, 0)[0] for row in rows]),
                    ]
                )
            else:
                merged_input[key] = torch.cat([value, torch.stack(rows)])
        merged_label = dict(label)
        merged_label["energy"] = torch.cat(
            [
                label["energy"].reshape(nf, 1),
                torch.stack([f["energy"] for f in picks]).to(
                    dev, label["energy"].dtype
                ),
            ]
        ).reshape(nf + r, *label["energy"].shape[1:])
        merged_label["force"] = like(
            torch.cat(
                [
                    pad(label["force"].reshape(nf, nloc, 3), m, 0.0),
                    stack("force", 0.0).to(dev, label["force"].dtype),
                ]
            ),
            label["force"],
        )
        if "virial" in label:
            merged_label["virial"] = torch.cat(
                [
                    label["virial"].reshape(nf, 9),
                    torch.zeros(
                        r, 9, dtype=label["virial"].dtype, device=label["virial"].device
                    ),
                ]
            ).reshape(nf + r, *label["virial"].shape[1:])
        return merged_input, merged_label, m

    def restore(value, nf, nloc, m, r):  # noqa: ANN001, ANN202
        """Cut a prediction of the merged batch (nf + r frames padded to m atoms) back to the batch's own nf frames and nloc atoms."""
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim == 0
            or value.shape[0] != nf + r
        ):
            return value
        value = value[:nf]
        if value.ndim >= 2 and value.shape[1] == m and m != nloc:
            return value[:, :nloc]
        if value.ndim == 2 and value.shape[1] == m * 3 and m != nloc:
            return value.reshape(nf, m, 3)[:, :nloc].reshape(nf, nloc * 3)
        return value

    def forward(self, input_dict, model, label, natoms, learning_rate, mae=False):  # noqa: ANN001, ANN202
        if not model.training or "force" not in label or "box" not in input_dict:
            return original(self, input_dict, model, label, natoms, learning_rate, mae)
        if any(
            isinstance(v, torch.Tensor)
            and (v.ndim == 0 or v.shape[0] != input_dict["atype"].shape[0])
            for k, v in input_dict.items()
            if k not in ("coord", "atype", "box")
        ):
            raise ValueError(
                "stratified replay: every tensor field of the batch must have the frame axis first"
            )
        state["calls"] += 1
        nf, nloc = input_dict["atype"].shape
        with torch.no_grad():
            hard = closest_ratio(input_dict, model) < ratio
        if bool(hard.any()):
            state["batches_with_hard"] += 1
            store(input_dict, label, hard)
        picks = pick(nloc)
        w_frame = torch.where(
            hard,
            torch.full_like(hard, stream_weight, dtype=torch.float32),
            torch.ones(nf, dtype=torch.float32, device=hard.device),
        )
        w_virial = torch.ones(nf, dtype=torch.float32, device=hard.device)
        if picks:
            input_dict, label, natoms = merged(input_dict, label, picks)
            r = len(picks)
            w_frame = torch.cat(
                [
                    w_frame,
                    torch.full(
                        (r,), replay_weight, dtype=torch.float32, device=hard.device
                    ),
                ]
            )
            w_virial = torch.cat(
                [w_virial, torch.zeros(r, dtype=torch.float32, device=hard.device)]
            )
            state["replayed"] += r
        pred, loss, more = original(
            self,
            input_dict,
            _WeightedModel(model, w_frame, w_virial),
            label,
            natoms,
            learning_rate,
            mae,
        )
        if picks:
            pred = {
                k: restore(v, nf, nloc, natoms, len(picks)) for k, v in pred.items()
            }
        if state["calls"] % 1000 == 0 or state["calls"] == 10:
            message = (
                f"Stratified replay at call {state['calls']}: {state['kept']} hard frames kept from {state['batches_with_hard']} batches "
                f"({len(state['buffer'])} in the buffer), {state['replayed']} replays, {state['evicted']} frames retired after {count} replays."
            )
            log.info(message)
            print(f"[stratified_replay] {message}", file=sys.stderr, flush=True)
        return pred, loss, more

    EnergyStdLoss.forward = forward
    message = (
        f"Stratified replay installed: closest pair below {ratio:.2f} of the covalent contact, stream weight {stream_weight:g}, "
        f"{count} replays of weight {replay_weight:.4f} each, up to {replay} frames appended per batch, buffer {size} frames of up to {width} atoms."
    )
    log.info(message)
    print(f"[stratified_replay] {message}", file=sys.stderr, flush=True)


def install_weight_sensitivity(
    weight: float, eps: float = 0.01, every: int = 10
) -> None:
    """Gain control at anchors (ledger E8): a bound on the model's off-data sensitivity to its weights.

    Every ``every`` training calls the batch is transformed by the dense anchor generators (Gaussian
    shell, collision, dilation, deletion; one per frame) into off-data configurations, the parameters
    are perturbed by a random relative amount (delta = eps * |theta| * xi with xi standard normal, one
    draw per call), and the change of the predicted forces under that perturbation is measured on the
    anchor frames and on the data frames as the root mean square over the real atoms, s_anchor and
    s_data. The penalty ``weight * relu(s_anchor / s_data - 1)^2`` is added to the batch's loss: the
    model's response to a weight change off the data may not exceed its response on the data, which
    is the lever by which the ordinary updates move the off-data function. The gradient flows through
    the anchor forwards at theta and at theta + delta (``torch.func.functional_call`` for the latter);
    s_data is a constant of the call and the perturbation itself is a constant. No reference and no
    label enter, and evaluation needs nothing.
    """
    import numpy as np
    import torch

    from deepmd.dpmodel.utils.safeguard import (
        resolve_safeguard_params,
        transform_batch,
    )
    from deepmd.pt.loss.ener import (
        EnergyStdLoss,
    )

    original = EnergyStdLoss.forward
    rng = np.random.default_rng(0)
    params = resolve_safeguard_params(
        {"strength": 1.0, "generators": "dense_sparse"}, finetune=False
    )
    state = {"calls": 0, "events": 0, "ratio": 0.0, "penalty": 0.0}

    def force_rms(delta_force, atype):  # noqa: ANN001, ANN202
        real = (atype >= 0).reshape(-1)
        change = delta_force.reshape(-1, 3)[real]
        return torch.sqrt(change.square().sum(-1).mean() + 1e-30)

    def forward(self, input_dict, model, label, natoms, learning_rate, mae=False):  # noqa: ANN001, ANN202
        out = original(self, input_dict, model, label, natoms, learning_rate, mae)
        if not model.training or "force" not in label:
            return out
        state["calls"] += 1
        if state["calls"] % every != 0:
            return out
        model_pred, loss, more = out
        anchor = transform_batch(input_dict, params, rng, force=label["force"])
        anchor = {k: v for k, v in anchor.items() if v is not None}
        perturbed = {
            name: p + eps * p.detach().abs() * torch.randn_like(p)
            for name, p in model.named_parameters()
        }
        # The model differentiates its energy with respect to the coordinates
        # inside the forward, so no forward runs under ``no_grad``; the data
        # response is detached instead.
        data_plus = torch.func.functional_call(model, perturbed, (), input_dict)[
            "force"
        ].detach()
        s_data = force_rms(
            data_plus - model_pred["force"].detach(), input_dict["atype"]
        )
        anchor_pred = model(**anchor)["force"]
        anchor_plus = torch.func.functional_call(model, perturbed, (), anchor)["force"]
        s_anchor = force_rms(anchor_plus - anchor_pred, anchor["atype"])
        penalty = weight * torch.relu(s_anchor / s_data - 1.0).square()
        state["events"] += 1
        state["ratio"] += float(s_anchor / s_data)
        state["penalty"] += float(penalty)
        if state["calls"] == every or state["calls"] % 2000 == 0:
            n = state["events"]
            message = (
                f"Weight sensitivity at call {state['calls']}: anchor/data force response ratio {state['ratio'] / n:.3f}, "
                f"mean penalty {state['penalty'] / n:.4f} over {n} events."
            )
            log.info(message)
            print(f"[weight_sensitivity] {message}", file=sys.stderr, flush=True)
            state.update(events=0, ratio=0.0, penalty=0.0)
        return model_pred, loss + penalty, more

    EnergyStdLoss.forward = forward
    message = f"Weight-sensitivity bound installed: relative perturbation {eps:g}, weight {weight:g}, every {every} calls."
    log.info(message)
    print(f"[weight_sensitivity] {message}", file=sys.stderr, flush=True)


def install_gain_control(
    kappa: float,
    share: float,
    every: int,
    n_probes: int,
    slice_frames: int,
    delta: float,
) -> None:
    """Gain control at the batch's own compressed pairs (ledger E32).

    The quantity that leads the tear of the function at compressed contacts is its gain: the
    root-mean-square change of the forces on label-free probes under a random weight perturbation
    the size of one optimizer step, over the same change on the batch's own frames. Every ``every``
    training calls the closest pair of each of the first ``n_probes`` frames of the batch (ranked by
    distance over covalent contact, under the minimum image) is placed alone in its frame's cell and
    compressed to 0.45 of the contact, the parameters are perturbed by an isotropic random vector of
    norm ``delta`` (one draw per event), and the force change under that perturbation is measured on
    the probes (s_probe, through both forwards) and on a slice of ``slice_frames`` batch frames
    (s_data, a constant of the event). The penalty relu(s_probe / s_data - kappa)² bounds the probes'
    gain relative to the data's. Its parameter gradient is 10²-10⁴ times the data gradient's and nearly
    orthogonal to it, so it is not weighted but given a fixed share of the update: the gradient that
    reaches the optimizer is g_data + share · ‖g_data‖ · g_pen / ‖g_pen‖, realized through a surrogate
    term of zero value whose gradient is the rescaled penalty gradient; ‖g_data‖ is the total gradient
    norm that the trainer's clipping measured at the previous step (recorded through
    :func:`clip_grad_norm_`; it changes slowly from step to step), so no second backward pass of the
    compiled data graph is needed. No reference, no label and no anchor generator enter; evaluation
    needs nothing.
    """
    import torch
    from ase.data import (
        atomic_numbers,
        covalent_radii,
    )

    from deepmd.pt.loss.ener import (
        EnergyStdLoss,
    )
    from deepmd.pt.train import training as training_module

    original = EnergyStdLoss.forward
    original_clip = training_module.clip_grad_norm_
    generator = torch.Generator().manual_seed(0)
    state: dict = {
        "calls": 0,
        "events": 0,
        "ratio": 0.0,
        "penalty": 0.0,
        "active": 0,
        "radii": None,
        "data_norm": None,
    }

    def clip_and_record(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        total_norm = original_clip(*args, **kwargs)
        state["data_norm"] = total_norm.detach()
        return total_norm

    training_module.clip_grad_norm_ = clip_and_record

    def radii_for(model, device):  # noqa: ANN001, ANN202
        if state["radii"] is None:
            type_map = model.get_type_map()
            state["radii"] = torch.tensor(
                [covalent_radii[atomic_numbers[e]] for e in type_map],
                dtype=torch.float32,
                device=device,
            )
        return state["radii"]

    def compressed_probes(input_dict, model):  # noqa: ANN001, ANN202
        """The closest pair of each of the first ``n_probes`` frames, alone in its cell, compressed to 0.45 of the contact."""
        coord, atype, box = input_dict["coord"], input_dict["atype"], input_dict["box"]
        nf = min(n_probes, atype.shape[0])
        xyz = coord[:nf].reshape(nf, -1, 3).to(torch.float32)
        types = atype[:nf]
        real = types >= 0
        radii = radii_for(model, xyz.device)[types.clamp(min=0)]
        cell = box[:nf].reshape(nf, 3, 3).to(xyz.device, torch.float32)
        d = xyz[:, :, None, :] - xyz[:, None, :, :]
        frac = torch.einsum("nijk,nkl->nijl", d, torch.linalg.inv(cell))
        frac = frac - torch.round(frac)
        d = torch.einsum("nijl,nlk->nijk", frac, cell)
        dist = d.norm(dim=-1)
        contact = radii[:, :, None] + radii[:, None, :]
        valid = (
            real[:, :, None]
            & real[:, None, :]
            & ~torch.eye(types.shape[1], dtype=torch.bool, device=xyz.device)
        )
        rel = torch.where(valid, dist / contact, torch.full_like(dist, float("inf")))
        flat = rel.flatten(1).argmin(dim=1)
        i, j = flat // types.shape[1], flat % types.shape[1]
        ar = torch.arange(nf, device=xyz.device)
        unit = d[ar, i, j] / dist[ar, i, j].clamp(min=1e-6).unsqueeze(-1)
        target = 0.45 * contact[ar, i, j]
        first = xyz[ar, i]
        second = first + unit * target.unsqueeze(-1)
        probe_coord = torch.stack([first, second], dim=1).to(coord.dtype)  # (nf, 2, 3)
        probe_atype = torch.stack([types[ar, i], types[ar, j]], dim=1)
        return {
            "coord": probe_coord.reshape(nf, 6) if coord.ndim == 2 else probe_coord,
            "atype": probe_atype,
            "box": box[:nf],
        }

    def force_rms(delta_force, atype):  # noqa: ANN001, ANN202
        real = (atype >= 0).reshape(-1)
        change = delta_force.reshape(-1, 3)[real]
        return torch.sqrt(change.square().sum(-1).mean() + 1e-30)

    def forward(self, input_dict, model, label, natoms, learning_rate, mae=False):  # noqa: ANN001, ANN202
        out = original(self, input_dict, model, label, natoms, learning_rate, mae)
        if not model.training or "force" not in label or "box" not in input_dict:
            return out
        state["calls"] += 1
        if state["calls"] % every != 0:
            return out
        model_pred, loss, more = out
        params = [p for p in model.parameters() if p.requires_grad]
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        # === Step 1. One isotropic perturbation of norm ``delta`` ===
        noise = [
            torch.randn(p.shape, generator=generator).to(p.device, p.dtype)
            for p in params
        ]
        norm = torch.sqrt(sum((n * n).sum() for n in noise))
        perturbed = {
            name: p.detach() + delta * n / norm
            for name, p, n in zip(names, params, noise, strict=True)
        }
        # === Step 2. The data's response on a slice of the batch (a constant) ===
        nf_slice = min(slice_frames, input_dict["atype"].shape[0])
        slice_in = {
            k: (
                v[:nf_slice]
                if isinstance(v, torch.Tensor)
                and v.ndim > 0
                and v.shape[0] == input_dict["atype"].shape[0]
                else v
            )
            for k, v in input_dict.items()
        }
        slice_plus = torch.func.functional_call(model, perturbed, (), slice_in)[
            "force"
        ].detach()
        s_data = force_rms(
            slice_plus
            - model_pred["force"][:nf_slice].detach().reshape(slice_plus.shape),
            slice_in["atype"],
        )
        # === Step 3. The probes' response, differentiable through both forwards ===
        probes = compressed_probes(input_dict, model)
        probe_pred = model(**probes)["force"]
        probe_plus = torch.func.functional_call(model, perturbed, (), probes)["force"]
        s_probe = force_rms(probe_plus - probe_pred, probes["atype"])
        ratio = s_probe / s_data
        penalty = torch.relu(ratio - kappa).square()
        state["events"] += 1
        state["ratio"] += float(ratio.detach())
        state["penalty"] += float(penalty)
        surrogate = loss.new_zeros(())
        if float(penalty) > 0.0 and state["data_norm"] is not None:
            # === Step 4. The fixed share: g_data + share · ‖g_data‖ · g_pen / ‖g_pen‖ ===
            g_pen = torch.autograd.grad(penalty, params, allow_unused=True)
            pen_norm = torch.sqrt(sum((g * g).sum() for g in g_pen if g is not None))
            coef = (
                share
                * state["data_norm"].to(pen_norm.device, pen_norm.dtype)
                / (pen_norm + 1e-30)
            )
            surrogate = sum(
                (coef * g.detach() * (p - p.detach())).sum()
                for g, p in zip(g_pen, params, strict=True)
                if g is not None
            )
            state["active"] += 1
        if state["calls"] == every or state["calls"] % 2000 == 0:
            n = max(state["events"], 1)
            message = (
                f"Gain control at call {state['calls']}: probe/data force response ratio {state['ratio'] / n:.2f}, "
                f"mean penalty {state['penalty'] / n:.3g}, active on {state['active']} of {n} events."
            )
            log.info(message)
            print(f"[gain_control] {message}", file=sys.stderr, flush=True)
            state.update(events=0, ratio=0.0, penalty=0.0, active=0)
        return model_pred, loss + surrogate, more

    EnergyStdLoss.forward = forward
    message = (
        f"Gain control installed: kappa {kappa:g}, share {share:g} of the data gradient, every {every} calls, "
        f"{n_probes} compressed-pair probes, a {slice_frames}-frame data slice, perturbation norm {delta:g}."
    )
    log.info(message)
    print(f"[gain_control] {message}", file=sys.stderr, flush=True)


def install_contradiction_filter(ratio: float, force_max: float) -> None:
    """Zero the loss contribution of the predeclared suspect frames (the counterfactual of ledger E7).

    A frame belongs to the suspect family when its closest pair (minimum image) lies below
    ``ratio`` of the covalent contact distance while the label forces on both atoms of the pair stay
    below ``force_max`` eV/Å: a geometry on which physics demands hundreds of eV/Å that the label does
    not carry. The batch keeps its content and the loss keeps its denominators; the flagged frames'
    labels are replaced by the model's own detached predictions, so their residuals, and with them
    their gradients, vanish exactly. The model is evaluated a second time on the batches that carry a
    flagged frame (about one batch in five at the family's share of 0.055% and 536 frames per batch).
    """
    import torch
    from ase.data import (
        atomic_numbers,
        covalent_radii,
    )

    from deepmd.pt.loss.ener import (
        EnergyStdLoss,
    )

    original = EnergyStdLoss.forward
    state = {"calls": 0, "flagged": 0, "batches": 0, "radii": None}

    def suspect_frames(input_dict, label, model):  # noqa: ANN001, ANN202
        coord, atype = input_dict["coord"], input_dict["atype"]
        nf, nloc = atype.shape
        xyz = coord.reshape(nf, nloc, 3).to(torch.float32)
        real = atype >= 0
        if state["radii"] is None:
            type_map = model.get_type_map()
            state["radii"] = torch.tensor(
                [covalent_radii[atomic_numbers[e]] for e in type_map],
                dtype=torch.float32,
                device=xyz.device,
            )
        radii = state["radii"][atype.clamp(min=0)]
        d = xyz[:, :, None, :] - xyz[:, None, :, :]
        box = input_dict.get("box")
        if box is not None:
            cell = box.reshape(nf, 3, 3).to(xyz.device, torch.float32)
            frac = torch.einsum("nijk,nkl->nijl", d, torch.linalg.inv(cell))
            frac = frac - torch.round(frac)
            d = torch.einsum("nijl,nlk->nijk", frac, cell)
        dist = d.norm(dim=-1)
        valid = (
            real[:, :, None]
            & real[:, None, :]
            & ~torch.eye(nloc, dtype=torch.bool, device=xyz.device)
        )
        rel = torch.where(
            valid,
            dist / (radii[:, :, None] + radii[:, None, :]),
            torch.full_like(dist, float("inf")),
        )
        k = rel.flatten(1).argmin(dim=1)
        i, j = k // nloc, k % nloc
        fmag = label["force"].reshape(nf, nloc, 3).to(torch.float32).norm(dim=-1)
        rows = torch.arange(nf, device=xyz.device)
        close = rel.flatten(1)[rows, k] < ratio
        quiet = torch.maximum(fmag[rows, i], fmag[rows, j]) < force_max
        return close & quiet

    def forward(self, input_dict, model, label, natoms, learning_rate, mae=False):  # noqa: ANN001, ANN202
        out = original(self, input_dict, model, label, natoms, learning_rate, mae)
        if not model.training or "force" not in label:
            return out
        state["calls"] += 1
        with torch.no_grad():
            flag = suspect_frames(input_dict, label, model)
        if bool(flag.any()):
            silenced = dict(label)
            for key in ("energy", "force", "virial"):
                if key in label and key in out[0]:
                    value = label[key].clone()
                    pred = out[0][key].detach().to(value.dtype).reshape(value.shape)
                    value[flag] = pred[flag]
                    silenced[key] = value
            # The first forward's graph is released before the second forward,
            # so the batch's peak memory stays that of one forward.
            del out
            out = original(
                self, input_dict, model, silenced, natoms, learning_rate, mae
            )
            state["flagged"] += int(flag.sum())
            state["batches"] += 1
        if state["calls"] % 10000 == 0:
            log.info(
                "Contradiction filter at call %d: %d frames silenced in %d batches.",
                state["calls"],
                state["flagged"],
                state["batches"],
            )
        return out

    EnergyStdLoss.forward = forward
    log.info(
        "Contradiction filter installed: frames whose closest pair lies below %.2f of the covalent contact with both label forces below %.0f eV/A contribute no loss.",
        ratio,
        force_max,
    )


def install_muon_unwhitened() -> None:
    """Replace every Muon matrix update by its unwhitened control (ledger E24).

    HybridMuon orthogonalizes the momentum matrix G of every 2-D parameter into its polar factor Q,
    which whitens the update: every singular direction of G receives the same step regardless of its
    singular value. The control keeps everything else of the optimizer — the momentum, the routing
    of parameters to Muon or Adam, the per-shape scale, the Magma damping, the decay and the
    schedule — and returns ``‖Q‖_F · G / ‖G‖_F`` in place of Q: the Frobenius norm of the
    orthogonalized update along the raw momentum direction. Every orthogonalization path of the
    optimizer is covered (the Gram Newton-Schulz path of the rectangular buckets and the standard and
    Triton paths of the square buckets). The Newton-Schulz iteration still runs, because ``‖Q‖_F`` is
    read from its result instead of being assumed equal to the square root of the rank; the patch is
    tensor-only, so the optimizer step can still be captured into one CUDA graph.
    """
    import torch

    from deepmd.pt.optimizer import hybrid_muon as hm

    def control(G: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        dims = (-2, -1)
        q_norm = Q.to(torch.float32).norm(dim=dims, keepdim=True)
        g_norm = G.to(torch.float32).norm(dim=dims, keepdim=True)
        return G * (q_norm / (g_norm + hm.NS_EPS)).to(G.dtype)

    gram_call = hm._GramNewtonSchulzOrthogonalizer.__call__
    flash_orth = hm._flash_newton_schulz_orth
    batched_orth = hm._batched_newton_schulz_orth

    def gram_control(self, X):  # noqa: ANN001, ANN202
        return control(X, gram_call(self, X))

    def flash_control(G, buf1, buf2):  # noqa: ANN001, ANN202
        return control(G, flash_orth(G, buf1, buf2))

    def batched_control(G):  # noqa: ANN001, ANN202
        return control(G, batched_orth(G))

    hm._GramNewtonSchulzOrthogonalizer.__call__ = gram_control
    hm._flash_newton_schulz_orth = flash_control
    hm._batched_newton_schulz_orth = batched_control
    log.info(
        "[muon_unwhitened] Muon matrix updates replaced by the unwhitened control ‖Q‖_F · G / ‖G‖_F on every orthogonalization path."
    )


def install_dens_inactive_head_freeze() -> None:
    """Freeze the fitting head that the active SeZM mode does not use (ledger E21, DeNS runs).

    A SeZM model carries two fitting heads, the energy-mode ``fitting_net`` and the DeNS-mode
    ``dens_fitting_net``; only the active one takes part in the loss, so under DistributedDataParallel
    the other head's parameters never receive a gradient and the reducer stops after the first
    iteration. Freezing the inactive head whenever the mode is switched keeps the reducer's parameter
    set equal to the set that produces the loss; the frozen head is not trained in either mode anyway.
    """
    from deepmd.pt.model.atomic_model.sezm_atomic_model import (
        SeZMAtomicModel,
    )

    original = SeZMAtomicModel.set_active_mode

    def set_active_mode(self, mode):  # noqa: ANN001, ANN202
        original(self, mode)
        active = self.get_active_mode()
        self.fitting_net.requires_grad_(active == "ener")
        if self.dens_fitting_net is not None:
            self.dens_fitting_net.requires_grad_(active == "dens")

    SeZMAtomicModel.set_active_mode = set_active_mode


def install_batch_error_log(
    dump_above: float = 500.0,
    max_dumps: int = 20,
    geometry_above: float = 100.0,
    tear_above: float = 1000.0,
    max_tears: int = 10,
) -> None:
    """Diagnostic: log the largest per-atom force error of every training batch.

    The ``pt`` loss is patched so that each training call (the wrapper is in
    evaluation mode during validation, which is skipped) appends one line to
    ``batch_max_error_rank<r>.log`` in the run directory: the training-step
    index, the largest per-atom force error of the batch in eV/A, and the
    number of atoms whose error exceeds 20 eV/A. Phantom atoms are excluded.
    Independent of the safeguard, so a plain run can carry the log.

    A batch whose largest per-atom error exceeds ``dump_above`` eV/A (a needle)
    is written to ``needle_rank<r>_step<n>.npz`` with its coordinates, cell,
    types, labels and the model's outputs, so that the needle can be attributed
    to the model or to the label. At most ``max_dumps`` batches are written per
    rank.

    Every atom above ``geometry_above`` is also recorded in a JSON-lines sidecar
    with its exact periodic contacts and both exclusion-zone classifications.
    These geometry records are independent of the batch and model snapshot limits.

    A batch whose largest per-atom error exceeds ``tear_above`` eV/A (a tear) also
    has the model written as it is at that forward pass, before the optimizer
    step, to ``tear_rank<r>_step<n>.pt`` in the format of a training checkpoint
    (the wrapper's state, loadable by ``diagnose/repro_spike.py``): the event at
    its own weights, so that it can be replayed through the production and the
    eager path and scanned around its geometry. At most ``max_tears`` per rank.
    """
    import functools

    import numpy as np
    import torch

    from deepmd.pt.loss.ener import (
        EnergyStdLoss,
    )
    from deepmd.pt.train.training import (
        Trainer,
    )
    from deepmd.pt.train.wrapper import (
        ModelWrapper,
    )

    original = EnergyStdLoss.forward
    original_run = Trainer.run
    original_wrapper_forward = ModelWrapper.forward
    state: dict = {
        "step": 0,
        "start_step": 0,
        "file": None,
        "dumps": 0,
        "tears": 0,
        "wrapper": None,
    }

    @functools.wraps(original_run)
    def run(self: Trainer) -> None:
        """Use the trainer's restored optimizer step as the diagnostic origin."""
        state["step"] = state["start_step"] = self.start_step
        original_run(self)

    Trainer.run = run

    # The loss is called from inside the wrapper's forward, so the wrapper seen last is the one
    # whose loss is being evaluated; ``wraps`` keeps the forward's signature, which the safeguard
    # inspects to select its loss-free call (``skip_loss``).
    @functools.wraps(original_wrapper_forward)
    def wrapper_forward(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        state["wrapper"] = self
        return original_wrapper_forward(self, *args, **kwargs)

    ModelWrapper.forward = wrapper_forward

    def write_needle_geometry(
        input_dict: dict[str, torch.Tensor],
        label: dict[str, torch.Tensor],
        prediction: dict[str, torch.Tensor],
        err: torch.Tensor,
        rank: int,
    ) -> float:
        """Record exact contacts of every large-error atom independently of snapshot limits."""
        from diagnose.needle_geometry import (
            frame_geometry,
        )

        model = state["wrapper"].model["Default"]
        types = input_dict["atype"].detach().cpu().numpy()
        errors = err.reshape(types.shape).cpu().numpy()
        coordinates = (
            input_dict["coord"].detach().reshape(*types.shape, 3).double().cpu().numpy()
        )
        forces = (
            prediction["force"].detach().reshape(*types.shape, 3).double().cpu().numpy()
        )
        labels = label["force"].detach().reshape(*types.shape, 3).double().cpu().numpy()
        box = input_dict.get("box")
        boxes = (
            box.detach().reshape(-1, 3, 3).double().cpu().numpy()
            if box is not None
            else None
        )
        records = []
        for frame in np.flatnonzero((errors > geometry_above).any(axis=1)):
            for item in frame_geometry(
                coordinates[frame],
                types[frame],
                errors[frame],
                forces[frame],
                labels[frame],
                None if boxes is None else boxes[frame],
                model.get_type_map(),
                geometry_above,
                model.get_rcut(),
            ):
                records.append({"frame": int(frame), **item})
        event = {
            "step": state["step"],
            "session_start_step": state["start_step"],
            "step_kind": "global_zero_based",
            "bad_atoms": records,
        }
        with open(f"needle_geometry_rank{rank}.jsonl", "a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        distance = max(records, key=lambda item: item["force_error_eV_A"])[
            "nearest_distance_A"
        ]
        return float("inf") if distance is None else distance

    def dump_batch(rank: int, step: int, input_dict, label, out, err) -> None:  # noqa: ANN001
        arrays = {"step": np.array(step), "per_atom_error": err.cpu().numpy()}
        for key in ("coord", "atype", "box", "fparam", "aparam"):
            if key in input_dict and input_dict[key] is not None:
                arrays[key] = input_dict[key].detach().cpu().numpy()
        for key in ("energy", "force", "virial"):
            if key in label:
                arrays["label_" + key] = label[key].detach().cpu().numpy()
            if key in out:
                arrays["model_" + key] = out[key].detach().cpu().numpy()
        np.savez(f"needle_rank{rank}_step{step}.npz", **arrays)

    def forward(self, input_dict, model, label, natoms, learning_rate, mae=False):  # noqa: ANN001, ANN202
        out = original(self, input_dict, model, label, natoms, learning_rate, mae)
        if model.training and "force" in label and "force" in out[0]:
            err = torch.linalg.vector_norm(
                out[0]["force"].detach().reshape(-1, 3) - label["force"].reshape(-1, 3),
                dim=-1,
            )
            err = torch.where(
                input_dict["atype"].reshape(-1) >= 0, err, torch.zeros_like(err)
            )
            peak, above = float(err.max()), int((err > 20.0).sum())
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )
            if state["file"] is None:
                state["file"] = open(f"batch_max_error_rank{rank}.log", "a")
                state["file"].write(
                    f"# session start_step={state['start_step']} step_kind=global_zero_based\n"
                )
                state["file"].write(
                    "# step  largest per-atom force error (eV/A)  atoms above 20 eV/A  lr  nearest-neighbour distance of the worst atom (A, when the error exceeds the geometry threshold)\n"
                )
            nearest = (
                write_needle_geometry(input_dict, label, out[0], err, rank)
                if peak > geometry_above
                else None
            )
            state["file"].write(
                f"{state['step']:8d} {peak:10.3f} {above:5d} {learning_rate:.3e} {'-' if nearest is None else f'{nearest:.3f}'}\n"
            )
            state["file"].flush()
            if peak > dump_above and state["dumps"] < max_dumps:
                dump_batch(rank, state["step"], input_dict, label, out[0], err)
                state["dumps"] += 1
                log.warning(
                    f"Needle at training call {state['step']}: largest per-atom force error {peak:.1f} eV/A; batch written to needle_rank{rank}_step{state['step']}.npz"
                )
            if (
                peak > tear_above
                and state["tears"] < max_tears
                and state["wrapper"] is not None
            ):
                torch.save(
                    {"model": state["wrapper"].state_dict(), "step": state["step"]},
                    f"tear_rank{rank}_step{state['step']}.pt",
                )
                state["tears"] += 1
                log.warning(
                    f"Tear at training call {state['step']}: the model at this forward pass written to tear_rank{rank}_step{state['step']}.pt"
                )
            state["step"] += 1
        return out

    EnergyStdLoss.forward = forward
    log.info("Batch-error log installed: batch_max_error_rank<r>.log")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="training input file")
    ap.add_argument(
        "--backend",
        choices=["pt", "pt-expt"],
        default="pt",
        help="the DeePMD-kit backend the training runs on (pt-expt for the DPA4C descriptor)",
    )
    ap.add_argument(
        "--refresh",
        type=int,
        required=True,
        help="replace the reference by a copy of the model every this many steps",
    )
    ap.add_argument(
        "--refresh-steps",
        default=None,
        help="ledger E3: an explicit comma-separated list of hand-over steps (for example 10000,60000) replacing the "
        "periodic --refresh/--max-refreshes rule; --refresh is still required and only sets the anchor bookkeeping",
    )
    ap.add_argument(
        "--max-refreshes",
        type=int,
        default=0,
        help="at most this many refreshes; 1 replaces the initial reference once "
        "and keeps that copy for the rest of the run (0 = unlimited)",
    )
    ap.add_argument(
        "--dense-after-refresh",
        action="store_true",
        help="after the first refresh, resolve the generators as for a fine-tune "
        "(dense and sparse set) instead of the sparse pretraining set",
    )
    ap.add_argument(
        "--collision-after-refresh",
        action="store_true",
        help="after the first refresh, keep the sparse generators and add the "
        "collision generator (0.3 A against the force), without the Gaussian shell",
    )
    ap.add_argument(
        "--near-after-refresh",
        action="store_true",
        help="after the first refresh, use only the near generators (Gaussian shell and collision)",
    )
    ap.add_argument(
        "--label-tube-after-refresh",
        action="store_true",
        help="after the first refresh, use the dense and sparse set (or the near set with "
        "--near-after-refresh) and widen every anchor atom's tube by the reference's error "
        "|F_ref - F_label| on that atom in the frame the anchor was made from",
    )
    ap.add_argument(
        "--label-tube-near-only",
        action="store_true",
        help="with --label-tube-after-refresh, widen the tube only on the anchor frames that kept "
        "the data's density (shell, collision); dilated and thinned frames keep the fixed tube",
    )
    ap.add_argument(
        "--pair-term",
        action="store_true",
        help="add the reference-free pair-consistency term at every anchor event: the closest pair "
        "of every batch frame, evaluated alone, is held to the label's radial force on that pair",
    )
    ap.add_argument(
        "--pair-term-anchors",
        action="store_true",
        help="with --pair-term, also hold the closest pair of a collision-compressed copy of every frame, alone, "
        "to the model's own radial force on it inside the compressed frame (reaches inside the wall)",
    )
    ap.add_argument(
        "--pair-term-tol",
        type=float,
        default=5.0,
        help="absolute tolerance of the pair term in eV/A",
    )
    ap.add_argument(
        "--pair-term-rel",
        type=float,
        default=1.0,
        help="relative tolerance of the pair term",
    )
    ap.add_argument(
        "--pair-term-contact-ratio",
        type=float,
        default=0.0,
        help="E20: closest pairs below this fraction of the covalent contact distance take the tight tube of the pair term (0 = one tube for every pair)",
    )
    ap.add_argument(
        "--pair-term-contact-tol",
        type=float,
        default=1.0,
        help="E20: absolute tolerance of the pair term on contacts, in eV/A",
    )
    ap.add_argument(
        "--pair-term-contact-rel",
        type=float,
        default=0.5,
        help="E20: relative tolerance of the pair term on contacts",
    )
    ap.add_argument(
        "--anchor-descent",
        type=int,
        default=0,
        help="after the generators, move every anchor frame along the model's own forces for this "
        "many steepest-descent steps before the hinge (0 = off)",
    )
    ap.add_argument(
        "--anchor-descent-step",
        type=float,
        default=0.05,
        help="largest per-atom move of one descent step, in A",
    )
    ap.add_argument(
        "--ensemble-refs",
        nargs="+",
        default=None,
        help="at the hand-over, the reference is the mean of the model's own copy and these checkpoints "
        "(independently trained models of the same architecture at the hand-over step)",
    )
    ap.add_argument(
        "--mixing-loops",
        type=int,
        default=1,
        help="E13: apply the gated SO(2) mixing layers of every block this many times (weight-tied)",
    )
    ap.add_argument(
        "--block-loops",
        type=int,
        default=1,
        help="E13: apply the interaction blocks this many times (weight-tied)",
    )
    ap.add_argument(
        "--stay-or-step",
        action="store_true",
        help="E14c: the SO(2) depth attention chooses between the current state and the stepped state at every mixing layer",
    )
    ap.add_argument(
        "--post-add-norm",
        action="store_true",
        help="E14d: the SO(2) mixing chain's norms move from the layer input to the residual stream after every add",
    )
    ap.add_argument(
        "--log-batch-error",
        action="store_true",
        help="diagnostic: write the largest per-atom force error of every data batch to batch_max_error_rank<r>.log",
    )
    ap.add_argument(
        "--attnres-keys",
        choices=["l0", "degrees"],
        default="l0",
        help="E14 (h): keys of the SO(2) depth attention from the l = 0 scalars only, or from every degree's norm as well",
    )
    ap.add_argument(
        "--attnres-weights",
        choices=["shared", "per_degree"],
        default="shared",
        help="E14 (i): one softmax over the sources shared by every degree, or one softmax per degree",
    )
    ap.add_argument(
        "--attnres-value-norm",
        action="store_true",
        help="E14 (j): RMS-normalize every source's degree blocks (with a learned per-degree gain) before the combination",
    )
    ap.add_argument(
        "--agg-logit-bound",
        type=float,
        default=0.0,
        help="E16: bound the logits of the envelope-gated edge aggregation by b·tanh(L/b) with this b (0 = off)",
    )
    ap.add_argument(
        "--repulsion-hinge",
        type=float,
        default=0.0,
        help="E11: weight of the inequality relu(-f_r)^2 on the closest pair of every collision-compressed batch frame whose pair lies "
        "below 0.55 of the covalent contact distance (0 = off)",
    )
    ap.add_argument(
        "--anchor-draws",
        type=int,
        default=1,
        help="E4: draw every anchor frame this many times and train on the draw with the largest hinge excess",
    )
    ap.add_argument(
        "--repulsion-ratio",
        type=float,
        default=0.55,
        help="E11: the fraction of the covalent contact distance below which the inequality applies",
    )
    ap.add_argument(
        "--dimer-hinge",
        type=float,
        default=0.0,
        help="E58: weight of the inequality relu(-f_r)^2 on isolated two-atom frames of the batch's element pairs, built in vacuum at every anchor event (0 = off)",
    )
    ap.add_argument(
        "--dimer-ratio",
        type=str,
        default="0.3,0.5",
        help="E58: the separation range of the isolated pairs as fractions of the covalent contact distance, lo,hi",
    )
    ap.add_argument(
        "--dimer-count",
        type=int,
        default=32,
        help="E58: isolated pairs per anchor event",
    )
    ap.add_argument(
        "--dimer-contacts",
        type=str,
        default=None,
        help="JSON of pair_contact_stats.py; the dimer inequality's reference distance is then the data's 5th-percentile closest distance instead of the covalent contact (unused by E58/E59)",
    )
    ap.add_argument(
        "--data-sign-hinge",
        type=float,
        default=0.0,
        help="E59: weight of the inequality relu(-f_r)^2 on the isolated copy of every batch frame's most compressed pair whose label pushes it apart by more than --data-sign-margin (0 = off)",
    )
    ap.add_argument(
        "--data-sign-margin",
        type=float,
        default=5.0,
        help="E59: the label's radial repulsion, in eV/A, above which a compressed pair is carried into the sparse limit",
    )
    ap.add_argument(
        "--data-sign-count",
        type=int,
        default=32,
        help="E59: the fixed number of vacuum pair frames per event (the most compressed qualifying pairs, repeated and masked to fill)",
    )
    ap.add_argument(
        "--tail-penalty",
        type=float,
        default=0.0,
        help="E67: weight of f_r^2 on isolated two-atom frames of the batch's element pairs built in vacuum at every anchor event with separations in --tail-range, where the physical force is zero (0 = off)",
    )
    ap.add_argument(
        "--tail-range",
        type=str,
        default="4.0,6.0",
        help="E67: the separation range of the tail pairs in A, lo,hi (lo is raised to 1.6 covalent contacts for pairs whose contact reaches it)",
    )
    ap.add_argument(
        "--tail-count", type=int, default=24, help="E67: tail pairs per anchor event"
    )
    ap.add_argument(
        "--gauss-inner",
        type=float,
        default=0.0,
        help="E70: initialize Gaussian centres on linspace(0, this radius), with proportional width unless --gauss-width is given; centres remain trainable unless --fixed-radial is set (0 = full cutoff)",
    )
    ap.add_argument(
        "--gauss-width",
        type=float,
        default=0.0,
        help="set an independent Gaussian width in Angstrom (0 = initial centre spacing)",
    )
    ap.add_argument(
        "--adam-route",
        default="",
        help="E110: comma-separated substrings of parameter names; every matching parameter takes "
        "the hybrid optimizer's Adam route (no weight decay) instead of Muon, extending the model's "
        "own adam_ naming convention (e.g. radial_embedding.net.0.matrix)",
    )
    ap.add_argument(
        "--gauss-floor",
        type=float,
        default=0.0,
        help="E102: lower endpoint of the Gaussian centres in Angstrom, so that they lie on "
        "linspace(floor, --gauss-inner, n_radial); removes the channels centred where the data "
        "have no pair distances (0 = centres start at zero)",
    )
    ap.add_argument(
        "--gauss-scales",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="use fixed zero-centred Gaussians with geometrically spaced widths in Angstrom",
    )
    ap.add_argument(
        "--single-envelope",
        action="store_true",
        help="E69: return the bare radial basis while retaining the downstream edge envelope",
    )
    ap.add_argument(
        "--fixed-radial",
        action="store_true",
        help="E71: keep radial centres or frequencies fixed throughout training",
    )
    ap.add_argument(
        "--output-envelope",
        action="store_true",
        help="E73: apply the cutoff to the final per-edge value entering normalized attention",
    )
    ap.add_argument(
        "--separate-attention-mass",
        action="store_true",
        help="E87: normalize neighbor selection separately from total envelope mass",
    )
    ap.add_argument(
        "--edge-type-neighbors",
        action="store_true",
        help="E88: weight the additive edge-type channel by the mass of other neighbors",
    )
    ap.add_argument(
        "--outer-radial-gate",
        type=float,
        default=0.0,
        help="E95: multiply the radial channels whose centre lies beyond this radius (A) by the other-neighbour mass fraction of the edge (0 = off)",
    )
    ap.add_argument(
        "--seed-radial-gate",
        action="store_true",
        help="E89: gate the environment seed's radial projection by its type projection",
    )
    ap.add_argument(
        "--isolated-reference",
        type=str,
        default="",
        help="E133: JSON table {symbol: isolated-atom energy in eV}; the bias of every tabulated element is "
        "set to its value and the fitting output is measured from its own isolated-atom value, so that an "
        "atom without neighbours has exactly the reference energy at every step",
    )
    ap.add_argument(
        "--pair-fade",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("R_LO", "R_HI"),
        help="E132: multiply the radial basis and the type feature of an edge by g_ij + (1 - g_ij) s(r_ij), "
        "with s a quintic smoothstep from one at R_LO to zero at R_HI (A), so that a nearly isolated pair "
        "fades out continuously instead of at a channel boundary (R_HI <= R_LO = off)",
    )
    ap.add_argument(
        "--fitting-rmsnorm",
        action="store_true",
        help="E74: normalize the descriptor entering the energy head with parameter-free RMSNorm, epsilon 1e-5",
    )
    ap.add_argument(
        "--plain-fitting",
        action="store_true",
        help="use ordinary MLP hidden layers at the configured fitting widths and activation",
    )
    ap.add_argument(
        "--squared-reference-loss",
        action="store_true",
        help="use squared force differences over all trusted anchors, without a tolerance region",
    )
    ap.add_argument(
        "--original-reference",
        action="store_true",
        help="fine-tuning: retain the pretrained teacher's original attention and fitting equations while constraining the student",
    )
    ap.add_argument(
        "--anchor-source",
        choices=["periodic", "clusters"],
        default=None,
        help="E77: paired periodic or mixed finite-cluster sources for the four anchor transforms",
    )
    ap.add_argument(
        "--cluster-selection",
        choices=["random", "nearest"],
        default="random",
        help="select random atoms or the centre atom's nearest neighbours within finite-cluster anchor sources",
    )
    ap.add_argument(
        "--anchor-source-start",
        type=int,
        default=0,
        help="absolute optimizer step at which the selected anchor source becomes active",
    )
    ap.add_argument(
        "--pair-anchors",
        type=float,
        nargs=3,
        metavar=("FRACTION", "R_LO", "R_HI"),
        default=None,
        help="E152: write isolated pairs of the frame's own atoms into this fraction of every anchor batch, at separations drawn log-uniformly from [R_LO, R_HI] A, held to the reference's pair curve",
    )
    ap.add_argument(
        "--anchor-replay",
        type=str,
        default=None,
        help="fine-tuning: sample some anchor source frames from this training LMDB",
    )
    ap.add_argument(
        "--anchor-replay-fraction",
        type=float,
        default=0.5,
        help="fraction of anchor frames drawn from the replay LMDB; zero supplies the paired source control",
    )
    ap.add_argument(
        "--uma-envelope",
        action="store_true",
        help="E68: the cutoff envelope becomes UMA's DimeNet polynomial (C2 at the cutoff, three to ten times more amplitude over the outer third) instead of SeZM's C3 form",
    )
    ap.add_argument(
        "--fit-silu",
        action="store_true",
        help="E66: the fitting network's gated hidden layers become Linear -> SiLU at the same hidden width (sezm_fit_silu)",
    )
    ap.add_argument(
        "--hard-replay",
        type=int,
        default=0,
        help="E6: replay this many hard-contact frames from a per-rank buffer through the data loss every --hard-replay-every calls (0 = off)",
    )
    ap.add_argument(
        "--hard-replay-ratio",
        type=float,
        default=0.55,
        help="E6: a frame is hard when its closest pair is below this fraction of the covalent contact distance",
    )
    ap.add_argument(
        "--hard-replay-width",
        type=int,
        default=128,
        help="E6: frames are padded to this many atoms; larger frames are not kept",
    )
    ap.add_argument(
        "--hard-replay-size",
        type=int,
        default=512,
        help="E6: size of the per-rank buffer of hard frames",
    )
    ap.add_argument(
        "--drop-contradictory",
        type=float,
        default=0.0,
        help="E7 counterfactual: frames whose closest pair lies below this fraction of the covalent contact with both label forces small contribute no loss (0 = off)",
    )
    ap.add_argument(
        "--drop-contradictory-force",
        type=float,
        default=60.0,
        help="E7 counterfactual: the label-force bound of the suspect family, in eV/A",
    )
    ap.add_argument(
        "--muon-unwhitened",
        action="store_true",
        help="E24: every Muon matrix update becomes the unwhitened control with the orthogonalized update's Frobenius norm along the raw momentum direction",
    )
    ap.add_argument(
        "--pair-zone",
        type=str,
        default="",
        help="E29: zone bridging at the pair's own scale, LO,HI as fractions of the covalent contact (the frozen zone ends at LO, the transition at HI; the ZBL term is enveloped by the same switch); the input must carry bridging_method ZBL",
    )
    ap.add_argument(
        "--readout-norm",
        type=float,
        default=0.0,
        help="E30: apply the equivariant RMS norm with this eps to the read-out's input (0 = off)",
    )
    ap.add_argument(
        "--radial-norm",
        type=float,
        default=0.0,
        help="E38: scale the radial basis vector to this fixed norm (in 1/Å) before its cutoff envelope (0 = off)",
    )
    ap.add_argument(
        "--radial-norm-decay-from",
        type=float,
        default=0.0,
        help="E41: with --radial-norm, keep the fixed norm up to this distance (in Å) and let it fall as 1/r beyond (0 = fixed everywhere)",
    )
    ap.add_argument(
        "--block-post-add-norm",
        type=float,
        default=0.0,
        help="E45: normalize the residual stream after the SO(2) add of every block with the soft equivariant RMS norm of this eps (0 = off)",
    )
    ap.add_argument(
        "--so2-knee",
        type=float,
        default=0.0,
        help="E37: put the post-SO(2) soft norm's knee at this message RMS and clamp its scale there (0 = off)",
    )
    ap.add_argument(
        "--node-knee",
        type=float,
        default=0.0,
        help="E60: put the knee of every block's pre-SO(2) and pre-feed-forward node norm at this RMS and clamp their scales there (0 = off)",
    )
    ap.add_argument(
        "--const-degree",
        type=float,
        default=0.0,
        help="E64: replace the per-node 1/sqrt(degree) of the neighbour sum by this constant for every node (0 = off)",
    )
    ap.add_argument(
        "--node-anchor",
        type=float,
        default=0.0,
        help="E64: add a learnable l=0 anchor of this initial RMS after every block's pre-SO(2) and pre-feed-forward norm (0 = off)",
    )
    ap.add_argument(
        "--inner-floor",
        type=str,
        default="",
        help="E49: floor on the pair distance seen by the descriptor, as r_inner,r_outer in A (the package's C3 InnerClamp used alone; empty = off)",
    )
    ap.add_argument(
        "--readout-knee",
        type=float,
        default=0.0,
        help="E48: install the read-out norm with its knee at this RMS (eps = knee², scale initialised and clamped at the knee; overrides --readout-norm; 0 = off)",
    )
    ap.add_argument(
        "--gain-control",
        type=float,
        default=0.0,
        help="E32: bound the probe/data force response ratio at kappa = this value (0 = off)",
    )
    ap.add_argument(
        "--gain-share",
        type=float,
        default=0.3,
        help="E32: share of the data gradient's norm given to the penalty gradient",
    )
    ap.add_argument(
        "--gain-every",
        type=int,
        default=10,
        help="E32: apply the bound every this many training calls",
    )
    ap.add_argument(
        "--gain-probes",
        type=int,
        default=32,
        help="E32: number of compressed-pair probes per event",
    )
    ap.add_argument(
        "--gain-slice",
        type=int,
        default=16,
        help="E32: number of batch frames in the data slice",
    )
    ap.add_argument(
        "--gain-delta",
        type=float,
        default=0.115,
        help="E32: norm of the weight perturbation (one optimizer step)",
    )
    ap.add_argument(
        "--weight-sensitivity",
        type=float,
        default=0.0,
        help="E8: weight of the bound on the anchor/data ratio of the force response to a random relative weight perturbation (0 = off)",
    )
    ap.add_argument(
        "--weight-sensitivity-eps",
        type=float,
        default=0.01,
        help="E8: relative size of the weight perturbation",
    )
    ap.add_argument(
        "--weight-sensitivity-every",
        type=int,
        default=10,
        help="E8: apply the bound every this many training calls",
    )
    ap.add_argument(
        "--hard-replay-weight",
        type=float,
        default=1.0,
        help="E6: weight of the replay loss (the inverse of the replays per buffered frame makes it importance sampling)",
    )
    ap.add_argument(
        "--hard-replay-every",
        type=int,
        default=1,
        help="E6: replay every this many training calls",
    )
    ap.add_argument(
        "--stratified-replay",
        type=int,
        default=0,
        help="E23: append up to this many buffered hard frames to every batch, each hard frame trained with the stream weight on arrival and with (1 - stream weight) / count on each of its count replays (0 = off)",
    )
    ap.add_argument(
        "--stratified-stream-weight",
        type=float,
        default=0.1,
        help="E23: weight of a hard frame's natural occurrence",
    )
    ap.add_argument(
        "--stratified-count",
        type=int,
        default=10,
        help="E23: replays per buffered hard frame before it retires",
    )
    ap.add_argument(
        "--needle-geometry-above",
        type=float,
        default=100.0,
        help="with --log-batch-error: record exact contacts of every atom above this force error in eV/A, and the worst atom's nearest distance in the batch log",
    )
    ap.add_argument(
        "--tear-state-above",
        type=float,
        default=1000.0,
        help="diagnostic: with --log-batch-error, a batch whose largest per-atom force error exceeds this (eV/A) has the model saved at that forward pass to tear_rank<r>_step<n>.pt",
    )
    ap.add_argument(
        "--needle-dump-above",
        type=float,
        default=500.0,
        help="with --log-batch-error: write every batch whose largest per-atom force error exceeds this many eV/A "
        "to needle_rank<r>_step<n>.npz (coordinates, cell, types, labels, model outputs)",
    )
    ap.add_argument(
        "--type-substitution",
        type=float,
        nargs=2,
        default=None,
        metavar=("ATOMS", "FRAMES"),
        help="after the geometric transform of every anchor event, replace the type of this "
        "fraction of the real atoms in this fraction of the anchor frames by a type drawn "
        "uniformly from the type map (E99: holds the response at every type embedding to the "
        "reference's)",
    )
    ap.add_argument(
        "--compose-generators",
        action="store_true",
        help="apply every enabled generator in sequence to every anchor frame (deletion, shell, "
        "collision, dilation) instead of drawing one generator per frame",
    )
    ap.add_argument(
        "--dilate-after-refresh",
        type=float,
        default=None,
        help="after the first refresh, upper bound of the dilation factor (the package default is 2.0; "
        "1.0 removes the dilation generator, or the dilation stage of a composed anchor)",
    )
    ap.add_argument(
        "--pair-squeeze",
        type=str,
        default=None,
        help="after the drawn generator, close one pair per anchor frame (a random real atom and its nearest "
        "neighbour) to a distance drawn uniformly from 'lo,hi' in A (ledger E55)",
    )
    ap.add_argument(
        "--pair-squeeze-ratio",
        type=str,
        default=None,
        help="after the drawn generator, close one pair per anchor frame to a fraction of its covalent "
        "contact drawn uniformly from 'lo,hi' (E112: the same physical regime for every pair type)",
    )
    ap.add_argument(
        "--fstep-after-refresh",
        type=float,
        default=None,
        help="after the first refresh, largest displacement of the collision generator in A "
        "(the package default is 0.3; a pair closes by up to twice this distance)",
    )
    ap.add_argument(
        "--rel-after-refresh",
        type=float,
        default=None,
        help="after the first refresh, relative tolerance of the tube (the package default is 0.1)",
    )
    ap.add_argument(
        "--pin-sparse-after-refresh",
        action="store_true",
        help="after the first refresh, alternate anchor events between the sparse "
        "generators with a closed tube (the model pinned to the reference far from "
        "the data) and the dense generators with the ordinary tube",
    )
    args, dp_args = ap.parse_known_args()
    if args.squared_reference_loss and (
        args.label_tube_after_refresh
        or args.pin_sparse_after_refresh
        or args.rel_after_refresh is not None
    ):
        ap.error(
            "--squared-reference-loss cannot combine with tolerance-modifying reference options"
        )
    if args.plain_fitting and args.fitting_rmsnorm:
        ap.error(
            "--plain-fitting uses the portable standard MLP format and cannot combine with --fitting-rmsnorm"
        )
    if args.cluster_selection != "random" and args.anchor_source != "clusters":
        ap.error("--cluster-selection requires --anchor-source clusters")
    if args.anchor_source_start < 0 or (
        args.anchor_source_start and args.anchor_source is None
    ):
        ap.error(
            "--anchor-source-start requires --anchor-source and a non-negative step"
        )
    if not 0.0 <= args.anchor_replay_fraction <= 1.0:
        ap.error("--anchor-replay-fraction must be in [0, 1]")
    if args.anchor_replay is not None and args.anchor_source is not None:
        ap.error(
            "--anchor-replay and --anchor-source are separate source-distribution experiments"
        )
    if args.original_reference:
        if (
            args.refresh != 0
            or args.single_envelope
            or args.gauss_inner
            or args.gauss_scales
            or not any(
                flag in dp_args for flag in ("--init-model", "--finetune", "--restart")
            )
        ):
            ap.error(
                "--original-reference requires pretrained initialization or restart, no refresh, and the original radial equation"
            )
        from original_reference import install as install_original_reference

        install_original_reference()
    if args.squared_reference_loss:
        from reference_loss import install as install_reference_loss

        install_reference_loss()
    # ``--refresh 0`` leaves the package untouched: a restarted run whose
    # checkpoint already carries the handed-over reference keeps it.
    if args.refresh > 0:
        install(
            args.refresh,
            args.max_refreshes,
            args.dense_after_refresh,
            args.collision_after_refresh,
            args.pin_sparse_after_refresh,
            args.near_after_refresh,
            args.label_tube_after_refresh,
            args.rel_after_refresh,
            args.compose_generators,
            args.dilate_after_refresh,
            args.label_tube_near_only,
            args.pair_term,
            args.pair_term_tol,
            args.pair_term_rel,
            args.anchor_descent,
            args.anchor_descent_step,
            args.ensemble_refs,
            args.pair_term_anchors,
            refresh_steps={int(s) for s in args.refresh_steps.split(",")}
            if args.refresh_steps
            else None,
            repulsion_weight=args.repulsion_hinge,
            repulsion_ratio=args.repulsion_ratio,
            anchor_draws=args.anchor_draws,
            pair_contact_ratio=args.pair_term_contact_ratio,
            pair_contact_tol=args.pair_term_contact_tol,
            pair_contact_rel=args.pair_term_contact_rel,
            fstep_after_refresh=args.fstep_after_refresh,
            type_substitution=tuple(args.type_substitution)
            if args.type_substitution
            else None,
            pair_squeeze=tuple(float(v) for v in args.pair_squeeze.split(","))
            if args.pair_squeeze
            else None,
            pair_squeeze_ratio=tuple(
                float(v) for v in args.pair_squeeze_ratio.split(",")
            )
            if args.pair_squeeze_ratio
            else None,
            dimer_weight=args.dimer_hinge,
            dimer_ratio=tuple(float(v) for v in args.dimer_ratio.split(",")),
            dimer_count=args.dimer_count,
            dimer_contacts=args.dimer_contacts,
            data_sign_weight=args.data_sign_hinge,
            data_sign_margin=args.data_sign_margin,
            data_sign_count=args.data_sign_count,
            tail_weight=args.tail_penalty,
            tail_range=tuple(float(v) for v in args.tail_range.split(",")),
            tail_count=args.tail_count,
        )
    if args.anchor_source is not None:
        from cluster_anchors import install as install_cluster_anchors

        install_cluster_anchors(
            args.anchor_source,
            float(
                json.loads(Path(args.input).read_text())["model"]["descriptor"]["rcut"]
            ),
            args.cluster_selection,
            args.anchor_source_start,
        )
    if args.pair_anchors is not None:
        from pair_anchors import install as install_pair_anchors

        install_pair_anchors(
            float(
                json.loads(Path(args.input).read_text())["model"]["descriptor"]["rcut"]
            ),
            *args.pair_anchors,
        )
    if args.anchor_replay is not None:
        from replay_anchors import install as install_replay_anchors

        install_replay_anchors(
            args.anchor_replay,
            json.loads(Path(args.input).read_text())["model"]["type_map"],
            args.anchor_replay_fraction,
        )
    if args.hard_replay > 0:
        install_hard_replay(
            args.hard_replay,
            args.hard_replay_ratio,
            args.hard_replay_width,
            args.hard_replay_size,
            args.hard_replay_every,
            args.hard_replay_weight,
        )
    if args.stratified_replay > 0:
        install_stratified_replay(
            args.stratified_replay,
            args.hard_replay_ratio,
            args.hard_replay_width,
            args.hard_replay_size,
            args.stratified_count,
            args.stratified_stream_weight,
        )
    if args.weight_sensitivity > 0.0:
        install_weight_sensitivity(
            args.weight_sensitivity,
            args.weight_sensitivity_eps,
            args.weight_sensitivity_every,
        )
    if args.gain_control > 0.0:
        install_gain_control(
            args.gain_control,
            args.gain_share,
            args.gain_every,
            args.gain_probes,
            args.gain_slice,
            args.gain_delta,
        )
    if args.drop_contradictory > 0.0:
        install_contradiction_filter(
            args.drop_contradictory, args.drop_contradictory_force
        )
    if args.muon_unwhitened:
        install_muon_unwhitened()
    if json.loads(Path(args.input).read_text())["loss"].get("type") == "dens":
        install_dens_inactive_head_freeze()
    if args.log_batch_error and args.backend == "pt-expt":
        install_batch_error_log_dpmodel()
    elif args.log_batch_error:
        install_batch_error_log(
            args.needle_dump_above,
            geometry_above=args.needle_geometry_above,
            tear_above=args.tear_state_above,
        )
    if args.mixing_loops > 1 or args.block_loops > 1:
        from sezm_loops import install as install_loops

        install_loops(args.mixing_loops, args.block_loops)
    if args.stay_or_step or args.post_add_norm:
        from sezm_attnres import install as install_attnres

        install_attnres(post_add_norm=args.post_add_norm)
    attnres_variant = (
        args.attnres_keys != "l0"
        or args.attnres_weights != "shared"
        or args.attnres_value_norm
    )
    if attnres_variant:
        from sezm_attnres import (
            install_variant,
        )

        install_variant(
            args.attnres_keys, args.attnres_weights, args.attnres_value_norm
        )
    if args.agg_logit_bound > 0.0:
        from sezm_aggregation import install as install_logit_bound

        install_logit_bound(args.agg_logit_bound)
    pair_zone = (
        tuple(float(v) for v in args.pair_zone.split(",")) if args.pair_zone else ()
    )
    if args.readout_norm > 0.0 or args.readout_knee > 0.0:
        import sezm_readout_norm

        sezm_readout_norm.install(args.readout_norm, args.readout_knee)
    if args.radial_norm > 0.0:
        from sezm_radial_norm import install as install_radial_norm

        install_radial_norm(args.radial_norm, args.radial_norm_decay_from)
    if args.block_post_add_norm > 0.0:
        import sezm_postadd_norm

        sezm_postadd_norm.install(args.block_post_add_norm)
    inner_floor = (
        tuple(float(v) for v in args.inner_floor.split(",")) if args.inner_floor else ()
    )
    if inner_floor:
        import sezm_inner_floor

        sezm_inner_floor.install(inner_floor[0], inner_floor[1])
    if args.so2_knee > 0.0:
        import sezm_so2_knee

        sezm_so2_knee.install(args.so2_knee)
    if args.node_knee > 0.0:
        import sezm_node_knee

        sezm_node_knee.install(args.node_knee)
    if args.const_degree > 0.0 or args.node_anchor > 0.0:
        import sezm_esen_count

        sezm_esen_count.install(args.const_degree, args.node_anchor)
    if args.fit_silu:
        import sezm_fit_silu

        sezm_fit_silu.install()
    # The depth-attention residual's forward reads its batch dimension as a Python integer,
    # which torch.compile cannot guard under dynamic shapes; the shape-agnostic replacement is
    # numerically identical and is installed for every run.
    import sezm_attnres_dynamic

    sezm_attnres_dynamic.install()
    if args.uma_envelope:
        import sezm_uma_envelope

        sezm_uma_envelope.install()
    from radial_experiment import install as install_radial_experiment

    install_radial_experiment(
        gauss_inner=args.gauss_inner,
        single_envelope=args.single_envelope,
        fixed_radial=args.fixed_radial,
        gauss_width=args.gauss_width,
        gauss_scales=None if args.gauss_scales is None else tuple(args.gauss_scales),
        gauss_floor=args.gauss_floor,
    )
    from adam_route import install as install_adam_route

    install_adam_route([p for p in args.adam_route.split(",") if p])
    from message_envelope import install as install_message_envelope

    install_message_envelope(args.output_envelope)
    from attention_normalization import install as install_attention_normalization

    install_attention_normalization(args.separate_attention_mass)
    from edge_type_neighbors import install as install_edge_type_neighbors

    install_edge_type_neighbors(args.edge_type_neighbors)
    from seed_radial_gate import install as install_seed_radial_gate

    install_seed_radial_gate(args.seed_radial_gate)
    from outer_radial_gate import install as install_outer_radial_gate

    install_outer_radial_gate(args.outer_radial_gate)
    from pair_fade import install as install_pair_fade

    install_pair_fade(*args.pair_fade)
    from isolated_reference import install as install_isolated_reference

    install_isolated_reference(args.isolated_reference or None)
    from fitting_normalization import install as install_fitting_normalization

    install_fitting_normalization(args.fitting_rmsnorm)
    from plain_fitting import install as install_plain_fitting

    install_plain_fitting(args.plain_fitting)
    if args.so2_knee > 0.0 or args.readout_knee > 0.0 or args.node_knee > 0.0:
        # The knee scales are projected at the clipping seam, once per step, before the optimizer step.
        from deepmd.pt.train import training as training_module

        original_clip_for_knee = training_module.clip_grad_norm_

        def clip_and_clamp_knee(*clip_args, **clip_kwargs):  # noqa: ANN002, ANN003, ANN202
            if args.so2_knee > 0.0:
                sezm_so2_knee.clamp_scales(args.so2_knee)
            if args.readout_knee > 0.0:
                sezm_readout_norm.clamp_scales(args.readout_knee)
            if args.node_knee > 0.0:
                sezm_node_knee.clamp_scales(args.node_knee)
            return original_clip_for_knee(*clip_args, **clip_kwargs)

        training_module.clip_grad_norm_ = clip_and_clamp_knee
    if pair_zone:
        from sezm_zone import install as install_zone

        install_zone(
            pair_zone[0],
            pair_zone[1],
            json.loads(Path(args.input).read_text())["model"]["type_map"],
        )
    if (
        args.mixing_loops > 1
        or args.block_loops > 1
        or args.stay_or_step
        or args.post_add_norm
        or attnres_variant
        or args.agg_logit_bound > 0.0
        or pair_zone
        or args.readout_norm > 0.0
        or args.readout_knee > 0.0
        or args.radial_norm > 0.0
        or args.so2_knee > 0.0
        or args.node_knee > 0.0
        or args.const_degree > 0.0
        or args.node_anchor > 0.0
        or args.block_post_add_norm > 0.0
        or inner_floor
        or args.uma_envelope
        or args.single_envelope
        or args.gauss_inner > 0.0
        or args.gauss_width > 0.0
        or args.gauss_floor > 0.0
        or bool(args.adam_route)
        or args.gauss_scales is not None
        or args.fixed_radial
        or args.output_envelope
        or args.separate_attention_mass
        or args.edge_type_neighbors
        or args.seed_radial_gate
        or args.outer_radial_gate > 0.0
        or args.pair_fade[1] > args.pair_fade[0]
        or bool(args.isolated_reference)
        or args.fitting_rmsnorm
        or args.plain_fitting
        or args.squared_reference_loss
    ):
        # Experiment settings record the forward equations and training objective.
        # Evaluators restore the forward patches from this record (diagnose/repro_spike.py).
        Path("patches.json").write_text(
            json.dumps(
                {
                    "mixing_loops": args.mixing_loops,
                    "block_loops": args.block_loops,
                    "stay_or_step": bool(args.stay_or_step),
                    "post_add_norm": bool(args.post_add_norm),
                    "attnres_keys": args.attnres_keys,
                    "attnres_weights": args.attnres_weights,
                    "attnres_value_norm": bool(args.attnres_value_norm),
                    "agg_logit_bound": float(args.agg_logit_bound),
                    "pair_zone": list(pair_zone),
                    "readout_norm": float(args.readout_norm),
                    "readout_knee": float(args.readout_knee),
                    "radial_norm": float(args.radial_norm),
                    "radial_norm_decay_from": float(args.radial_norm_decay_from),
                    "so2_knee": float(args.so2_knee),
                    "node_knee": float(args.node_knee),
                    "const_degree": float(args.const_degree),
                    "node_anchor": float(args.node_anchor),
                    "block_post_add_norm": float(args.block_post_add_norm),
                    "inner_floor": list(inner_floor),
                    "uma_envelope": bool(args.uma_envelope),
                    "single_envelope": bool(args.single_envelope),
                    "gauss_inner": float(args.gauss_inner),
                    "gauss_width": float(args.gauss_width),
                    "gauss_floor": float(args.gauss_floor),
                    "adam_route": [p for p in args.adam_route.split(",") if p],
                    "gauss_scales": args.gauss_scales,
                    "fixed_radial": bool(args.fixed_radial),
                    "output_envelope": bool(args.output_envelope),
                    "separate_attention_mass": bool(args.separate_attention_mass),
                    "edge_type_neighbors": bool(args.edge_type_neighbors),
                    "seed_radial_gate": bool(args.seed_radial_gate),
                    "outer_radial_gate": float(args.outer_radial_gate),
                    "pair_fade": [float(args.pair_fade[0]), float(args.pair_fade[1])],
                    "isolated_reference": str(Path(args.isolated_reference).resolve())
                    if args.isolated_reference
                    else "",
                    "fitting_rmsnorm": bool(args.fitting_rmsnorm),
                    "plain_fitting": bool(args.plain_fitting),
                    "squared_reference_loss": bool(args.squared_reference_loss),
                    "pair_zone_type_map": json.loads(Path(args.input).read_text())[
                        "model"
                    ]["type_map"]
                    if pair_zone
                    else [],
                }
            )
        )
    # The backend entry point re-parses ``sys.argv``, so the command line is
    # rewritten to what ``dp --pt train`` expects.
    sys.argv = [sys.argv[0], "--" + args.backend, "train", args.input, *dp_args]
    from deepmd.main import main as dp_main

    dp_main()


if __name__ == "__main__":
    main()

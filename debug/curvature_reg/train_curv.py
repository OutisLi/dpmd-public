# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, T201, ANN001, ANN002, ANN003, ANN201, ANN202
"""
Train with a label-free curvature regularizer added by monkey patch.

The regularizer is grafted onto ``EnergyStdLoss.forward`` at run time so that
the installed package stays untouched; every option is a command-line flag of
this launcher, and the training input is an ordinary DeePMD-kit input file.

Rationale
---------
Energy and force labels pin the value and the gradient of the fitted surface
at each training configuration and leave its second derivative free, which is
the degree of freedom in which a high-capacity model develops sharp spurious
features between configurations.  Those features are detected without labels.
With ``u = F/|F|`` the unit force direction over all ``3N`` coordinates, the
directional curvature ``c = u^T H u`` of the model energy is read either from
the second-order Taylor remainder of one extra energy evaluation,

    c = 2 [E(x + eps u) - E(x) + eps |F|] / eps^2,

or per atom from the force difference ``g = [F(x) - F(x + eps u)] / eps``,
which approximates ``H u``.  A real potential cannot be arbitrarily stiff:
away from repulsive walls its curvature is bounded by the stiffest bond force
constant ``k0``, and on a wall of decay length ``rho0`` a force ``F`` implies a
curvature of about ``F / rho0``.  The ceiling ``kappa = k0 + F_max / rho0``
therefore separates physical stiffness from spurious sharpness, and the
penalty is the squared logarithmic excess ``relu(log(|c| / kappa))^2``: zero on
every physical configuration, and growing gently on defects so that one badly
bent frame cannot dominate the gradient.

The anchor is the training configuration itself, or a Gaussian perturbation
of it when ``--sigma > 0`` so that the surface between configurations is
probed as well.  Only the curvature is differentiated: the probe direction and
the ceiling are detached.
"""

from __future__ import (
    annotations,
)

import argparse
import math
import sys
from typing import (
    ClassVar,
)

import torch


class CurvConfig:
    """Regularizer settings shared with the patched loss."""

    pref = 0.0
    freq = 5
    eps = 0.01
    sigma = 0.0
    k0 = 300.0
    rho0 = 0.1
    estimator = "energy"
    sigma_list: ClassVar[list[float]] = []
    strain = 0.0
    dilate = 0.0
    dilate_min = 0.0
    drop = 0.0
    drop_min = 0.0
    squeeze = 0.0
    fcap = 0.0
    anchor_skips = 0
    fstep = 0.0
    edge_norm_keep = ""
    edge_norm_neutralized = False
    ref_ckpt = ""
    ref_alternate = False
    ref_step = 0
    ref_refresh = 0
    ref_refresh_max = 0
    dense_after_refresh = False
    squeeze_after_refresh = 0.0
    tol_after_refresh = None
    rel_after_refresh = None
    pref_after_refresh = None
    families_after_refresh = False
    near_after_refresh = False
    collision_after_refresh = False
    fstep_after_refresh = None
    dilate_after_refresh = None
    drop_after_refresh = None
    families: ClassVar[list[dict]] = []
    ref_refreshes = 0
    ref_calls = 0
    ref_pref = 0.0
    ref_tol = 1.0
    ref_rel = 0.1
    ref_fscale = 0.0
    ref_model = None
    last_ref_penalty = None
    buffer_size = 0
    buffer_frac = 0.5
    ascend_steps = 0
    ascend_eta = 0.05
    ascend_always = False
    buffer: ClassVar[list] = []
    force_anchor_forward = False
    last_x = None
    last_g = None
    last_ratio = None
    probe_frames = 0
    probe_max_atoms = 0
    sequential = False
    probe_early = False
    md_steps = 0
    md_temp = 1000.0
    md_dt = 0.5
    md_gamma = 0.01
    spec_cap = 0.0
    spec_every = 10
    spec_calls = 0
    spec_state: dict | None = None
    spec_frac = torch.nan
    calls = 0
    stats: ClassVar[dict[str, torch.Tensor]] = {}


# Velocity unit conversion: sqrt(eV/amu) expressed in Angstrom/fs.
_VEL_UNIT = 0.09822694788
_KB = 8.617333262e-5  # eV/K


def squeeze_anchor(x, atype, box, factor):
    """
    Push one random atom of every frame toward its nearest neighbour.

    The atom is moved along the line to its nearest neighbour so that their
    distance becomes ``f`` times the current one, with ``f`` drawn uniformly
    from ``[factor, 1]``.  This samples the short-contact extrapolation region
    of every element pair present in the batch, where a surface with no data
    roughens with training (design note §3), without labels or search.
    Phantom atoms (``atype < 0``) are excluded from the choice of atoms and
    of neighbours.

    Parameters
    ----------
    x : torch.Tensor
        Coordinates with shape (nf, nloc, 3).
    atype : torch.Tensor
        Atom types with shape (nf, nloc); negative entries are phantoms.
    box : torch.Tensor
        Cells with shape (nf, 9) or (nf, 3, 3), possibly on the CPU.
    factor : float
        Lower bound of the compression factor.

    Returns
    -------
    torch.Tensor
        Displaced coordinates with the shape of ``x``.
    """
    nf, nloc, _ = x.shape
    cell = box.reshape(nf, 3, 3).to(x.dtype).to(x.device)
    real = atype >= 0
    inv = torch.linalg.inv(cell)
    d = (
        x[:, None, :, :] - x[:, :, None, :]
    )  # (nf, nloc, nloc, 3), d[b, i, j] = x_j - x_i
    frac = torch.einsum("bijk,bkl->bijl", d, inv)
    frac = frac - torch.round(frac)
    d = torch.einsum("bijk,bkl->bijl", frac, cell)
    r = d.norm(dim=-1)  # (nf, nloc, nloc)
    big = torch.finfo(r.dtype).max
    r = r.masked_fill(~real[:, None, :], big).masked_fill(~real[:, :, None], big)
    r = r + torch.eye(nloc, device=x.device, dtype=r.dtype) * big
    # One random real atom per frame; its nearest real neighbour.
    scores = torch.rand(nf, nloc, device=x.device).masked_fill(~real, -1.0)
    a = scores.argmax(dim=1)  # (nf,)
    ar = torch.arange(nf, device=x.device)
    rj = r[ar, a]  # (nf, nloc)
    b = rj.argmin(dim=1)
    vec = d[ar, a, b]  # x_b - x_a, minimum image, (nf, 3)
    f = factor + (1.0 - factor) * torch.rand(nf, 1, device=x.device, dtype=x.dtype)
    x = x.clone()
    x[ar, a] = x[ar, a] + vec * (1.0 - f)
    return x


def md_anchor(input_dict, model):
    """
    Displace the batch along its own dynamics to generate anchors.

    The model's own Langevin dynamics is the sampler that reaches the narrow,
    directional defects that random perturbations miss (§9 of the design
    note): a short segment of it is run from every frame of the batch, with
    the parameters frozen and the model in evaluation mode, and the end
    configuration is returned as the anchor.  Phantom atoms carry zero force
    and a unit mass, so they stay where they are.

    Parameters
    ----------
    input_dict : dict[str, torch.Tensor]
        Model inputs of the current batch.
    model : torch.nn.Module
        The model, in training mode on entry and on exit.

    Returns
    -------
    torch.Tensor
        Anchor coordinates with the shape of ``input_dict["coord"]``.
    """
    from ase.data import (
        atomic_masses,
        atomic_numbers,
    )

    cfg = CurvConfig
    coord = input_dict["coord"]
    atype = input_dict["atype"]
    nf = atype.shape[0]
    x = coord.reshape(nf, -1, 3).detach().clone()
    type_map = model.get_type_map()
    mass_table = torch.tensor(
        [atomic_masses[atomic_numbers[s]] for s in type_map],
        dtype=x.dtype,
        device=x.device,
    )
    mass = torch.where(
        atype >= 0, mass_table[atype.clamp_min(0)], torch.ones_like(x[..., 0])
    ).unsqueeze(-1)
    v = torch.randn_like(x) * torch.sqrt(_KB * cfg.md_temp / mass) * _VEL_UNIT
    v = v * (atype >= 0).unsqueeze(-1)
    dt = cfg.md_dt
    c1 = math.exp(-cfg.md_gamma * dt)
    c2 = torch.sqrt((1.0 - c1**2) * _KB * cfg.md_temp / mass) * _VEL_UNIT
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.requires_grad_(False)
    model.eval()
    try:

        def acc(pos):
            f = model(**{**input_dict, "coord": pos.reshape(coord.shape)})[
                "force"
            ].reshape(nf, -1, 3)
            return f.detach() / mass * _VEL_UNIT**2

        a = acc(x)
        for _ in range(cfg.md_steps):
            v = v + 0.5 * dt * a
            x = x + dt * v
            a = acc(x)
            v = v + 0.5 * dt * a
            v = c1 * v + c2 * torch.randn_like(v)
    finally:
        model.train()
        for p in params:
            p.requires_grad_(True)
    return x.reshape(coord.shape)


def curvature_penalty(input_dict, model, model_pred):
    """
    Penalize non-physical curvature of the predicted energy surface.

    Parameters
    ----------
    input_dict : dict[str, torch.Tensor]
        Model inputs of the current batch.
    model : torch.nn.Module
        The model, in training mode.
    model_pred : dict[str, torch.Tensor]
        Predictions at the training configurations.

    Returns
    -------
    tuple[torch.Tensor, dict[str, torch.Tensor]]
        The penalty averaged over frames, and display statistics: the largest
        logarithmic excess and the fraction of probed frames (or atoms) above
        the ceiling.
    """
    cfg = CurvConfig
    if cfg.probe_frames > 0 and cfg.probe_frames < model_pred["energy"].shape[0]:
        # Probe a random subset of the batch's frames: the cost of the term
        # scales with the number of probed frames, not with the batch.
        nf_all = model_pred["energy"].shape[0]
        sub = torch.randperm(nf_all, device=model_pred["energy"].device)[
            : cfg.probe_frames
        ]
        sub_cpu = sub.cpu()
        input_dict = {
            k: (
                v[sub_cpu if v.device.type == "cpu" else sub]
                if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == nf_all
                else v
            )
            for k, v in input_dict.items()
        }
        model_pred = {
            k: (
                v[sub]
                if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == nf_all
                else v
            )
            for k, v in model_pred.items()
        }
    coord = input_dict["coord"]
    nf = model_pred["energy"].shape[0]
    x = coord.reshape(nf, -1, 3)
    if cfg.drop > 0.0:
        # Sparse-environment anchors: a random fraction of the real atoms, in
        # [drop_min, drop], is turned into phantoms, which the model ignores.
        # The remaining atoms see fewer neighbours than any training frame
        # shows, the direction of extrapolation in which the isolated dimer
        # fails; a positive lower bound keeps every anchor away from the data.
        atype = input_dict["atype"]
        real = atype >= 0
        frac = cfg.drop_min + (cfg.drop - cfg.drop_min) * torch.rand(
            nf, 1, device=atype.device
        )
        dropped = real & (torch.rand(atype.shape, device=atype.device) < frac)
        input_dict = {**input_dict, "atype": atype.masked_fill(dropped, -1)}
    if cfg.squeeze > 0.0 and input_dict.get("box") is not None:
        x = squeeze_anchor(x, input_dict["atype"], input_dict["box"], cfg.squeeze)
    if cfg.md_steps > 0:
        x = md_anchor(input_dict, model).reshape(nf, -1, 3)
    if cfg.sigma_list:
        # Multi-scale shell: every frame draws its own perturbation amplitude,
        # so one probed batch samples several distances from the data.
        amp = torch.tensor(cfg.sigma_list, dtype=x.dtype, device=x.device)[
            torch.randint(len(cfg.sigma_list), (nf, 1, 1), device=x.device)
        ]
        x = x + amp * torch.randn_like(x)
    elif cfg.sigma > 0.0:
        x = x + cfg.sigma * torch.randn_like(x)
    if cfg.fstep > 0.0:
        # Virtual collision: every atom is pushed *against* the force it feels
        # by a random distance up to fstep, which is how thermal motion
        # reaches the compressed configurations where dynamics fails.
        f_dir = model_pred["force"].reshape(nf, -1, 3).detach()
        f_dir = f_dir / f_dir.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        step = cfg.fstep * torch.rand(nf, x.shape[1], 1, dtype=x.dtype, device=x.device)
        x = x - step * f_dir
    box = input_dict.get("box")
    rescale = (cfg.strain > 0.0 or cfg.dilate > 0.0) and box is not None
    if rescale:
        # Isotropic rescaling by a random factor in [1 - strain, 1 + dilate]:
        # compression is the density extrapolation that Case A's dynamics
        # performed, dilation the low-coordination one in which the isolated
        # dimer fails.  ``dilate_min`` raises the lower end to 1 + dilate_min
        # (dilation only), so that no anchor stays within reach of the data.
        lo = 1.0 - cfg.strain + cfg.dilate_min
        s_ = lo + (1.0 + cfg.dilate - lo) * torch.rand(
            nf, 1, 1, dtype=x.dtype, device=x.device
        )
        x = x * s_
        box = box.reshape(nf, -1) * s_.reshape(nf, 1).to(box.device)
        box = box.reshape(input_dict["box"].shape)
    if (
        cfg.force_anchor_forward
        or cfg.md_steps > 0
        or cfg.sigma > 0.0
        or cfg.sigma_list
        or cfg.squeeze > 0.0
        or cfg.fstep > 0.0
        or cfg.drop > 0.0
        or rescale
    ):
        anchor = model(**{**input_dict, "coord": x.reshape(coord.shape), "box": box})
    else:
        anchor = model_pred
    cfg.last_x = x.detach()
    eps = cfg.eps
    if cfg.ref_model is not None and cfg.ref_pref > 0.0:
        anchor_box = box if rescale else input_dict.get("box")
        fmax_frame = (
            model_pred["force"].reshape(nf, -1, 3).norm(dim=-1).amax(dim=1).detach()
        )
        cfg.last_ref_penalty = cfg.ref_pref * reference_penalty(
            input_dict, x, anchor_box, anchor, fmax_frame
        )
    else:
        cfg.last_ref_penalty = None
    f0 = anchor["force"].reshape(nf, -1, 3)
    fnorm = f0.reshape(nf, -1).norm(dim=-1)  # (nf,)
    u = (f0 / fnorm.clamp_min(1e-12).reshape(nf, 1, 1)).detach()
    probe = model(
        **{
            **input_dict,
            "coord": (x + eps * u).reshape(coord.shape),
            "box": box if rescale else input_dict.get("box"),
        }
    )
    if cfg.estimator == "energy":
        # Per-atom differencing avoids cancellation between two large total
        # energies; the sum over atoms is the total energy change.
        de = (probe["atom_energy"] - anchor["atom_energy"]).reshape(nf, -1).sum(-1)
        curv = 2.0 * (de + eps * fnorm) / eps**2  # (nf,)
        fmax = f0.norm(dim=-1).amax(dim=-1).detach()  # (nf,)
        if cfg.fcap > 0.0:
            fmax = fmax.clamp_max(cfg.fcap)
        ratio = curv.abs() / (cfg.k0 + fmax / cfg.rho0)
        excess = torch.relu(torch.log(ratio.clamp_min(1e-12)))
        penalty = excess.square().mean()
    else:
        g = (f0 - probe["force"].reshape(nf, -1, 3)) / eps  # (nf, nloc, 3) ~ H u
        fat = f0.norm(dim=-1).detach()
        if cfg.fcap > 0.0:
            # A physical force never exceeds fcap in any region a stable
            # simulation may visit, so the ceiling must not grow with a
            # runaway prediction: beyond fcap the allowed curvature is fixed
            # and a larger predicted force only raises the ratio.
            fat = fat.clamp_max(cfg.fcap)
        ratio = g.norm(dim=-1) / (cfg.k0 + fat / cfg.rho0)
        excess = torch.relu(torch.log(ratio.clamp_min(1e-12)))
        penalty = excess.square().sum() / nf
        cfg.last_g = g.detach()
    cfg.last_ratio = ratio.detach()
    stats = {
        "curv_excess": excess.detach().max(),
        "curv_frac": (ratio.detach() > 1.0).to(excess.dtype).mean(),
    }
    if cfg.last_ref_penalty is not None:
        # The curvature term is weighted by ``pref`` outside; undo that weight
        # for the reference term so that ``ref_pref`` is its own coefficient.
        penalty = penalty + cfg.last_ref_penalty / max(cfg.pref, 1e-30)
    return penalty, stats


def reference_penalty(input_dict, anchor_x, box, anchor_pred, fmax_frame):
    """
    Hinge on the force deviation from a frozen reference model at the anchor.

    A region without data roughens because nothing holds the function there
    while the parameters move to fit data elsewhere.  Holding the off-manifold
    forces within ``ref_tol`` of a smooth reference (an early checkpoint or a
    slow average) forbids that drift without asking for accuracy: the reference
    may be wrong, it is only required to be smooth.

    Parameters
    ----------
    input_dict : dict[str, torch.Tensor]
        Model inputs of the (sub)batch.
    anchor_x : torch.Tensor
        Anchor coordinates with shape (nf, nloc, 3).
    box : torch.Tensor or None
        Cells used for the anchor.
    anchor_pred : dict[str, torch.Tensor]
        The model's prediction at the anchor, carrying the parameter graph.
    fmax_frame : torch.Tensor
        Largest force on the training frame each anchor was generated from,
        shape (nf,): the force scale of the chemistry around the anchor, which
        sets the frame-dependent part of the tolerance (``ref_fscale``).

    Returns
    -------
    torch.Tensor
        Mean over atoms of ``relu(|F - F_ref| - tol - rel |F_ref|)^2``.
    """
    cfg = CurvConfig
    out = cfg.ref_model(
        **{
            **input_dict,
            "coord": anchor_x.reshape(input_dict["coord"].shape),
            "box": box,
        }
    )
    f_ref = out["force"].detach().reshape(anchor_x.shape[0], -1, 3)
    f = anchor_pred["force"].reshape(anchor_x.shape[0], -1, 3)
    # The tolerance has an absolute and a relative part: on a steep wall the
    # reference force is large and its mixed-precision evaluation differs from
    # the training-mode one by a fixed fraction, which is not a deviation.
    # Atoms on which the reference itself predicts an unphysical force (above
    # ``fcap``) are outside the region where it is a reference, and are skipped.
    dev = (f - f_ref).norm(dim=-1)
    fn = f_ref.norm(dim=-1)
    trusted = (
        (fn <= cfg.fcap) if cfg.fcap > 0.0 else torch.ones_like(fn, dtype=torch.bool)
    )
    tol = (
        cfg.ref_tol
        + cfg.ref_rel * fn
        + cfg.ref_fscale * fmax_frame.to(fn.dtype).reshape(-1, 1)
    )
    excess = torch.relu(dev - tol).square() * trusted
    return excess.sum() / trusted.sum().clamp_min(1)


def adaptive_probe(sub_in, model, sub_pred):
    """
    Probe one frame with a replay buffer of flagged anchors and curvature ascent.

    The plateau of §9.2 comes from anchors landing on a defect only in passing.
    Two devices raise the pressure on what has been found: an anchor whose
    ratio exceeded the ceiling is kept in a buffer and probed again on later
    steps until it falls below the ceiling; and from a flagged anchor a few
    steps are taken along ``g = H u``, the direction in which the force grows
    fastest, so that the probe climbs toward the core of the defect instead of
    staying on its flank.  Everything is still generated from training frames.

    Returns
    -------
    tuple[torch.Tensor, dict[str, torch.Tensor]]
        The summed (unweighted) penalty of the probes made for this frame and
        the display statistics of the last probe.
    """
    cfg = CurvConfig
    from_buffer = bool(cfg.buffer) and torch.rand(()) < cfg.buffer_frac
    if from_buffer:
        entry = cfg.buffer[int(torch.randint(len(cfg.buffer), ()))]
        sub_in = entry["input"]
        sub_pred = entry["pred"]
        fixed_x = entry["x"]
    else:
        fixed_x = None
    penalty, stats, x_used, ratio = curvature_penalty_x(
        sub_in, model, sub_pred, fixed_x
    )
    total = penalty
    worst_x, worst_ratio = x_used, float(ratio.max())
    for _ in range(cfg.ascend_steps if (worst_ratio > 1.0 or cfg.ascend_always) else 0):
        g = cfg.last_g
        step = cfg.ascend_eta * g / g.norm().clamp_min(1e-12)
        x_new = (worst_x + step).detach()
        penalty2, stats2, x_used2, ratio2 = curvature_penalty_x(
            sub_in, model, sub_pred, x_new
        )
        total = total + penalty2
        stats = stats2
        if float(ratio2.max()) > worst_ratio:
            worst_x, worst_ratio = x_used2, float(ratio2.max())
    if cfg.buffer_size > 0:
        if from_buffer:
            if worst_ratio <= 1.0:
                # Remove by identity: entries hold tensors, so equality tests are undefined.
                cfg.buffer = [e for e in cfg.buffer if e is not entry]
            else:
                entry["x"], entry["ratio"] = worst_x.detach(), worst_ratio
        elif worst_ratio > 1.0:
            keep = {
                "input": {
                    k: (v.detach() if torch.is_tensor(v) else v)
                    for k, v in sub_in.items()
                },
                "pred": {
                    k: (v.detach() if torch.is_tensor(v) else v)
                    for k, v in sub_pred.items()
                },
                "x": worst_x.detach(),
                "ratio": worst_ratio,
            }
            if len(cfg.buffer) < cfg.buffer_size:
                cfg.buffer.append(keep)
            else:
                j = min(range(len(cfg.buffer)), key=lambda i: cfg.buffer[i]["ratio"])
                if cfg.buffer[j]["ratio"] < worst_ratio:
                    cfg.buffer[j] = keep
    stats["curv_buf"] = torch.tensor(float(len(cfg.buffer)))
    return total, stats


def curvature_penalty_x(input_dict, model, model_pred, fixed_x):
    """``curvature_penalty`` variant that can start from given anchor coordinates and returns them."""
    cfg = CurvConfig
    saved = (
        cfg.sigma,
        cfg.sigma_list,
        cfg.strain,
        cfg.squeeze,
        cfg.fstep,
        cfg.md_steps,
    )
    if fixed_x is not None:
        cfg.sigma, cfg.sigma_list, cfg.strain, cfg.squeeze, cfg.fstep, cfg.md_steps = (
            0.0,
            [],
            0.0,
            0.0,
            0.0,
            0,
        )
        input_dict = {**input_dict, "coord": fixed_x.reshape(input_dict["coord"].shape)}
        # A fixed anchor is off the data, so its prediction must be recomputed.
        cfg.force_anchor_forward = True
    try:
        penalty, stats = curvature_penalty(input_dict, model, model_pred)
    finally:
        cfg.sigma, cfg.sigma_list, cfg.strain, cfg.squeeze, cfg.fstep, cfg.md_steps = (
            saved
        )
        cfg.force_anchor_forward = False
    return penalty, stats, cfg.last_x, cfg.last_ratio


def detached_forward(model, sub_in):
    """Predictions of the model at ``sub_in`` with no graph retained (forces still need autograd to be built)."""
    with torch.enable_grad():
        pred = model(**sub_in)
    return {k: (v.detach() if torch.is_tensor(v) else v) for k, v in pred.items()}


def sequential_penalty(input_dict, model, model_pred=None):
    """
    Probe the selected frames one at a time and back-propagate each penalty at
    once, so that a single anchor/probe graph is alive at any time; this caps
    the memory of the term at one frame's worth regardless of batch size.

    Parameters
    ----------
    input_dict : dict[str, torch.Tensor]
        Model inputs of the current batch.
    model : torch.nn.Module
        The model, in training mode.
    model_pred : dict[str, torch.Tensor] | None
        Predictions at the training configurations.  ``None`` means the term
        is evaluated *before* the training forward: the predictions of each
        probed frame are then recomputed without a graph, and the anchor is
        always re-evaluated, so that the probe graph and the batch graph never
        coexist and the peak memory is the larger of the two instead of their
        sum.

    Returns
    -------
    tuple[torch.Tensor, dict[str, torch.Tensor]]
        The detached total penalty (already weighted) and the display
        statistics aggregated over the probed frames.
    """
    cfg = CurvConfig
    nf_all = input_dict["coord"].shape[0]
    device = input_dict["coord"].device
    k = cfg.probe_frames if 0 < cfg.probe_frames < nf_all else nf_all
    sel = torch.randperm(nf_all)[:k].tolist()
    if cfg.probe_max_atoms > 0:
        # Large frames are skipped: their anchor and probe graphs would not fit
        # next to the training graph of the batch.
        nreal = (input_dict["atype"] >= 0).sum(dim=1)
        sel = [i for i in sel if int(nreal[i]) <= cfg.probe_max_atoms]
        if not sel:
            return torch.zeros((), device=device), {
                "curv_excess": torch.tensor(0.0),
                "curv_frac": torch.tensor(0.0),
            }
        k = len(sel)
    total = torch.zeros((), device=device)
    excess_max, frac_sum = 0.0, 0.0
    for i in sel:
        sub_in = {
            kk: (
                v[i : i + 1]
                if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == nf_all
                else v
            )
            for kk, v in input_dict.items()
        }
        if model_pred is None:
            sub_pred = detached_forward(model, sub_in)
        else:
            sub_pred = {
                kk: (
                    v[i : i + 1]
                    if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == nf_all
                    else v
                )
                for kk, v in model_pred.items()
            }
        saved = (cfg.probe_frames, cfg.force_anchor_forward)
        cfg.probe_frames = 0
        cfg.force_anchor_forward = cfg.force_anchor_forward or model_pred is None
        try:
            if cfg.buffer_size > 0 or cfg.ascend_steps > 0:
                pen, st = adaptive_probe(sub_in, model, sub_pred)
            else:
                pen, st = curvature_penalty(sub_in, model, sub_pred)
        finally:
            cfg.probe_frames, cfg.force_anchor_forward = saved
        (cfg.pref * pen / k).backward()
        total = total + (cfg.pref * pen / k).detach()
        excess_max = max(excess_max, float(st["curv_excess"]))
        frac_sum += float(st["curv_frac"])
        del pen
    # A pathological anchor draw can poison the accumulated gradients with
    # non-finite values (observed as sudden NaN divergence at extreme peak
    # rates). In the probe-early layout the buffers hold only this event's
    # anchor gradients here, so dropping them discards exactly the poisoned
    # event; the data step of the same iteration proceeds unchanged.
    if any(
        p.grad is not None and not torch.isfinite(p.grad).all()
        for p in model.parameters()
    ):
        for p in model.parameters():
            p.grad = None
        cfg.anchor_skips += 1
        print(
            f"[curv] non-finite anchor gradient at call {cfg.calls}: "
            f"anchor event discarded ({cfg.anchor_skips} total)"
        )
    if not torch.isfinite(total).all():
        total = torch.zeros_like(total)
    stats = {
        "curv_excess": torch.tensor(excess_max),
        "curv_frac": torch.tensor(frac_sum / k),
    }
    if cfg.buffer_size > 0:
        stats["curv_buf"] = torch.tensor(float(len(cfg.buffer)))
    return total, stats


def neutralize_edge_norm_sites(model, keep: str) -> int:
    """
    Remove the edge_norm-gated norms of every site except ``keep``.

    ``edge_norm = False`` does not construct the radial hidden RMSNorms, the
    cross-focus competition norms or the FiLM scale/shift norms (and sets the
    SO(2) post-norm eps to 1, which is irrelevant under the pre-norm
    sandwich).  This helper reproduces exactly that structure per site on a
    model built with ``edge_norm = True``: the modules of every site not in
    ``keep`` are replaced by ``nn.Identity`` — the same forward the False
    branch produces — so a run with one site kept isolates that site under
    the true configuration semantics.  Returns the number of removed modules.
    """
    import torch.nn as nn

    kept = {v.strip() for v in keep.split(",") if v.strip()}
    modules = dict(model.named_modules())
    n = 0
    for name, m in modules.items():
        cls = type(m).__name__
        site = None
        if ".radial_embedding." in name and cls == "RMSNorm":
            site = "radial"
        elif name.endswith("focus_compete_norm") and cls == "ScalarRMSNorm":
            site = "focus"
        elif (
            name.endswith(("film_scale_norm", "film_shift_norm"))
            and cls == "ScalarRMSNorm"
        ):
            site = "film"
        if site is None or site in kept:
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = modules[parent_name]
        if isinstance(parent, nn.Sequential):
            parent[int(attr)] = nn.Identity()
        else:
            setattr(parent, attr, nn.Identity())
        n += 1
    return n


def spectral_projection(model, cfg) -> torch.Tensor:
    """
    Clip the spectral norm of every weight matrix to a multiple of its initial value.

    The curvature of the composed network is bounded by the product of the
    per-layer operator norms, and along the Pro checkpoint sequence those norms
    grow monotonically (gates and SO(2) mixing weights by 1.5-2x, the radial
    embedding by 2x) while the parameter norm does not.  This projection is a
    hard constraint ``sigma_max(W) <= spec_cap * sigma_max(W_0)`` applied to
    every parameter with at least two dimensions (viewed as ``(d0, -1)``),
    enforced by rescaling the tensor whenever a warm-started power iteration
    finds it above the cap.  It touches no anchors and costs two matrix-vector
    products per tensor.

    Parameters
    ----------
    model : torch.nn.Module
        The model being trained; its parameters are modified in place.
    cfg : type[CurvConfig]
        Holds ``spec_cap`` and the per-tensor state (cap, power-iteration vector).

    Returns
    -------
    torch.Tensor
        Fraction of constrained tensors that were rescaled in this call.
    """
    with torch.no_grad():
        if cfg.spec_state is None:
            cfg.spec_state = {}
            for name, p in model.named_parameters():
                if p.ndim < 2 or p.numel() < 64:
                    continue
                m = p.reshape(p.shape[0], -1)
                sigma0 = torch.linalg.matrix_norm(m.float(), ord=2)
                u = torch.randn(m.shape[0], device=p.device, dtype=torch.float32)
                cfg.spec_state[name] = (cfg.spec_cap * sigma0, u / u.norm())
        clipped = 0
        params = dict(model.named_parameters())
        for name, (cap, u) in cfg.spec_state.items():
            p = params[name]
            m = p.reshape(p.shape[0], -1).float()
            for _ in range(2):
                v = m.t() @ u
                v = v / (v.norm() + 1e-30)
                u = m @ v
                sigma = u.norm()
                u = u / (sigma + 1e-30)
            cfg.spec_state[name] = (cap, u)
            if sigma > cap:
                p.mul_((cap / sigma).to(p.dtype))
                clipped += 1
        return torch.tensor(clipped / max(len(cfg.spec_state), 1))


def install_final_norm() -> None:
    """
    Graft an equivariant RMS norm onto the descriptor read-out (architecture variant).

    The read-out of ``DescrptSeZM`` hands the raw residual stream's scalar slice,
    plus its residual FFN, to the fitting network; nothing bounds its magnitude,
    so a stream inflated by the multiplicative units off the data reaches the
    energy unattenuated. This variant normalizes the read-out — the final norm a
    language model places in front of its unembedding — on a quantity that never
    vanishes (it carries the node's own embedding), so the Jacobian stays bounded.
    """
    from deepmd.pt.model.descriptor.sezm import (
        DescrptSeZM,
    )
    from deepmd.pt.model.descriptor.sezm_nn.norm import (
        EquivariantRMSNorm,
    )

    orig_init = DescrptSeZM.__init__
    orig_readout = DescrptSeZM._apply_readout

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.final_norm = EquivariantRMSNorm(
            lmax=0,
            channels=self.channels,
            n_focus=1,
            eps=1e-5,
            dtype=self.compute_dtype,
            trainable=True,
        )

    def _apply_readout(self, x, n_rows):
        return self.final_norm(orig_readout(self, x, n_rows))

    DescrptSeZM.__init__ = __init__
    DescrptSeZM._apply_readout = _apply_readout


def install_patch() -> None:
    """Graft the regularizer onto ``EnergyStdLoss.forward``."""
    import deepmd.pt.loss.ener as ener_mod

    orig = ener_mod.EnergyStdLoss.forward

    def forward(self, input_dict, model, label, natoms, learning_rate, mae=False):
        cfg = CurvConfig
        active = model.training and not self.inference
        if cfg.edge_norm_keep and not cfg.edge_norm_neutralized and active:
            n_ = neutralize_edge_norm_sites(model, cfg.edge_norm_keep)
            cfg.edge_norm_neutralized = True
            print(
                f"[curv] edge-norm sites removed except {cfg.edge_norm_keep!r}: {n_} modules"
            )
        # The operator-norm constraint is applied before the forward pass, which
        # is equivalent to projecting after the previous optimizer step.
        if cfg.spec_cap > 0.0 and active:
            cfg.spec_calls += 1
            if (cfg.spec_calls - 1) % cfg.spec_every == 0:
                cfg.spec_frac = spectral_projection(model, cfg)
        if cfg.ref_pref > 0.0 and active:
            # ``--ref-ckpt init``: the reference is the function the run starts
            # from — the pretrained model of a fine-tune, or the smooth random
            # initialization of a pretraining — frozen at step ``ref_step``
            # (the first step by default); the term is inactive before that.
            # With ``--ref-refresh N`` the reference is replaced by a fresh
            # copy of the model every N steps after the freeze, so the term
            # holds the function near its own recent past instead of its
            # starting point.
            cfg.ref_calls += 1
            step_index = cfg.ref_calls - 1
            freeze = cfg.ref_model is None and step_index == cfg.ref_step
            refresh = (
                cfg.ref_refresh > 0
                and cfg.ref_model is not None
                and step_index > cfg.ref_step
                and (step_index - cfg.ref_step) % cfg.ref_refresh == 0
                and (
                    cfg.ref_refresh_max == 0 or cfg.ref_refreshes < cfg.ref_refresh_max
                )
            )
            if freeze or refresh:
                import copy

                cfg.ref_model = copy.deepcopy(model).eval()
                for p_ in cfg.ref_model.parameters():
                    p_.requires_grad_(False)
                if refresh:
                    cfg.ref_refreshes += 1
                    print(
                        f"reference refreshed at step {step_index} (refresh {cfg.ref_refreshes})",
                        flush=True,
                    )
                    if cfg.dense_after_refresh:
                        # A reference that carries the data's physics admits the
                        # dense generators of the fine-tune recipe next to the
                        # sparse ones: the run continues as a fine-tune of itself.
                        cfg.sigma_list = [0.1, 0.2]
                        cfg.fstep = 0.3
                        cfg.dilate, cfg.dilate_min = 1.0, 0.0
                        cfg.drop, cfg.drop_min = 0.9, 0.0
                        print(
                            "generators switched to the dense and sparse set",
                            flush=True,
                        )
                    if cfg.near_after_refresh:
                        # Only the near generators from here on: the Gaussian
                        # shell and the collision, which compress the pairs of
                        # the data frames into the wall; no dilation, no deletion.
                        cfg.sigma_list = [0.1, 0.2]
                        cfg.fstep = 0.3
                        cfg.dilate, cfg.dilate_min = 0.0, 0.0
                        cfg.drop, cfg.drop_min = 0.0, 0.0
                        print(
                            "generators switched to the near set (shell, collision)",
                            flush=True,
                        )
                    if cfg.collision_after_refresh:
                        # The sparse generators plus the collision, without the
                        # Gaussian shell that sits on every data frame.
                        cfg.fstep = 0.3
                        print(
                            "collision generator switched on next to the sparse set",
                            flush=True,
                        )
                    if cfg.fstep_after_refresh is not None:
                        # The collision amplitude after the refresh, applied after
                        # the generator set so that it overrides the set's own
                        # value; 0 removes the collision generator from the set.
                        cfg.fstep = cfg.fstep_after_refresh
                        print(
                            f"collision step after the refresh set to {cfg.fstep} A",
                            flush=True,
                        )
                    if cfg.dilate_after_refresh is not None:
                        # The dilation range after the refresh (upper bound of
                        # the scale factor minus one); 0 removes the dilation.
                        cfg.dilate = cfg.dilate_after_refresh
                        print(
                            f"dilation after the refresh set to {cfg.dilate}",
                            flush=True,
                        )
                    if cfg.drop_after_refresh is not None:
                        # The deletion range after the refresh (upper bound of
                        # the deleted fraction); 0 removes the deletion.
                        cfg.drop = cfg.drop_after_refresh
                        print(
                            f"deletion after the refresh set to {cfg.drop}", flush=True
                        )
                    if cfg.squeeze_after_refresh > 0.0:
                        # Pair compression reaches the short-range region the
                        # other generators leave to extrapolation; it is only
                        # admissible once the reference carries a wall there.
                        cfg.squeeze = cfg.squeeze_after_refresh
                        print(
                            f"pair compression switched on (fraction {cfg.squeeze})",
                            flush=True,
                        )
                    if (
                        cfg.tol_after_refresh is not None
                        or cfg.rel_after_refresh is not None
                    ):
                        # A narrower tube after the hand-over keeps the hinge
                        # active on the anchors, so that the weights shaping the
                        # far-from-data function stay pinned as they were under
                        # the random reference, now to a function that carries
                        # the learnt physics.
                        if cfg.tol_after_refresh is not None:
                            cfg.ref_tol = cfg.tol_after_refresh
                        if cfg.rel_after_refresh is not None:
                            cfg.ref_rel = cfg.rel_after_refresh
                        print(
                            f"reference tube switched to tol {cfg.ref_tol}, rel {cfg.ref_rel}",
                            flush=True,
                        )
                    if cfg.pref_after_refresh is not None:
                        cfg.ref_pref = cfg.pref_after_refresh
                        print(
                            f"reference weight switched to {cfg.ref_pref}", flush=True
                        )
                    if cfg.families_after_refresh:
                        # Two anchor families alternate from here on: the far
                        # family keeps the run's sparse generators with the
                        # tube closed, so that the model is pinned to the
                        # reference where the data do not reach; the near
                        # family (Gaussian shell, collision) keeps the tube,
                        # inside which the data keep shaping the model.
                        far = {
                            "sigma_list": list(cfg.sigma_list),
                            "fstep": cfg.fstep,
                            "dilate": cfg.dilate,
                            "dilate_min": cfg.dilate_min,
                            "drop": cfg.drop,
                            "drop_min": cfg.drop_min,
                            "ref_tol": 0.0,
                            "ref_rel": 0.0,
                        }
                        near = {
                            "sigma_list": [0.1, 0.2],
                            "fstep": 0.3,
                            "dilate": 0.0,
                            "dilate_min": 0.0,
                            "drop": 0.0,
                            "drop_min": 0.0,
                            "ref_tol": cfg.ref_tol,
                            "ref_rel": cfg.ref_rel,
                        }
                        cfg.families = [far, near]
                        print(
                            "anchor families alternate: far (sparse, closed tube) / near (shell, collision, tube)",
                            flush=True,
                        )
        due, penalty = False, None
        if cfg.pref != 0.0 and active:
            cfg.calls += 1
            due = (cfg.calls - 1) % cfg.freq == 0
            if due and cfg.families:
                for key, value in cfg.families[
                    ((cfg.calls - 1) // cfg.freq) % 2
                ].items():
                    setattr(cfg, key, value)
            if due and cfg.probe_early:
                # The probes run before the batch forward, so the two graphs
                # never coexist; the gradients are accumulated immediately and
                # only the detached value is added to the loss below.
                penalty, cfg.stats = sequential_penalty(input_dict, model, None)
        model_pred, loss, more_loss = orig(
            self, input_dict, model, label, natoms, learning_rate, mae
        )
        if cfg.spec_cap > 0.0:
            more_loss["spec_frac"] = cfg.spec_frac if model.training else torch.nan
        if cfg.pref == 0.0 or self.inference:
            return model_pred, loss, more_loss
        if model.training:
            if due and penalty is None:
                if cfg.sequential:
                    penalty, cfg.stats = sequential_penalty(
                        input_dict, model, model_pred
                    )  # detached, gradients accumulated
                else:
                    raw, cfg.stats = curvature_penalty(input_dict, model, model_pred)
                    penalty = cfg.pref * raw
            if penalty is not None:
                if cfg.ref_alternate and due:
                    # Alternation: a due step trains on the anchors alone — the
                    # step's loss is the reference term (whose gradients are
                    # already accumulated by the probe-early path) and the data
                    # loss contributes nothing, exactly a training step whose
                    # batch and labels come from the generator and the frozen
                    # reference instead of the dataset.
                    loss = loss * 0.0 + penalty.to(loss.dtype)
                else:
                    loss = loss + penalty.to(loss.dtype)
            keys = (
                ("curv_excess", "curv_frac", "curv_buf")
                if cfg.buffer_size > 0
                else ("curv_excess", "curv_frac")
            )
            for key in keys:
                more_loss[key] = cfg.stats.get(key, torch.nan)
        else:
            more_loss["curv_excess"] = torch.nan
            more_loss["curv_frac"] = torch.nan
            if cfg.buffer_size > 0:
                more_loss["curv_buf"] = torch.nan
        more_loss["rmse"] = torch.sqrt(loss.detach())
        return model_pred, loss, more_loss

    ener_mod.EnergyStdLoss.forward = forward


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Unrecognized arguments are passed through to `dp --pt train`.",
    )
    ap.add_argument("input", help="training input file")
    ap.add_argument(
        "--pref", type=float, default=0.0, help="regularizer weight; 0 disables"
    )
    ap.add_argument(
        "--freq", type=int, default=5, help="apply every this many training steps"
    )
    ap.add_argument(
        "--eps",
        type=float,
        default=0.01,
        help="probe displacement in Angstrom (3N norm)",
    )
    ap.add_argument(
        "--sigma",
        type=float,
        default=0.0,
        help="Gaussian anchor perturbation per coordinate in Angstrom",
    )
    ap.add_argument(
        "--k0",
        type=float,
        default=300.0,
        help="curvature ceiling at zero force, eV/Angstrom^2",
    )
    ap.add_argument(
        "--rho0", type=float, default=0.1, help="wall decay length, Angstrom"
    )
    ap.add_argument("--estimator", choices=["energy", "force"], default="energy")
    ap.add_argument(
        "--strain",
        type=float,
        default=0.0,
        help="anchors are additionally compressed isotropically by a random factor in [1 - strain, 1]",
    )
    ap.add_argument(
        "--dilate",
        type=float,
        default=0.0,
        help="anchors are additionally expanded isotropically by a random factor in [1, 1 + dilate] (low-coordination anchors)",
    )
    ap.add_argument(
        "--dilate-min",
        type=float,
        default=0.0,
        help="lower end of the dilation range: factors in [1 + dilate_min, 1 + dilate]",
    )
    ap.add_argument(
        "--drop",
        type=float,
        default=0.0,
        help="a random fraction, up to this, of each anchor frame's atoms is removed (turned into phantoms)",
    )
    ap.add_argument(
        "--drop-min",
        type=float,
        default=0.0,
        help="lower end of the removed fraction: fractions in [drop_min, drop]",
    )
    ap.add_argument(
        "--sigma-list",
        type=str,
        default="",
        help="comma-separated Gaussian amplitudes in Angstrom; each anchor frame draws one at random (overrides --sigma)",
    )
    ap.add_argument(
        "--ref-ckpt",
        type=str,
        default="",
        help="checkpoint of a frozen smooth reference model, or 'init' for the model's own initial state; off-manifold forces are held within --ref-tol of it",
    )
    ap.add_argument(
        "--edge-norm-keep",
        default="",
        help="with edge_norm=True in the model: comma-separated sites (radial, focus, film) whose gated norms stay active; every other site is neutralized (eps -> 1)",
    )
    ap.add_argument(
        "--ref-alternate",
        action="store_true",
        help="a due step trains on the anchors alone (whole-batch anchors, data loss zeroed) instead of adding the reference term to the data loss",
    )
    ap.add_argument(
        "--ref-step",
        type=int,
        default=0,
        help="with --ref-ckpt init: freeze the reference at this training step instead of the first one; the reference term is inactive before it",
    )
    ap.add_argument(
        "--ref-refresh",
        type=int,
        default=0,
        help="with --ref-ckpt init: replace the reference by a fresh copy of the model every this many steps after the freeze (0 keeps the initial reference)",
    )
    ap.add_argument(
        "--ref-refresh-max",
        type=int,
        default=0,
        help="with --ref-refresh: at most this many refreshes; 1 replaces the initial reference once and keeps that copy for the rest of the run (0 = unlimited)",
    )
    ap.add_argument(
        "--dense-after-refresh",
        action="store_true",
        help="after the first refresh, switch the anchor generators to the dense and sparse set of the fine-tune recipe (0.1/0.2 A shells, 0.3 A collisions, dilation to 2x, deletion to 90%%)",
    )
    ap.add_argument(
        "--squeeze-after-refresh",
        type=float,
        default=0.0,
        help="after the first refresh, switch on pair compression (--squeeze) with this lower fraction",
    )
    ap.add_argument(
        "--tol-after-refresh",
        type=float,
        default=None,
        help="after the first refresh, absolute tolerance of the reference tube (eV/A); 0 pins the anchors to the reference",
    )
    ap.add_argument(
        "--rel-after-refresh",
        type=float,
        default=None,
        help="after the first refresh, relative tolerance of the reference tube",
    )
    ap.add_argument(
        "--pref-after-refresh",
        type=float,
        default=None,
        help="after the first refresh, weight of the reference term",
    )
    ap.add_argument(
        "--near-after-refresh",
        action="store_true",
        help="after the first refresh, use only the near generators (Gaussian shell, collision)",
    )
    ap.add_argument(
        "--collision-after-refresh",
        action="store_true",
        help="after the first refresh, add the collision generator (0.3 A against the force) to the sparse set",
    )
    ap.add_argument(
        "--fstep-after-refresh",
        type=float,
        default=None,
        help="after the first refresh, collision amplitude in A, overriding the generator set's value (0 removes the collision generator)",
    )
    ap.add_argument(
        "--dilate-after-refresh",
        type=float,
        default=None,
        help="after the first refresh, upper bound of the dilation (scale factor minus one), overriding the generator set's value (0 removes the dilation)",
    )
    ap.add_argument(
        "--drop-after-refresh",
        type=float,
        default=None,
        help="after the first refresh, upper bound of the deleted fraction, overriding the generator set's value (0 removes the deletion)",
    )
    ap.add_argument(
        "--families-after-refresh",
        action="store_true",
        help="after the first refresh, alternate anchor events between the run's sparse "
        "generators with a closed tube and the dense generators (shell, collision) with the tube",
    )
    ap.add_argument(
        "--ref-pref",
        type=float,
        default=0.0,
        help="weight of the reference-consistency hinge (0 disables)",
    )
    ap.add_argument(
        "--ref-tol",
        type=float,
        default=1.0,
        help="tolerated force deviation from the reference at the anchors, eV/Angstrom",
    )
    ap.add_argument(
        "--ref-rel",
        type=float,
        default=0.1,
        help="additional tolerance as a fraction of the reference force (mixed-precision slack on steep walls)",
    )
    ap.add_argument(
        "--ref-fscale",
        type=float,
        default=0.0,
        help="additional tolerance as a fraction of the largest force on the training frame the anchor came from (the local force scale of the data)",
    )
    ap.add_argument(
        "--fstep",
        type=float,
        default=0.0,
        help="push every atom against its force by a random distance up to this, in Angstrom (virtual collision anchors)",
    )
    ap.add_argument(
        "--fcap",
        type=float,
        default=0.0,
        help="cap on the force entering the ceiling, in eV/Angstrom (0 = uncapped)",
    )
    ap.add_argument(
        "--squeeze",
        type=float,
        default=0.0,
        help="if > 0, one random atom per anchor frame is pushed toward its nearest neighbour to a fraction drawn from [squeeze, 1] of their distance",
    )
    ap.add_argument(
        "--probe-frames",
        type=int,
        default=0,
        help="probe only this many random frames of each probed batch (0 = all)",
    )
    ap.add_argument(
        "--buffer-size",
        type=int,
        default=0,
        help="replay buffer of flagged anchors (0 disables); with --sequential",
    )
    ap.add_argument(
        "--buffer-frac",
        type=float,
        default=0.5,
        help="probability that a probed frame is drawn from the buffer",
    )
    ap.add_argument(
        "--ascend-steps",
        type=int,
        default=0,
        help="curvature-ascent steps from a flagged anchor along g = H u",
    )
    ap.add_argument(
        "--ascend-eta",
        type=float,
        default=0.05,
        help="ascent step length in Angstrom (3N norm)",
    )
    ap.add_argument(
        "--ascend-always",
        action="store_true",
        help="climb from every anchor, not only from flagged ones (a search for defects the generators miss)",
    )
    ap.add_argument(
        "--probe-max-atoms",
        type=int,
        default=0,
        help="with --sequential, skip probing frames with more real atoms than this (0 = no limit)",
    )
    ap.add_argument(
        "--sequential",
        action="store_true",
        help="probe frames one at a time with immediate backward, capping the term's memory at one frame",
    )
    ap.add_argument(
        "--probe-early",
        action="store_true",
        help="with --sequential, probe before the batch forward so the probe and batch graphs never coexist (peak memory = max, not sum)",
    )
    ap.add_argument(
        "--md-steps",
        type=int,
        default=0,
        help="Langevin steps run from each frame with the frozen model to generate anchors; 0 disables",
    )
    ap.add_argument(
        "--md-temp", type=float, default=1000.0, help="anchor dynamics temperature in K"
    )
    ap.add_argument(
        "--md-dt", type=float, default=0.5, help="anchor dynamics time step in fs"
    )
    ap.add_argument(
        "--md-gamma", type=float, default=0.01, help="Langevin friction in 1/fs"
    )
    ap.add_argument(
        "--spec-cap",
        type=float,
        default=0.0,
        help="clip every weight matrix's spectral norm to this multiple of its initial value (0 disables); independent of --pref",
    )
    ap.add_argument(
        "--spec-every",
        type=int,
        default=10,
        help="apply the spectral clip every this many training steps",
    )
    ap.add_argument(
        "--final-norm",
        action="store_true",
        help="architecture variant: equivariant RMS norm on the descriptor read-out (new parameters; from-scratch training only)",
    )
    ap.add_argument(
        "--muon-ridge",
        type=float,
        default=0.0,
        help="optimizer variant: ridge tau (in units of the root-mean-square singular value of the update) on HybridMuon's orthogonalization; singular directions below tau are damped instead of normalized (0 = plain Muon)",
    )
    ap.add_argument(
        "--muon-spectral",
        action="store_true",
        help="optimizer control: replace HybridMuon's orthogonalization by the momentum scaled by its largest singular value (Muon's maximal step, no whitening)",
    )
    ap.add_argument(
        "--adam-eps",
        type=float,
        default=None,
        help="optimizer variant: denominator floor of HybridMuon's Adam-routed parameters (module default 1e-20)",
    )
    args, dp_args = ap.parse_known_args()
    for key in (
        "pref",
        "freq",
        "eps",
        "sigma",
        "strain",
        "dilate",
        "dilate_min",
        "drop",
        "drop_min",
        "squeeze",
        "fcap",
        "fstep",
        "probe_frames",
        "probe_max_atoms",
        "sequential",
        "buffer_size",
        "buffer_frac",
        "ascend_steps",
        "ascend_eta",
        "ascend_always",
        "k0",
        "rho0",
        "estimator",
        "md_steps",
        "md_temp",
        "md_dt",
        "md_gamma",
        "spec_cap",
        "spec_every",
        "probe_early",
    ):
        setattr(CurvConfig, key, getattr(args, key))
    CurvConfig.sigma_list = [float(v) for v in args.sigma_list.split(",") if v.strip()]
    if (
        args.freq < 1
        or args.eps <= 0.0
        or args.rho0 <= 0.0
        or not math.isfinite(args.pref)
    ):
        raise ValueError("freq must be >= 1 and eps, rho0 must be positive.")
    install_patch()
    if args.final_norm:
        install_final_norm()
    if args.muon_ridge > 0.0 or args.muon_spectral or args.adam_eps is not None:
        from muon_ridge import install as install_muon_ridge

        install_muon_ridge(args.muon_ridge, args.adam_eps, args.muon_spectral)
    for key in (
        "edge_norm_keep",
        "ref_ckpt",
        "ref_alternate",
        "ref_step",
        "ref_refresh",
        "ref_refresh_max",
        "dense_after_refresh",
        "squeeze_after_refresh",
        "tol_after_refresh",
        "rel_after_refresh",
        "pref_after_refresh",
        "families_after_refresh",
        "near_after_refresh",
        "collision_after_refresh",
        "fstep_after_refresh",
        "dilate_after_refresh",
        "drop_after_refresh",
        "ref_pref",
        "ref_tol",
        "ref_rel",
        "ref_fscale",
    ):
        setattr(CurvConfig, key, getattr(args, key))
    if args.ref_ckpt and args.ref_ckpt != "init" and args.ref_pref > 0.0:
        from pathlib import Path as _Path

        sys.path.insert(0, str(_Path(__file__).resolve().parent / "diagnose"))
        from repro_spike import load_model as _load_model

        # The reference stays in evaluation mode with frozen parameters.
        ref, _ = _load_model(_Path(args.ref_ckpt), "cuda")
        for p_ in ref.parameters():
            p_.requires_grad_(False)
        CurvConfig.ref_model = ref
    from deepmd.main import main as dp_main

    # The backend entry point re-parses ``sys.argv``, so the command line is
    # rewritten to the plain ``dp`` form before handing over.
    sys.argv = [sys.argv[0], "--pt", "train", args.input, *dp_args]
    dp_main()


if __name__ == "__main__":
    main()

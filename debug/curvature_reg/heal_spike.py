# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, T201, ANN001, ANN201, ANN202
"""
Whether the curvature penalty can remove a known spurious peak.

The hydrogen model carries a +285 eV peak on the 49-48 bond of frame s17097.
This experiment fine-tunes that model with the label-free curvature penalty
anchored on the seven archived pathological frames, while the predictions on
healthy frames are held to their original values by self-distillation from a
frozen copy.  It isolates one question -- whether the hinge gradient flattens
a spike without disturbing the healthy surface -- from the separate question
of whether training-time anchors would have reached the spike at all.

Before and after fine-tuning it reports the largest force and the curvature
ratio on the pathological frames, the largest force a hill climb finds near
s17096, and the radial energy profile of atom 49 along the 49-48 bond.
"""

from __future__ import (
    annotations,
)

import argparse
import copy
import sys
from pathlib import (
    Path,
)

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "diagnose"))
from repro_spike import (
    load_model,
    read_xyz,
)
from spike_density import (
    adversarial_search,
    random_search,
)
from traj_curvature import (
    read_dump,
)

ROOT = Path("/nas/outisli/Software/deepmd-kit/temp/normal")
BAD = [f"last_normal_s{s}" for s in range(17091, 17097)] + ["onset_1_s17097"]
CONTROL = ["control_s5000", "control_s12000"]


def batch(model, pos, cell, atype, device):
    """Batched forward; returns energy (nf,), atom energies (nf, N), forces (nf, N, 3)."""
    nf = pos.shape[0]
    c = torch.as_tensor(pos, dtype=torch.float32, device=device)
    b = (
        torch.as_tensor(cell, dtype=torch.float32, device=device)
        .expand(nf, 3, 3)
        .contiguous()
    )
    t = (
        torch.as_tensor(atype, dtype=torch.long, device=device)
        .expand(nf, -1)
        .contiguous()
    )
    out = model(c, t, box=b)
    return (
        out["energy"].reshape(nf),
        out["atom_energy"].reshape(nf, -1),
        out["force"].reshape(nf, -1, 3),
    )


def penalty_energy(model, pos, cell, atype, eps, k0, rho0, device):
    """Log-hinge curvature penalty on one frame; returns (penalty, ratio, F_max)."""
    pos_t = torch.as_tensor(pos, dtype=torch.float32, device=device).reshape(1, -1, 3)
    e0, ae0, f0 = batch(model, pos_t, cell, atype, device)
    fnorm = f0.reshape(-1).norm()
    u = (f0 / fnorm.clamp_min(1e-12)).detach()
    _, aep, _ = batch(model, pos_t + eps * u, cell, atype, device)
    curv = 2.0 * ((aep - ae0).sum() + eps * fnorm) / eps**2
    fmax = f0.norm(dim=-1).max().detach()
    ratio = curv.abs() / (k0 + fmax / rho0)
    excess = torch.relu(torch.log(ratio.clamp_min(1e-12)))
    return excess.square(), float(ratio), float(fmax)


def penalty_force(model, pos, cell, atype, eps, k0, rho0, device):
    """Per-atom log-hinge on the force difference; returns (penalty, max ratio, F_max)."""
    pos_t = torch.as_tensor(pos, dtype=torch.float32, device=device).reshape(1, -1, 3)
    _, _, f0 = batch(model, pos_t, cell, atype, device)
    fnorm = f0.reshape(-1).norm()
    u = (f0 / fnorm.clamp_min(1e-12)).detach()
    _, _, fp = batch(model, pos_t + eps * u, cell, atype, device)
    g = (f0 - fp) / eps
    ratio = g.norm(dim=-1) / (k0 + f0.norm(dim=-1).detach() / rho0)
    excess = torch.relu(torch.log(ratio.clamp_min(1e-12)))
    return excess.square().sum(), float(ratio.max()), float(f0.norm(dim=-1).max())


def penalty_stats(model, frames, cell, atype, eps, k0, rho0, device):
    """Detached per-frame (penalty, ratio, F_max) over a list of frames."""
    rows = []
    for pos in frames:
        pen, ratio, fmax = penalty_energy(
            model, pos, cell, atype, eps, k0, rho0, device
        )
        rows.append((float(pen.detach()), ratio, fmax))
        del pen
    return np.array(rows)


def bond_scan(model, pos, cell, atype, i, j, device, rs):
    """Move atom ``i`` along the ``i->j`` bond; return (r, E, radial force on i)."""
    d = pos[j] - pos[i]
    d -= cell[0, 0] * np.round(d / cell[0, 0])
    r0 = np.linalg.norm(d)
    n = d / r0
    rows = []
    for r in rs:
        p = pos.copy()
        p[i] = pos[i] + n * (r0 - r)
        e, _, f = batch(model, p[None], cell, atype, device)
        rows.append((r, float(e[0]), float(-(f[0, i].detach().cpu().numpy() @ n))))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=Path("/nas/outisli/Software/deepmd-kit/temp/model_ema.ckpt.pt"),
    )
    ap.add_argument("--dump", type=Path, default=ROOT / "dump.traj")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--k0", type=float, default=300.0)
    ap.add_argument("--rho0", type=float, default=0.1)
    ap.add_argument(
        "--anchor-steps",
        type=int,
        nargs="*",
        default=None,
        help="dump steps used as anchors instead of the pathological frames",
    )
    ap.add_argument(
        "--sigma",
        type=float,
        default=0.0,
        help="Gaussian perturbation of the anchors per coordinate, in Angstrom",
    )
    ap.add_argument(
        "--estimator",
        choices=["energy", "force"],
        default="force",
        help="curvature estimator used for the training penalty",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--out", type=Path, default=Path(__file__).with_name("runs") / "heal"
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)

    model, type_map = load_model(args.ckpt, args.device)
    frozen = copy.deepcopy(model).eval()
    for p in frozen.parameters():
        p.requires_grad_(False)

    def frame(tag):
        pos, cell, species = read_xyz(ROOT / f"normal_{tag}.xyz")
        return pos, cell, np.array([type_map.index(s) for s in species])

    bad = [frame(t) for t in BAD]
    cell = bad[0][1]
    atype = bad[0][2]
    bad_pos = np.stack([b[0] for b in bad])
    healthy_steps = list(range(2000, 16000, 2000))
    anchor_steps = args.anchor_steps or []
    dumped = read_dump(args.dump, set(healthy_steps) | set(anchor_steps))
    anchors = np.stack([dumped[s] for s in anchor_steps]) if anchor_steps else bad_pos
    rng = np.random.default_rng(0)
    healthy_pos = np.stack(
        [frame(t)[0] for t in CONTROL] + [dumped[s] for s in healthy_steps]
    )
    print(
        f"{len(anchors)} anchors ({'dump steps ' + str(anchor_steps) if anchor_steps else 'pathological frames'}, sigma={args.sigma}, estimator={args.estimator}), {len(healthy_pos)} healthy distillation frames"
    )

    # Pseudo-labels from the frozen model on the healthy frames.
    e_ref, f_ref = [], []
    for pos in healthy_pos:
        e, _, f = batch(frozen, pos[None], cell, atype, args.device)
        e_ref.append(float(e[0]))
        f_ref.append(f[0].detach())
        del e, f

    def report(tag):
        model.eval()
        bad_stats = penalty_stats(
            model, bad_pos, cell, atype, args.eps, args.k0, args.rho0, args.device
        )
        h_stats = penalty_stats(
            model, healthy_pos, cell, atype, args.eps, args.k0, args.rho0, args.device
        )
        de, df = 0.0, 0.0
        for k, pos in enumerate(healthy_pos):
            e_h, _, f_h = batch(model, pos[None], cell, atype, args.device)
            de = max(de, abs(float(e_h[0]) - e_ref[k]) / atype.shape[0])
            df = max(df, float((f_h[0].detach() - f_ref[k]).norm(dim=-1).max()))
            del e_h, f_h
        print(
            f"[{tag}] bad frames: penalty {bad_stats[:, 0].mean():.3f}, ratio {np.round(bad_stats[:, 1], 2)}, Fmax {np.round(bad_stats[:, 2], 1)}"
        )
        print(
            f"[{tag}] healthy frames: max ratio {h_stats[:, 1].max():.2f}, max Fmax {h_stats[:, 2].max():.2f}, drift |dE|/N {de * 1000:.2f} meV/atom, max |dF| {df:.3f} eV/A"
        )
        s0, f, _ = adversarial_search(
            model, bad_pos[5], cell, atype, 0.05, 40, args.device
        )
        gen = torch.Generator().manual_seed(0)
        rnd = random_search(model, bad_pos[5], cell, atype, 0.02, 60, args.device, gen)
        print(
            f"[{tag}] s17096 hill climb 0.05 A: {s0:.1f} -> {f:.1f}; random 0.02 A: median {np.median(rnd):.1f} max {rnd.max():.1f} frac>100 {(rnd > 100).mean():.2f}"
        )
        rows = bond_scan(
            model,
            bad_pos[6],
            cell,
            atype,
            49,
            48,
            args.device,
            np.arange(0.6, 1.16, 0.075),
        )
        print(
            f"[{tag}] s17097 atom 49 along 49-48: "
            + "  ".join(f"r={r:.3f}:E={e:.1f},F={fr:.0f}" for r, e, fr in rows)
        )
        model.train()

    report("before")
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    model.train()
    nb, nh = len(anchors), len(healthy_pos)
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        distill_tot, pen_tot, ratio_max, fmax_max = 0.0, 0.0, 0.0, 0.0
        for k, pos in enumerate(healthy_pos):
            e_h, _, f_h = batch(model, pos[None], cell, atype, args.device)
            distill = ((e_h[0] - e_ref[k]) / atype.shape[0]).square() + (
                f_h[0] - f_ref[k]
            ).square().mean()
            (distill / nh).backward()
            distill_tot += float(distill.detach()) / nh
            del e_h, f_h, distill
        for pos in anchors:
            if args.sigma > 0.0:
                pos = pos + rng.normal(size=pos.shape) * args.sigma
            estimator = penalty_force if args.estimator == "force" else penalty_energy
            pen, ratio, fmax = estimator(
                model, pos, cell, atype, args.eps, args.k0, args.rho0, args.device
            )
            (args.lam * pen / nb).backward()
            pen_tot += float(pen.detach()) / nb
            ratio_max, fmax_max = max(ratio_max, ratio), max(fmax_max, fmax)
            del pen
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % 10 == 0 or step == 1:
            print(
                f"step {step:4d} distill {distill_tot:.2e} penalty {pen_tot:.3f} anchors: max ratio {ratio_max:.2f} max Fmax {fmax_max:.1f}"
            )
    report("after")
    torch.save(model.state_dict(), args.out / "healed_state.pt")


if __name__ == "__main__":
    main()

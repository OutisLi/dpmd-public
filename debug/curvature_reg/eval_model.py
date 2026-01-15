# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, T201, ANN001, ANN201, ANN202
"""
Evaluate a trained model for accuracy, curvature statistics, spike density
and short-MD stability on held-out OMat24 frames.

Four groups of numbers are produced for one checkpoint:

* accuracy: energy MAE per atom and force MAE against the DFT labels;
* curvature: the force-direction curvature ``c`` and per-atom force Lipschitz
  constant along the force, at the frames and at Gaussian perturbations of
  them, expressed as a ratio to the physical ceiling ``k0 + F_max / rho0``;
* spike density: the largest force found by random displacements and by a
  (1+1) hill climb inside a displacement ball, relative to the frame's own
  largest force;
* MD stability: Langevin dynamics started from the frames, counting runs in
  which the largest force or the energy leaves a physical range.

All quantities are label-free except the accuracy group, so the same script
serves as an acceptance test for models trained with and without the
curvature regularizer.
"""

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
from ase.data import (
    atomic_masses,
    atomic_numbers,
)

sys.path.insert(0, str(Path(__file__).resolve().parent / "diagnose"))
from repro_spike import (
    load_model,
)
from spike_density import (
    adversarial_search,
    random_search,
)

KB = 8.617333262e-5  # eV/K
FS = 0.09822694788  # sqrt(eV/amu)/Angstrom in 1/fs: velocity unit conversion factor


def evaluate(model, pos, cell, atype, device):
    """Return energy (float), atomic energies (N,), forces (N, 3) as float64."""
    c = torch.tensor(pos, dtype=torch.float32, device=device).reshape(1, -1, 3)
    b = torch.tensor(cell, dtype=torch.float32, device=device).reshape(1, 3, 3)
    t = torch.tensor(atype, dtype=torch.long, device=device).reshape(1, -1)
    out = model(c, t, box=b)
    e = float(out["energy"].detach().reshape(-1)[0])
    ae = out["atom_energy"].detach().reshape(-1).double().cpu().numpy()
    f = out["force"].detach().reshape(-1, 3).double().cpu().numpy()
    return e, ae, f


def curvature(model, pos, cell, atype, eps, device):
    """Force-direction curvature by energy probe, per-atom Lipschitz max, and F_max."""
    e0, ae0, f0 = evaluate(model, pos, cell, atype, device)
    fn = float(np.linalg.norm(f0))
    u = f0 / max(fn, 1e-12)
    ep, aep, fp = evaluate(model, pos + eps * u, cell, atype, device)
    c = 2.0 * ((aep - ae0).sum() + eps * fn) / eps**2
    g = (f0 - fp) / eps
    return (
        c,
        float(np.linalg.norm(g, axis=-1).max()),
        float(np.linalg.norm(f0, axis=-1).max()),
    )


def langevin(
    model, pos, cell, atype, symbols, temp, dt, steps, gamma, device, seed, fmax_limit
):
    """
    Run Langevin dynamics and report the largest force and energy excursion.

    Returns
    -------
    dict
        ``fmax``: largest per-atom force seen; ``e_drift``: largest deviation of
        the potential energy per atom from its running start value in eV;
        ``exploded``: whether ``fmax`` exceeded ``fmax_limit``; ``min_dist``:
        smallest interatomic distance reached.
    """
    rng = np.random.default_rng(seed)
    mass = np.array([atomic_masses[atomic_numbers[s]] for s in symbols])[:, None]
    n = len(mass)
    vel = rng.normal(size=(n, 3)) * np.sqrt(KB * temp / mass) * FS  # Angstrom/fs
    vel -= vel.mean(0)
    x = pos.copy()
    e0, _, f = evaluate(model, x, cell, atype, device)
    acc = f / mass * FS**2
    c1 = np.exp(-gamma * dt)
    c2 = np.sqrt((1 - c1**2) * KB * temp / mass) * FS
    fmax = np.linalg.norm(f, axis=-1).max()
    e_drift = 0.0
    min_dist = np.inf
    inv = np.linalg.inv(cell)
    for k in range(steps):
        vel += 0.5 * dt * acc
        x += dt * vel
        e, _, f = evaluate(model, x, cell, atype, device)
        acc = f / mass * FS**2
        vel += 0.5 * dt * acc
        vel = c1 * vel + c2 * rng.normal(size=(n, 3))
        fm = np.linalg.norm(f, axis=-1).max()
        fmax = max(fmax, fm)
        e_drift = max(e_drift, abs(e - e0) / n)
        if k % 50 == 0:
            d = x[:, None, :] - x[None, :, :]
            frac = d @ inv
            frac -= np.round(frac)
            d = frac @ cell
            r = np.linalg.norm(d, axis=-1) + np.eye(n) * 1e9
            min_dist = min(min_dist, r.min())
        if fm > fmax_limit or not np.isfinite(e):
            return {
                "fmax": float(fmax),
                "e_drift": float(e_drift),
                "exploded": True,
                "min_dist": float(min_dist),
                "steps": k + 1,
            }
    return {
        "fmax": float(fmax),
        "e_drift": float(e_drift),
        "exploded": False,
        "min_dist": float(min_dist),
        "steps": steps,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument(
        "--frames", type=Path, default=Path(__file__).with_name("heldout.npz")
    )
    ap.add_argument("--n-acc", type=int, default=600)
    ap.add_argument("--n-curv", type=int, default=300)
    ap.add_argument("--n-search", type=int, default=100)
    ap.add_argument("--n-md", type=int, default=60)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--sigma", type=float, default=0.03)
    ap.add_argument("--k0", type=float, default=300.0)
    ap.add_argument("--rho0", type=float, default=0.1)
    ap.add_argument("--md-temp", type=float, default=1500.0)
    ap.add_argument("--md-steps", type=int, default=2000)
    ap.add_argument("--md-dt", type=float, default=0.5)
    ap.add_argument("--fmax-limit", type=float, default=100.0)
    ap.add_argument(
        "--state-dict",
        type=Path,
        default=None,
        help="optional state dict loaded over the checkpoint's model",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, type_map = load_model(args.ckpt, args.device)
    if args.state_dict is not None:
        model.load_state_dict(torch.load(args.state_dict, map_location=args.device))
    data = np.load(args.frames, allow_pickle=True)
    assert list(data["type_map"]) == list(type_map)
    nfr = len(data["energy"])
    # Object arrays of equal-length frames collapse to a 3-d object array;
    # normalize every frame to plain numeric arrays once.
    coords = [np.asarray(data["coord"][i], dtype=np.float64) for i in range(nfr)]
    boxes = [np.asarray(data["box"][i], dtype=np.float64) for i in range(nfr)]
    atypes = [np.asarray(data["atype"][i], dtype=np.int64) for i in range(nfr)]
    forces = [np.asarray(data["force"][i], dtype=np.float64) for i in range(nfr)]
    rng = np.random.default_rng(0)
    res = {}

    # === Step 1. Accuracy ===
    e_err, f_err = [], []
    for i in range(min(args.n_acc, nfr)):
        pos, cell, atype = coords[i], boxes[i], atypes[i]
        e, _, f = evaluate(model, pos, cell, atype, args.device)
        e_err.append(abs(e - data["energy"][i]) / len(atype))
        f_err.append(np.abs(f - forces[i]).mean())
    res["mae_e_per_atom"] = float(np.mean(e_err))
    res["mae_f"] = float(np.mean(f_err))
    print(
        f"accuracy: MAE_E {res['mae_e_per_atom'] * 1000:.2f} meV/atom, MAE_F {res['mae_f'] * 1000:.1f} meV/A"
    )

    # === Step 2. Curvature at frames and at perturbed frames ===
    def ratio_stats(tag, perturb):
        ratios, lratios = [], []
        for i in range(min(args.n_curv, nfr)):
            pos, cell, atype = coords[i].copy(), boxes[i], atypes[i]
            if perturb:
                pos = pos + rng.normal(size=pos.shape) * args.sigma
            c, lmax, fmax = curvature(model, pos, cell, atype, args.eps, args.device)
            kappa = args.k0 + fmax / args.rho0
            ratios.append(abs(c) / kappa)
            lratios.append(lmax / kappa)
        ratios, lratios = np.array(ratios), np.array(lratios)
        res[f"curv_{tag}_frac_flagged"] = float((ratios > 1).mean())
        res[f"curv_{tag}_ratio_p50"] = float(np.median(ratios))
        res[f"curv_{tag}_ratio_p99"] = float(np.quantile(ratios, 0.99))
        res[f"curv_{tag}_ratio_max"] = float(ratios.max())
        res[f"lip_{tag}_frac_flagged"] = float((lratios > 1).mean())
        res[f"lip_{tag}_ratio_max"] = float(lratios.max())
        print(
            f"curvature[{tag}]: flagged {res[f'curv_{tag}_frac_flagged']:.3f}, ratio p50 {np.median(ratios):.3f} p99 {np.quantile(ratios, 0.99):.2f} max {ratios.max():.2f}; per-atom Lipschitz flagged {res[f'lip_{tag}_frac_flagged']:.3f} max {lratios.max():.2f}"
        )

    ratio_stats("frames", False)
    ratio_stats("perturbed", True)

    # === Step 3. Spike search ===
    amp2, amp5, ampa, absa = [], [], [], []
    for i in range(min(args.n_search, nfr)):
        pos, cell, atype = coords[i], boxes[i], atypes[i]
        _, _, f = evaluate(model, pos, cell, atype, args.device)
        f0 = max(np.linalg.norm(f, axis=-1).max(), 0.1)
        gen = torch.Generator().manual_seed(i)
        r2 = random_search(model, pos, cell, atype, 0.02, 10, args.device, gen).max()
        r5 = random_search(model, pos, cell, atype, 0.05, 10, args.device, gen).max()
        _, adv, _ = adversarial_search(model, pos, cell, atype, 0.05, 20, args.device)
        amp2.append(r2 / f0)
        amp5.append(r5 / f0)
        ampa.append(adv / f0)
        absa.append(adv)
    for tag, v in [("rand002", amp2), ("rand005", amp5), ("adv005", ampa)]:
        v = np.array(v)
        res[f"spike_{tag}_amp_p50"] = float(np.median(v))
        res[f"spike_{tag}_amp_p90"] = float(np.quantile(v, 0.9))
        res[f"spike_{tag}_amp_max"] = float(v.max())
    absa = np.array(absa)
    res["spike_adv005_fmax_p90"] = float(np.quantile(absa, 0.9))
    res["spike_adv005_fmax_max"] = float(absa.max())
    res["spike_adv005_frac_gt50"] = float((absa > 50).mean())
    print(
        f"spikes: rand0.02 amp p50/p90/max {np.median(amp2):.2f}/{np.quantile(amp2, 0.9):.2f}/{max(amp2):.2f}; adv0.05 amp p50/p90/max {np.median(ampa):.2f}/{np.quantile(ampa, 0.9):.2f}/{max(ampa):.2f}; adv fmax p90 {res['spike_adv005_fmax_p90']:.1f} max {absa.max():.1f} frac>50 {res['spike_adv005_frac_gt50']:.3f}"
    )

    # === Step 4. MD stability ===
    if args.n_md <= 0:
        out = args.out or args.ckpt.parent / "eval.json"
        out.write_text(json.dumps(res, indent=1))
        print(f"saved {out}")
        return
    runs = []
    for i in range(min(args.n_md, nfr)):
        pos, cell, atype = coords[i].copy(), boxes[i], atypes[i]
        symbols = [type_map[t] for t in atype]
        runs.append(
            langevin(
                model,
                pos,
                cell,
                atype,
                symbols,
                args.md_temp,
                args.md_dt,
                args.md_steps,
                0.01,
                args.device,
                i,
                args.fmax_limit,
            )
        )
    fm = np.array([r["fmax"] for r in runs])
    res["md_frac_exploded"] = float(np.mean([r["exploded"] for r in runs]))
    res["md_fmax_p50"] = float(np.median(fm))
    res["md_fmax_p90"] = float(np.quantile(fm, 0.9))
    res["md_fmax_max"] = float(fm.max())
    res["md_e_drift_p90"] = float(np.quantile([r["e_drift"] for r in runs], 0.9))
    print(
        f"MD({args.md_temp:.0f} K, {args.md_steps} x {args.md_dt} fs): exploded {res['md_frac_exploded']:.3f}, fmax p50/p90/max {np.median(fm):.1f}/{np.quantile(fm, 0.9):.1f}/{fm.max():.1f}, e_drift p90 {res['md_e_drift_p90']:.3f} eV/atom"
    )

    out = args.out or args.ckpt.parent / "eval.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()

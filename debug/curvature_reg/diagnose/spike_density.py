# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, T201, ANN001
"""
Density of spurious force spikes around a trajectory.

The diagnosis of the crash located a smooth but non-physical energy peak of
about 285 eV in one atom pair's local surface. Whether that is a rare accident
or a pervasive property of the fit decides which remedy can work: a rare
defect calls for targeted data, while a dense one calls for a regularizer that
suppresses the whole class.

Two searches are run from archived frames. The random search perturbs every
atom by a Gaussian displacement and records the largest force, which measures
how much of the neighbourhood of the trajectory is spoiled. The adversarial
search takes a few gradient-ascent steps on the largest force, which measures
how easily an optimizer -- and by extension a training-time regularizer --
can find the defects.

The separation between the normal force scale and the spike scale is the other
output that matters: a regularizer keyed on a force threshold only works if
the two populations do not overlap.
"""

from __future__ import (
    annotations,
)

import argparse
from pathlib import (
    Path,
)

import numpy as np
import torch
from repro_spike import (
    load_model,
    read_xyz,
)


def forces_of(
    model, pos: torch.Tensor, cell: torch.Tensor, atype: torch.Tensor
) -> torch.Tensor:
    """
    Evaluate forces, keeping the graph for adversarial ascent.

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    pos : torch.Tensor
        Positions in Angstrom with shape (1, N, 3), requiring grad.
    cell : torch.Tensor
        Cell in Angstrom with shape (1, 3, 3).
    atype : torch.Tensor
        Element indices with shape (1, N).

    Returns
    -------
    torch.Tensor
        Forces in eV/Angstrom with shape (N, 3).
    """
    return model(pos, atype, box=cell)["force"].reshape(-1, 3)


def random_search(
    model,
    pos0: np.ndarray,
    cell: np.ndarray,
    atype: np.ndarray,
    sigma: float,
    trials: int,
    device: str,
    gen: torch.Generator,
) -> np.ndarray:
    """
    Sample the largest force under Gaussian displacements.

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    pos0 : np.ndarray
        Reference positions in Angstrom with shape (N, 3).
    cell : np.ndarray
        Cell in Angstrom with shape (3, 3).
    atype : np.ndarray
        Element indices with shape (N,).
    sigma : float
        Standard deviation of the per-coordinate displacement in Angstrom.
    trials : int
        Number of samples.
    device : str
        Torch device string.
    gen : torch.Generator
        CPU generator for reproducible displacements.

    Returns
    -------
    np.ndarray
        Largest force magnitude of each trial, in eV/Angstrom.
    """
    base = torch.tensor(pos0, dtype=torch.float32, device=device)
    c = torch.tensor(cell, dtype=torch.float32, device=device).reshape(1, 3, 3)
    t = torch.tensor(atype, dtype=torch.long, device=device).reshape(1, -1)
    out = np.zeros(trials)
    for k in range(trials):
        noise = torch.randn(base.shape, generator=gen).to(device) * sigma
        p = (base + noise).reshape(1, -1, 3).requires_grad_(True)
        f = forces_of(model, p, c, t)
        out[k] = float(f.norm(dim=-1).max().detach())
    return out


def adversarial_search(
    model,
    pos0: np.ndarray,
    cell: np.ndarray,
    atype: np.ndarray,
    radius: float,
    steps: int,
    device: str,
) -> tuple[float, float, float]:
    """
    Ascend the largest force within a displacement ball.

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    pos0 : np.ndarray
        Reference positions in Angstrom with shape (N, 3).
    cell : np.ndarray
        Cell in Angstrom with shape (3, 3).
    atype : np.ndarray
        Element indices with shape (N,).
    radius : float
        Maximum per-atom displacement in Angstrom.
    steps : int
        Number of ascent steps.
    device : str
        Torch device string.

    Returns
    -------
    tuple[float, float, float]
        Force at the start, force at the end, and the largest per-atom
        displacement actually used, in Angstrom.
    """
    base = torch.tensor(pos0, dtype=torch.float32, device=device)
    c = torch.tensor(cell, dtype=torch.float32, device=device).reshape(1, 3, 3)
    t = torch.tensor(atype, dtype=torch.long, device=device).reshape(1, -1)
    gen = torch.Generator().manual_seed(1)

    def peak(delta: torch.Tensor) -> float:
        p = (base + delta).reshape(1, -1, 3).requires_grad_(True)
        return float(forces_of(model, p, c, t).norm(dim=-1).max().detach())

    # The model rebuilds its own graph over the coordinates, so the force it
    # returns is not differentiable in an externally supplied displacement.
    # A (1+1) hill climb needs only forward evaluations and answers the same
    # question: how readily does a search find a spike near the trajectory.
    delta = torch.zeros_like(base)
    start = peak(delta)
    best = start
    for _ in range(steps):
        cand = delta + torch.randn(base.shape, generator=gen).to(device) * (
            radius / 3.0
        )
        cand.clamp_(-radius, radius)
        value = peak(cand)
        if value > best:
            best, delta = value, cand
    return start, best, float(delta.abs().max())


def main() -> None:
    """Run both searches on archived frames and print the statistics."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("/nas/outisli/Software/deepmd-kit/temp/normal"),
    )
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=Path("/nas/outisli/Software/deepmd-kit/temp/model_ema.ckpt.pt"),
    )
    ap.add_argument(
        "--frames",
        nargs="+",
        default=["normal_control_s5000.xyz", "normal_last_normal_s17096.xyz"],
    )
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--adv-radius", type=float, default=0.05)
    ap.add_argument("--adv-steps", type=int, default=12)
    ap.add_argument("--threshold", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    model, type_map = load_model(args.ckpt, args.device)

    for name in args.frames:
        pos, cell, species = read_xyz(args.root / name)
        atype = np.array([type_map.index(s) for s in species])
        gen = torch.Generator().manual_seed(args.seed)

        print(f"\n=== {name} ===")
        print(
            f"  {'sigma (A)':>10} {'median |F|':>11} {'p90':>9} {'max':>9} "
            f"{'frac >' + str(int(args.threshold)):>10}"
        )
        print("  " + "-" * 54)
        for sigma in args.sigmas:
            vals = random_search(
                model, pos, cell, atype, sigma, args.trials, args.device, gen
            )
            print(
                f"  {sigma:10.3f} {np.median(vals):11.2f} "
                f"{np.percentile(vals, 90):9.2f} {vals.max():9.2f} "
                f"{(vals > args.threshold).mean():10.2f}"
            )

        s, f, d = adversarial_search(
            model, pos, cell, atype, args.adv_radius, args.adv_steps, args.device
        )
        print(
            f"  adversarial ascent within {args.adv_radius} A "
            f"({args.adv_steps} steps): |F|max {s:.2f} -> {f:.2f}, "
            f"max displacement {d:.4f} A"
        )


if __name__ == "__main__":
    main()

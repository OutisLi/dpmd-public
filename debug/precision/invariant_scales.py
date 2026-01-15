# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, T201
"""
Spatial sensitivity of invariant families, measured.

A quantity ``h`` entering a reduced-precision region turns into a staircase in
the coordinates whose tread width is ``eps * L`` with
``L = |h| / |dh/dx|``. A finite difference of step ``h_fd`` is blind to that
staircase only while ``eps * L > h_fd``, so ``L`` decides which invariants may
cross into bf16 and which must stay in fp32.

The estimate is geometric rather than empirical for the angular families --
an angular coordinate changes at rate ``1/r`` under a displacement, and an
order-``k`` contraction of degree-``l`` moments multiplies that by ``k*l`` --
but the constants matter, so this probe measures ``L`` directly by autograd on
a real neighbourhood for four families:

``radial``
    ``n[m, t] = sum_j u_m(r_ij)``, wide-envelope occupancy per scale and
    neighbour element. The proposed slow channel.
``gram``
    ``G[l, a, b] = sum_mu S[l, mu, a] * S[l, mu, b]``, the exact channel Gram
    of the degree-``l`` moments. Order 2.
``bispectrum``
    ``b[l1, l2, l3] = sum C * S[l1] * S[l2] * S[l3]``. Order 3.
``quartic``
    ``|G[2] @ S[1]|^2``. Order 4.

Only families whose measured ``L`` clears ``h_fd / eps`` may enter a bf16
trunk without any freezing or caching.
"""

from __future__ import (
    annotations,
)

import argparse

import torch
from torch import (
    Tensor,
)

EPS_BF16 = 2.0**-8
EPS_FP16 = 2.0**-11


def switch_c2(r: Tensor, rcut: float, rcut_smth: float) -> Tensor:
    """
    Evaluate the quintic C2 cutoff switch.

    Parameters
    ----------
    r : Tensor
        Distances in Angstrom, any shape.
    rcut : float
        Outer cutoff in Angstrom.
    rcut_smth : float
        Inner cutoff in Angstrom, below which the switch is unity.

    Returns
    -------
    Tensor
        Switch values in [0, 1], same shape as ``r``.
    """
    u = ((r - rcut_smth) / (rcut - rcut_smth)).clamp(0.0, 1.0)
    return 1.0 - u**3 * (10.0 - 15.0 * u + 6.0 * u**2)


def harmonics(vec: Tensor, lmax: int) -> list[Tensor]:
    """
    Evaluate unnormalized real solid harmonics of a unit direction.

    Written in closed form through degree four, which is what a fused kernel
    would use; overall normalization is irrelevant to a ratio like ``L``.

    Parameters
    ----------
    vec : Tensor
        Direction vectors with shape (E, 3); need not be normalized.
    lmax : int
        Maximum degree, at most 4.

    Returns
    -------
    list[Tensor]
        Degree blocks, entry ``l`` with shape (E, 2*l+1).
    """
    n = vec / vec.pow(2).sum(-1, keepdim=True).clamp_min(1e-24).sqrt()
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    out = [torch.ones_like(x)[:, None]]
    if lmax >= 1:
        out.append(torch.stack([y, z, x], dim=-1))
    if lmax >= 2:
        out.append(
            torch.stack(
                [
                    x * y,
                    y * z,
                    z * z - (x * x + y * y) / 2.0,
                    x * z,
                    (x * x - y * y) / 2.0,
                ],
                dim=-1,
            )
        )
    if lmax >= 3:
        r2 = x * x + y * y + z * z
        out.append(
            torch.stack(
                [
                    y * (3 * x * x - y * y),
                    x * y * z,
                    y * (5 * z * z - r2),
                    z * (5 * z * z - 3 * r2),
                    x * (5 * z * z - r2),
                    z * (x * x - y * y),
                    x * (x * x - 3 * y * y),
                ],
                dim=-1,
            )
        )
    if lmax >= 4:
        r2 = x * x + y * y + z * z
        out.append(
            torch.stack(
                [
                    x * y * (x * x - y * y),
                    y * z * (3 * x * x - y * y),
                    x * y * (7 * z * z - r2),
                    y * z * (7 * z * z - 3 * r2),
                    35 * z**4 - 30 * z * z * r2 + 3 * r2 * r2,
                    x * z * (7 * z * z - 3 * r2),
                    (x * x - y * y) * (7 * z * z - r2),
                    x * z * (x * x - 3 * y * y),
                    x**4 - 6 * x * x * y * y + y**4,
                ],
                dim=-1,
            )
        )
    return out[: lmax + 1]


def length_scale(feat: Tensor, pos: Tensor) -> tuple[float, float]:
    """
    Measure ``L = |h| / |dh/dx|`` of a feature vector by autograd.

    The gradient is taken with respect to the centre atom's position, which is
    the displacement a finite-difference derivative applies.

    Parameters
    ----------
    feat : Tensor
        Feature vector with shape (F,), a differentiable function of ``pos``.
    pos : Tensor
        Position tensor with shape (N, 3) carrying ``requires_grad``.

    Returns
    -------
    tuple[float, float]
        Median and 10th-percentile length scale in Angstrom over the feature
        components. The low percentile is the one that matters: a single
        fast-varying component pollutes the whole trunk.
    """
    scales = []
    for k in range(feat.numel()):
        (g,) = torch.autograd.grad(feat.flatten()[k], pos, retain_graph=True)
        grad_norm = float(g[0].norm())
        value = abs(float(feat.flatten()[k]))
        if grad_norm > 1e-12 and value > 1e-10:
            scales.append(value / grad_norm)
    if not scales:
        return float("nan"), float("nan")
    t = torch.tensor(scales)
    return float(t.median()), float(t.quantile(0.1))


def build_cluster(n_atoms: int, seed: int, device: str) -> Tensor:
    """
    Generate a random cluster with a physical minimum spacing.

    Parameters
    ----------
    n_atoms : int
        Number of atoms.
    seed : int
        Random seed.
    device : str
        Torch device string.

    Returns
    -------
    Tensor
        Positions in Angstrom with shape (N, 3).
    """
    g = torch.Generator().manual_seed(seed)
    box = (n_atoms / 0.09) ** (1.0 / 3.0)
    pts: list[Tensor] = []
    while len(pts) < n_atoms:
        c = torch.rand(3, generator=g) * box
        if all((c - p).norm() > 2.2 for p in pts):
            pts.append(c)
    return torch.stack(pts).to(device=device, dtype=torch.float64)


def main() -> None:
    """Measure and tabulate the length scale of each invariant family."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-atoms", type=int, default=40)
    ap.add_argument("--rcut", type=float, default=6.0)
    ap.add_argument("--lmax", type=int, default=4)
    ap.add_argument("--n-chan", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--h-fd",
        type=float,
        nargs="+",
        default=[0.01, 0.03],
        help="finite-difference steps to compare against (phonopy, phono3py)",
    )
    args = ap.parse_args()

    device = "cpu"
    pos = build_cluster(args.n_atoms, args.seed, device).requires_grad_(True)

    d = pos[0:1] - pos[1:]
    r = d.pow(2).sum(-1).clamp_min(1e-24).sqrt()
    keep = r < args.rcut
    d, r = d[keep], r[keep]
    print(f"centre atom has {int(keep.sum())} neighbours inside rcut={args.rcut}\n")

    # === Slow family: wide-envelope radial occupancy ===
    # Scales spanning [0.2, 0.9] * rcut, i.e. transition widths of 0.6 to 4.8 A.
    slow_scales = torch.linspace(0.2, 0.9, args.n_chan, dtype=torch.float64)
    radial = torch.stack(
        [switch_c2(r, args.rcut, float(s) * args.rcut).sum() for s in slow_scales]
    )

    # === Fast families: moments of the narrow radial basis times harmonics ===
    # A Bessel-like narrow basis, the width the fast path actually uses.
    centres = torch.linspace(1.0, args.rcut, args.n_chan, dtype=torch.float64)
    width = args.rcut / (2 * args.n_chan)
    rad_chan = torch.exp(-0.5 * ((r[:, None] - centres[None, :]) / width) ** 2)
    env = switch_c2(r, args.rcut, 0.9 * args.rcut)[:, None]
    ylm = harmonics(d, args.lmax)
    # S[l] has shape (2l+1, n_chan)
    moments = [torch.einsum("em,ec->mc", y, rad_chan * env) for y in ylm]

    grams = []
    for l_deg in range(1, args.lmax + 1):
        g = moments[l_deg].t() @ moments[l_deg]
        iu = torch.triu_indices(args.n_chan, args.n_chan)
        grams.append(g[iu[0], iu[1]])
    gram = torch.cat(grams)

    bispec = []
    for l1 in range(1, args.lmax + 1):
        for l2 in range(l1, args.lmax + 1):
            for l3 in range(l2, args.lmax + 1):
                if l3 > l1 + l2 or (l1 + l2 + l3) % 2:
                    continue
                # A single probe channel per triple is enough to expose the
                # scaling; the real layout uses low-rank probes.
                bispec.append(
                    (
                        moments[l1][:, 0].sum()
                        * moments[l2][:, 0].sum()
                        * moments[l3][:, 0].sum()
                    ).reshape(1)
                )
    bispectrum = torch.cat(bispec)

    g2 = moments[2].t() @ moments[2]
    quartic = (g2 @ moments[1].t().reshape(args.n_chan, -1)[:, :1]).pow(2).flatten()

    print(f"  {'family':14s} {'order':>5s} {'dim':>5s} {'L median':>10s} {'L p10':>9s}")
    print("  " + "-" * 48)
    rows = [
        ("radial (l=0)", 1, radial),
        ("gram (l>=1)", 2, gram),
        ("bispectrum", 3, bispectrum),
        ("quartic", 4, quartic),
    ]
    results = {}
    for name, order, feat in rows:
        med, p10 = length_scale(feat, pos)
        results[name] = p10
        print(f"  {name:14s} {order:>5d} {feat.numel():>5d} {med:>10.3f} {p10:>9.3f}")

    print("\n  required L for a bf16 trunk with no freezing (L > h_fd / eps):")
    for h in args.h_fd:
        need = h / EPS_BF16
        print(f"    h_fd = {h:.3f} A  ->  L > {need:6.2f} A")
    print("\n  margin of the radial family (L_p10 / required):")
    for h in args.h_fd:
        print(
            f"    h_fd = {h:.3f} A  ->  {results['radial (l=0)'] / (h / EPS_BF16):6.2f}x"
        )


if __name__ == "__main__":
    main()

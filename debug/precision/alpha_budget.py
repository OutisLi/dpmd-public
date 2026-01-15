# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, T201
"""
How much modulation authority a bf16 trunk can be given without freezing.

The trunk reads slowly varying invariants, so its rounding staircase has a
tread of ``eps * L``. Measured on a real neighbourhood the widest invariant
family reaches only ``L ~ 3.2`` Angstrom, giving a tread of ``1.3e-2``
Angstrom in bf16 -- the same order as the 0.01-0.03 Angstrom displacement
phonopy and phono3py use. The staircase is therefore visible to a finite
difference no matter how the input is designed, and the only remaining lever
is the strength with which the trunk enters the energy:

    E_i = sum_c (1 + alpha * gamma_i,c) * phi_i,c

with ``gamma`` from the quantized trunk and ``phi`` from an fp32 fast path.
Every derivative error scales with ``alpha``, so this probe measures the
coefficient that keeps FC2 and FC3 inside a target budget with no caching,
no freezing, and no refresh -- only a ``detach`` on the trunk input.

The reference is the same model with the trunk in fp32, so the numbers isolate
the arithmetic, not the architecture.
"""

from __future__ import (
    annotations,
)

import argparse
import math

import torch
from probe import (
    Quantizer,
    build_edges,
    make_cluster,
    switch_c2,
)
from torch import (
    Tensor,
    nn,
)


class ModulatedMLIP(nn.Module):
    """
    fp32 fast path with a reduced-precision slow-channel modulation.

    Parameters
    ----------
    n_types : int
        Number of element types.
    n_rbf : int
        Radial basis size of the fast path.
    n_scales : int
        Number of wide envelopes feeding the slow channel.
    width : int
        Trunk width.
    n_layers : int
        Trunk depth.
    n_coef : int
        Coefficient/basis width.
    rcut : float
        Outer cutoff in Angstrom.
    """

    def __init__(
        self,
        n_types: int = 3,
        n_rbf: int = 16,
        n_scales: int = 4,
        width: int = 128,
        n_layers: int = 3,
        n_coef: int = 64,
        rcut: float = 6.0,
    ) -> None:
        super().__init__()
        self.rcut = rcut
        self.n_coef = n_coef
        # Fast path: narrow Bessel-like basis, the sharp geometry.
        self.register_buffer(
            "freqs",
            torch.arange(1, n_rbf + 1, dtype=torch.float32) * (math.pi / rcut),
        )
        # Slow channel: wide envelopes only, the flattest invariants available.
        self.register_buffer(
            "scales", torch.linspace(0.15, 0.55, n_scales, dtype=torch.float32) * rcut
        )
        n_te = 8
        self.type_emb = nn.Embedding(n_types, n_te)
        self.basis = nn.Linear(n_rbf + n_te, n_coef)
        self.proj_in = nn.Linear(n_scales * n_types + n_te, width)
        self.trunk = nn.ModuleList(nn.Linear(width, width) for _ in range(n_layers))
        self.to_gamma = nn.Linear(width, n_coef)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.5 / math.sqrt(m.in_features))
                nn.init.zeros_(m.bias)

    def slow_summary(self, pos: Tensor, atype: Tensor) -> Tensor:
        """
        Build the wide-envelope occupancy summary, grouped by neighbour type.

        Parameters
        ----------
        pos : Tensor
            Positions in Angstrom with shape (N, 3).
        atype : Tensor
            Element indices with shape (N,).

        Returns
        -------
        Tensor
            Summary with shape (N, n_scales * n_types + n_te).
        """
        geo = build_edges(pos, self.rcut, 0.9 * self.rcut)
        n_types = int(self.type_emb.num_embeddings)
        occ = pos.new_zeros(pos.shape[0], self.scales.numel(), n_types)
        for k, s in enumerate(self.scales):
            w = switch_c2(geo.dist, self.rcut, float(s))
            src_type = atype.index_select(0, geo.idx_j)
            occ[:, k, :].index_put_((geo.idx_i, src_type), w, accumulate=True)
        return torch.cat([occ.flatten(1), self.type_emb(atype)], dim=-1)

    def forward(self, pos: Tensor, atype: Tensor, q: Quantizer, alpha: float) -> Tensor:
        """
        Evaluate the total energy.

        Parameters
        ----------
        pos : Tensor
            Positions in Angstrom with shape (N, 3).
        atype : Tensor
            Element indices with shape (N,).
        q : Quantizer
            Quantizer applied to the trunk GEMM operands.
        alpha : float
            Modulation strength.

        Returns
        -------
        Tensor
            Total energy in eV, a scalar.
        """
        geo = build_edges(pos, self.rcut, 0.9 * self.rcut)
        te = self.type_emb(atype)
        # Fast path, fp32 throughout.
        z = geo.dist[:, None] * self.freqs[None, :]
        rbf = (
            torch.sinc(z / math.pi)
            * switch_c2(geo.dist, self.rcut, 0.9 * self.rcut)[:, None]
        )
        phi = pos.new_zeros(pos.shape[0], self.n_coef)
        phi.index_add_(0, geo.idx_i, self.basis(torch.cat([rbf, te[geo.idx_j]], -1)))
        if alpha == 0.0:
            return phi.sum()
        # Slow channel: detached input, so d gamma / d x = 0 and the force
        # comes only from the fp32 path.
        h = torch.nn.functional.linear(
            q(self.slow_summary(pos, atype).detach()),
            q(self.proj_in.weight),
            self.proj_in.bias,
        )
        for layer in self.trunk:
            h = h + torch.tanh(
                torch.nn.functional.linear(q(h), q(layer.weight), layer.bias)
            )
        gamma = torch.tanh(self.to_gamma(h))
        return ((1.0 + alpha * gamma) * phi).sum()


def forces(
    model: ModulatedMLIP, pos: Tensor, atype: Tensor, q: Quantizer, alpha: float
) -> Tensor:
    """
    Analytic forces by autograd.

    Parameters
    ----------
    model : ModulatedMLIP
        The potential.
    pos : Tensor
        Positions in Angstrom with shape (N, 3).
    atype : Tensor
        Element indices with shape (N,).
    q : Quantizer
        Trunk quantizer.
    alpha : float
        Modulation strength.

    Returns
    -------
    Tensor
        Forces in eV/Angstrom with shape (N, 3).
    """
    x = pos.detach().clone().requires_grad_(True)
    (g,) = torch.autograd.grad(model(x, atype, q, alpha), x)
    return -g


def scan_fc(
    model: ModulatedMLIP,
    pos: Tensor,
    atype: Tensor,
    q: Quantizer,
    alpha: float,
    idx: int,
    direction: int,
    step: float,
    n_half: int,
    stride2: int,
    stride3: int,
) -> tuple[Tensor, Tensor]:
    """
    Collect FC2 and FC3 estimators from finite differences of the force.

    Parameters
    ----------
    model : ModulatedMLIP
        The potential.
    pos : Tensor
        Reference positions in Angstrom with shape (N, 3).
    atype : Tensor
        Element indices with shape (N,).
    q : Quantizer
        Trunk quantizer.
    alpha : float
        Modulation strength.
    idx : int
        Scanned atom index.
    direction : int
        Scanned Cartesian direction.
    step : float
        Grid spacing in Angstrom.
    n_half : int
        Grid points on each side.
    stride2, stride3 : int
        Grid strides for the FC2 and FC3 differences.

    Returns
    -------
    tuple[Tensor, Tensor]
        FC2 values in eV/Angstrom^2 and FC3 values in eV/Angstrom^3, one per
        admissible centre.
    """
    offs = torch.arange(-n_half, n_half + 1, dtype=torch.float64) * step
    f = torch.zeros(offs.numel(), dtype=torch.float64)
    for k, o in enumerate(offs):
        p = pos.clone()
        p[idx, direction] += float(o)
        f[k] = float(forces(model, p, atype, q, alpha)[idx, direction])
    lo, hi = max(stride2, stride3), f.numel() - max(stride2, stride3)
    c = torch.arange(lo, hi)
    fc2 = -(f[c + stride2] - f[c - stride2]) / (2.0 * stride2 * step)
    fc3 = -(f[c + stride3] - 2.0 * f[c] + f[c - stride3]) / (stride3 * step) ** 2
    return fc2, fc3


def main() -> None:
    """Sweep the modulation strength and report FC2/FC3 degradation."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-atoms", type=int, default=48)
    ap.add_argument("--n-types", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=float, default=0.005)
    ap.add_argument("--n-half", type=int, default=14)
    ap.add_argument("--stride-fc2", type=int, default=2, help="FC2 h = stride*step")
    ap.add_argument("--stride-fc3", type=int, default=6, help="FC3 h = stride*step")
    ap.add_argument(
        "--alphas", type=float, nargs="+", default=[1.0, 0.3, 0.1, 0.03, 0.01]
    )
    ap.add_argument("--formats", nargs="+", default=["bf16", "fp8_e4m3"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    # The bit-level rounding operator works on float32; an fp32 reference is
    # accurate enough here anyway, contributing about 1e-6 of relative FC3
    # noise at these step sizes.
    pos, atype = make_cluster(args.n_atoms, args.n_types, args.seed, args.device)
    model = ModulatedMLIP(n_types=args.n_types).to(args.device)
    model.eval()
    fp32 = Quantizer("fp32", low_backward=False)

    print(f"system: {args.n_atoms} atoms, {args.n_types} types")
    print(
        f"FC2 h = {args.stride_fc2 * args.step:.3f} A, "
        f"FC3 h = {args.stride_fc3 * args.step:.3f} A"
    )
    print("trunk input: wide-envelope occupancy only, detached; no freezing\n")
    print(
        f"  {'format':10s} {'alpha':>6s} {'FC2 rms':>10s} {'FC3 rms':>10s} {'FC3 max':>10s}"
    )
    print("  " + "-" * 50)

    for fmt in args.formats:
        q = Quantizer(fmt, low_backward=False)
        for alpha in args.alphas:
            ref2, ref3 = scan_fc(
                model,
                pos,
                atype,
                fp32,
                alpha,
                0,
                0,
                args.step,
                args.n_half,
                args.stride_fc2,
                args.stride_fc3,
            )
            got2, got3 = scan_fc(
                model,
                pos,
                atype,
                q,
                alpha,
                0,
                0,
                args.step,
                args.n_half,
                args.stride_fc2,
                args.stride_fc3,
            )
            s2 = float(ref2.abs().mean())
            s3 = float(ref3.abs().mean())
            e2 = float((got2 - ref2).pow(2).mean().sqrt()) / s2
            e3rms = float((got3 - ref3).pow(2).mean().sqrt()) / s3
            e3max = float((got3 - ref3).abs().max()) / s3
            print(f"  {fmt:10s} {alpha:6.3f} {e2:10.3e} {e3rms:10.3e} {e3max:10.3e}")


if __name__ == "__main__":
    main()

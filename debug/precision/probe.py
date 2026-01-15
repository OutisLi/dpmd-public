# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, T201, ANN001, ANN205
"""
Propagation of reduced-precision rounding noise into PES derivatives.

The experiment isolates one question: which derivative estimators survive
reduced-precision arithmetic in a machine-learning interatomic potential, and
how that survival depends on *where* the reduced-precision region sits in the
architecture.

Rounding is modelled by quantize-dequantize, which reproduces the tensor-core
contract exactly: operands are rounded to the reduced format, accumulation
stays in fp32. The backward pass is selectable, because the two options carry
different physics -- a low-precision backward matches ``torch.autocast``,
while an fp32 backward matches a forward-only quantization scheme.

Architectures
-------------
``standard``
    The conventional layout: smooth geometric features feed a reduced-
    precision trunk whose output is read out to energy. Every coordinate
    dependence passes through the quantized region, so rounding noise is a
    step function of the coordinates with tread width ``eps * L``, where
    ``L`` is the length scale over which the trunk activations vary.

``split``
    Coefficient/geometry separation. The reduced-precision trunk consumes
    only slowly varying inputs (element identity and a smooth coordination
    number) and emits per-atom coefficients; the sharp geometric dependence
    is carried by an fp32 basis; energy is the fp32 inner product of the two.
    Rounding noise enters as a slowly varying perturbation of the
    coefficients, so its tread width grows by the ratio of the slow-channel
    to fast-channel length scales.

``frozen``
    The ``split`` architecture with the coefficients evaluated once on a
    reference configuration and held fixed. Rounding noise becomes an exact
    constant in the coefficients, so the energy remains an analytic function
    of the coordinates and every derivative order is clean by construction.

Derivative probes
-----------------
A single atom is scanned along one Cartesian direction on a uniform grid, and
the same scan supplies every estimator by varying the stride:

- ``FD1(E)``  first derivative from finite-differenced energy, the estimator
  behind an energy-based smoothness test;
- ``FC2``     second derivative from finite-differenced analytic force, the
  estimator behind phonopy;
- ``FC3``     third derivative from twice-differenced analytic force, the
  estimator behind phono3py and thus thermal conductivity;
- analytic force, Hessian diagonal and third derivative from autograd.
"""

from __future__ import (
    annotations,
)

import argparse
import math
from dataclasses import (
    dataclass,
)

import torch
from torch import (
    Tensor,
    nn,
)

# Explicit mantissa bits of each format; the exponent range is irrelevant here
# because per-tensor scaling puts every operand inside the representable band.
MANTISSA_BITS = {
    "fp32": 23,
    "tf32": 10,
    "fp16": 10,
    "bf16": 7,
    "fp8_e4m3": 3,
    "fp8_e5m2": 2,
    "fp4_e2m1": 1,
}


def round_to_format(x: Tensor, bits: int) -> Tensor:
    """
    Round a float32 tensor to ``bits`` explicit mantissa bits.

    Round-to-nearest-even is applied on the mantissa field, which reproduces
    the rounding a tensor core performs on its operands.

    Parameters
    ----------
    x : Tensor
        Input tensor, float32, with any shape.
    bits : int
        Number of explicit mantissa bits to retain, in ``[1, 23]``.

    Returns
    -------
    Tensor
        Rounded tensor, float32, same shape as ``x``.
    """
    if bits >= 23:
        return x
    drop = 23 - bits
    i = x.detach().contiguous().view(torch.int32)
    # Round-to-nearest-even: add half an ulp of the retained field, biased by
    # the retained field's low bit, then clear the dropped bits.
    half = torch.tensor(1 << (drop - 1), dtype=torch.int32, device=x.device)
    low = (i >> drop) & 1
    i = i + half - 1 + low
    mask = torch.tensor(-(1 << drop), dtype=torch.int32, device=x.device)
    return (i & mask).view(torch.float32)


class Quantizer:
    """
    Quantize-dequantize operator with a selectable backward precision.

    Parameters
    ----------
    fmt : str
        Key into ``MANTISSA_BITS``.
    low_backward : bool
        If True the rounding is applied to the gradient as well, matching
        ``torch.autocast``. If False the backward pass is fp32, matching a
        forward-only quantization scheme.
    """

    def __init__(self, fmt: str, low_backward: bool) -> None:
        self.bits = MANTISSA_BITS[fmt]
        self.low_backward = low_backward
        self.enabled = self.bits < 23

    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply quantize-dequantize to ``x``.

        Parameters
        ----------
        x : Tensor
            Input tensor of any shape.

        Returns
        -------
        Tensor
            Quantized-dequantized tensor, same shape and dtype as ``x``.
        """
        if not self.enabled:
            return x
        if self.low_backward:
            return _QuantLowBackward.apply(x, self.bits)
        return x + (round_to_format(x, self.bits) - x).detach()


class _QuantLowBackward(torch.autograd.Function):
    """
    Quantize-dequantize that also rounds the incoming gradient.

    The gradient is returned in straight-through form so that the operator
    remains twice differentiable: rounding has zero derivative almost
    everywhere, which is precisely how a tensor core behaves when a
    reduced-precision backward GEMM feeds a second-order graph.
    """

    @staticmethod
    def forward(ctx, x: Tensor, bits: int) -> Tensor:
        ctx.bits = bits
        return round_to_format(x, bits)

    @staticmethod
    def backward(ctx, g: Tensor):
        return g + (round_to_format(g, ctx.bits) - g).detach(), None


def switch_c2(r: Tensor, rcut: float, rcut_smth: float) -> Tensor:
    """
    Evaluate the quintic C2 cutoff switch used by DeePMD descriptors.

    Parameters
    ----------
    r : Tensor
        Pair distances in Angstrom, with shape (E,).
    rcut : float
        Outer cutoff in Angstrom, where the switch and its first two
        derivatives vanish.
    rcut_smth : float
        Inner cutoff in Angstrom, below which the switch is unity.

    Returns
    -------
    Tensor
        Switch values in [0, 1], with shape (E,).
    """
    u = ((r - rcut_smth) / (rcut - rcut_smth)).clamp(0.0, 1.0)
    return 1.0 - u**3 * (10.0 - 15.0 * u + 6.0 * u**2)


@dataclass
class Geometry:
    """Edge geometry of one configuration."""

    idx_i: Tensor
    idx_j: Tensor
    dist: Tensor
    switch: Tensor


def build_edges(pos: Tensor, rcut: float, rcut_smth: float) -> Geometry:
    """
    Build the full neighbour list of a cluster inside the cutoff.

    Parameters
    ----------
    pos : Tensor
        Atomic positions in Angstrom, with shape (N, 3).
    rcut : float
        Outer cutoff in Angstrom.
    rcut_smth : float
        Inner cutoff in Angstrom.

    Returns
    -------
    Geometry
        Edge indices, distances and switch values.
    """
    n = pos.shape[0]
    d = pos[:, None, :] - pos[None, :, :]
    r = d.pow(2).sum(-1).clamp_min(1e-12).sqrt()
    eye = torch.eye(n, dtype=torch.bool, device=pos.device)
    keep = (r < rcut) & ~eye
    idx_i, idx_j = keep.nonzero(as_tuple=True)
    dist = r[idx_i, idx_j]
    return Geometry(idx_i, idx_j, dist, switch_c2(dist, rcut, rcut_smth))


class ToyMLIP(nn.Module):
    """
    Minimal interatomic potential with a precision-controlled trunk.

    The three architectures share the trunk shape and parameter count, so the
    comparison isolates the placement of the reduced-precision region.

    Parameters
    ----------
    n_types : int
        Number of element types.
    arch : str
        One of ``"standard"``, ``"split"``, ``"frozen"``.
    n_rbf : int
        Number of radial basis functions.
    width : int
        Trunk width.
    n_layers : int
        Number of residual trunk layers.
    n_coef : int
        Coefficient/basis dimension of the split architectures.
    rcut : float
        Outer cutoff in Angstrom.
    rcut_smth : float
        Inner cutoff in Angstrom.
    """

    def __init__(
        self,
        n_types: int = 3,
        arch: str = "standard",
        n_rbf: int = 32,
        width: int = 256,
        n_layers: int = 4,
        n_coef: int = 64,
        rcut: float = 6.0,
        rcut_smth: float = 5.0,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.rcut = rcut
        self.rcut_smth = rcut_smth
        self.n_coef = n_coef

        # Radial basis: width chosen so neighbouring centres overlap, giving a
        # feature length scale of roughly rcut / n_rbf.
        centres = torch.linspace(0.5, rcut, n_rbf)
        self.register_buffer("centres", centres)
        self.rbf_width = float(rcut / n_rbf)

        n_te = 16
        self.type_emb = nn.Embedding(n_types, n_te)
        trunk_in = (n_rbf + n_te) if arch == "standard" else (n_te + n_te)
        self.proj_in = nn.Linear(trunk_in, width)
        self.trunk = nn.ModuleList([nn.Linear(width, width) for _ in range(n_layers)])
        if arch == "standard":
            self.readout = nn.Linear(width, 1)
        else:
            self.to_coef = nn.Linear(width, n_coef)
            self.basis = nn.Linear(n_rbf + n_te, n_coef)

        # Keep activations O(1) so per-tensor scaling is not the limiting
        # factor; the residual scale mimics a trained deep trunk.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.5 / math.sqrt(m.in_features))
                nn.init.zeros_(m.bias)

    def radial(self, geo: Geometry) -> Tensor:
        """
        Evaluate switched radial basis functions on every edge.

        Parameters
        ----------
        geo : Geometry
            Edge geometry.

        Returns
        -------
        Tensor
            Radial features, with shape (E, n_rbf).
        """
        z = (geo.dist[:, None] - self.centres[None, :]) / self.rbf_width
        return torch.exp(-0.5 * z * z) * geo.switch[:, None]

    def coefficients(self, pos: Tensor, atype: Tensor, q: Quantizer) -> Tensor:
        """
        Evaluate the slow-channel coefficients of the split architectures.

        The slow channel sees only element identity and a smooth coordination
        number, whose length scale is ``rcut - rcut_smth`` rather than the
        radial-basis width. Exposed separately so the ``frozen`` architecture
        can evaluate it once on a reference configuration.

        Parameters
        ----------
        pos : Tensor
            Atomic positions in Angstrom, with shape (N, 3).
        atype : Tensor
            Element indices, with shape (N,).
        q : Quantizer
            Quantize-dequantize operator for the trunk.

        Returns
        -------
        Tensor
            Per-atom coefficients, with shape (N, n_coef).
        """
        geo = build_edges(pos, self.rcut, self.rcut_smth)
        te = self.type_emb(atype)
        coord = torch.zeros(
            pos.shape[0], te.shape[-1], dtype=pos.dtype, device=pos.device
        )
        coord.index_add_(0, geo.idx_i, geo.switch[:, None] * te[geo.idx_j])
        return self.to_coef(self._trunk(torch.cat([te, coord], -1), q))

    def _trunk(self, x: Tensor, q: Quantizer) -> Tensor:
        """
        Run the residual trunk with every GEMM operand quantized.

        Parameters
        ----------
        x : Tensor
            Trunk input, with shape (N, trunk_in).
        q : Quantizer
            Quantize-dequantize operator applied to both GEMM operands.

        Returns
        -------
        Tensor
            Trunk output, with shape (N, width).
        """
        h = torch.nn.functional.linear(q(x), q(self.proj_in.weight), self.proj_in.bias)
        for layer in self.trunk:
            h = h + torch.tanh(
                torch.nn.functional.linear(q(h), q(layer.weight), layer.bias)
            )
        return h

    def forward(
        self,
        pos: Tensor,
        atype: Tensor,
        q: Quantizer,
        frozen_coef: Tensor | None = None,
    ) -> Tensor:
        """
        Evaluate the total energy.

        Parameters
        ----------
        pos : Tensor
            Atomic positions in Angstrom, with shape (N, 3).
        atype : Tensor
            Element indices, with shape (N,).
        q : Quantizer
            Quantize-dequantize operator for the trunk.
        frozen_coef : Tensor | None
            Pre-computed coefficients with shape (N, n_coef); required by the
            ``frozen`` architecture and ignored otherwise.

        Returns
        -------
        Tensor
            Total energy in eV, a scalar.
        """
        geo = build_edges(pos, self.rcut, self.rcut_smth)
        te = self.type_emb(atype)
        rbf = self.radial(geo)
        edge_feat = torch.cat([rbf, te[geo.idx_j]], dim=-1)
        n = pos.shape[0]

        if self.arch == "standard":
            msg = torch.zeros(
                n, edge_feat.shape[-1], dtype=pos.dtype, device=pos.device
            )
            msg.index_add_(0, geo.idx_i, edge_feat)
            return self.readout(self._trunk(msg, q)).sum()

        # Fast channel: the sharp geometric dependence, entirely in fp32.
        phi = torch.zeros(n, self.n_coef, dtype=pos.dtype, device=pos.device)
        phi.index_add_(0, geo.idx_i, self.basis(edge_feat))

        theta = (
            frozen_coef if frozen_coef is not None else self.coefficients(pos, atype, q)
        )
        return (theta * phi).sum()


def make_cluster(
    n_atoms: int, n_types: int, seed: int, device: str
) -> tuple[Tensor, Tensor]:
    """
    Generate a random cluster with a physically plausible minimum spacing.

    Parameters
    ----------
    n_atoms : int
        Number of atoms.
    n_types : int
        Number of element types.
    seed : int
        Random seed.
    device : str
        Torch device string.

    Returns
    -------
    tuple[Tensor, Tensor]
        Positions in Angstrom with shape (N, 3), and element indices with
        shape (N,).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    box = (n_atoms / 0.03) ** (1.0 / 3.0)
    pos: list[Tensor] = []
    while len(pos) < n_atoms:
        cand = torch.rand(3, generator=g) * box
        if all((cand - p).norm() > 2.2 for p in pos):
            pos.append(cand)
    positions = torch.stack(pos).to(device=device, dtype=torch.float32)
    atype = torch.randint(0, n_types, (n_atoms,), generator=g).to(device)
    return positions, atype


def energy_and_force(
    model: ToyMLIP,
    pos: Tensor,
    atype: Tensor,
    q: Quantizer,
    frozen_coef: Tensor | None,
) -> tuple[float, Tensor]:
    """
    Evaluate the energy and the analytic force by autograd.

    Parameters
    ----------
    model : ToyMLIP
        The potential.
    pos : Tensor
        Positions in Angstrom, with shape (N, 3).
    atype : Tensor
        Element indices, with shape (N,).
    q : Quantizer
        Trunk quantizer.
    frozen_coef : Tensor | None
        Frozen coefficients for the ``frozen`` architecture.

    Returns
    -------
    tuple[float, Tensor]
        Energy in eV, and forces in eV/Angstrom with shape (N, 3).
    """
    x = pos.detach().clone().requires_grad_(True)
    e = model(x, atype, q, frozen_coef)
    (g,) = torch.autograd.grad(e, x)
    return float(e), -g


def analytic_derivatives(
    model: ToyMLIP,
    pos: Tensor,
    atype: Tensor,
    q: Quantizer,
    frozen_coef: Tensor | None,
    idx: int,
    direction: int,
) -> tuple[float, float]:
    """
    Evaluate the analytic second and third derivative along one coordinate.

    Parameters
    ----------
    model : ToyMLIP
        The potential.
    pos : Tensor
        Positions in Angstrom, with shape (N, 3).
    atype : Tensor
        Element indices, with shape (N,).
    q : Quantizer
        Trunk quantizer.
    frozen_coef : Tensor | None
        Frozen coefficients for the ``frozen`` architecture.
    idx : int
        Index of the scanned atom.
    direction : int
        Cartesian direction of the scan.

    Returns
    -------
    tuple[float, float]
        Second derivative in eV/Angstrom^2 and third derivative in
        eV/Angstrom^3.
    """
    x = pos.detach().clone().requires_grad_(True)
    e = model(x, atype, q, frozen_coef)
    (g1,) = torch.autograd.grad(e, x, create_graph=True)
    (g2,) = torch.autograd.grad(g1[idx, direction], x, create_graph=True)
    (g3,) = torch.autograd.grad(g2[idx, direction], x)
    return float(g2[idx, direction]), float(g3[idx, direction])


@dataclass
class ScanResult:
    """Derivative estimators collected on one displacement scan."""

    force_analytic: Tensor
    fd1_energy: Tensor
    fc2_force: Tensor
    fc3_force: Tensor
    hess_analytic: float
    third_analytic: float


def run_scan(
    model: ToyMLIP,
    pos: Tensor,
    atype: Tensor,
    q: Quantizer,
    idx: int,
    direction: int,
    step: float,
    n_half: int,
    stride_fc2: int,
    stride_fc3: int,
) -> ScanResult:
    """
    Scan one coordinate and assemble every derivative estimator.

    Parameters
    ----------
    model : ToyMLIP
        The potential.
    pos : Tensor
        Reference positions in Angstrom, with shape (N, 3).
    atype : Tensor
        Element indices, with shape (N,).
    q : Quantizer
        Trunk quantizer.
    idx : int
        Index of the scanned atom.
    direction : int
        Cartesian direction of the scan.
    step : float
        Grid spacing in Angstrom.
    n_half : int
        Number of grid points on each side of the reference.
    stride_fc2 : int
        Grid stride used for the second-derivative difference.
    stride_fc3 : int
        Grid stride used for the third-derivative difference.

    Returns
    -------
    ScanResult
        Estimators evaluated on the interior of the scan.
    """
    # The frozen architecture evaluates its coefficients once, on the
    # reference configuration, and reuses them across the whole scan.
    frozen = None
    if model.arch == "frozen":
        frozen = model.coefficients(pos, atype, q).detach()

    offsets = torch.arange(-n_half, n_half + 1, dtype=torch.float64) * step
    energies = torch.zeros(offsets.numel(), dtype=torch.float64)
    forces = torch.zeros(offsets.numel(), dtype=torch.float64)
    for k, off in enumerate(offsets):
        p = pos.clone()
        p[idx, direction] += float(off)
        e, f = energy_and_force(model, p, atype, q, frozen)
        energies[k] = e
        forces[k] = float(f[idx, direction])

    # FD1 from energy at the finest stride: the estimator an energy-based
    # smoothness test uses.
    fd1 = -(energies[2:] - energies[:-2]) / (2.0 * step)
    # FC2 and FC3 from the analytic force, the phonopy/phono3py estimators.
    # Quantization treads are intermittent: a single centre may sit inside a
    # tread and see no error at all. Every admissible centre on the scan is
    # therefore evaluated, and the caller reduces over them.
    s2, s3 = stride_fc2, stride_fc3
    lo, hi = max(s2, s3), forces.numel() - max(s2, s3)
    c = torch.arange(lo, hi)
    fc2 = -(forces[c + s2] - forces[c - s2]) / (2.0 * s2 * step)
    fc3 = -(forces[c + s3] - 2.0 * forces[c] + forces[c - s3]) / (s3 * step) ** 2

    h_ana, t_ana = analytic_derivatives(model, pos, atype, q, frozen, idx, direction)
    return ScanResult(
        force_analytic=forces,
        fd1_energy=fd1,
        fc2_force=fc2,
        fc3_force=fc3,
        hess_analytic=h_ana,
        third_analytic=t_ana,
    )


def main() -> None:
    """Run the precision sweep and print the derivative error table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-atoms", type=int, default=64)
    ap.add_argument("--n-types", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--atom-idx", type=int, default=0)
    ap.add_argument("--direction", type=int, default=0)
    ap.add_argument(
        "--step", type=float, default=0.0025, help="grid spacing in Angstrom"
    )
    ap.add_argument("--n-half", type=int, default=12)
    ap.add_argument(
        "--stride-fc2", type=int, default=4, help="FC2 stride (h = stride * step)"
    )
    ap.add_argument(
        "--stride-fc3", type=int, default=8, help="FC3 stride (h = stride * step)"
    )
    ap.add_argument(
        "--formats",
        nargs="+",
        default=["fp32", "tf32", "bf16", "fp8_e4m3"],
    )
    ap.add_argument(
        "--architectures",
        nargs="+",
        default=["standard", "split", "frozen"],
    )
    ap.add_argument(
        "--low-backward",
        action="store_true",
        help="round the gradient as torch.autocast does",
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    pos, atype = make_cluster(args.n_atoms, args.n_types, args.seed, args.device)
    fp32 = Quantizer("fp32", low_backward=False)

    print(f"# system: {args.n_atoms} atoms, {args.n_types} types, seed {args.seed}")
    print(f"# scan: atom {args.atom_idx} dir {args.direction}, step {args.step} A")
    print(
        f"# FD1 h={args.step:.4f} A, FC2 h={args.stride_fc2 * args.step:.4f} A, "
        f"FC3 h={args.stride_fc3 * args.step:.4f} A"
    )
    print(
        f"# backward: {'low precision (autocast-like)' if args.low_backward else 'fp32'}"
    )

    for arch in args.architectures:
        torch.manual_seed(args.seed + 1)
        model = ToyMLIP(n_types=args.n_types, arch=arch).to(args.device)
        model.eval()

        ref = run_scan(
            model,
            pos,
            atype,
            fp32,
            args.atom_idx,
            args.direction,
            args.step,
            args.n_half,
            args.stride_fc2,
            args.stride_fc3,
        )
        f_ref = float(ref.force_analytic[args.n_half])
        scale2 = float(ref.fc2_force.abs().mean())
        scale3 = float(ref.fc3_force.abs().mean())
        # Consistency of the fp32 reference itself bounds what the probes can
        # resolve; report it so the low-precision rows can be read against it.
        fd1_ref_err = float((ref.fd1_energy - ref.force_analytic[1:-1]).abs().max())

        print(f"\n=== arch = {arch} ===")
        print(
            f"  fp32 reference:  F = {f_ref:+.4f} eV/A   "
            f"FC2 = {scale2:.3g} eV/A^2   FC3 = {scale3:.3g} eV/A^3   "
            f"({ref.fc2_force.numel()} centres)"
        )
        print(
            f"  fp32 |F_ana - FD1(E)| = {fd1_ref_err:.3e} eV/A   "
            f"analytic FC2 = {ref.hess_analytic:+.4f}, FC3 = {ref.third_analytic:+.4f}"
        )
        print(
            f"  {'format':10s} {'dF_ana':>10s} {'|F-FD1E|':>10s} "
            f"{'FC2 rms':>10s} {'FC3 rms':>10s} {'FC3 max':>10s} "
            f"{'FC2 ana':>10s} {'FC3 ana':>10s}"
        )
        for fmt in args.formats:
            q = Quantizer(fmt, low_backward=args.low_backward)
            res = run_scan(
                model,
                pos,
                atype,
                q,
                args.atom_idx,
                args.direction,
                args.step,
                args.n_half,
                args.stride_fc2,
                args.stride_fc3,
            )
            d_force = abs(float(res.force_analytic[args.n_half]) - f_ref)
            fd1_err = float((res.fd1_energy - res.force_analytic[1:-1]).abs().max())
            e2 = (res.fc2_force - ref.fc2_force).abs()
            e3 = (res.fc3_force - ref.fc3_force).abs()
            d_h = abs(res.hess_analytic - ref.hess_analytic) / max(
                abs(ref.hess_analytic), 1e-30
            )
            d_t = abs(res.third_analytic - ref.third_analytic) / max(
                abs(ref.third_analytic), 1e-30
            )
            print(
                f"  {fmt:10s} {d_force:10.3e} {fd1_err:10.3e} "
                f"{float(e2.pow(2).mean().sqrt()) / scale2:10.3e} "
                f"{float(e3.pow(2).mean().sqrt()) / scale3:10.3e} "
                f"{float(e3.max()) / scale3:10.3e} "
                f"{d_h:10.3e} {d_t:10.3e}"
            )


if __name__ == "__main__":
    main()

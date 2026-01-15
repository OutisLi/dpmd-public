# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, T201
"""
What a bf16 quantization tread is, in numbers.

The demonstration uses the highest-frequency component of the SeZM Bessel
basis, ``phi(r) = sin(w r) / r`` with ``w = n_radial * pi / rcut``, because that
component has the shortest length scale in the whole descriptor and therefore
the narrowest tread.

Part 1 prints the raw and the bf16-rounded value on a displacement grid fine
enough to resolve individual treads, so the staircase is directly visible.

Part 2 contrasts the two ways rounding can enter the energy, which is the
distinction that decides whether a potential-energy surface stays smooth:

``coordinate path``
    ``E(x) = c * Q(phi(x))``. Rounding sits on the coordinate dependence, so
    the energy itself becomes a staircase in ``x``.

``coefficient``
    ``E(x) = Q(c) * phi(x)``. Rounding sits on a coefficient while the
    coordinate dependence is evaluated in fp32, so the energy stays an
    analytic function of ``x`` whose parameters are off by a relative
    ``epsilon``.

Both carry the same error magnitude. Only the second survives differentiation.
"""

from __future__ import (
    annotations,
)

import argparse
import math

import numpy as np
import torch


def bf16(x: np.ndarray) -> np.ndarray:
    """
    Round a float64 array to bfloat16 and back.

    Parameters
    ----------
    x : np.ndarray
        Input array of any shape.

    Returns
    -------
    np.ndarray
        Values rounded to bfloat16 precision, returned as float64.
    """
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    return t.to(torch.bfloat16).to(torch.float64).numpy()


def phi(r: np.ndarray, w: float) -> np.ndarray:
    """
    Evaluate the Bessel radial component ``sin(w r) / r``.

    Parameters
    ----------
    r : np.ndarray
        Distances in Angstrom, with any shape.
    w : float
        Angular frequency in 1/Angstrom.

    Returns
    -------
    np.ndarray
        Basis values, same shape as ``r``.
    """
    return np.sin(w * r) / r


def phi_derivatives(r: float, w: float) -> tuple[float, float]:
    """
    Evaluate the first and third analytic derivatives of ``sin(w r) / r``.

    Parameters
    ----------
    r : float
        Distance in Angstrom.
    w : float
        Angular frequency in 1/Angstrom.

    Returns
    -------
    tuple[float, float]
        First derivative in 1/Angstrom^2 and third derivative in
        1/Angstrom^4.
    """
    s, c = math.sin(w * r), math.cos(w * r)
    d1 = w * c / r - s / r**2
    d3 = -(w**3) * c / r + 3 * w**2 * s / r**2 + 6 * w * c / r**3 - 6 * s / r**4
    return d1, d3


def show_tread(r0: float, w: float, step: float, n: int) -> None:
    """
    Print the bf16 staircase around a reference distance.

    Parameters
    ----------
    r0 : float
        Reference distance in Angstrom.
    w : float
        Angular frequency in 1/Angstrom.
    step : float
        Grid spacing in Angstrom.
    n : int
        Number of grid points to print.
    """
    r = r0 + np.arange(n) * step
    exact = phi(r, w)
    rounded = bf16(exact)

    d1, _ = phi_derivatives(r0, w)
    val = float(exact[0])
    # The ulp of a bf16 number is 2^-8 of its binade, so it follows the
    # exponent of the value itself.
    ulp = 2.0 ** (math.floor(math.log2(abs(val))) - 7)
    tread = ulp / abs(d1)

    print("=== Part 1. What one bf16 tread looks like ===")
    print(f"phi(r) = sin(w r)/r,  w = {w:.4f} /A   (n_radial=16, rcut=6)")
    print(f"at r0 = {r0:.4f} A:  phi = {val:.10f},  dphi/dr = {d1:+.4f} /A^2")
    print(f"  bf16 ulp at this magnitude : {ulp:.3e}")
    print(f"  predicted tread  eps*L     : {tread:.3e} A")
    print(f"  grid step used             : {step:.1e} A\n")
    print(f"  {'r (A)':>10} {'phi exact':>16} {'phi in bf16':>16}   step")
    prev = rounded[0]
    for k in range(n):
        mark = ""
        if rounded[k] != prev:
            mark = "  <-- tread boundary"
            prev = rounded[k]
        print(f"  {r[k]:10.6f} {exact[k]:16.10f} {rounded[k]:16.10f}{mark}")

    jumps = int(np.count_nonzero(np.diff(rounded)))
    span = (n - 1) * step
    print(
        f"\n  {jumps} boundaries over {span:.2e} A "
        f"=> measured tread {span / max(jumps, 1):.3e} A"
    )


def fd1(f: np.ndarray) -> np.ndarray:
    """Central first difference of a 5-point stencil stack."""
    return (f[3] - f[1]) / 2.0


def fd3(f: np.ndarray) -> np.ndarray:
    """Central third difference of a 5-point stencil stack."""
    return (f[4] - 2.0 * f[3] + 2.0 * f[1] - f[0]) / 2.0


def compare_placements(
    r0: float, w: float, coeff: float, n_centres: int, spread: float
) -> None:
    """
    Compare rounding on the coordinate path against rounding on a coefficient.

    Parameters
    ----------
    r0 : float
        Centre of the sampling window in Angstrom.
    w : float
        Angular frequency in 1/Angstrom.
    coeff : float
        The fp32 coefficient multiplying the basis.
    n_centres : int
        Number of sampling centres over which errors are averaged.
    spread : float
        Half-width of the centre distribution in Angstrom.
    """
    centres = r0 + np.linspace(-spread, spread, n_centres)
    coeff_q = float(bf16(np.array([coeff]))[0])

    print("\n\n=== Part 2. The two fates of a finite difference ===")
    print(f"coefficient c = {coeff:.10f}  ->  bf16(c) = {coeff_q:.10f}")
    print(f"relative coefficient error: {abs(coeff_q - coeff) / coeff:.3e}")
    print(f"averaged over {n_centres} centres near r0 = {r0:.3f} A\n")
    print(
        f"  {'h_fd (A)':>10} | {'A: 1st':>10} {'A: 3rd':>10} "
        f"| {'B: 1st':>10} {'B: 3rd':>10}"
    )
    print(f"  {'':->10} | {'':->21} | {'':->21}")

    for h in (1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2):
        err = {"A1": [], "A3": [], "B1": [], "B3": []}
        for c0 in centres:
            offsets = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) * h
            r = c0 + offsets
            exact = phi(r, w)
            # Case A: rounding on the coordinate dependence.
            e_a = coeff * bf16(exact)
            # Case B: rounding on the coefficient, geometry in fp32.
            e_b = coeff_q * exact

            d1_ref, d3_ref = phi_derivatives(float(c0), w)
            d1_ref *= coeff
            d3_ref *= coeff
            err["A1"].append((fd1(e_a) / h - d1_ref) / abs(d1_ref))
            err["A3"].append((fd3(e_a) / h**3 - d3_ref) / abs(d3_ref))
            err["B1"].append((fd1(e_b) / h - d1_ref) / abs(d1_ref))
            err["B3"].append((fd3(e_b) / h**3 - d3_ref) / abs(d3_ref))

        def rms(key: str) -> float:
            return float(np.sqrt(np.mean(np.square(err[key]))))

        print(
            f"  {h:10.1e} | {rms('A1'):10.3e} {rms('A3'):10.3e} "
            f"| {rms('B1'):10.3e} {rms('B3'):10.3e}"
        )
    print(
        "\n  A = quantization on the coordinate path; "
        "B = quantization on the coefficient."
    )
    print("  Both carry the same O(eps) error magnitude.")


def main() -> None:
    """Run both parts of the demonstration."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rcut", type=float, default=6.0)
    ap.add_argument("--n-radial", type=int, default=16)
    ap.add_argument("--r0", type=float, default=3.1)
    ap.add_argument("--step", type=float, default=1e-4)
    ap.add_argument("--n-rows", type=int, default=18)
    ap.add_argument("--coeff", type=float, default=0.7)
    ap.add_argument("--n-centres", type=int, default=64)
    ap.add_argument("--spread", type=float, default=0.02)
    args = ap.parse_args()

    w = args.n_radial * math.pi / args.rcut
    show_tread(args.r0, w, args.step, args.n_rows)
    compare_placements(args.r0, w, args.coeff, args.n_centres, args.spread)


if __name__ == "__main__":
    main()

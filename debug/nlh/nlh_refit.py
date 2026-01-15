# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Regenerate the analytical bridging coefficients shipped with DeePMD-kit.

The bridging pair potential of the ``nlh`` mode has the Nordlund-Lehtola-Hobler
form, a screened Coulomb series in the unscaled separation,

.. math::

   V_{ab}(r) = \frac{k_e Z_a Z_b}{r}\,\varphi_{ab}(r),
   \qquad \varphi_{ab}(r) = \sum_{k=1}^{4} a_{abk}\,e^{-b_{abk} r},

and this script produces the table of :math:`(a_k, b_k)` that DeePMD-kit
installs as ``deepmd/dpmodel/atomic_model/nlh_coefficients.npz``.  The
coefficients are fitted here rather than copied from the publication: the fit
imposes three conditions that the published coefficients do not carry, namely

``a_k >= 0``
    every amplitude is non-negative, which makes :math:`\varphi` positive and
    :math:`V` strictly decreasing at every separation;
``sum_k a_k = 1``
    the screening function reaches the unscreened Coulomb limit
    :math:`\varphi(0) = 1` exactly;
``V(5 A) <= 1 meV`` and ``V(6 A) <= 0.1 meV``
    the term stays negligible over the whole range of a neighbour list, which
    is required because it is summed on every edge with no switching function.

Rows are fitted to the reference that covers their element pair:

=============================== ===========================================
pair                            reference
=============================== ===========================================
``Z1 <= 92`` and ``Z2 <= 92``   the self-consistent DFT (DMol) pair energies
``Z1 = 5, Z2 = 10`` (B-Ne)      the Hartree-Fock/MP2 screening function
``Z1 > 92`` or ``Z2 > 92``      the ZBL universal potential
=============================== ===========================================

Input data
----------
The reference curves come from the open-data package of the publication,
Zenodo record 10.5281/zenodo.14172633 version 1.0, released 16 November 2024,
file ``nlh_potentials_opendata.tar.gz`` (md5
``ee14b690a86a37ee6a55a181561930c9``).  Unpack it and pass the resulting
directory with ``--data``.  Only ``dmol/original_data/`` and ``mp2/`` are read.
The package is licensed CC BY 4.0; the attribution and the statement that the
generated coefficients are modified material travel with the table, in its
``meta`` entry and in the module that loads it.

Usage
-----
::

    python debug/nlh/nlh_refit.py fit \
        --data /path/to/nlh_potentials_opendata \
        --out deepmd/dpmodel/atomic_model/nlh_coefficients.npz

    python debug/nlh/nlh_refit.py report \
        --data /path/to/nlh_potentials_opendata \
        --table deepmd/dpmodel/atomic_model/nlh_coefficients.npz

``fit`` writes the table and ``report`` recomputes its accuracy and tail
figures.  ``fit`` exposes the settings of the objective and of the tail bound
(``--lo-ev``, ``--cap-ev``, ``--terms``, ``--eps5``, ``--eps6``) so that the
alternatives weighed while choosing them can be reproduced; the defaults are
the values of the shipped table.  Both accept ``--jobs``, and ``report``
accepts ``--stride`` to score a subset of the pairs.

Runtime
-------
``fit`` evaluates 7021 element pairs and takes roughly ten CPU-hours: about
eleven minutes wall clock on sixty worker processes, of which eight minutes are
the 4278 DMol rows and three minutes the 2743 ZBL rows.  ``report`` takes about
five minutes on one process.  The result is deterministic: every random start
is drawn from a generator seeded by the element pair, and results are stored by
pair index, so the worker count does not affect the output.
"""

# ruff: noqa: T201

import argparse
import itertools
import json
import os
import re
import sys
import time
from multiprocessing import (
    Pool,
)
from pathlib import (
    Path,
)

import numpy as np
from scipy.optimize import (
    minimize,
)

# === Physical constants ===
# Coulomb constant and Bohr radius as the reference evaluator of the source
# package defines them, so that the screening functions derived from the
# published pair energies match the published screening functions.
KE_EV_A = 1.0 / (4.0 * np.pi * 8.8541878188e-12) * 1.602176634e-19 / 1e-10
A_BOHR = 0.5291772109

# === ZBL universal screening function ===
ZBL_A = np.array([0.18175, 0.50986, 0.28022, 0.028171])
ZBL_B = np.array([3.1998, 0.94229, 0.4029, 0.20162])

# === Fit settings ===
Z_MAX = 118  # highest nuclear charge in the preset type maps
Z_REFERENCE = 92  # highest nuclear charge covered by the reference data
N_TERM = 4  # exponential slots the fused kernels evaluate
LO_EV = 10.0  # lowest pair energy that enters the fit
CAP_EV = 100.0  # energy below which the relative weight stops growing
N_GRID = 200  # sample points of the fit grid
TAIL_RADII = (5.0, 6.0)  # angstrom
TAIL_EPS = (1.0e-3, 1.0e-4)  # eV
B_LO, B_HI = 0.05, 1000.0  # bounds on a decay rate, angstrom^-1
B_NE = (5, 10)  # the pair whose DMol curve is superseded by MP2

_SUBSETS = {
    n: [tuple(s) for k in range(1, n + 1) for s in itertools.combinations(range(n), k)]
    for n in range(1, N_TERM + 1)
}


# ---------------------------------------------------------------------------
# Reference curves
# ---------------------------------------------------------------------------
def zbl_screening(x: np.ndarray) -> np.ndarray:
    """Evaluate the ZBL universal screening function at reduced radii.

    Parameters
    ----------
    x : np.ndarray
        Reduced radii ``r / a_screen`` with shape (M,), unitless.

    Returns
    -------
    np.ndarray
        Screening function with shape (M,), unitless.
    """
    return (ZBL_A * np.exp(-ZBL_B * np.asarray(x)[..., None])).sum(-1)


def zbl_screening_length(z1: int, z2: int) -> float:
    """Screening length of the ZBL universal potential in angstrom."""
    return 0.88534 * A_BOHR / (z1**0.23 + z2**0.23)


def zbl_curve(z1: int, z2: int) -> tuple[np.ndarray, np.ndarray]:
    """Tabulate the ZBL universal potential of one pair.

    Parameters
    ----------
    z1, z2 : int
        Nuclear charges.

    Returns
    -------
    r : np.ndarray
        Radii with shape (240,) in Å, spanning the range of the reference grid.
    v : np.ndarray
        Pair energies with shape (240,) in eV.
    """
    a = zbl_screening_length(z1, z2)
    r = np.geomspace(0.002, 8.0, 240)
    return r, KE_EV_A * z1 * z2 / r * zbl_screening(r / a)


def load_dmol(data_dir: Path) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Load the DMol pair energies of every available element pair.

    Parameters
    ----------
    data_dir : Path
        Root of the unpacked open-data package.

    Returns
    -------
    dict
        Maps ``(Z1, Z2)`` with ``Z1 <= Z2`` to ascending radii in Å and pair
        energies in eV.

    Raises
    ------
    FileNotFoundError
        If the expected subdirectory is absent.
    """
    root = data_dir / "dmol" / "original_data"
    if not root.is_dir():
        raise FileNotFoundError(
            f"{root} not found; pass the unpacked package to --data"
        )
    out = {}
    for path in sorted(root.glob("energies.*")):
        m = re.fullmatch(r"energies\.(\d+)\.(\d+)", path.name)
        if m is None:
            continue
        z1, z2 = sorted((int(m.group(1)), int(m.group(2))))
        d = np.loadtxt(path)
        order = np.argsort(d[:, 0])
        out[(z1, z2)] = (d[order, 0], d[order, 1])
    return out


def load_mp2_pair(data_dir: Path, z1: int, z2: int) -> tuple[np.ndarray, np.ndarray]:
    """Load one MP2 screening function and convert it to a pair energy.

    Parameters
    ----------
    data_dir : Path
        Root of the unpacked open-data package.
    z1, z2 : int
        Nuclear charges, in either order.

    Returns
    -------
    r : np.ndarray
        Radii in Å, with the zero-separation entry removed.
    v : np.ndarray
        Pair energies in eV.
    """
    a, b = sorted((z1, z2))
    d = np.loadtxt(data_dir / "mp2" / f"screening_mp2_{a}_{b}.dat")
    keep = d[:, 0] > 0.0
    r, phi = d[keep, 0], d[keep, 1]
    return r, KE_EV_A * z1 * z2 / r * phi


# ---------------------------------------------------------------------------
# The fit problem of one element pair
# ---------------------------------------------------------------------------
def _radius_at_energy(r: np.ndarray, v: np.ndarray, threshold: float) -> float:
    """Radius at which a tabulated pair energy crosses ``threshold``.

    The crossing is taken log-linearly between the two bracketing grid points,
    which keeps the fit window independent of how finely the reference is
    tabulated.
    """
    j = int(np.flatnonzero(v >= threshold)[-1])
    if j + 1 >= len(r):
        return float(r[j])
    lo, hi = np.log(v[j]), np.log(max(v[j + 1], 1e-12))
    return float(r[j] + (np.log(threshold) - lo) / (hi - lo) * (r[j + 1] - r[j]))


class PairProblem:
    r"""Fit and evaluation grids of one element pair.

    The reference screening function is :math:`\varphi = V r / (k_e Z_1 Z_2)`,
    resampled from the tabulated pair energy onto ``n_grid`` points spread
    uniformly in :math:`r` between the innermost reference point and the radius
    at which the pair energy falls to ``lo_eV``.  Residuals are weighted by
    :math:`1/\varphi`, so the objective measures a relative deviation; below
    ``cap_eV`` the weight is held at its value there, which keeps the outermost
    points, where the reference bends towards the bonding minimum, from
    dominating the sum.

    Parameters
    ----------
    z1, z2 : int
        Nuclear charges.
    r, v : np.ndarray
        Reference radii in Å and pair energies in eV, ascending in ``r``.
    lo_eV : float
        Lowest pair energy that enters the fit.
    cap_eV : float
        Energy below which the relative weight stops growing.
    n_grid : int
        Number of sample points.
    """

    def __init__(
        self,
        z1: int,
        z2: int,
        r: np.ndarray,
        v: np.ndarray,
        lo_eV: float = LO_EV,
        cap_eV: float = CAP_EV,
        n_grid: int = N_GRID,
    ) -> None:
        self.z1, self.z2 = int(z1), int(z2)
        finite = np.isfinite(r) & np.isfinite(v)
        r, v = r[finite], v[finite]
        # The reference turns attractive near the bonding minimum; the
        # screening function is defined on the repulsive branch alone.
        nonpositive = np.flatnonzero(v <= 0.0)
        end = int(nonpositive[0]) if nonpositive.size else len(r)
        self.r_ref, self.v_ref = r[:end], v[:end]
        self.phi_ref_tab = self.v_ref * self.r_ref / (KE_EV_A * self.z1 * self.z2)
        self.log_phi_tab = np.log(self.phi_ref_tab)
        self.r_lo = float(r[0])
        self.r_cap = _radius_at_energy(self.r_ref, self.v_ref, cap_eV)
        self.r_hi = _radius_at_energy(self.r_ref, self.v_ref, lo_eV)
        self.grid = np.linspace(self.r_lo, self.r_hi, n_grid)
        self.phi_ref = self._interpolate(self.grid)
        phi_cap = float(self._interpolate(np.array([self.r_cap]))[0])
        self.weight = np.where(
            self.grid <= self.r_cap, 1.0 / self.phi_ref, 1.0 / phi_cap
        )
        self.target = self.phi_ref * self.weight

    def _interpolate(self, r: np.ndarray) -> np.ndarray:
        """Reference screening function at arbitrary radii, log-linearly."""
        return np.exp(np.interp(r, self.r_ref, self.log_phi_tab))

    def scoring_grid(
        self, threshold: float, n_grid: int = 400
    ) -> tuple[np.ndarray, np.ndarray]:
        """Grid and reference used to score a fit above ``threshold`` eV."""
        g = np.linspace(
            self.r_lo, _radius_at_energy(self.r_ref, self.v_ref, threshold), n_grid
        )
        return g, self._interpolate(g)

    def tail_bounds(
        self, radii: tuple[float, ...], eps: tuple[float, ...]
    ) -> np.ndarray:
        r"""Right-hand sides of the tail inequalities.

        The requirement :math:`V(r_t) \le \epsilon_t` is linear in the
        amplitudes, :math:`\sum_k a_k e^{-b_k r_t} \le \epsilon_t r_t /
        (k_e Z_1 Z_2)`, and this returns those right-hand sides.
        """
        return np.asarray(eps) * np.asarray(radii) / (KE_EV_A * self.z1 * self.z2)


# ---------------------------------------------------------------------------
# Inner problem: the amplitudes at fixed decay rates
# ---------------------------------------------------------------------------
def _equality_solve(
    gram: np.ndarray, rhs: np.ndarray, rows: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Minimise ``a' G a / 2 - h' a`` under the equalities ``[1; rows] a = [1; values]``.

    Returns the minimiser and the multipliers of those equality rows; the
    stationarity condition is ``G a - h + E' mu = 0`` with ``E = [1; rows]``.
    """
    n = gram.shape[0]
    m = 1 + len(values)
    kkt = np.empty((n + m, n + m))
    kkt[:n, :n] = gram
    kkt[:n, :n].flat[:: n + 1] += 1e-12 * (np.trace(gram) / n + 1e-300)
    equalities = np.vstack([np.ones(n), rows]) if len(values) else np.ones((1, n))
    kkt[:n, n:] = equalities.T
    kkt[n:, :n] = equalities
    kkt[n:, n:] = 0.0
    target = np.concatenate([rhs, [1.0], values])
    try:
        x = np.linalg.solve(kkt, target)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(kkt, target, rcond=None)[0]
    return x[:n], x[n:]


def _active_set(
    gram: np.ndarray,
    rhs: np.ndarray,
    rows: np.ndarray,
    values: np.ndarray,
    active: list[int],
    n: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Non-negative minimiser with the tail rows ``active`` held as equalities.

    Indices leave the free set while their amplitude is negative and re-enter
    while their reduced gradient is, which is the standard active-set loop for
    a non-negative least-squares problem.
    """
    free = list(range(n))
    tail_rows = rows[active] if active else np.zeros((0, n))
    tail_values = values[active] if active else np.zeros(0)
    for _ in range(4 * n):
        if not free:
            return None, None
        amp, mu = _equality_solve(
            gram[np.ix_(free, free)],
            rhs[free],
            tail_rows[:, free],
            tail_values,
        )
        if not np.all(np.isfinite(amp)):
            return None, None
        if amp.min() < -1e-11:
            free.pop(int(np.argmin(amp)))
            continue
        a = np.zeros(n)
        a[free] = np.clip(amp, 0.0, None)
        if len(free) == n:
            return a, mu
        equalities = np.vstack([np.ones(n), tail_rows])
        reduced = gram @ a - rhs + equalities.T @ mu
        held = [j for j in range(n) if j not in free]
        candidate = held[int(np.argmin(reduced[held]))]
        if reduced[candidate] < -1e-11 * (abs(rhs).max() + 1e-300):
            free = sorted([*free, candidate])
            continue
        return a, mu
    return None, None


def solve_amplitudes(
    design: np.ndarray, target: np.ndarray, rows: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, float]:
    """Solve the amplitude problem exactly at fixed decay rates.

    Minimises ``||design a - target||^2`` subject to ``sum a = 1``, ``a >= 0``
    and ``rows a <= values``.  The program is convex, so a point satisfying the
    Karush-Kuhn-Tucker conditions is its global minimum; candidates are
    generated by an active-set loop over the tail rows and certified against
    those conditions, and an uncertified problem falls back to an exhaustive
    enumeration of active sets.

    Parameters
    ----------
    design : np.ndarray
        Weighted design matrix with shape (M, n); column ``k`` is
        ``weight * exp(-b_k r)``.
    target : np.ndarray
        Weighted reference with shape (M,).
    rows : np.ndarray
        Tail rows with shape (T, n); ``rows[t, k] = exp(-b_k r_t)``.
    values : np.ndarray
        Tail right-hand sides with shape (T,).

    Returns
    -------
    a : np.ndarray
        Minimising amplitudes with shape (n,).
    chi2 : float
        Objective value at the minimum.
    """
    n = design.shape[1]
    gram, rhs = design.T @ design, design.T @ target
    offset = float(target @ target)
    n_tail = len(values)

    def objective(a: np.ndarray) -> float:
        return float(a @ gram @ a - 2.0 * rhs @ a + offset)

    a, _ = _equality_solve(gram, rhs, np.zeros((0, n)), np.zeros(0))
    if (
        np.all(np.isfinite(a))
        and a.min() >= -1e-12
        and (n_tail == 0 or np.all(rows @ a <= values + 1e-15))
    ):
        return np.clip(a, 0.0, None), objective(a)

    scale = float(abs(rhs).max()) + 1e-300
    best, best_value = None, np.inf
    for k in range(n_tail + 1):
        for active in itertools.combinations(range(n_tail), k):
            a, mu = _active_set(gram, rhs, rows, values, list(active), n)
            if a is None or (n_tail and np.any(rows @ a > values * (1 + 1e-9) + 1e-14)):
                continue
            equalities = (
                np.vstack([np.ones(n), rows[list(active)]]) if k else np.ones((1, n))
            )
            reduced = gram @ a - rhs + equalities.T @ mu
            free = a > 1e-12
            if (
                (k and mu[1:].min() < -1e-9 * scale)
                or (free.any() and np.abs(reduced[free]).max() > 1e-7 * scale)
                or ((~free).any() and reduced[~free].min() < -1e-9 * scale)
            ):
                continue
            value = objective(a)
            if value < best_value:
                best, best_value = a, value
    if best is not None:
        return best, best_value

    for support in _SUBSETS[n]:
        sub_gram = gram[np.ix_(support, support)]
        sub_rhs = rhs[list(support)]
        for k in range(min(n_tail, len(support) - 1) + 1):
            for active in itertools.combinations(range(n_tail), k):
                sub_rows = (
                    rows[np.ix_(list(active), list(support))]
                    if k
                    else np.zeros((0, len(support)))
                )
                amp, _ = _equality_solve(
                    sub_gram, sub_rhs, sub_rows, values[list(active)]
                )
                if not np.all(np.isfinite(amp)) or amp.min() < -1e-10:
                    continue
                a = np.zeros(n)
                a[list(support)] = np.clip(amp, 0.0, None)
                if n_tail and np.any(rows @ a > values + 1e-12):
                    continue
                value = objective(a)
                if value < best_value:
                    best, best_value = a, value
    if best is None:
        a = np.zeros(n)
        a[int(np.argmin(rows.sum(0))) if n_tail else 0] = 1.0
        return a, objective(a)
    return best, best_value


# ---------------------------------------------------------------------------
# Outer problem: the decay rates
# ---------------------------------------------------------------------------
def _starting_rates(problem: PairProblem, n_term: int) -> list[np.ndarray]:
    """Decay-rate sets the outer search starts from.

    The list mixes the ZBL rates of the pair, geometric ladders scaled by the
    width of the fit window, and random sets drawn from a generator seeded by
    the element pair, so the result depends on the pair alone.
    """
    out = [np.sort(ZBL_B / zbl_screening_length(problem.z1, problem.z2))[::-1]]
    scale = 3.0 / max(problem.r_hi, 1e-3)
    for ratio in (1.8, 2.6, 4.0, 6.0):
        out.append(scale * np.array([ratio**3, ratio**2, ratio, 1.0]))
    rng = np.random.default_rng(7919 * problem.z1 + 104729 * problem.z2)
    for _ in range(10):
        lo = rng.uniform(np.log(0.8), np.log(8.0))
        hi = rng.uniform(np.log(20.0), np.log(500.0))
        out.append(np.sort(np.exp(rng.uniform(lo, hi, 4)))[::-1])
    return [np.clip(np.sort(b)[::-1][:n_term], B_LO, B_HI) for b in out]


def fit_pair(
    problem: PairProblem,
    n_term: int = N_TERM,
    radii: tuple[float, ...] = TAIL_RADII,
    eps: tuple[float, ...] = TAIL_EPS,
    n_refine: int = 3,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit one element pair under the amplitude, sum and tail conditions.

    The amplitudes enter linearly, so the problem is solved by variable
    projection: :func:`solve_amplitudes` gives the exact optimum at fixed decay
    rates, and the reduced objective it leaves behind is minimised over the
    logarithms of those rates by a multi-start Nelder-Mead search followed by a
    Powell polish.

    Parameters
    ----------
    problem : PairProblem
        The pair's fit grid and reference.
    n_term : int
        Number of exponential terms to fit.
    radii : tuple of float
        Radii in Å at which the long-range energy is bounded.
    eps : tuple of float
        Bounds in eV, one per radius.
    n_refine : int
        Number of starts, ranked by the reduced objective, that are refined.

    Returns
    -------
    a, b : np.ndarray
        Amplitudes and decay rates in Å⁻¹, shape (4,), sorted by decreasing
        rate, with unused slots zero.
    chi2 : float
        Weighted objective at the optimum.
    """
    radii_arr = np.asarray(radii, dtype=float)
    bounds = problem.tail_bounds(radii, eps)
    grid, weight, target = problem.grid, problem.weight, problem.target

    def reduced(log_b: np.ndarray) -> float:
        b = np.exp(np.clip(log_b, np.log(B_LO), np.log(B_HI)))
        design = weight[:, None] * np.exp(-np.outer(grid, b))
        rows = np.exp(-np.outer(radii_arr, b))
        return solve_amplitudes(design, target, rows, bounds)[1]

    ranked = sorted(
        ((reduced(np.log(b)), b) for b in _starting_rates(problem, n_term)),
        key=lambda item: item[0],
    )
    best_value, best_log_b = np.inf, None
    for _, start in ranked[:n_refine]:
        simplex = minimize(
            reduced,
            np.log(start),
            method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-14, "maxfev": 2000},
        )
        polish = minimize(
            reduced,
            simplex.x,
            method="Powell",
            options={"xtol": 1e-5, "ftol": 1e-14, "maxfev": 2000},
        )
        value = min(simplex.fun, polish.fun)
        if value < best_value:
            best_value = value
            best_log_b = polish.x if polish.fun <= simplex.fun else simplex.x

    b = np.exp(np.clip(best_log_b, np.log(B_LO), np.log(B_HI)))
    design = problem.weight[:, None] * np.exp(-np.outer(problem.grid, b))
    rows = np.exp(-np.outer(radii_arr, b))
    a, chi2 = solve_amplitudes(design, problem.target, rows, bounds)
    a = np.clip(a, 0.0, None)
    a = a / a.sum()
    a = np.pad(a, (0, N_TERM - n_term))
    b = np.pad(b, (0, N_TERM - n_term))
    order = np.argsort(-b)
    a, b = a[order], b[order]
    keep = a > 1e-12
    return np.where(keep, a, 0.0), np.where(keep, b, 0.0), chi2


def screening(a: np.ndarray, b: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Screening function of an amplitude and rate set at radii ``r``."""
    return (np.asarray(a) * np.exp(-np.asarray(b) * np.asarray(r)[..., None])).sum(-1)


def energy(a: np.ndarray, b: np.ndarray, r: np.ndarray, z1: int, z2: int) -> np.ndarray:
    """Pair energy in eV of an amplitude and rate set at radii ``r``."""
    return KE_EV_A * z1 * z2 / np.asarray(r) * screening(a, b, r)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
_DMOL: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
_SETTINGS: dict[str, object] = {}


def _reference_for(data_dir: Path, z1: int, z2: int) -> tuple[np.ndarray, np.ndarray]:
    """Reference curve of one pair, from whichever source covers it."""
    if (z1, z2) == B_NE:
        return load_mp2_pair(data_dir, z1, z2)
    if max(z1, z2) > Z_REFERENCE:
        return zbl_curve(z1, z2)
    return _DMOL[(z1, z2)]


def _fit_one(args: tuple[Path, int, int]) -> tuple[int, int, np.ndarray, np.ndarray]:
    data_dir, z1, z2 = args
    r, v = _reference_for(data_dir, z1, z2)
    problem = PairProblem(
        z1, z2, r, v, lo_eV=_SETTINGS["lo_ev"], cap_eV=_SETTINGS["cap_ev"]
    )
    a, b, _ = fit_pair(problem, n_term=_SETTINGS["terms"], eps=_SETTINGS["eps"])
    return z1, z2, a, b


def _init_worker(dmol: dict, settings: dict) -> None:
    _DMOL.update(dmol)
    _SETTINGS.update(settings)


def _metadata(settings: dict) -> dict:
    """Provenance and conditions recorded inside the generated table."""
    eps5, eps6 = settings["eps"]
    return {
        "form": "V(r) = k_e Z1 Z2 phi(r) / r, phi(r) = sum_k a_k exp(-b_k r)",
        "units": {"a": "unitless, sum_k a_k = 1", "b": "1/angstrom"},
        "conditions": {
            "amplitudes": "a_k >= 0",
            "normalisation": "sum_k a_k = 1, so phi(0) = 1",
            "tail": f"V(5 A) <= {eps5:g} eV and V(6 A) <= {eps6:g} eV",
        },
        "objective": {
            "reference": "screening function of the reference pair energies",
            "grid": f"{N_GRID} points uniform in r from the innermost reference "
            f"point to r(V = {settings['lo_ev']:g} eV)",
            "weight": f"1/phi_ref(r), held constant below V = "
            f"{settings['cap_ev']:g} eV",
            "terms": settings["terms"],
        },
        "rows": {
            "Z1 <= 92 and Z2 <= 92": "fitted to the DMol pair energies",
            "Z1 = 5, Z2 = 10": "fitted to the MP2 screening function",
            "Z1 > 92 or Z2 > 92": "fitted to the ZBL universal potential",
        },
        "source_data": {
            "title": 'Data sets for the publication "Repulsive interatomic '
            'potentials calculated at three levels of theory"',
            "authors": ["K. Nordlund", "G. Hobler", "S. Lehtola"],
            "doi": "10.5281/zenodo.14172633",
            "version": "1.0",
            "released": "2024-11-16",
            "archive": "nlh_potentials_opendata.tar.gz",
            "md5": "ee14b690a86a37ee6a55a181561930c9",
            "licence": "CC BY 4.0, https://creativecommons.org/licenses/by/4.0/",
        },
        "source_paper": {
            "citation": "K. Nordlund, S. Lehtola and G. Hobler, Repulsive "
            "interatomic potentials calculated at three levels of theory, "
            "Phys. Rev. A 111, 032818 (2025)",
            "doi": "10.1103/PhysRevA.111.032818",
            "erratum": "Phys. Rev. A 112, 059901 (2025), doi 10.1103/cdrk-x7my",
            "licence": "CC BY 4.0, https://creativecommons.org/licenses/by/4.0/",
        },
        "modification": "MODIFIED material under CC BY 4.0 section 3(a)(1)(B). "
        "These are not the coefficients published by Nordlund, Lehtola and "
        "Hobler: they are an independent refit of the same open reference data, "
        "performed under the conditions listed above, and they differ from the "
        "published values. B-Ne is fitted to the MP2 screening function and "
        "every pair containing an element with Z > 92 to the ZBL universal "
        "potential, as recorded under 'rows'. Neither the authors of the "
        "reference data nor the American Physical Society endorse this work.",
        "attribution": "Reference data: K. Nordlund, G. Hobler and S. Lehtola, "
        "Zenodo doi 10.5281/zenodo.14172633 v1.0 (2024), CC BY 4.0. "
        "Functional form: K. Nordlund, S. Lehtola and G. Hobler, "
        "Phys. Rev. A 111, 032818 (2025), doi 10.1103/PhysRevA.111.032818, "
        "CC BY 4.0. Licence text: https://creativecommons.org/licenses/by/4.0/",
    }


def command_fit(args: argparse.Namespace) -> None:
    """Fit every element pair and write the table."""
    data_dir = Path(args.data)
    dmol = load_dmol(data_dir)
    expected = Z_REFERENCE * (Z_REFERENCE + 1) // 2
    if len(dmol) != expected:
        raise ValueError(
            f"expected {expected} DMol curves under {data_dir}, found {len(dmol)}"
        )
    settings = {
        "lo_ev": args.lo_ev,
        "cap_ev": args.cap_ev,
        "terms": args.terms,
        "eps": (args.eps5, args.eps6),
    }
    pairs = [(z1, z2) for z1 in range(1, Z_MAX + 1) for z2 in range(z1, Z_MAX + 1)]
    print(f"fitting {len(pairs)} element pairs on {args.jobs} processes", flush=True)
    start = time.perf_counter()
    a_table = np.zeros((len(pairs), N_TERM))
    b_table = np.zeros((len(pairs), N_TERM))
    index = {pair: i for i, pair in enumerate(pairs)}
    with Pool(args.jobs, initializer=_init_worker, initargs=(dmol, settings)) as pool:
        done = 0
        for z1, z2, a, b in pool.imap_unordered(
            _fit_one, [(data_dir, z1, z2) for z1, z2 in pairs], chunksize=8
        ):
            a_table[index[(z1, z2)]] = a
            b_table[index[(z1, z2)]] = b
            done += 1
            if done % 500 == 0:
                print(
                    f"  {done}/{len(pairs)}  {time.perf_counter() - start:.0f} s",
                    flush=True,
                )

    charges = np.array(pairs, dtype=np.uint8)
    _verify(charges, a_table, b_table, settings["eps"])
    np.savez_compressed(
        args.out,
        pairs=charges,
        a=a_table,
        b=b_table,
        meta=np.array(json.dumps(_metadata(settings), indent=1)),
    )
    size = os.path.getsize(args.out) / 1024
    print(
        f"wrote {args.out}: {len(pairs)} rows, {size:.1f} KiB, "
        f"{time.perf_counter() - start:.0f} s"
    )


def _verify(
    charges: np.ndarray, a: np.ndarray, b: np.ndarray, eps: tuple[float, ...]
) -> None:
    """Check the conditions the fit is supposed to guarantee.

    A row on which a tail bound is active lands on it to the conditioning of
    its own least-squares system, so the bounds are checked to a relative
    tolerance of 1e-6.  That is two orders below the float32 resolution of the
    bound itself, which is the precision the table is stored and evaluated at.
    """
    tolerance = 1e-6
    if a.min() < 0.0:
        raise ValueError("negative amplitude in the fitted table")
    deviation = float(np.abs(a.sum(1) - 1.0).max())
    if deviation > 1e-12:
        raise ValueError(f"amplitudes deviate from unit sum by {deviation:.2e}")
    if not np.all((b > 0.0) | (a <= 0.0)):
        raise ValueError("a live term carries a non-positive decay rate")
    z1 = charges[:, 0].astype(float)
    z2 = charges[:, 1].astype(float)
    for radius, bound in zip(TAIL_RADII, eps, strict=True):
        v = KE_EV_A * z1 * z2 / radius * (a * np.exp(-b * radius)).sum(1)
        excess = v.max() / bound - 1.0
        if excess > tolerance:
            raise ValueError(
                f"V({radius} A) reaches {v.max():.3e} eV, above {bound:g} eV"
            )
        print(
            f"  max V({radius:g} A) = {v.max():.6e} eV "
            f"(bound {bound:g} eV, relative excess {max(excess, 0.0):.2e})"
        )
    used = (a > 0.0).sum(1)
    print(
        "  slots used: "
        + "  ".join(f"{k}:{int((used == k).sum())}" for k in (1, 2, 3, 4))
    )


def command_report(args: argparse.Namespace) -> None:
    """Recompute the accuracy and tail figures of the shipped table."""
    data_dir = Path(args.data)
    table = np.load(args.table, allow_pickle=False)
    charges, a_table, b_table = table["pairs"], table["a"], table["b"]
    dmol = load_dmol(data_dir)
    # ``_reference_for`` reads the reference curves from module state, which the
    # worker initializer fills during a fit; this command runs in one process.
    _DMOL.update(dmol)
    index = {(int(x), int(y)): i for i, (x, y) in enumerate(charges)}

    thresholds = (10.0, 30.0, 100.0, 1000.0)
    errors = {t: [] for t in thresholds}
    residual, tail4, tail5, tail6 = [], [], [], []
    scored = sorted(dmol)[:: args.stride]
    print(f"scoring {len(scored)} pairs against the DMol reference", flush=True)
    for z1, z2 in scored:
        r, v = _reference_for(data_dir, z1, z2)
        problem = PairProblem(z1, z2, r, v)
        i = index[(z1, z2)]
        a, b = a_table[i], b_table[i]
        for t in thresholds:
            grid, reference = problem.scoring_grid(t)
            errors[t].append(
                100.0 * np.sqrt(np.mean((screening(a, b, grid) / reference - 1.0) ** 2))
            )
        tail4.append(float(energy(a, b, 4.0, z1, z2)))
        tail5.append(float(energy(a, b, 5.0, z1, z2)))
        tail6.append(float(energy(a, b, 6.0, z1, z2)))
        rr, vv = dmol[(z1, z2)]
        window = (rr > 0.3) & (rr < 6.0)
        k = int(np.argmin(vv[window]))
        if vv[window][k] < 0.0:
            residual.append(float(energy(a, b, rr[window][k], z1, z2)))

    print(f"\n{'metric':22s} {'median':>10s} {'p90':>10s} {'p99':>10s} {'max':>10s}")
    for t in thresholds:
        e = np.array(errors[t])
        print(
            f"{f'E{int(t)} / %':22s} {np.median(e):10.2f} "
            f"{np.percentile(e, 90):10.2f} {np.percentile(e, 99):10.2f} {e.max():10.2f}"
        )
    res = np.array(residual)
    print(
        f"{'V(r_eq) / eV':22s} {np.median(res):10.3f} {np.percentile(res, 90):10.3f} "
        f"{np.percentile(res, 99):10.3f} {res.max():10.3f}"
    )
    for label, values in (("V(4 A)", tail4), ("V(5 A)", tail5), ("V(6 A)", tail6)):
        arr = np.array(values)
        print(
            f"{label + ' / eV':22s} {np.median(arr):10.2e} "
            f"{np.percentile(arr, 90):10.2e} {np.percentile(arr, 99):10.2e} "
            f"{arr.max():10.2e}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fit_parser = sub.add_parser("fit", help="fit every pair and write the table")
    fit_parser.add_argument("--data", required=True, help="unpacked open-data package")
    fit_parser.add_argument("--out", required=True, help="output .npz path")
    fit_parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    fit_parser.add_argument(
        "--lo-ev", type=float, default=LO_EV, help="lowest pair energy in the fit"
    )
    fit_parser.add_argument(
        "--cap-ev",
        type=float,
        default=CAP_EV,
        help="energy below which the weight is held",
    )
    fit_parser.add_argument(
        "--terms", type=int, default=N_TERM, help="number of exponential terms"
    )
    fit_parser.add_argument(
        "--eps5", type=float, default=TAIL_EPS[0], help="bound on V(5 A) in eV"
    )
    fit_parser.add_argument(
        "--eps6", type=float, default=TAIL_EPS[1], help="bound on V(6 A) in eV"
    )
    fit_parser.set_defaults(func=command_fit)

    report_parser = sub.add_parser("report", help="score an existing table")
    report_parser.add_argument(
        "--data", required=True, help="unpacked open-data package"
    )
    report_parser.add_argument("--table", required=True, help="table .npz path")
    report_parser.add_argument(
        "--stride", type=int, default=1, help="score every n-th pair"
    )
    report_parser.set_defaults(func=command_report)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: LGPL-3.0-or-later
"""Reproduce the measurements quoted in ``doc/outisli/nlh.md``.

Each section is independent and is selected by name on the command line; with
no argument every section that has the data it needs is run.

===========  ==============================================================
section      what it measures
===========  ==============================================================
``metric``   the point-set convention behind the publication's E30 and E10
             columns, checked against the printed values
``window``   the pair energies at the bridging window's two radii, which fix
             the energy range the analytical term has to be accurate over
``pathology``the published table's long-range tail, the fit quality of the
             rows that carry it, and the same fit without the tail bound
``erratum``  the v1.0 coefficient file against the current v4 one
``compare``  ZBL, the published coefficients and a shipped table, on
             accuracy, bond-length residual, tail and shape
``terms``    three exponential terms against four
``bne``      the B-Ne pair against DMol and against MP2
``zext``     rows above Z = 92, by holding out Z in 89..92
``runtime``  float32 evaluation, the CPU exponent clamp, file size and the
             cost of building the kernel table
===========  ==============================================================

Usage
-----
::

    python debug/nlh/nlh_evaluate.py \
        --data /path/to/nlh_potentials_opendata \
        --table deepmd/dpmodel/atomic_model/nlh_coefficients.npz \
        [--v4 /path/to/nlh_potentials_opendata_v4] \
        [--alt name=path.npz ...] [section ...]

``--alt`` adds a further table to the ``compare`` section, which is how the
fit-range and tail-bound sweeps are reproduced: generate each variant with
``nlh_refit.py fit`` and its own ``--lo-ev``, ``--cap-ev`` or ``--eps5``
``--eps6``, then pass them all here.  ``terms`` reads the table named
``three_term`` if one is supplied.

Runtime: ``compare`` takes about a minute per table and ``zext`` about three
minutes; the rest are seconds.
"""

# ruff: noqa: T201

import argparse
import sys
import time
from pathlib import (
    Path,
)

import numpy as np
from nlh_refit import (
    B_NE,
    KE_EV_A,
    N_TERM,
    TAIL_EPS,
    TAIL_RADII,
    ZBL_A,
    ZBL_B,
    PairProblem,
    energy,
    fit_pair,
    load_dmol,
    load_mp2_pair,
    screening,
    zbl_screening,
    zbl_screening_length,
)

FRACTION_INNER, FRACTION_OUTER = 0.26, 0.80
THRESHOLDS = (10.0, 30.0, 100.0, 1000.0)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def load_published(path: Path) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Read the publication's coefficient file into padded four-slot rows."""
    out = {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        key = (int(f[0]), int(f[1]))
        out[key] = (
            np.array([float(f[2]), float(f[4]), float(f[6]), 0.0]),
            np.array([float(f[3]), float(f[5]), float(f[7]), 0.0]),
            float(f[8].rstrip("%")),
            float(f[9].rstrip("%")),
        )
    return out


def zbl_row(z1: int, z2: int) -> tuple[np.ndarray, np.ndarray]:
    """The ZBL universal potential in the table's own coefficient form."""
    return ZBL_A.copy(), ZBL_B / zbl_screening_length(z1, z2)


def table_rows(path: Path) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Read a generated table into a dictionary keyed by element pair."""
    d = np.load(path, allow_pickle=False)
    return {
        (int(x), int(y)): (d["a"][i], d["b"][i]) for i, (x, y) in enumerate(d["pairs"])
    }


def rms_relative(
    a: np.ndarray, b: np.ndarray, grid: np.ndarray, ref: np.ndarray
) -> float:
    """RMS relative deviation of a fit from a reference screening function."""
    return 100.0 * float(np.sqrt(np.mean((screening(a, b, grid) / ref - 1.0) ** 2)))


def slope(a: np.ndarray, b: np.ndarray, r: float, z1: int, z2: int) -> float:
    """Radial derivative of the pair energy in eV/Å."""
    e = np.asarray(a) * np.exp(-np.asarray(b) * r)
    return -KE_EV_A * z1 * z2 * float((e * (np.asarray(b) + 1.0 / r)).sum()) / r


def _quantiles(x: np.ndarray) -> tuple[float, float, float, float]:
    x = np.asarray(x)[np.isfinite(x)]
    return (
        float(np.median(x)),
        float(np.percentile(x, 90)),
        float(np.percentile(x, 99)),
        float(x.max()),
    )


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def section_metric(ctx: dict) -> None:
    """Identify the point set behind the publication's error columns."""
    published, dmol = ctx["published"], ctx["dmol"]
    conventions = {"native": [], "uniform-in-r": []}
    printed = {"native": [], "uniform-in-r": []}
    for threshold, column in ((30.0, 2), (10.0, 3)):
        conventions = {k: [] for k in conventions}
        printed = {k: [] for k in printed}
        for key, (a, b, e30, e10) in published.items():
            if key not in dmol or key == B_NE:
                continue
            r, v = dmol[key]
            problem = PairProblem(key[0], key[1], r, v)
            mask = np.isfinite(v) & (v >= threshold)
            conventions["native"].append(
                rms_relative(
                    a,
                    b,
                    r[mask],
                    v[mask] * r[mask] / (KE_EV_A * key[0] * key[1]),
                )
            )
            conventions["uniform-in-r"].append(
                rms_relative(a, b, *problem.scoring_grid(threshold))
            )
            printed["native"].append(e30 if column == 2 else e10)
            printed["uniform-in-r"].append(e30 if column == 2 else e10)
        label = "E30" if column == 2 else "E10"
        ref = np.array(printed["native"])
        print(f"{label}: printed median {np.median(ref):.2f}%")
        for name, values in conventions.items():
            values = np.array(values)
            diff = values - ref
            print(
                f"   {name:14s} median {np.median(values):5.2f}%  "
                f"bias {diff.mean():+6.3f} pp  median |diff| "
                f"{np.median(np.abs(diff)):5.3f} pp  "
                f"correlation {np.corrcoef(values, ref)[0, 1]:.4f}"
            )


def section_window(ctx: dict) -> None:
    """Map the bridging window onto reference pair energies."""
    from deepmd.utils.element_radii import (
        COVALENT_RADII,
    )

    dmol = ctx["dmol"]
    rows = []
    for (z1, z2), (r, v) in sorted(dmol.items()):
        d_ref = COVALENT_RADII[z1 - 1] + COVALENT_RADII[z2 - 1]
        finite = np.isfinite(v) & (v > 0)
        rr, vv = r[finite], v[finite]
        v_in = float(np.exp(np.interp(FRACTION_INNER * d_ref, rr, np.log(vv))))
        v_out = float(np.exp(np.interp(FRACTION_OUTER * d_ref, rr, np.log(vv))))
        rows.append((FRACTION_INNER * d_ref, FRACTION_OUTER * d_ref, v_in, v_out))
    arr = np.array(rows)
    print(f"pairs covered: {len(arr)}")
    print(
        f"inner radius : min {arr[:, 0].min():.3f}  median {np.median(arr[:, 0]):.3f}  "
        f"max {arr[:, 0].max():.3f} A"
    )
    q = np.percentile(arr[:, 2], [0, 5, 50, 95, 100])
    print(
        f"pair energy at the inner radius: min {q[0]:.4g}  p5 {q[1]:.4g}  "
        f"median {q[2]:.4g}  p95 {q[3]:.4g}  max {q[4]:.4g} eV"
    )
    print(
        f"   below 10 eV for {int((arr[:, 2] < 10).sum())} pairs, "
        f"below 30 eV for {int((arr[:, 2] < 30).sum())} pairs"
    )
    q = np.percentile(arr[:, 3], [0, 50, 100])
    print(
        f"pair energy at the outer radius: min {q[0]:.4g}  median {q[1]:.4g}  "
        f"max {q[2]:.4g} eV"
    )


def section_pathology(ctx: dict) -> None:
    """The published tail, whose rows carry it, and the same fit unbounded."""
    published = ctx["published"]
    keys = sorted(published)
    e30 = np.array([published[k][2] for k in keys])
    e10 = np.array([published[k][3] for k in keys])
    flat = np.array(
        [bool(((published[k][1] == 0) & (published[k][0] > 0)).any()) for k in keys]
    )
    v6 = np.array([energy(*published[k][:2], 6.0, *k) for k in keys])
    slow = v6 > 1e-4
    print(f"rows with a zero exponent and a positive amplitude: {int(flat.sum())}")
    print(f"rows with V(6 A) above 0.1 meV                    : {int(slow.sum())}")
    for name, mask in (
        ("zero-exponent rows", flat),
        ("V(6 A) > 0.1 meV rows", slow),
        ("all other rows", ~slow),
    ):
        p30 = np.median([(e30 < x).mean() for x in e30[mask]])
        p10 = np.median([(e10 < x).mean() for x in e10[mask]])
        print(
            f"   {name:22s} n={int(mask.sum()):4d}  printed E30 median "
            f"{np.median(e30[mask]):5.2f}% ({100 * p30:.0f}th pct)  "
            f"E10 median {np.median(e10[mask]):5.2f}% ({100 * p10:.0f}th pct)"
        )
    print(
        f"   {'overall':22s} n={len(keys):4d}  printed E30 median "
        f"{np.median(e30):5.2f}%           E10 median {np.median(e10):5.2f}%"
    )

    print("\nthe same fit with and without the tail bound, on a sample of pairs:")
    sample = keys[:: ctx["stride"] * 4]
    for label, eps in (("unbounded", (0.0, 0.0)), ("bounded", TAIL_EPS)):
        rates, tails = [], []
        for z1, z2 in sample:
            r, v = ctx["dmol"][(z1, z2)]
            problem = PairProblem(z1, z2, r, v)
            radii = TAIL_RADII if eps[0] > 0 else ()
            a, b, _ = fit_pair(problem, radii=radii, eps=eps if eps[0] > 0 else ())
            live = a > 1e-4
            rates.append(float(b[live].min()) if live.any() else np.nan)
            tails.append(energy(a, b, 5.0, z1, z2))
        rates, tails = np.array(rates), np.array(tails)
        print(
            f"   {label:10s} slowest surviving rate min {np.nanmin(rates):7.4f} 1/A, "
            f"{int((rates < 1).sum()):4d} of {len(sample)} pairs below 1 1/A, "
            f"{int((tails > 1.0001e-3).sum()):4d} above 1 meV at 5 A, "
            f"worst V(5 A) {tails.max():.4g} eV"
        )


def section_erratum(ctx: dict) -> None:
    """The v1.0 coefficient file against the current v4 one."""
    if ctx["v4"] is None:
        print("no --v4 package supplied; skipped")
        return
    new = load_published(ctx["v4"] / "nlh" / "nlh_coeffs.dat")
    old = ctx["published"]
    changed = [
        k for k in sorted(old) if not np.allclose(old[k][0], new[k][0], atol=1e-9)
    ]
    print(f"rows that differ: {len(changed)} -> {changed}")
    for label, table in (("v1.0", old), ("v4", new)):
        keys = sorted(table)
        v4_ = np.array([energy(*table[k][:2], 4.0, *k) for k in keys])
        v5 = np.array([energy(*table[k][:2], 5.0, *k) for k in keys])
        v6 = np.array([energy(*table[k][:2], 6.0, *k) for k in keys])
        worst = keys[int(v6.argmax())]
        flat = sum(bool(((table[k][1] == 0) & (table[k][0] > 0)).any()) for k in keys)
        print(
            f"   {label:5s} V(4A)>10meV {int((v4_ > 1e-2).sum()):5d}  "
            f"V(5A)>1meV {int((v5 > 1e-3).sum()):5d}  "
            f"V(6A)>0.1meV {int((v6 > 1e-4).sum()):5d}  "
            f"worst V(6A) {v6.max():7.4f} eV at Z{worst[0]}-Z{worst[1]}  "
            f"zero-exponent rows {flat}"
        )


def section_compare(ctx: dict) -> None:
    """Score every supplied table against the reference, side by side."""
    dmol, published = ctx["dmol"], ctx["published"]
    tables = {"ZBL": None, "NLH-published": None, **ctx["tables"]}
    keys = sorted(dmol)[:: ctx["stride"]]
    shape_grid = np.concatenate(
        [np.geomspace(1e-4, 2.0, 400), np.linspace(2.01, 20.0, 300)]
    )
    metrics = {
        name: {
            k: []
            for k in (
                "E10",
                "E30",
                "E100",
                "E1000",
                "Veq",
                "V4",
                "V5",
                "V6",
                "pos",
                "mono",
                "slope6",
            )
        }
        for name in tables
    }
    for z1, z2 in keys:
        r, v = dmol[(z1, z2)]
        problem = PairProblem(z1, z2, r, v)
        grids = {t: problem.scoring_grid(t) for t in THRESHOLDS}
        finite = np.isfinite(v)
        window = finite & (r > 0.3) & (r < 6.0)
        k = int(np.argmin(v[window]))
        r_eq = float(r[window][k]) if v[window][k] < 0 else np.nan
        for name in tables:
            if name == "ZBL":
                a, b = zbl_row(z1, z2)
            elif name == "NLH-published":
                a, b = published[(z1, z2)][:2]
            else:
                a, b = ctx["tables"][name][(z1, z2)]
            m = metrics[name]
            for t in THRESHOLDS:
                m[f"E{int(t)}"].append(rms_relative(a, b, *grids[t]))
            m["Veq"].append(energy(a, b, r_eq, z1, z2) if np.isfinite(r_eq) else np.nan)
            for radius, field in ((4.0, "V4"), (5.0, "V5"), (6.0, "V6")):
                m[field].append(energy(a, b, radius, z1, z2))
            curve = energy(a, b, shape_grid, z1, z2)
            m["pos"].append(float(curve.min()))
            m["mono"].append(float(np.diff(curve).max()))
            m["slope6"].append(abs(slope(a, b, 6.0, z1, z2)))

    names = list(tables)
    print(f"scored on {len(keys)} pairs\n")
    print(f"{'metric':18s}" + "".join(f"{n:>22s}" for n in names))
    for field, label in (
        ("E10", "E10 / %"),
        ("E30", "E30 / %"),
        ("E100", "E100 / %"),
        ("E1000", "E1000 / %"),
        ("Veq", "V(r_eq) / eV"),
    ):
        row = f"{label:18s}"
        for n in names:
            med, _, p99, _ = _quantiles(np.array(metrics[n][field]))
            row += f"{med:12.3f} ({p99:7.3f})"
        print(row)
    for field, label in (
        ("V4", "worst V(4 A)"),
        ("V5", "worst V(5 A)"),
        ("V6", "worst V(6 A)"),
        ("slope6", "worst |dV/dr|(6 A)"),
    ):
        print(
            f"{label:18s}"
            + "".join(f"{np.nanmax(metrics[n][field]):22.3e}" for n in names)
        )
    for field, label in (
        ("E30", "worst E30 / %"),
        ("E100", "worst E100 / %"),
        ("Veq", "worst V(r_eq) / eV"),
    ):
        print(
            f"{label:18s}"
            + "".join(f"{np.nanmax(metrics[n][field]):22.2f}" for n in names)
        )
    for label, field, threshold in (
        ("V(5 A) > 1 meV", "V5", 1e-3 * (1 + 1e-6)),
        ("V(6 A) > 0.1 meV", "V6", 1e-4 * (1 + 1e-6)),
        ("V(r_eq) > 1 eV", "Veq", 1.0),
    ):
        print(
            f"{'pairs ' + label:18s}"
            + "".join(
                f"{int(np.nansum(np.array(metrics[n][field]) > threshold)):22d}"
                for n in names
            )
        )
    print(
        f"{'pairs V <= 0':18s}"
        + "".join(f"{int((np.array(metrics[n]['pos']) <= 0).sum()):22d}" for n in names)
    )
    print(
        f"{'pairs not falling':18s}"
        + "".join(
            f"{int((np.array(metrics[n]['mono']) >= 0).sum()):22d}" for n in names
        )
    )

    reference = "NLH-published"
    print(f"\nper-pair wins against {reference}:")
    for n in names:
        if n == reference:
            continue
        wins = []
        for field in ("E30", "E100", "E1000", "Veq"):
            x = np.array(metrics[n][field])
            y = np.array(metrics[reference][field])
            ok = np.isfinite(x) & np.isfinite(y)
            wins.append(f"{field} {100 * (x[ok] < y[ok]).mean():.1f}%")
        print(f"   {n:22s} " + "  ".join(wins))


def section_terms(ctx: dict) -> None:
    """Three exponential terms against four, on the same objective."""
    four = ctx["tables"].get("shipped")
    three = ctx["tables"].get("three_term")
    if four is None or three is None:
        print("needs --table and --alt three_term=<three-term table>; skipped")
        return
    keys = sorted(ctx["dmol"])[:: ctx["stride"]]
    chi3, chi4, used = [], [], []
    for z1, z2 in keys:
        problem = PairProblem(z1, z2, *ctx["dmol"][(z1, z2)])
        for store, out in ((three, chi3), (four, chi4)):
            a, b = store[(z1, z2)]
            res = problem.weight * (screening(a, b, problem.grid) - problem.phi_ref)
            out.append(float(res @ res))
        used.append(int((four[(z1, z2)][0] > 1e-12).sum()))
    ratios = np.array(chi3) / np.maximum(np.array(chi4), 1e-300)
    print(
        f"chi2(3 terms)/chi2(4 terms) over {len(keys)} pairs: "
        f"median {np.median(ratios):.4f}  p90 {np.percentile(ratios, 90):.4f}  "
        f"p99 {np.percentile(ratios, 99):.3f}  max {ratios.max():.2f}"
    )
    print(
        f"pairs where the fourth term reduces chi2 by >1%: "
        f"{int((ratios > 1.01).sum())}, by >10%: {int((ratios > 1.10).sum())}"
    )
    used = np.array(used)
    print(
        "slots used by the four-slot fit: "
        + "  ".join(f"{k}:{int((used == k).sum())}" for k in (1, 2, 3, 4))
    )


def section_bne(ctx: dict) -> None:
    """The B-Ne pair, and how far its DMol curve sits from MP2."""
    data_dir, dmol = ctx["data"], ctx["dmol"]
    print("ratio of the DMol to the MP2 screening function, where V >= 100 eV:")
    deviations = []
    for key in sorted(dmol):
        path = data_dir / "mp2" / f"screening_mp2_{key[0]}_{key[1]}.dat"
        if not path.exists():
            continue
        r, v = dmol[key]
        finite = np.isfinite(v) & (v > 0)
        rr, vv = r[finite], v[finite]
        phi_d = vv * rr / (KE_EV_A * key[0] * key[1])
        rm, vm = load_mp2_pair(data_dir, *key)
        phi_m = vm * rm / (KE_EV_A * key[0] * key[1])
        hi = rr[vv >= 100.0].max()
        probe = np.linspace(0.02, hi, 40)
        ratio = np.interp(probe, rr, phi_d) / np.interp(probe, rm, phi_m)
        deviations.append((float(np.abs(ratio - 1).max()), key))
    deviations.sort(reverse=True)
    print(f"   {len(deviations)} pairs have an MP2 reference; the worst are")
    for value, key in deviations[:6]:
        print(f"      Z{key[0]}-Z{key[1]}: {value:.1%}")
    print(f"   median over all of them: {np.median([v for v, _ in deviations]):.2%}")

    r_m, v_m = load_mp2_pair(data_dir, *B_NE)
    problem_m = PairProblem(*B_NE, r_m, v_m)
    r_d, v_d = dmol[B_NE]
    problem_d = PairProblem(*B_NE, r_d, v_d)
    a_m, b_m, _ = fit_pair(problem_m)
    a_d, b_d, _ = fit_pair(problem_d)
    rows = {
        "published (fitted to MP2)": ctx["published"][B_NE][:2],
        "refit to DMol": (a_d, b_d),
        "refit to MP2": (a_m, b_m),
    }
    if "shipped" in ctx["tables"]:
        rows["shipped"] = ctx["tables"]["shipped"][B_NE]
    print(
        f"\n{'row':28s} {'E30 vs MP2':>11s} {'E100 vs MP2':>12s} "
        f"{'E30 vs DMol':>12s} {'V(5 A)/eV':>11s}"
    )
    for name, (a, b) in rows.items():
        print(
            f"{name:28s} {rms_relative(a, b, *problem_m.scoring_grid(30.0)):11.2f} "
            f"{rms_relative(a, b, *problem_m.scoring_grid(100.0)):12.2f} "
            f"{rms_relative(a, b, *problem_d.scoring_grid(30.0)):12.2f} "
            f"{energy(a, b, 5.0, *B_NE):11.3e}"
        )


def section_zext(ctx: dict) -> None:
    """Rows above Z = 92, judged by holding out the heaviest fitted pairs."""
    dmol, shipped = ctx["dmol"], ctx["tables"].get("shipped")
    if shipped is None:
        print("needs --table; skipped")
        return
    keys = sorted(dmol)
    reduced = np.geomspace(0.02, 45.0, 64)
    log_ratio = {}
    for key in keys:
        a, b = shipped[key]
        length = zbl_screening_length(*key)
        log_ratio[key] = np.log(
            np.maximum(screening(a, b, reduced * length), 1e-30)
            / zbl_screening(reduced)
        )
    held = [k for k in keys if max(k) >= 89]
    train = [k for k in keys if max(k) <= 88]
    print(f"training pairs {len(train)}, held out {len(held)}")

    def design(pairs: list[tuple[int, int]], degree: int) -> np.ndarray:
        u1 = np.array([k[0] for k in pairs], float) ** (1 / 3)
        u2 = np.array([k[1] for k in pairs], float) ** (1 / 3)
        p, q = u1 + u2, u1 * u2
        cols = [np.ones_like(p)]
        for d in range(1, degree + 1):
            cols.extend(p ** (d - j) * q**j for j in range(d + 1))
        return np.stack(cols, -1)

    models = {"ZBL universal": {k: np.zeros(len(reduced)) for k in held}}
    y_train = np.stack([log_ratio[k] for k in train])
    for degree in (1, 2, 3):
        coef, *_ = np.linalg.lstsq(design(train, degree), y_train, rcond=None)
        pred = design(held, degree) @ coef
        models[f"extrapolated, degree {degree}"] = dict(zip(held, pred, strict=True))
    nearest = {}
    train_z = np.array(train, float)
    for key in held:
        j = int(np.argmin(np.abs(train_z - np.array(key, float)).sum(1)))
        nearest[key] = log_ratio[train[j]]
    models["nearest pair in Z"] = nearest
    models["the fitted row"] = {k: log_ratio[k] for k in held}

    print(
        f"\n{'source of the row':30s} {'E30 med':>9s} {'E30 p99':>9s} "
        f"{'E100 med':>9s} {'E1000 med':>10s}"
    )
    problems = {k: PairProblem(k[0], k[1], *dmol[k]) for k in held}
    for name, model in models.items():
        scores = {t: [] for t in (30.0, 100.0, 1000.0)}
        for key in held:
            length = zbl_screening_length(*key)
            for t in scores:
                grid, ref = problems[key].scoring_grid(t)
                phi = np.exp(
                    np.interp(grid / length, reduced, model[key])
                ) * zbl_screening(grid / length)
                scores[t].append(100 * np.sqrt(np.mean((phi / ref - 1.0) ** 2)))
        print(
            f"{name:30s} {np.median(scores[30.0]):9.2f} "
            f"{np.percentile(scores[30.0], 99):9.2f} "
            f"{np.median(scores[100.0]):9.2f} {np.median(scores[1000.0]):10.2f}"
        )

    print("\nplain ZBL against the shipped rows above Z = 92:")
    print(
        f"{'pair':>10s} {'ZBL V(5A)':>12s} {'ZBL V(6A)':>12s} "
        f"{'shipped V(5A)':>14s} {'shipped V(6A)':>14s} {'E30 vs ZBL':>11s}"
    )
    for z1, z2 in ((92, 92), (94, 94), (100, 100), (118, 118), (1, 118), (8, 118)):
        length = zbl_screening_length(z1, z2)
        a_z, b_z = zbl_row(z1, z2)
        r = np.geomspace(0.002, 8.0, 240)
        problem = PairProblem(z1, z2, r, energy(a_z, b_z, r, z1, z2))
        a, b = shipped[(z1, z2)]
        print(
            f"{f'{z1}-{z2}':>10s} {energy(a_z, b_z, 5.0, z1, z2):12.3e} "
            f"{energy(a_z, b_z, 6.0, z1, z2):12.3e} "
            f"{energy(a, b, 5.0, z1, z2):14.3e} {energy(a, b, 6.0, z1, z2):14.3e} "
            f"{rms_relative(a, b, *problem.scoring_grid(30.0)):10.3f}%"
        )
        del length


def section_runtime(ctx: dict) -> None:
    """Float32 evaluation, the CPU exponent clamp, file size and build cost."""
    shipped = ctx["tables"].get("shipped")
    if shipped is None:
        print("needs --table; skipped")
        return
    keys = sorted(shipped)

    def f32(a: np.ndarray, b: np.ndarray, r: float, z1: int, z2: int) -> float:
        amp = np.float32(KE_EV_A * z1 * z2 * np.asarray(a, np.float32))
        rate = np.asarray(b, np.float32)
        acc = np.float32(0.0)
        for k in range(N_TERM):
            acc = np.float32(
                acc
                + np.float32(amp[k])
                * np.float32(np.exp(np.float32(-rate[k] * np.float32(r))))
            )
        return float(acc) / r

    radii = np.geomspace(0.05, 2.0, 25)
    errors = []
    for key in keys[:: ctx["stride"] * 4]:
        a, b = shipped[key]
        for r in radii:
            exact = float(energy(a, b, r, *key))
            if abs(exact) > 1e-12:
                errors.append(abs(f32(a, b, float(r), *key) / exact - 1.0))
    errors = np.array(errors)
    print(
        f"float32 evaluation error: median {np.median(errors):.2e}  "
        f"p99 {np.percentile(errors, 99):.2e}  max {errors.max():.2e}"
    )

    worst = 0.0
    for key in keys[:: ctx["stride"] * 4]:
        a, b = shipped[key]
        for r in np.geomspace(0.02, 1.0, 12):
            total = float(screening(a, b, r))
            clamped = float((a * np.exp(-np.minimum(b * r, 80.0))).sum())
            if total > 0:
                worst = max(worst, abs(clamped / total - 1.0))
    print(
        f"largest decay rate in the table: {max(b.max() for _, b in shipped.values()):.1f} 1/A"
    )
    print(f"relative change from the CPU exponent clamp at 80: {worst:.3e}")

    path = Path(ctx["table_paths"]["shipped"])
    print(f"table file: {path.stat().st_size / 1024:.1f} KiB")
    start = time.perf_counter()
    for _ in range(50):
        d = np.load(path, allow_pickle=False)
        _ = d["a"], d["b"], d["pairs"]
    print(f"load time: {(time.perf_counter() - start) / 50 * 1e3:.2f} ms")

    lut_a = np.zeros((119, 119, N_TERM))
    lut_b = np.zeros((119, 119, N_TERM))
    for (z1, z2), (a, b) in shipped.items():
        lut_a[z1, z2] = lut_a[z2, z1] = a
        lut_b[z1, z2] = lut_b[z2, z1] = b
    charges = np.arange(1, 119)
    for n_type in (118, 2):
        z = charges[:n_type]
        start = time.perf_counter()
        for _ in range(50):
            table = np.zeros((n_type + 1, n_type + 1, 8))
            table[:n_type, :n_type, :4] = (
                KE_EV_A
                * (z[:, None] * z[None, :])[:, :, None]
                * lut_a[z[:, None], z[None, :]]
            )
            table[:n_type, :n_type, 4:] = lut_b[z[:, None], z[None, :]]
            table = table.reshape(-1, 8).astype(np.float32)
        elapsed = (time.perf_counter() - start) / 50
        print(
            f"kernel table for {n_type} types: {elapsed * 1e3:.2f} ms, "
            f"{table.nbytes / 1024:.0f} KiB"
        )


SECTIONS = {
    "metric": section_metric,
    "window": section_window,
    "pathology": section_pathology,
    "erratum": section_erratum,
    "compare": section_compare,
    "terms": section_terms,
    "bne": section_bne,
    "zext": section_zext,
    "runtime": section_runtime,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="unpacked open-data package")
    parser.add_argument("--table", help="the shipped table .npz")
    parser.add_argument("--v4", help="unpacked v4 open-data package")
    parser.add_argument(
        "--alt",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="a further table to compare, as name=path",
    )
    parser.add_argument("--stride", type=int, default=1, help="score every n-th pair")
    parser.add_argument("sections", nargs="*", choices=[*SECTIONS, []], default=[])
    args = parser.parse_args(argv)

    data_dir = Path(args.data)
    table_paths = {}
    if args.table:
        table_paths["shipped"] = args.table
    for item in args.alt:
        name, _, path = item.partition("=")
        table_paths[name] = path
    ctx = {
        "data": data_dir,
        "dmol": load_dmol(data_dir),
        "published": load_published(data_dir / "nlh" / "nlh_coeffs.dat"),
        "v4": Path(args.v4) if args.v4 else None,
        "tables": {name: table_rows(Path(p)) for name, p in table_paths.items()},
        "table_paths": table_paths,
        "stride": max(args.stride, 1),
    }
    for name in args.sections or list(SECTIONS):
        print("=" * 88)
        print(f"--- {name} ---")
        SECTIONS[name](ctx)
        print()


if __name__ == "__main__":
    sys.exit(main())

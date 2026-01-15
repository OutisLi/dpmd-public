# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201
"""Compare staged release checkpoints on the force-spike criteria.

For every ``run:step`` the staged outputs under ``runs/<run>/`` are read: the 24-pair
isolated-pair survey (``survey_full_<step>``), the all-element homonuclear scan
(``homo_scan/ema_<step>``), the compressed-trimer probe (``trimer_probe.jsonl``), the
complete sub001 validation scan (``accuracy_sub001_<step>.npz``) and the held-out
spike/MD evaluation (``eval_<step>.json``). Missing outputs print as ``-``.

Pair readings follow section 4 of the design document: A1 original (|F| <= 0.05 eV/A
over 4-7 A), A1 revised (from 4.5 A: monotone within 5 meV and |F| <= 14.4/r^2), the
whole-range form proposed at closure (monotone from the principal well outward and
|F| <= 1.1 x 14.4/r^2 from max(1.5 contacts, 3 A) to 7 A), A2 (no hole outside 0.6
contacts, read as reference-flagged minima outside the exclusion), A3 (the H2 well),
A4 (no additional extremum beyond 1.5 A). The barrier beyond the principal well is the
largest energy above the dissociation limit outside the well, in eV.
"""

from __future__ import (
    annotations,
)

import argparse
import json
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import numpy as np
from ase.data import (
    atomic_numbers,
    covalent_radii,
)
from scipy.signal import (
    find_peaks,
)

ROOT = Path(__file__).resolve().parent / "runs"
COULOMB_UNIT = 14.399645
ABSENT = {"He", "Ne", "Ar", "Kr", "Xe", "Rn", "Po", "At", "Fr", "Ra"}


def whole_range(pair: dict[str, Any], r: np.ndarray, e: np.ndarray, f: np.ndarray) -> dict[str, float | bool]:
    """Whole-range tail rule of section 6.6 and the barrier beyond the principal well."""
    contact = pair["contact"]
    outward = r >= pair["principal_r"]
    minima, _ = find_peaks(-e[outward], prominence=0.005)
    maxima, _ = find_peaks(e[outward], prominence=0.005)
    window = r >= max(1.5 * contact, 3.0)
    ratio = np.abs(f[window]) / (1.1 * COULOMB_UNIT / r[window] ** 2)
    k = int(np.argmax(ratio))
    beyond = r > pair["principal_r"]
    b = int(np.argmax(e[beyond])) if beyond.any() else 0
    return {
        "extrema": int(len(minima) + len(maxima)),
        "ratio": float(ratio[k]),
        "ratio_r": float(r[window][k]),
        "pass": bool(len(minima) + len(maxima) == 0 and ratio[k] <= 1.0),
        "barrier": float(e[beyond][b]) if beyond.any() else 0.0,
        "barrier_r": float(r[beyond][b]) if beyond.any() else 0.0,
    }


def survey(run: Path, step: int) -> dict[str, Any] | None:
    path = run / f"survey_full_{step}.json"
    if not path.exists():
        return None
    pairs = json.loads(path.read_text())["pairs"]
    with np.load(run / f"survey_full_{step}.npz") as data:
        curves = {k: data[k] for k in data.files}
    for pair in pairs:
        prefix = pair["pair"].replace("-", "_")
        pair["whole"] = whole_range(
            pair, curves[f"{prefix}_r"], curves[f"{prefix}_energy"], curves[f"{prefix}_force"]
        )
    hydrogen = next(p for p in pairs if p["pair"] == "H-H")
    return {
        "pairs": pairs,
        "A1_original_fail": sum(not p["tail_pass"] for p in pairs),
        "A1_revised_fail": sum(not p["tail45_pass"] for p in pairs),
        "whole_range_fail": sum(not p["whole"]["pass"] for p in pairs),
        "A2_flagged_outside": sum(
            any(x["outside_06_contact"] for x in p["reference_flagged_minima"]) for p in pairs
        ),
        "inner_holes_info": sum(p["inner_hole"] for p in pairs),
        "A4_extra": sum(bool(p["additional_extrema_beyond_15A"]) for p in pairs),
        "barrier_over_50meV": sum(p["whole"]["barrier"] > 0.05 for p in pairs),
        "H2": f"{hydrogen['principal_energy']:.3f}@{hydrogen['principal_r']:.3f}",
        "max_tail_force": max(p["tail_max_force"] for p in pairs),
    }


def homo(run: Path, step: int) -> dict[str, Any] | None:
    path = run / "homo_scan" / f"ema_{step}.json"
    if not path.exists():
        return None
    pairs = json.loads(path.read_text())["pairs"]
    data = [p for p in pairs if p["pair"].split("-")[0] not in ABSENT]
    absent = [p for p in pairs if p["pair"].split("-")[0] in ABSENT]
    deepest = min(data, key=lambda p: p["principal_energy"])
    inner = min(data, key=lambda p: p["minimum_energy"])
    return {
        "n": len(pairs),
        "catastrophic_data": sum(p["principal_energy"] < -10.0 for p in data),
        "catastrophic_absent": sum(p["principal_energy"] < -10.0 for p in absent),
        "inner_holes_data": sum(p["inner_hole"] for p in data),
        "inner_deep_data": sum(p["minimum_energy"] < -10.0 and not p["principal_r"] == p["minimum_r"] for p in data),
        "revised_fail_data": sum(not p["tail45_pass"] for p in data),
        "extra_data": sum(bool(p["additional_extrema_beyond_15A"]) for p in data),
        "deepest_data": f"{deepest['pair']} {deepest['principal_energy']:.1f}@{deepest['principal_r']:.2f}",
        "inner_data": f"{inner['pair']} {inner['minimum_energy']:.0f}@{inner['minimum_r'] / inner['contact']:.2f}c",
    }


def trimer(run: Path, step: int) -> dict[str, Any] | None:
    path = run / "trimer_probe.jsonl"
    if not path.exists():
        return None
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    records = [x for x in records if x["label"].endswith(f"ema {step}")]
    if not records:
        return None
    record = records[-1]
    data = {k: v for k, v in record["pairs"].items() if k.split("-")[0] not in ABSENT}
    worst = max(data.items(), key=lambda kv: kv[1]["trimer_max_force"])
    return {
        "worst": f"{worst[0]} {worst[1]['trimer_max_force']:.0f}/{abs(worst[1]['pair_radial_force']):.0f}",
        "over_100": sum(v["trimer_max_force"] > 100.0 for v in data.values()),
        "n": len(data),
    }


def accuracy(run: Path, step: int) -> dict[str, Any] | None:
    prefix = run / f"accuracy_sub001_{step}"
    if not prefix.with_suffix(".done").exists():
        return None
    with np.load(prefix.with_suffix(".npz")) as data:
        w = data["natoms"]
        finite = np.isfinite(data["f_rmse"])
        rmse = float(np.sqrt(np.sum(w[finite] * data["f_rmse"][finite] ** 2) / w[finite].sum()))
        mae = float(np.sum(w[finite] * data["f_mae"][finite]) / w[finite].sum())
        comp = float(np.sum(w[finite] * data["f_component_mae"][finite]) / w[finite].sum())
        energy = float(np.nanmean(data["e_err_atom"]) * 1000)
        maxerr = data["f_maxerr"]
        worst = int(np.nanargmax(maxerr))
        return {
            "frames": int(len(w)),
            "nan": int((~finite).sum()),
            "e_mae": energy,
            "f_mae": mae,
            "f_comp_mae": comp,
            "f_rmse": rmse,
            "gt1": int((maxerr > 1).sum()),
            "gt5": int((maxerr > 5).sum()),
            "gt20": int((maxerr > 20).sum()),
            "gt100": int((maxerr > 100).sum()),
            "worst": f"{maxerr[worst]:.1f}@{int(data['idx'][worst])}",
        }


def evaluation(run: Path, step: int) -> dict[str, Any] | None:
    path = run / f"eval_{step}.json"
    return json.loads(path.read_text()) if path.exists() else None


def fmt(value: Any, spec: str = "") -> str:
    if value is None:
        return "-"
    return format(value, spec) if spec else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="run:step")
    parser.add_argument("--pairs", action="store_true", help="print the per-pair tables")
    parser.add_argument("--root", type=Path, default=ROOT, help="directory holding the staged runs")
    args = parser.parse_args()
    jobs = [(name.rsplit(":", 1)[0], int(name.rsplit(":", 1)[1])) for name in args.runs]
    width = 18
    columns = [f"{name}@{step}" for name, step in jobs]
    results = {}
    for name, step in jobs:
        run = args.root / name
        results[(name, step)] = {
            "survey": survey(run, step),
            "homo": homo(run, step),
            "trimer": trimer(run, step),
            "acc": accuracy(run, step),
            "eval": evaluation(run, step),
        }

    def row(label: str, getter) -> None:
        cells = []
        for key in results:
            try:
                cells.append(getter(results[key]))
            except (KeyError, TypeError):
                cells.append("-")
        print(f"{label:44s}" + "".join(f"{c:>{width}s}" for c in cells))

    print(f"{'reading':44s}" + "".join(f"{c[-width + 1:]:>{width}s}" for c in columns))
    print("-- isolated pairs (24), EMA weights")
    row("A1 original fail (|F|>0.05 in 4-7 A) /24", lambda x: fmt(x["survey"]["A1_original_fail"]))
    row("A1 revised fail (4.5 A: extrema, 14.4/r^2) /24", lambda x: fmt(x["survey"]["A1_revised_fail"]))
    row("whole-range fail (well->7 A, 1.1x Coulomb) /24", lambda x: fmt(x["survey"]["whole_range_fail"]))
    row("barrier beyond well > 50 meV /24", lambda x: fmt(x["survey"]["barrier_over_50meV"]))
    row("A4 additional extrema beyond 1.5 A /24", lambda x: fmt(x["survey"]["A4_extra"]))
    row("A2 deep minima outside 0.6 contacts /24", lambda x: fmt(x["survey"]["A2_flagged_outside"]))
    row("inner holes inside 0.6 contacts (info) /24", lambda x: fmt(x["survey"]["inner_holes_info"]))
    row("A3 H2 well E@r (eV, A)", lambda x: fmt(x["survey"]["H2"]))
    row("largest 4-7 A tail force (eV/A)", lambda x: fmt(x["survey"]["max_tail_force"], ".3f"))
    print("-- homonuclear scan (94 elements; 10 absent from OMat24 are informational)")
    row("wells < -10 eV outside 0.6 contacts, data /84", lambda x: fmt(x["homo"]["catastrophic_data"]))
    row("wells < -10 eV outside 0.6 contacts, absent /10", lambda x: fmt(x["homo"]["catastrophic_absent"]))
    row("inner holes inside 0.6 contacts, data /84 (info)", lambda x: fmt(x["homo"]["inner_holes_data"]))
    row("inner holes < -10 eV, data /84 (info)", lambda x: fmt(x["homo"]["inner_deep_data"]))
    row("deepest inner hole, data (info)", lambda x: fmt(x["homo"]["inner_data"]))
    row("A1 revised fail, elements with data /84", lambda x: fmt(x["homo"]["revised_fail_data"]))
    row("additional extrema, elements with data /84", lambda x: fmt(x["homo"]["extra_data"]))
    row("deepest well, elements with data", lambda x: fmt(x["homo"]["deepest_data"]))
    print("-- compressed trimer (0.45 contacts + third atom at 2 A; inside ZBL range, info)")
    row("worst data pair: trimer/pair-alone force (eV/A)", lambda x: fmt(x["trimer"]["worst"]))
    row("data pairs with trimer force > 100 eV/A /16", lambda x: fmt(x["trimer"]["over_100"]))
    print("-- accuracy on the complete sub001val set (21742 frames)")
    row("energy MAE (meV/atom)", lambda x: fmt(x["acc"]["e_mae"], ".2f"))
    row("force MAE, vector (eV/A)", lambda x: fmt(x["acc"]["f_mae"], ".4f"))
    row("force MAE, component (eV/A)", lambda x: fmt(x["acc"]["f_comp_mae"], ".4f"))
    row("force RMSE (eV/A)", lambda x: fmt(x["acc"]["f_rmse"], ".4f"))
    row("frames with max atom error > 1 eV/A", lambda x: fmt(x["acc"]["gt1"]))
    row("frames with max atom error > 5 eV/A", lambda x: fmt(x["acc"]["gt5"]))
    row("frames with max atom error > 20 eV/A", lambda x: fmt(x["acc"]["gt20"]))
    row("frames with max atom error > 100 eV/A", lambda x: fmt(x["acc"]["gt100"]))
    row("worst frame err@idx", lambda x: fmt(x["acc"]["worst"]))
    row("frames lost to OOM (NaN)", lambda x: fmt(x["acc"]["nan"]))
    print("-- held-out 600 frames: curvature, spike search, Langevin MD at 1500 K")
    row("curvature ratio > 1, frames", lambda x: fmt(x["eval"]["curv_frames_frac_flagged"], ".3f"))
    row("curvature ratio > 1, perturbed", lambda x: fmt(x["eval"]["curv_perturbed_frac_flagged"], ".3f"))
    row("adversarial 0.05 A: Fmax p90 (eV/A)", lambda x: fmt(x["eval"]["spike_adv005_fmax_p90"], ".1f"))
    row("adversarial 0.05 A: Fmax max (eV/A)", lambda x: fmt(x["eval"]["spike_adv005_fmax_max"], ".1f"))
    row("adversarial 0.05 A: frac Fmax > 50", lambda x: fmt(x["eval"]["spike_adv005_frac_gt50"], ".3f"))
    row("MD exploded fraction (60 runs)", lambda x: fmt(x["eval"]["md_frac_exploded"], ".3f"))
    row("MD Fmax p90 / max (eV/A)", lambda x: f"{x['eval']['md_fmax_p90']:.1f} / {x['eval']['md_fmax_max']:.1f}")
    row("MD energy drift p90 (eV/atom)", lambda x: fmt(x["eval"]["md_e_drift_p90"], ".3f"))

    if not args.pairs:
        return
    first = next(v["survey"] for v in results.values() if v["survey"])
    names = [p["pair"] for p in first["pairs"]]
    print("\n-- per pair: principal well E@r | barrier beyond the well (eV@A)")
    for pair in names:
        cells = []
        for key, val in results.items():
            s = val["survey"]
            p = next((q for q in s["pairs"] if q["pair"] == pair), None) if s else None
            cells.append("-" if p is None else f"{p['principal_energy']:+.2f}@{p['principal_r']:.2f} B{p['whole']['barrier']:.3f}@{p['whole']['barrier_r']:.1f}")
        print(f"{pair:8s}" + "".join(f"{c:>{width + 8}s}" for c in cells))
    print("\n-- per pair (inside 0.6 contacts, informational): lowest E@ratio | wall E(0.4c)-E(min), E(0.5c)-E(min) | attractive inside ratio")
    for pair in names:
        cells = []
        for key, val in results.items():
            s_ = val["survey"]
            p = next((q for q in s_["pairs"] if q["pair"] == pair), None) if s_ else None
            if p is None:
                cells.append("-")
                continue
            att = "-" if p["inner_attractive_to"] is None else f"{p['inner_attractive_to']:.2f}"
            cells.append(f"{p['minimum_energy']:+.1f}@{p['minimum_r'] / p['contact']:.2f}c w{p['wall_04']:.1f}/{p['wall_05']:.1f} a{att}")
        print(f"{pair:8s}" + "".join(f"{c:>{width + 14}s}" for c in cells))
    print("\n-- per pair: A1 revised max |F| in 4.5-7 A (eV/A@A, extrema) [pass/FAIL] | whole-range ratio to 1.1x Coulomb (@A, extrema) [pass/FAIL]")
    for pair in names:
        cells = []
        for key, val in results.items():
            s = val["survey"]
            p = next((q for q in s["pairs"] if q["pair"] == pair), None) if s else None
            if p is None:
                cells.append("-")
                continue
            w = p["whole"]
            cells.append(
                f"{p['tail45_max_force']:.2f}@{p['tail45_max_force_r']:.1f},{p['tail45_extrema']} {'ok' if p['tail45_pass'] else 'FAIL'} | "
                f"{w['ratio']:.2f}x@{w['ratio_r']:.1f},{w['extrema']} {'ok' if w['pass'] else 'FAIL'}"
            )
        print(f"{pair:8s}" + "".join(f"{c:>{width + 22}s}" for c in cells))


if __name__ == "__main__":
    main()

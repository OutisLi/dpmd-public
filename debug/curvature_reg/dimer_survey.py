# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201
"""Scan isolated pairs, retaining the curves and separating their principal well.

The 0.6-contact and absolute 1 Angstrom boundaries are reported separately.
Every resolved extremum is retained; one beyond 1.5 Angstrom is not automatically
unphysical, since the principal bond of many element pairs lies beyond it.
The additional-extremum count excludes only the principal well and is read
alongside that well's position and depth. Energy is measured relative to 7
Angstrom and the tail threshold is the absolute 0.05 eV/Angstrom of A1.
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

# Spectroscopic bond lengths and dissociation energies are diagnostic references.
# They are not training labels and are not assumed to equal the training DFT setup.
DIATOMIC = {
    "H-H": (0.741, 4.48),
    "O-O": (1.208, 5.12),
    "N-N": (1.098, 9.76),
    "C-C": (1.243, 6.21),
    "O-H": (0.970, 4.39),
    "Li-H": (1.595, 2.43),
    "Li-O": (1.69, 3.5),
    "Na-Cl": (2.361, 4.23),
    "Mg-O": (1.749, 2.6),
    "Al-Al": (2.70, 1.5),
    "Si-O": (1.510, 8.26),
    "Ca-O": (1.822, 4.1),
    "Ti-O": (1.620, 6.87),
    "Fe-O": (1.62, 4.2),
    "Cu-Cu": (2.220, 2.03),
    "Zn-S": (2.05, 2.1),
    "Ga-As": (2.53, 2.1),
    "Ag-Ag": (2.53, 1.65),
    "Au-Au": (2.472, 2.29),
    "Pb-H": (1.84, 1.6),
    "Cr-Cr": (1.679, 1.53),
    "Mo-Mo": (1.94, 4.5),
    "K-F": (2.171, 5.07),
    "Ni-Ni": (2.155, 2.0),
}


TAIL_START = 4.5
TAIL_PROMINENCE = 0.005
COULOMB_UNIT = 14.399645  # e^2 / (4 pi eps_0) in eV A


def tail_rule(r: np.ndarray, energy: np.ndarray, force: np.ndarray) -> dict[str, Any]:
    """Judge the far tail of an isolated pair on physical grounds (criterion A1, revised 9 September).

    Between ``TAIL_START`` and the last separation the energy must approach the
    dissociation limit monotonically -- no extremum with a prominence above
    ``TAIL_PROMINENCE`` -- and the force may nowhere exceed the force between
    two unit point charges, ``COULOMB_UNIT / r**2`` (0.71 eV/A at 4.5 A), the
    largest force two atoms can physically exert at that distance. A physical
    well's outer wall and an ionic pair's Coulomb tail pass; a slope toward a
    wrong limit, a barrier or a second well in the window fails.
    """
    window = r >= TAIL_START
    e = energy[window] - energy[-1]
    minima, _ = find_peaks(-e, prominence=TAIL_PROMINENCE)
    maxima, _ = find_peaks(e, prominence=TAIL_PROMINENCE)
    bound = COULOMB_UNIT / r[window] ** 2
    excess = np.abs(force[window]) - bound
    k = int(np.argmax(excess))
    monotone = len(minima) + len(maxima) == 0
    return {
        "tail45_monotone": bool(monotone),
        "tail45_extrema": int(len(minima) + len(maxima)),
        "tail45_max_force": float(np.abs(force[window]).max()),
        "tail45_max_force_r": float(r[window][int(np.argmax(np.abs(force[window])))]),
        "tail45_coulomb_excess": float(excess[k]),
        "tail45_coulomb_excess_r": float(r[window][k]),
        "tail45_pass": bool(monotone and excess[k] <= 0.0),
    }


def curve_summary(
    pair: str, contact: float, r: np.ndarray, energy: np.ndarray, force: np.ndarray
) -> dict[str, Any]:
    """Summarize one finite pair curve with energy zero at its final separation."""
    if not all(np.isfinite(value).all() for value in (r, energy, force)):
        raise ValueError(f"Non-finite isolated-pair curve for {pair}")
    e = energy - energy[-1]
    minima, minimum_properties = find_peaks(-e, prominence=1e-3)
    maxima, maximum_properties = find_peaks(e, prominence=1e-3)
    outside = r >= round(0.6 * contact, 12)
    inner = ~outside
    outer_indices = np.flatnonzero(outside)
    principal = int(outer_indices[np.argmin(e[outside])])
    tied_minima = minima[outside[minima] & (e[minima] == e[principal])]
    if len(tied_minima):
        principal = int(tied_minima[0])
    prominence = dict(zip(minima, minimum_properties["prominences"], strict=True))
    prominence.update(zip(maxima, maximum_properties["prominences"], strict=True))
    extrema = []
    for k in sorted(set(minima.tolist() + maxima.tolist())):
        extrema.append(
            {
                "kind": "minimum" if k in minima else "maximum",
                "r": float(r[k]),
                "ratio": float(r[k] / contact),
                "energy": float(e[k]),
                "prominence_eV": float(prominence[k]),
                "principal": k == principal,
                "outside_06_contact": bool(outside[k]),
                "outside_1A": bool(r[k] >= 1.0),
            }
        )
    reference = DIATOMIC.get(pair)
    flagged = []
    if reference is not None:
        bond, depth = reference
        flagged = [
            int(k)
            for k in minima
            if r[k] < contact
            and e[k] < -1.0
            and (r[k] < 0.85 * bond or e[k] < -(depth + 3.0))
        ]
    tail = (r >= 4.0) & (r <= 7.0)
    tail_indices = np.flatnonzero(tail)
    tail_index = int(tail_indices[np.argmax(np.abs(force[tail]))])
    attractive = np.flatnonzero(inner & (force < 0.0))
    deepest = int(np.argmin(e))
    extra = [x for x in extrema if x["r"] > 1.5 and not x["principal"]]
    extra_outer = [x for x in extrema if x["outside_06_contact"] and not x["principal"]]
    near_minima = [int(k) for k in minima if r[k] < 0.8 * contact]
    return {
        "pair": pair,
        "contact": contact,
        "energy_zero_r": float(r[-1]),
        "minimum_energy": float(e[deepest]),
        "minimum_r": float(r[deepest]),
        "principal_energy": float(e[principal]),
        "principal_r": float(r[principal]),
        "principal_at_inner_boundary": principal == int(outer_indices[0]),
        "inner_hole": bool(e[inner].min() < e[outside].min() - 0.05),
        "inner_attractive_to": float(r[attractive[-1]] / contact)
        if attractive.size
        else None,
        "wall_04": float(e[np.argmin(abs(r - 0.4 * contact))] - e.min()),
        "wall_05": float(e[np.argmin(abs(r - 0.5 * contact))] - e.min()),
        "near_minima": [
            {"energy": float(e[k]), "r": float(r[k]), "ratio": float(r[k] / contact)}
            for k in near_minima
        ],
        "reference_flagged_minima": [
            {
                "energy": float(e[k]),
                "r": float(r[k]),
                "outside_06_contact": bool(outside[k]),
                "outside_1A": bool(r[k] >= 1.0),
            }
            for k in flagged
        ],
        "tail_max_force": float(abs(force[tail_index])),
        "tail_peak_r": float(r[tail_index]),
        "tail_pass": bool(abs(force[tail_index]) <= 0.05),
        **tail_rule(r, energy, force),
        "extrema": extrema,
        "additional_extrema_beyond_15A": extra,
        "additional_extrema_outside_06contact": extra_outer,
    }


def print_pair(result: dict[str, Any]) -> None:
    """Print the principal well, reference flags, tail, and every resolved extremum."""
    pair = result["pair"]
    extras = result["additional_extrema_beyond_15A"]
    flags = result["reference_flagged_minima"]
    print(
        f"{pair:6s} contact {result['contact']:.2f} A | principal "
        f"{result['principal_energy']:+.6g} eV at {result['principal_r']:.4f} A | "
        f"inner hole {result['inner_hole']} | reference flags {len(flags)} "
        f"(outside 0.6c {sum(x['outside_06_contact'] for x in flags)}, "
        f"outside 1A {sum(x['outside_1A'] for x in flags)}) | "
        f"tail {result['tail_max_force']:.6g} eV/A at {result['tail_peak_r']:.4f} A | "
        f"additional extrema beyond 1.5A {len(extras)} | A1 revised {'pass' if result['tail45_pass'] else 'FAIL'} "
        f"(max {result['tail45_max_force']:.3f} eV/A at {result['tail45_max_force_r']:.2f} A, extrema {result['tail45_extrema']})"
    )
    for item in result["extrema"]:
        print(
            f"  {item['kind']} {item['energy']:+.6g} eV at {item['r']:.4f} A "
            f"({item['ratio']:.3f}c), principal={item['principal']}, "
            f"outside_0.6c={item['outside_06_contact']}, outside_1A={item['outside_1A']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("label")
    parser.add_argument("pairs", nargs="?", default=",".join(DIATOMIC))
    parser.add_argument(
        "--out",
        type=Path,
        help="Save all curves as NPZ and the readings as adjacent JSON",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--spacing", type=float, default=0.005, help="Maximum separation spacing in A"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.spacing <= 0 or args.batch_size <= 0:
        parser.error("spacing and batch size must be positive")
    if args.out is not None:
        for path in (args.out, args.out.with_suffix(".json")):
            if path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite the survey artifact: {path}"
                )
    import torch

    directory = Path(__file__).resolve().parent
    sys.path.insert(0, str(directory / "diagnose"))
    from heal_spike import (
        batch,
    )
    from repro_spike import (
        load_model,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, type_map = load_model(args.checkpoint, args.device)
    pairs = args.pairs.split(",")
    cell = np.eye(3) * 20.0
    curves: dict[str, np.ndarray] = {}
    results = []
    print(f"== {args.label}: {args.checkpoint}", flush=True)
    for pair in pairs:
        a, b = pair.split("-")
        if a not in type_map or b not in type_map:
            raise ValueError(f"Pair {pair} is absent from the checkpoint type map")
        contact = float(
            covalent_radii[atomic_numbers[a]] + covalent_radii[atomic_numbers[b]]
        )
        r = np.unique(
            np.round(
                np.concatenate(
                    [
                        np.arange(0.25, 1.6001, 0.01) * contact,
                        np.arange(0.25 * contact, 7.0, args.spacing),
                        [0.6 * contact, 1.0, 4.0, 6.0, 7.0],
                    ]
                ),
                12,
            )
        )
        positions = np.zeros((len(r), 2, 3))
        positions[:, 1, 0] = r
        types = np.array([type_map.index(a), type_map.index(b)])
        energies, forces = [], []
        for i in range(0, len(r), args.batch_size):
            energy, _, force = batch(
                model, positions[i : i + args.batch_size], cell, types, args.device
            )
            energies.append(energy.detach().double().cpu().numpy())
            forces.append(force[:, 1, 0].detach().double().cpu().numpy())
        energy, force = np.concatenate(energies), np.concatenate(forces)
        result = curve_summary(pair, contact, r, energy, force)
        results.append(result)
        print_pair(result)
        prefix = pair.replace("-", "_")
        curves.update(
            {
                f"{prefix}_r": r,
                f"{prefix}_energy": energy - energy[-1],
                f"{prefix}_force": force,
            }
        )
    tail_count = sum(not x["tail_pass"] for x in results)
    tail45_count = sum(not x["tail45_pass"] for x in results)
    extra_count = sum(bool(x["additional_extrema_beyond_15A"]) for x in results)
    print(
        f"summary {args.label}: {sum(x['inner_hole'] for x in results)} holes below 0.6 contacts "
        f"(informational), {tail_count} tails above the 0.05 eV/A of criterion A1, "
        f"{extra_count} pairs with additional extrema beyond 1.5 A, excluding the principal well "
        f"(energy zero at 7 A; maximum spacing {args.spacing:g} A, exact boundary samples); "
        f"A1 revised (from {TAIL_START:g} A: monotone within {TAIL_PROMINENCE * 1000:g} meV and |F| <= 14.4/r^2): "
        f"{tail45_count} failing",
        flush=True,
    )
    if args.out is not None:
        with args.out.open("xb") as stream:
            np.savez_compressed(stream, **curves)
        args.out.with_suffix(".json").write_text(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint.resolve()),
                    "label": args.label,
                    "pairs": results,
                    "tail_threshold": 0.05,
                    "extremum_resolution_eV": 0.001,
                    "extremum_measure": "topographic_prominence",
                    "maximum_spacing_A": args.spacing,
                    "batch_size": args.batch_size,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, T201, ANN001
"""
Reproduce the force spike observed in a high-pressure hydrogen trajectory.

The trajectory is 64 hydrogen atoms at 1.579 A^3/atom and 1000 K, i.e. about
260 GPa, run with a compiled ``pt2`` model under LAMMPS. The thermo log shows
an isolated event at step 17093 -- the pressure jumps by 34% and Pzz by 62%
for exactly one step while the potential energy moves only 8 meV/atom -- and
the run destabilizes four steps later.

A virial responds to a localized force error far more strongly than a total
energy does, which is the signature of a spike in a few atoms' forces. This
probe evaluates the model in the eager fp32 python path on the archived frames
and, more importantly, on interpolations between consecutive frames: a spike
that survives eager evaluation is in the model, and one that appears between
two frames localizes the geometry that triggers it.
"""

from __future__ import (
    annotations,
)

import argparse
import os
import sys
from pathlib import (
    Path,
)

import numpy as np
import torch


def read_xyz(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Read an extended-xyz frame with a diagonal or general lattice.

    Parameters
    ----------
    path : Path
        Frame file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[str]]
        Positions in Angstrom with shape (N, 3), the 3x3 cell in Angstrom, and
        the element symbols.
    """
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    comment = lines[1]
    start = comment.index('Lattice="') + len('Lattice="')
    cell = np.fromstring(comment[start : comment.index('"', start)], sep=" ").reshape(
        3, 3
    )
    species: list[str] = []
    pos = np.zeros((n, 3))
    for k, line in enumerate(lines[2 : 2 + n]):
        parts = line.split()
        species.append(parts[0])
        pos[k] = [float(v) for v in parts[1:4]]
    return pos, cell, species


def install_training_patches(ckpt: Path) -> None:
    """
    Install the structural monkey patches recorded next to a checkpoint.

    A run trained with ``train_refresh.py --mixing-loops``, ``--block-loops``,
    ``--stay-or-step`` or ``--post-add-norm`` records them in ``patches.json``
    in its run directory. The patched forward is part of the trained function,
    so every evaluation installs the same patches before the model is built.
    The file is looked up in the checkpoint's directory and its parent (the run
    directory holds ``ema_<step>.pt`` copies and the ``ckpt/`` subdirectory).
    """
    import json

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from attention_normalization import install as install_attention_normalization
    from edge_type_neighbors import install as install_edge_type_neighbors
    from fitting_normalization import install as install_fitting_normalization
    from isolated_reference import install as install_isolated_reference
    from message_envelope import install as install_message_envelope
    from outer_radial_gate import install as install_outer_radial_gate
    from pair_fade import install as install_pair_fade
    from plain_fitting import install as install_plain_fitting
    from radial_experiment import install as install_radial_experiment
    from seed_radial_gate import install as install_seed_radial_gate

    install_radial_experiment()
    install_message_envelope()
    install_attention_normalization()
    install_edge_type_neighbors()
    install_seed_radial_gate()
    install_outer_radial_gate()
    install_pair_fade()
    install_isolated_reference()
    install_fitting_normalization()
    install_plain_fitting()

    for directory in (ckpt.parent, ckpt.parent.parent):
        record = directory / "patches.json"
        if record.exists():
            break
    else:
        return
    patches = json.loads(record.read_text())
    if patches.get("stay_or_step") or patches.get("post_add_norm"):
        from sezm_attnres import install as install_attnres

        install_attnres(post_add_norm=bool(patches.get("post_add_norm")))
    keys, weights = (
        patches.get("attnres_keys", "l0"),
        patches.get("attnres_weights", "shared"),
    )
    if keys != "l0" or weights != "shared" or patches.get("attnres_value_norm"):
        from sezm_attnres import (
            install_variant,
        )

        install_variant(keys, weights, bool(patches.get("attnres_value_norm")))
    install_radial_experiment(
        gauss_inner=float(patches.get("gauss_inner", 0.0)),
        single_envelope=bool(patches.get("single_envelope")),
        fixed_radial=bool(patches.get("fixed_radial")),
        gauss_width=float(patches.get("gauss_width", 0.0)),
        gauss_scales=None
        if patches.get("gauss_scales") is None
        else tuple(patches["gauss_scales"]),
        gauss_floor=float(patches.get("gauss_floor", 0.0)),
    )
    install_message_envelope(bool(patches.get("output_envelope")))
    install_attention_normalization(bool(patches.get("separate_attention_mass")))
    install_edge_type_neighbors(bool(patches.get("edge_type_neighbors")))
    install_seed_radial_gate(bool(patches.get("seed_radial_gate")))
    install_outer_radial_gate(float(patches.get("outer_radial_gate", 0.0)))
    install_pair_fade(*patches.get("pair_fade", (0.0, 0.0)))
    install_isolated_reference(patches.get("isolated_reference") or None)
    install_fitting_normalization(bool(patches.get("fitting_rmsnorm")))
    install_plain_fitting(bool(patches.get("plain_fitting")))
    if patches.get("uma_envelope"):
        import sezm_uma_envelope

        sezm_uma_envelope.install()
    print(f"  training patches installed from {record}: {patches}")


def load_model(ckpt: Path, device: str):  # noqa: ANN201
    """
    Rebuild the model from a training checkpoint.

    Parameters
    ----------
    ckpt : Path
        Path to the checkpoint.
    device : str
        Torch device string.

    Returns
    -------
    tuple
        The model in eval mode and its type map.
    """
    from deepmd.pt.model.model import (
        get_model,
    )

    state = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    params = dict(state["_extra_state"]["model_params"])
    # Compilation is a deployment choice and would obscure an eager
    # reproduction; TF32 would add a second precision variable.
    params["use_compile"] = False
    params["enable_tf32"] = False
    install_training_patches(Path(ckpt))
    model = get_model(params).to(device)
    tensors = {
        k[len("model.Default.") :]: v
        for k, v in state.items()
        if k.startswith("model.Default.")
    }
    # A checkpoint trained under the ``dens`` loss carries its trained energy
    # in the DeNS fitting head; the ``ener`` head of such a checkpoint is the
    # untrained initialization. The model is rebuilt in the mode it was
    # trained in, read off the presence of the DeNS head's weights.
    if hasattr(model, "set_active_mode") and any(
        k.startswith("atomic_model.dens_fitting_net.") for k in tensors
    ):
        model.set_active_mode("dens")
    missing, unexpected = model.load_state_dict(tensors, strict=False)
    dropped = [k for k in missing if "buffer" not in k]
    if dropped:
        print(f"  note: {len(dropped)} missing keys, first: {dropped[:3]}")
    if unexpected:
        print(f"  note: {len(unexpected)} unexpected keys, first: {unexpected[:3]}")
    model.eval()
    keep = os.environ.get("CURV_EDGE_NORM_KEEP", "")
    if keep:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from train_curv import (
            neutralize_edge_norm_sites,
        )

        n = neutralize_edge_norm_sites(model, keep)
        print(f"[repro] edge-norm sites removed except {keep!r}: {n} modules")
    return model, params["type_map"]


def evaluate(
    model, pos: np.ndarray, cell: np.ndarray, atype: np.ndarray, device: str
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate energy, force and virial on one frame.

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    pos : np.ndarray
        Positions in Angstrom with shape (N, 3).
    cell : np.ndarray
        Cell matrix in Angstrom with shape (3, 3).
    atype : np.ndarray
        Element indices with shape (N,).
    device : str
        Torch device string.

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray]
        Total energy in eV, forces in eV/Angstrom with shape (N, 3), and the
        virial in eV with shape (3, 3).
    """
    c = torch.tensor(pos, dtype=torch.float64, device=device).reshape(1, -1, 3)
    b = torch.tensor(cell, dtype=torch.float64, device=device).reshape(1, 3, 3)
    t = torch.tensor(atype, dtype=torch.long, device=device).reshape(1, -1)
    out = model(c.to(torch.float32), t, box=b.to(torch.float32))
    e = float(out["energy"].reshape(-1)[0])
    f = out["force"].reshape(-1, 3).detach().cpu().numpy()
    v = out["virial"].reshape(3, 3).detach().cpu().numpy()
    return e, f, v


def main() -> None:
    """Evaluate the archived frames and interpolate across the spike."""
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
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--interp", type=int, default=0, help="samples between frames")
    ap.add_argument("--pair", type=str, default="17092,17093")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    print("loading model")
    model, type_map = load_model(args.ckpt, args.device)

    frames = sorted(args.root.glob("normal_*.xyz"))
    order = {"control": 0, "last": 1, "onset": 2}
    frames.sort(key=lambda p: (order.get(p.name.split("_")[1], 3), p.name))

    print(
        f"\n  {'frame':>34} {'E (eV)':>12} {'|F|max':>10} {'|F|rms':>10} {'tr(W)/3':>12}"
    )
    print("  " + "-" * 82)
    cache: dict[str, tuple] = {}
    for path in frames:
        pos, cell, species = read_xyz(path)
        atype = np.array([type_map.index(s) for s in species])
        e, f, v = evaluate(model, pos, cell, atype, args.device)
        cache[path.stem] = (pos, cell, atype)
        fmag = np.linalg.norm(f, axis=-1)
        print(
            f"  {path.stem:>34} {e:12.4f} {fmag.max():10.2f} "
            f"{np.sqrt((fmag**2).mean()):10.2f} {np.trace(v) / 3:12.1f}"
        )

    if args.interp <= 0:
        return

    a_tag, b_tag = args.pair.split(",")
    key_a = next(k for k in cache if k.endswith(f"s{a_tag}"))
    key_b = next(k for k in cache if k.endswith(f"s{b_tag}"))
    pos_a, cell, atype = cache[key_a]
    pos_b = cache[key_b][0]
    # Frames are consecutive MD steps, so the atoms move by well under the
    # cell size and a straight-line interpolation stays physical.
    print(f"\ninterpolating {key_a} -> {key_b}, {args.interp} samples")
    print(
        f"  {'t':>7} {'E (eV)':>12} {'|F|max':>10} {'atom':>5} {'tr(W)/3':>12} {'dE step':>11}"
    )
    print("  " + "-" * 62)
    prev_e = None
    for k in range(args.interp + 1):
        t = k / args.interp
        pos = (1.0 - t) * pos_a + t * pos_b
        e, f, v = evaluate(model, pos, cell, atype, args.device)
        fmag = np.linalg.norm(f, axis=-1)
        step = "" if prev_e is None else f"{e - prev_e:11.4f}"
        print(
            f"  {t:7.4f} {e:12.4f} {fmag.max():10.2f} {int(fmag.argmax()):5d} "
            f"{np.trace(v) / 3:12.1f} {step:>11}"
        )
        prev_e = e


if __name__ == "__main__":
    main()

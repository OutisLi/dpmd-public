# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, T201, ANN001
"""
Split the curvature of an isolated pair's energy curve into a head term and a
descriptor term.

The model is ``E(x) = sum_i G(h_i(x))``, with ``h_i`` the per-atom descriptor
output (the fitting network's input) and ``G`` the scalar fitting network.
Along the pair separation ``r`` of a two-atom frame in vacuum,

    d2E/dr2 = sum_i h_i'^T H_G(h_i) h_i'     (head term)
            + sum_i grad G(h_i) . h_i''      (descriptor term)

Both terms are evaluated by automatic differentiation and checked against the
second derivative of the model's own energy, itself checked against a central
finite difference of the radial force.

The descriptor term is obtained without ever forming ``h''`` explicitly: with
``g_i = grad G(h_i)`` held constant, ``d2/dr2 [ sum_i g_i . h_i(r) ] =
sum_i g_i . h_i''``.  The head term needs the full ``h_i'``, which is taken
one output channel at a time; every frame of the scan is an independent row of
one batch, so a single backward pass yields that channel's derivative at every
separation.

The run's energy path detaches the edge vectors before the force autograd, so
the differentiable scan enters ``SeZMModel.core_compute`` with
``atomic_output_only=True``: that branch skips the detach and returns the
per-atom energies with the graph to the separation intact.  Its energies are
identical to the public ``forward``.

Alongside the split, the script records what every RMS normalization inside the
descriptor divides by -- the degree-balanced mean square of its input, per node
-- so that a growing derivative can be compared with a collapsing denominator.
"""

from __future__ import (
    annotations,
)

import argparse
import sys
import time
from pathlib import (
    Path,
)
from typing import (
    ClassVar,
)

import numpy as np
import torch
from ase.data import (
    atomic_numbers,
    covalent_radii,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "diagnose"))
sys.path.insert(0, str(ROOT))

from repro_spike import (
    install_training_patches,
)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_model(ckpt: Path, precision: str = "float64"):  # noqa: ANN201
    """
    Rebuild a checkpoint's model at the requested precision on the CPU.

    The stored weights are float32; the network is instantiated at
    ``precision`` and ``load_state_dict`` casts them, so the whole forward runs
    in double precision without touching the trained function.

    Parameters
    ----------
    ckpt : Path
        Checkpoint path.
    precision : str
        Descriptor and fitting-net precision.

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
    params["use_compile"] = False
    params["enable_tf32"] = False
    params["descriptor"] = dict(params["descriptor"])
    params["descriptor"]["precision"] = precision
    params["descriptor"]["use_amp"] = False
    params["fitting_net"] = dict(params["fitting_net"])
    params["fitting_net"]["precision"] = precision
    install_training_patches(Path(ckpt))
    model = get_model(params).to("cpu")
    tensors = {
        k[len("model.Default.") :]: v
        for k, v in state.items()
        if k.startswith("model.Default.")
    }
    missing, unexpected = model.load_state_dict(tensors, strict=False)
    dropped = [k for k in missing if "buffer" not in k]
    if dropped or unexpected:
        print(f"  note: {len(dropped)} missing, {len(unexpected)} unexpected keys")
    model.eval()
    return model, params["type_map"]


# --------------------------------------------------------------------------
# Normalization denominators
# --------------------------------------------------------------------------
def equivariant_variance(norm, x: torch.Tensor) -> torch.Tensor:
    """
    Degree-balanced mean square that ``EquivariantRMSNorm`` divides by.

    The layer scales its input by ``rsqrt(v + eps)`` with ``v`` the value
    returned here, so its denominator is ``sqrt(v + eps)``.

    Parameters
    ----------
    norm : EquivariantRMSNorm
        The layer, read for its balance weights only.
    x : torch.Tensor
        Layer input with shape (N, D, F, C).

    Returns
    -------
    torch.Tensor
        Per-node, per-focus mean square with shape (N, F).
    """
    x = x.to(norm.balance_weight.dtype)
    x0 = x[:, :1]
    x0 = x0 - x0.mean(dim=-1, keepdim=True)
    v = x0.square().sum(dim=(1, 3)) * norm.balance_weight[0]
    xt = x[:, 1:]
    if xt.numel() > 0:
        v = v + torch.einsum("ndfc,d->nf", xt * xt, norm.balance_weight[1:])
    return v


def equivariant_replica(norm, x: torch.Tensor) -> torch.Tensor:
    """Re-evaluate ``EquivariantRMSNorm`` from the recorded variance, for checking."""
    x = x.to(norm.balance_weight.dtype)
    inv = torch.rsqrt(equivariant_variance(norm, x) + norm.eps)
    inv = inv.unsqueeze(1).unsqueeze(-1)
    x0 = x[:, :1]
    x0 = (x0 - x0.mean(dim=-1, keepdim=True)) * inv
    xt = x[:, 1:] * inv
    scale = torch.index_select(norm.adam_scale, 0, norm.expand_index).unsqueeze(0)
    x0 = x0 * scale[:, :1]
    x0 = x0 + norm.bias.reshape(1, 1, norm.n_focus, -1)
    if xt.numel() == 0:
        return x0
    return torch.cat([x0, xt * scale[:, 1:]], dim=1)


def readout_variance(norm, x: torch.Tensor) -> torch.Tensor:
    """
    Degree-balanced mean square that the read-out norm divides by.

    The read-out norm is the soft form: it scales by ``rsqrt(v + eps)`` with
    ``eps = 1``, so a node whose input mean square is far below one passes
    almost unchanged.

    Parameters
    ----------
    norm : ReadoutNorm
        The layer, read for its balance weights only.
    x : torch.Tensor
        Layer input with shape (N, D, 1, C).

    Returns
    -------
    torch.Tensor
        Per-node mean square with shape (N,).
    """
    x = x.to(norm.balance.dtype)
    x0 = x[:, :1]
    x0 = x0 - x0.mean(dim=-1, keepdim=True)
    xx = torch.cat([x0, x[:, 1:]], dim=1)
    return (xx * xx * norm.balance).sum(dim=(1, 3)).reshape(x.shape[0])


def readout_replica(norm, x: torch.Tensor) -> torch.Tensor:
    """Re-evaluate the read-out norm from the recorded variance, for checking."""
    x = x.to(norm.balance.dtype)
    x0 = x[:, :1]
    x0 = x0 - x0.mean(dim=-1, keepdim=True)
    xx = torch.cat([x0, x[:, 1:]], dim=1)
    v = readout_variance(norm, x).reshape(-1, 1, 1, 1)
    xx = xx * torch.rsqrt(v + norm.eps)
    scale = torch.index_select(norm.adam_scale, 0, norm.degree).unsqueeze(0)
    xx = xx * scale
    return torch.cat([xx[:, :1] + norm.bias.reshape(1, 1, 1, -1), xx[:, 1:]], dim=1)


class Probe:
    """
    Hooks that record the descriptor's internal quantities during one forward.

    Records, per node row (frames flattened as ``frame * nloc + atom``):
    the fitting network's input, every RMS norm's degree-balanced input mean
    square, and every interaction block's output tensor.
    """

    def __init__(self, model) -> None:
        self.desc = model.atomic_model.descriptor
        self.fit = model.atomic_model.fitting_net
        self.norms: list[tuple[str, torch.nn.Module, str]] = []
        for k, block in enumerate(self.desc.blocks):
            self._collect(f"b{k}.pre_so2", block.pre_so2_norm)
            self._collect(f"b{k}.post_so2", block.post_so2_norm)
            for j, norm in enumerate(block.pre_ffn_norms):
                self._collect(f"b{k}.pre_ffn{j}", norm)
            for j, norm in enumerate(block.post_ffn_norms):
                self._collect(f"b{k}.post_ffn{j}", norm)
        self._collect("readout", getattr(self.desc, "readout_norm", None))
        self.var: dict[str, torch.Tensor] = {}
        self.blk: dict[int, torch.Tensor] = {}
        self.h: torch.Tensor | None = None
        self.checked = False
        self._handles = []
        self._handles.append(
            self.fit.register_forward_pre_hook(
                lambda m, inp: self.__setattr__("h", inp[0])
            )
        )
        for name, norm, kind in self.norms:
            self._handles.append(
                norm.register_forward_hook(self._make_norm_hook(name, norm, kind))
            )
        for k, block in enumerate(self.desc.blocks):
            self._handles.append(block.register_forward_hook(self._make_block_hook(k)))

    def _collect(self, name: str, norm) -> None:
        if norm is None:
            return
        cls = type(norm).__name__
        if cls == "EquivariantRMSNorm":
            self.norms.append((name, norm, "equivariant"))
        elif cls == "ReadoutNorm":
            self.norms.append((name, norm, "readout"))

    def _make_norm_hook(self, name: str, norm, kind: str):  # noqa: ANN202
        def hook(mod, inp, out) -> None:
            x = inp[0]
            if kind == "equivariant":
                v = equivariant_variance(norm, x)[:, 0]
                replica = equivariant_replica(norm, x)
            else:
                v = readout_variance(norm, x)
                replica = readout_replica(norm, x)
            if not self.checked:
                err = float((replica - out).abs().max().detach())
                scale = float(out.abs().max().detach()) + 1e-30
                if err / scale > 1e-9:
                    raise RuntimeError(
                        f"recorded denominator of {name} does not reproduce its "
                        f"output: relative error {err / scale:.3e}"
                    )
            self.var[name] = v.detach()

        return hook

    def _make_block_hook(self, k: int):  # noqa: ANN202
        def hook(mod, inp, out) -> None:
            self.blk[k] = (out[0] if isinstance(out, tuple) else out).detach()

        return hook

    def close(self) -> None:
        """Remove every hook."""
        for handle in self._handles:
            handle.remove()
        self._handles = []


# --------------------------------------------------------------------------
# Scan
# --------------------------------------------------------------------------
def frames(r: torch.Tensor) -> torch.Tensor:
    """Two-atom frames along x with separation ``r``, shape (nf, 2, 3)."""
    z = torch.zeros_like(r)
    return torch.stack([torch.stack([z, z, z], -1), torch.stack([r, z, z], -1)], dim=1)


def energies(model, r: torch.Tensor, atype: torch.Tensor, box: torch.Tensor):  # noqa: ANN201
    """
    Per-atom energies of the pair frames, differentiable in ``r``.

    Returns the total energy per frame with shape (nf,).  The neighbour list is
    rebuilt inside, so its edge vectors carry the graph back to ``r``.
    """
    coord = frames(r)
    nf = r.shape[0]
    schema = model.build_neighbor_list(coord, atype, box.reshape(nf, 9))
    fit_ret = model.core_compute(
        schema.coord,
        schema.atype,
        schema.edge_index,
        schema.edge_vec,
        schema.edge_scatter_index,
        schema.edge_mask,
        atomic_output_only=True,
    )
    return fit_ret["energy"].reshape(nf, -1).sum(-1)


def dr_grad(
    y: torch.Tensor,
    r: torch.Tensor,
    grad_outputs: torch.Tensor | None = None,
    create_graph: bool = False,
) -> torch.Tensor:
    """
    Derivative with respect to the separations, zero where the graph misses them.

    Beyond the 6 A cutoff the pair has no edge at all, so the energy of those
    frames does not depend on ``r``; autograd then reports the leaf as unused.
    Those frames are flat by construction and their derivative is zero.

    Parameters
    ----------
    y : torch.Tensor
        Differentiated tensor.
    r : torch.Tensor
        Separations with shape (nf,).
    grad_outputs : torch.Tensor | None
        Seed for a non-scalar ``y``.
    create_graph : bool
        Whether to keep the graph of the derivative.

    Returns
    -------
    torch.Tensor
        Derivative with shape (nf,).
    """
    if not y.requires_grad:
        return torch.zeros_like(r)
    return torch.autograd.grad(
        y,
        r,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=True,
        materialize_grads=True,
    )[0]


def scan(model, type_map, pair: str, r_grid: np.ndarray, fd_step: float, chunk: int):  # noqa: ANN201
    """
    Run the full split on one element pair over a grid of separations.

    Parameters
    ----------
    model : torch.nn.Module
        The rebuilt model.
    type_map : list[str]
        The checkpoint's type map.
    pair : str
        Element pair such as ``"Cu-Cu"``.
    r_grid : np.ndarray
        Separations in Angstrom with shape (n,).
    fd_step : float
        Central-difference step in Angstrom for the second derivative of the
        energy and for the block-output derivatives.
    chunk : int
        Maximum number of frames per batch.

    Returns
    -------
    dict[str, np.ndarray]
        Every recorded curve, keyed by name.
    """
    a, b = pair.split("-")
    atype = torch.tensor([type_map.index(a), type_map.index(b)], dtype=torch.long)
    fitting = model.atomic_model.fitting_net
    probe = Probe(model)
    out: dict[str, list[np.ndarray]] = {}

    def push(key: str, value: np.ndarray) -> None:
        out.setdefault(key, []).append(value)

    for start in range(0, len(r_grid), chunk):
        rv = r_grid[start : start + chunk]
        nf = len(rv)
        at = atype.expand(nf, -1).contiguous()
        box = torch.eye(3, dtype=torch.float64).mul(20.0).expand(nf, 3, 3).contiguous()

        # === Energy, force and exact curvature ===
        r = torch.tensor(rv, dtype=torch.float64, requires_grad=True)
        e = energies(model, r, at, box)
        h = probe.h
        probe.checked = True
        var = {k: v.reshape(nf, -1).numpy() for k, v in probe.var.items()}
        blk0 = {k: v.reshape(nf, 2, -1).clone() for k, v in probe.blk.items()}
        de = dr_grad(e.sum(), r, create_graph=True)
        d2e = dr_grad(de.sum(), r)

        # === Head and descriptor terms ===
        hh = h.detach().clone().requires_grad_(True)
        y = fitting(hh, at)["energy"]
        grad_g = torch.autograd.grad(y.sum(), hh, create_graph=True)[0]

        n_ch = h.shape[1] * h.shape[2]
        hp = torch.zeros(nf, h.shape[1], h.shape[2], dtype=torch.float64)
        for idx in range(n_ch):
            seed = torch.zeros_like(h)
            seed.reshape(nf, -1)[:, idx] = 1.0
            hp.reshape(nf, -1)[:, idx] = dr_grad(h, r, grad_outputs=seed)

        hvp = torch.autograd.grad((grad_g * hp).sum(), hh, retain_graph=False)[0]
        head = (hp * hvp).sum(dim=(1, 2))

        s = (grad_g.detach() * h).sum()
        ds = dr_grad(s, r, create_graph=True)
        d2s = dr_grad(ds.sum(), r)

        # === Finite-difference reference and block-output derivatives ===
        fd: dict = {}
        for sign in (1.0, -1.0):
            tag = "+1" if sign > 0 else "-1"
            rs = torch.tensor(rv + sign * fd_step, dtype=torch.float64)
            rs.requires_grad_(True)
            es = energies(model, rs, at, box)
            fd["de" + tag] = dr_grad(es.sum(), rs).detach()
            fd["blk" + tag] = {
                k: v.reshape(nf, 2, -1).clone() for k, v in probe.blk.items()
            }
            fd["hdes" + tag] = probe.h.reshape(nf, 2, -1).detach().clone()
        d2e_fd = (fd["de+1"] - fd["de-1"]) / (2.0 * fd_step)
        hp_fd = (fd["hdes+1"] - fd["hdes-1"]) / (2.0 * fd_step)

        push("r", rv)
        push("e", e.detach().numpy())
        push("f", (-de).detach().numpy())
        push("d2e", d2e.detach().numpy())
        push("d2e_fd", d2e_fd.numpy())
        push("head", head.detach().numpy())
        push("desc", d2s.detach().numpy())
        push("hp_norm", hp.norm(dim=-1).detach().numpy())
        push("hp_fd_err", (hp - hp_fd).norm(dim=-1).numpy())
        push("h_norm", h.norm(dim=-1).detach().numpy())
        push("gradg_norm", grad_g.norm(dim=-1).detach().numpy())
        for k in blk0:
            d = (fd["blk+1"][k] - fd["blk-1"][k]) / (2.0 * fd_step)
            push(f"blk{k}_dnorm", d.norm(dim=-1).numpy())
            push(f"blk{k}_norm", blk0[k].norm(dim=-1).numpy())
        for k, v in var.items():
            push(f"var.{k}", v)
    probe.close()
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def hard_norm_names(data: dict[str, np.ndarray]) -> list[str]:
    """Names of the hard (eps 1e-5) norms recorded in a scan."""
    return [
        k[4:]
        for k in data
        if k.startswith("var.") and ("pre_so2" in k or "pre_ffn" in k)
    ]


def pick_indices(r: np.ndarray, picks: np.ndarray) -> list[int]:
    """Grid indices nearest to the requested separations, without repeats."""
    seen: list[int] = []
    for r_pick in picks:
        i = int(np.argmin(np.abs(r - r_pick)))
        if i not in seen:
            seen.append(i)
    return sorted(seen)


def table_rows(data: dict[str, np.ndarray], contact: float, picks: np.ndarray):  # noqa: ANN201
    """Assemble the printed table at the requested separations."""
    r = data["r"]
    hard = hard_norm_names(data)
    rows = []
    for i in pick_indices(r, picks):
        hv = {n: data[f"var.{n}"][i].min() for n in hard}
        worst = min(hv, key=hv.get)
        rows.append(
            {
                "i": i,
                "r": r[i],
                "ratio": r[i] / contact,
                "e": data["e"][i],
                "f": data["f"][i],
                "d2e_fd": data["d2e_fd"][i],
                "d2e": data["d2e"][i],
                "head": data["head"][i],
                "desc": data["desc"][i],
                "hp0": data["hp_norm"][i, 0],
                "hp1": data["hp_norm"][i, 1],
                "hard_rms": np.sqrt(hv[worst]),
                "hard_which": worst,
                "ro_rms": np.sqrt(data["var.readout"][i].min())
                if "var.readout" in data
                else np.nan,
            }
        )
    return rows


def print_table(pair: str, contact: float, rows: list[dict]) -> None:
    """Print one pair's compact table."""
    print(f"\n== {pair}  (covalent contact {contact:.3f} A)")
    print(
        f"{'r(A)':>7} {'c':>6} {'E-E7':>10} {'F_r':>11} {'d2E(FD)':>12} "
        f"{'d2E(AD)':>12} {'head':>12} {'desc':>12} {'|h1p|':>10} {'|h2p|':>10} "
        f"{'hardRMS':>9} {'which':>12} {'roRMS':>9}"
    )
    for row in rows:
        print(
            f"{row['r']:7.3f} {row['ratio']:6.3f} {row['e']:10.3f} {row['f']:11.3f} "
            f"{row['d2e_fd']:12.3f} {row['d2e']:12.3f} {row['head']:12.3f} "
            f"{row['desc']:12.3f} {row['hp0']:10.3f} {row['hp1']:10.3f} "
            f"{row['hard_rms']:9.4f} {row['hard_which']:>12} {row['ro_rms']:9.4f}"
        )


def print_norm_table(data: dict[str, np.ndarray], rows: list[dict]) -> None:
    """Print the per-norm denominators and the per-block derivative growth."""
    names = sorted(k[4:] for k in data if k.startswith("var."))
    blocks = sorted(
        int(k[3 : k.index("_")])
        for k in data
        if k.startswith("blk") and k.endswith("_norm")
    )
    header = f"{'r(A)':>7}"
    for name in names:
        header += f" {name:>13}"
    for k in blocks:
        header += f" {'|x' + str(k) + '|':>10} {'|dx' + str(k) + '/dr|':>12}"
    header += f" {'|h|':>10} {'|dh/dr|':>10}"
    print(
        "\n   RMS the norms divide by (sqrt of the degree-balanced mean square, "
        "minimum over the two atoms); block output norms and their d/dr"
    )
    print(header)
    for row in rows:
        i = row["i"]
        line = f"{row['r']:7.3f}"
        for name in names:
            line += f" {np.sqrt(data['var.' + name][i].min()):13.5f}"
        for k in blocks:
            line += (
                f" {data['blk' + str(k) + '_norm'][i].max():10.2f}"
                f" {data['blk' + str(k) + '_dnorm'][i].max():12.2f}"
            )
        line += f" {data['h_norm'][i].max():10.3f} {data['hp_norm'][i].max():10.2f}"
        print(line)


def default_picks(r: np.ndarray, e: np.ndarray, contact: float) -> np.ndarray:
    """Twelve representative separations: the well, its flanks, the wall, the tail."""
    inside = r < 1.6 * contact
    i_min = int(np.argmin(np.where(inside, e, np.inf)))
    r_min = r[i_min]
    picks = [
        r.min(),
        0.35 * contact,
        0.5 * contact,
        max(r.min(), 0.75 * r_min),
        max(r.min(), 0.9 * r_min),
        r_min,
        1.1 * r_min,
        1.3 * r_min,
        1.6 * contact,
        4.0,
        4.5,
        5.0,
        6.0,
    ]
    picks = [p for p in picks if r.min() - 1e-9 <= p <= r.max() + 1e-9]
    return np.unique(np.round(picks, 4))


def scalar_variance(norm, x: torch.Tensor) -> torch.Tensor:
    """Mean square that ``ScalarRMSNorm`` divides by, flattened over its rows."""
    return x.to(norm.adam_scale.dtype).square().mean(dim=-1).reshape(-1)


def reduced_variance(norm, x: torch.Tensor) -> torch.Tensor:
    """Degree-balanced mean square that ``ReducedEquivariantRMSNorm`` divides by."""
    x = x.to(norm.balance_weight.dtype)
    x0 = x[:, :, :1, :]
    x0 = x0 - x0.mean(dim=-1, keepdim=True)
    v = x0.square().sum(dim=(2, 3)) * norm.balance_weight[0]
    xt = x[:, :, 1:, :]
    if xt.numel() > 0:
        v = v + torch.einsum("efdc,d->ef", xt * xt, norm.balance_weight[1:])
    return v.reshape(-1)


class AllNormProbe:
    """
    Record the input mean square of every RMS normalization in the descriptor.

    Rows are nodes for the node-wise norms and edges for the norms inside the
    SO(2) convolution and the edge FiLM.  A padded edge carries an all-zero
    input, so rows whose mean square is exactly zero are dropped before the
    quantiles are taken.
    """

    KINDS: ClassVar[dict] = {
        "ScalarRMSNorm": scalar_variance,
        "EquivariantRMSNorm": lambda n, x: equivariant_variance(n, x).reshape(-1),
        "ReducedEquivariantRMSNorm": reduced_variance,
        "ReadoutNorm": lambda n, x: readout_variance(n, x).reshape(-1),
    }

    def __init__(self, model) -> None:
        self.stats: dict[str, tuple[float, float, float, int, int]] = {}
        self.eps: dict[str, float] = {}
        self._handles = []
        for name, module in model.atomic_model.descriptor.named_modules():
            fn = self.KINDS.get(type(module).__name__)
            if fn is None:
                continue
            self.eps[name] = float(module.eps)
            self._handles.append(
                module.register_forward_hook(self._make_hook(name, module, fn))
            )

    def _make_hook(self, name: str, module, fn):  # noqa: ANN202
        def hook(mod, inp, out) -> None:
            v = fn(module, inp[0]).detach()
            nz = v[v > 0]
            if nz.numel() == 0:
                nz = v
            self.stats[name] = (
                float(nz.min()),
                float(nz.median()),
                float(nz.max()),
                int(nz.numel()),
                int(v.numel()),
            )

        return hook

    def close(self) -> None:
        """Remove every hook."""
        for handle in self._handles:
            handle.remove()
        self._handles = []


def norm_sweep(model, type_map, pair: str, r_picks: np.ndarray) -> None:
    """
    Print the input RMS of every normalization, one separation at a time.

    One frame per forward keeps every recorded row inside the frame being
    scanned, which the batched scan cannot guarantee for the per-edge norms.
    """
    a, b = pair.split("-")
    atype = torch.tensor([[type_map.index(a), type_map.index(b)]], dtype=torch.long)
    box = torch.eye(3, dtype=torch.float64).mul(20.0).reshape(1, 3, 3)
    probe = AllNormProbe(model)
    rows = []
    for r_pick in r_picks:
        r = torch.tensor([float(r_pick)], dtype=torch.float64)
        with torch.no_grad():
            energies(model, r, atype, box)
        rows.append((float(r_pick), dict(probe.stats)))
    probe.close()
    names = sorted(rows[0][1])
    print(
        "\n   Smallest input RMS of every normalization (sqrt of the mean square "
        "it divides by, minimum over non-empty rows); the denominator is "
        "sqrt(RMS^2 + eps)"
    )
    header = f"{'r(A)':>7}"
    for name in names:
        header += f" {name.split('.')[-1][:11] + '@' + name[:6]:>19}"
    print("   eps: " + ", ".join(f"{n}={probe.eps[n]:g}" for n in names))
    print(header)
    for r_pick, stat in rows:
        line = f"{r_pick:7.3f}"
        for name in names:
            line += f" {np.sqrt(stat[name][0]):19.6f}"
        print(line)


def main() -> None:
    """Run the curvature split for every requested pair of one checkpoint."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs", type=str, required=True)
    ap.add_argument(
        "--grid",
        choices=("contact", "angstrom"),
        default="contact",
        help="contact: 0.25-1.6 covalent contacts at 0.01, then to rmax at 0.05 A; "
        "angstrom: rmin-rmax at drmin",
    )
    ap.add_argument("--rmin", type=float, default=0.4)
    ap.add_argument("--rmax", type=float, default=7.0)
    ap.add_argument("--dr", type=float, default=0.01)
    ap.add_argument("--fd-step", type=float, default=1.0e-3)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--picks", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(0)
    model, type_map = build_model(args.ckpt)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"checkpoint {args.ckpt}")
    print(f"output     {args.out}")
    print(f"precision  float64, CPU, FD step {args.fd_step} A")

    for pair in args.pairs.split(","):
        a, b = pair.split("-")
        if a not in type_map or b not in type_map:
            print(f"{pair}: not in type map")
            continue
        contact = covalent_radii[atomic_numbers[a]] + covalent_radii[atomic_numbers[b]]
        if args.grid == "contact":
            inner = np.arange(0.25, 1.6001, 0.01) * contact
            outer = np.arange(1.6 * contact + 0.05, args.rmax + 1e-9, 0.05)
            r_grid = np.concatenate([inner, outer])
        else:
            r_grid = np.arange(args.rmin, args.rmax + 1e-9, args.dr)
        # The energy zero is the pair at ``rmax``; beyond the 6 A cutoff the two
        # atoms no longer interact, so this is the separated-atom limit.
        r_grid = np.unique(np.round(np.append(r_grid, args.rmax), 10))
        t0 = time.time()
        data = scan(model, type_map, pair, r_grid, args.fd_step, args.chunk)
        data["e"] = data["e"] - data["e"][-1]
        span = time.time() - t0

        total = data["head"] + data["desc"]
        ref = data["d2e"]
        denom = np.maximum(np.abs(ref), 1.0)
        mismatch = np.abs(total - ref) / denom
        fd_gap = np.abs(data["d2e_fd"] - ref) / denom
        print(
            f"\n{pair}: {len(r_grid)} points in {span:.1f} s | "
            f"head+desc vs autograd d2E: max rel {mismatch.max():.2e}, "
            f"median {np.median(mismatch):.2e} | "
            f"FD vs autograd d2E: max rel {fd_gap.max():.2e}, "
            f"median {np.median(fd_gap):.2e} | "
            f"autograd h' vs FD h': max abs {data['hp_fd_err'].max():.2e}"
        )
        picks = (
            np.array([float(v) for v in args.picks.split(",")])
            if args.picks
            else default_picks(data["r"], data["e"], contact)
        )
        rows = table_rows(data, contact, picks)
        print_table(pair, contact, rows)
        print_norm_table(data, rows)
        norm_sweep(model, type_map, pair, np.array([row["r"] for row in rows]))

        tag = pair.replace("-", "")
        np.savez_compressed(
            args.out / f"curvature_split_{tag}.npz", contact=contact, **data
        )
        with (args.out / f"curvature_split_{tag}.csv").open("w") as fh:
            keys = list(rows[0].keys())
            fh.write(",".join(keys) + "\n")
            for row in rows:
                fh.write(
                    ",".join(
                        f"{row[k]:.6g}" if isinstance(row[k], float) else str(row[k])
                        for k in keys
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: LGPL-3.0-or-later
"""Pin the isolated-atom energy of every tabulated element to a reference value.

The atomic energy of the PT atomic model is ``E_i = b(t_i) + f(x_i)``, with
``b`` the per-element output bias fitted by least squares on the training data
and ``f`` the fitting network on the descriptor ``x_i`` of atom ``i``. Neither
term knows the energy of an atom without neighbours: ``b`` is an average over
the atoms of the data, and ``f(x_iso, t)``, the network on the descriptor of an
isolated atom of type ``t``, is an untrained constant. The isolated-atom limit
of the model is therefore arbitrary, and every extrapolation toward low
coordination (isolated pairs, clusters, surfaces) starts from it.

With this patch the bias of a tabulated element holds its isolated-atom energy
``E0(t)`` and the fitting output is measured from its own isolated value,

    E_i = E0(t_i) + f(x_i) - f(x_iso, t_i),

so that an atom without neighbours has the energy ``E0`` exactly, at every step
of training, and the network learns the change that neighbours bring. The
isolated values ``f(x_iso, t)`` are evaluated in the same forward pass from one
atom of every type with an empty edge list; they depend on the parameters (the
gradient flows through them) and not on the coordinates, so forces and virials
are unchanged. The TF backend's ``atom_ener`` option applies the same rule with
a zero descriptor as the isolated input; MACE reaches the same limit with
bias-free readouts on features that vanish without neighbours. Elements
without a tabulated value keep the fitted bias and receive no subtraction.

Bias adjustment at fine-tuning (``change-by-statistic``) shifts the stored
bias by the least-squares residual of the new data, so ``E0`` follows the
energy zero of the new data; only the from-scratch statistic
(``set-by-statistic``) is replaced by the table.
"""

from __future__ import (
    annotations,
)

import json
from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    import torch

_table: dict[str, float] = {}
_installed = False


def install(path: str | None = None) -> None:
    """Load the reference table ``{symbol: energy in eV}`` and patch the SeZM atomic model."""
    global _table, _installed
    _table = {} if not path else {k: float(v) for k, v in json.load(open(path)).items()}
    if _installed or not _table:
        return
    from deepmd.pt.model.atomic_model.base_atomic_model import (
        BaseAtomicModel,
    )
    from deepmd.pt.model.atomic_model.sezm_atomic_model import (
        SeZMAtomicModel,
    )

    original_change = BaseAtomicModel.change_out_bias
    original_apply = SeZMAtomicModel.apply_out_stat

    def change_out_bias(
        self: Any,
        sample_merged: Any,
        stat_file_path: Any = None,
        bias_adjust_mode: str = "change-by-statistic",
    ) -> None:
        original_change(self, sample_merged, stat_file_path, bias_adjust_mode)
        if bias_adjust_mode == "set-by-statistic":
            assign_reference(self)

    def apply_out_stat(self: Any, ret: dict, atype: torch.Tensor) -> dict:
        ret = original_apply(self, ret, atype)
        if "energy" in ret and any(symbol in _table for symbol in self.get_type_map()):
            correction = isolated_output(self, ret["energy"].dtype)  # (ntypes,)
            ret["energy"] = ret["energy"] - correction[atype][..., None]
        return ret

    BaseAtomicModel.change_out_bias = change_out_bias
    SeZMAtomicModel.apply_out_stat = apply_out_stat
    _installed = True


def reference_vector(
    model: Any, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(E0, pinned)`` over the model's type map; ``E0`` is zero where unpinned."""
    import torch

    type_map = model.get_type_map()
    e0 = torch.tensor(
        [_table.get(symbol, 0.0) for symbol in type_map], dtype=dtype, device=device
    )
    pinned = torch.tensor(
        [symbol in _table for symbol in type_map], dtype=torch.bool, device=device
    )
    return e0, pinned


def assign_reference(model: Any) -> None:
    """Overwrite the energy bias of every tabulated element with its reference energy."""
    import torch

    bias = model.out_bias  # (n_out, ntypes, max_out_size)
    idx = model._get_bias_index("energy")
    e0, pinned = reference_vector(model, bias.dtype, bias.device)
    with torch.no_grad():
        bias[idx, :, 0] = torch.where(pinned, e0, bias[idx, :, 0])


def isolated_output(model: Any, dtype: torch.dtype) -> torch.Tensor:
    """Return the fitting output ``f(x_iso, t)`` of every type, zero where the type is not pinned.

    One atom of every type is passed through the descriptor without any edge
    and through the fitting network, exactly as an atom without neighbours is
    evaluated by the model; the output is differentiable with respect to the
    parameters. Two auxiliary atoms of type 0 joined by a single edge are
    appended to the system because several kernels of the eager path reject
    an empty edge list; they share no edge with the isolated atoms, so they
    cannot influence them, and their outputs are discarded.
    """
    import torch

    device = model.out_bias.device
    ntypes = len(model.get_type_map())
    edge_dtype = model.descriptor.compute_dtype
    n_nodes = ntypes + 2
    atype = torch.cat(
        [
            torch.arange(ntypes, device=device),
            torch.zeros(2, dtype=torch.long, device=device),
        ]
    )[None, :]  # (1, n_nodes)
    edge_vec = torch.zeros((1, 3), dtype=edge_dtype, device=device)
    edge_vec[0, 0] = 0.5 * model.descriptor.get_rcut()
    descriptor, _ = model.descriptor.forward_with_edges(
        extended_coord=torch.zeros((1, n_nodes, 3), dtype=edge_dtype, device=device),
        extended_atype=atype,
        edge_index=torch.tensor(
            [[ntypes], [ntypes + 1]], dtype=torch.long, device=device
        ),
        edge_vec=edge_vec,
        edge_mask=torch.ones((1,), dtype=torch.bool, device=device),
    )
    bare = (
        model.fitting_net(descriptor, atype)["energy"]
        .reshape(n_nodes)[:ntypes]
        .to(dtype)
    )
    _, pinned = reference_vector(model, dtype, device)
    return torch.where(pinned, bare, torch.zeros_like(bare))


def table() -> dict[str, float]:
    return dict(_table)

# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared helpers: build the Neo model from a plain-run checkpoint on CPU, the run's loss, needle batches."""

from __future__ import (
    annotations,
)

import json
import logging
import os
import sys
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

if TYPE_CHECKING:
    import torch

    from deepmd.pt.loss.ener import (
        EnergyStdLoss,
    )

sys.path.insert(0, "/nas/outisli/Software/deepmd-kit")

RUN = "/nas/outisli/Software/deepmd-kit/debug/curvature_reg/runs/neo_plain_2gpu_log"

CKPT_RUN = os.environ.get("CKPT_RUN", RUN)
log = logging.getLogger(__name__)


def build(step: int, amp: bool = False) -> tuple[torch.nn.Module, dict[str, Any]]:
    import torch

    from deepmd.pt.model.model import (
        get_model,
    )

    # The run's structural patches (patches.json) are part of the trained function; the shared
    # loader of the evaluation chain installs them before the model is built.
    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnose")
    )
    from repro_spike import (
        install_training_patches,
    )

    path = f"{CKPT_RUN}/ckpt/model.ckpt-{step}.pt"
    install_training_patches(Path(path))
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck["model"]
    params = dict(state["_extra_state"]["model_params"])
    params["use_compile"] = False
    params["enable_tf32"] = False
    params["descriptor"] = dict(params["descriptor"])
    params["descriptor"]["use_amp"] = amp
    model = get_model(params)
    tensors = {
        k[len("model.Default.") :]: v
        for k, v in state.items()
        if k.startswith("model.Default.")
    }
    missing, unexpected = model.load_state_dict(tensors, strict=False)
    dropped = [k for k in missing if "buffer" not in k]
    if dropped or unexpected:
        log.warning(
            "Checkpoint keys: missing %s, unexpected %s", dropped[:5], unexpected[:5]
        )
    model.eval()
    return model, ck


def make_loss() -> EnergyStdLoss:
    from deepmd.pt.loss.ener import (
        EnergyStdLoss,
    )

    cfg = json.load(open(f"{RUN}/input.json"))
    lp = dict(cfg["loss"])
    lp.pop("type", None)
    lp["starter_learning_rate"] = cfg["learning_rate"]["start_lr"]
    return EnergyStdLoss(**lp)


def load_needle(
    name: str,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor | float],
    np.ndarray,
    np.lib.npyio.NpzFile,
]:
    import torch

    d = np.load(f"{RUN}/{name}")
    nf, nloc = d["atype"].shape
    inp = {
        "coord": torch.tensor(d["coord"], dtype=torch.float32).reshape(nf, nloc, 3),
        "atype": torch.tensor(d["atype"], dtype=torch.long),
        "box": torch.tensor(d["box"], dtype=torch.float32).reshape(nf, 9),
    }
    lab = {
        "energy": torch.tensor(d["label_energy"], dtype=torch.float32).reshape(nf, 1),
        "force": torch.tensor(d["label_force"], dtype=torch.float32).reshape(
            nf, nloc, 3
        ),
        "virial": torch.tensor(d["label_virial"], dtype=torch.float32).reshape(nf, 9),
        "find_energy": 1.0,
        "find_force": 1.0,
        "find_virial": 1.0,
    }
    err = d["per_atom_error"].reshape(nf, nloc)
    return inp, lab, err, d


def subset(
    inp: dict[str, torch.Tensor],
    lab: dict[str, torch.Tensor | float],
    idx: np.ndarray | list[int] | torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | float]]:
    import torch

    idx = torch.as_tensor(idx)
    return (
        {k: v[idx] for k, v in inp.items()},
        {k: (v[idx] if isinstance(v, torch.Tensor) else v) for k, v in lab.items()},
    )


def batch_loss(
    loss_fn: EnergyStdLoss,
    model: torch.nn.Module,
    inp: dict[str, torch.Tensor],
    lab: dict[str, torch.Tensor | float],
    lr: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
    nloc = inp["atype"].shape[1]
    pred, loss, more = loss_fn(inp, model, lab, nloc, lr)
    return pred, loss, more


def flat_grad(model: torch.nn.Module) -> torch.Tensor:
    import torch

    return torch.cat(
        [
            (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).double()
            for p in model.parameters()
        ]
    )


def forces(
    model: torch.nn.Module, inp: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    # The model differentiates its energy inside the forward; no ``no_grad`` here.
    out = model(inp["coord"], inp["atype"], box=inp["box"])
    return out["force"].detach().double(), out["energy"].detach().double()

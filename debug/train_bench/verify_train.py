#!/usr/bin/env python
# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, T201, TC002
"""
End-to-end gradient equivalence between the dense and the fused training paths.

The operator-level checks in ``gradcheck.py`` prove each fused kernel against
its own reference. This script closes the remaining gap: it runs a whole
training step of a real model twice, once through the dense reference path and
once through the fused kernels, and compares the loss together with the gradient
of every parameter. Both runs share one model instance, one set of weights and
one batch, and differ only in the dispatch level, so any discrepancy is
attributable to the fused path alone.

The comparison defaults to float32 because bfloat16 rounding masks the
differences that matter; ``--amp`` reproduces the real training dtype and needs
a correspondingly loose tolerance.

Examples
--------
Compare level 1 against the dense path in float32::

    python debug/train_bench/verify_train.py work/air.json --level 1
"""

from __future__ import (
    annotations,
)

import argparse
import json
import logging
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import torch

from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)


def set_train_path(model: torch.nn.Module, level: int, cuda: bool) -> int:
    """
    Select the training dispatch on every module that carries one.

    The Triton level is read at construction to bind the kernels but consulted
    at call time to choose the path, so flipping it on a built model switches
    the dispatch without disturbing the weights. The fused CUDA convolution is
    a bound callable; it is stashed aside to disable and restored to enable,
    which keeps the mutually exclusive dispatch of ``forward_attention``
    intact.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose SO(2) modules carry ``triton_train_level``.
    level : int
        Triton dispatch level to install.
    cuda : bool
        Whether the fused CUDA convolution stays bound.

    Returns
    -------
    int
        Number of modules updated.
    """
    count = 0
    for module in model.modules():
        if hasattr(module, "triton_train_level"):
            module.triton_train_level = level
            count += 1
        if hasattr(module, "_cuda_value_train"):
            if not hasattr(module, "_cuda_value_train_stash"):
                module._cuda_value_train_stash = module._cuda_value_train
            module._cuda_value_train = module._cuda_value_train_stash if cuda else None
        if hasattr(module, "_grid_pair_train_fn"):
            if not hasattr(module, "_grid_pair_train_stash"):
                module._grid_pair_train_stash = module._grid_pair_train_fn
            module._grid_pair_train_fn = module._grid_pair_train_stash if cuda else None
    return count


def run_step(trainer: Any, batch: tuple[dict, dict], lr: float) -> tuple[float, dict]:
    """
    Run one forward and backward pass and collect the parameter gradients.

    Parameters
    ----------
    trainer : Any
        Initialized trainer.
    batch : tuple[dict, dict]
        Model inputs and labels.
    lr : float
        Learning-rate value handed to the loss.

    Returns
    -------
    tuple[float, dict]
        The loss value and a mapping from parameter name to detached gradient.
    """
    input_dict, label_dict = batch
    trainer.wrapper.zero_grad(set_to_none=True)
    _, loss, _ = trainer.wrapper(
        **input_dict, cur_lr=lr, label=label_dict, task_key="Default"
    )
    loss.backward()
    grads = {
        name: param.grad.detach().clone()
        for name, param in trainer.wrapper.named_parameters()
        if param.grad is not None
    }
    return float(loss.detach()), grads


def discrepancy(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> dict[str, float]:
    """
    Measure the per-parameter gradient difference between two runs.

    The difference is expressed relative to the largest gradient component
    across the whole model rather than to each parameter's own magnitude: a
    parameter whose gradient is orders of magnitude below the rest carries a
    large *own* relative error while contributing nothing to the optimizer step.

    Parameters
    ----------
    reference : dict[str, torch.Tensor]
        Gradients from the reference run.
    candidate : dict[str, torch.Tensor]
        Gradients from the run under test.

    Returns
    -------
    dict[str, float]
        Per-parameter error relative to the global gradient scale.

    Raises
    ------
    KeyError
        If the two runs produced gradients for different parameter sets,
        beyond parameters one path proves constant: a parameter absent from
        one run whose gradient is identically zero in the other is treated
        as zero on both sides (a single-branch grid router's softmax is
        identically one, so the fused product never records it).
    """
    reference = dict(reference)
    candidate = dict(candidate)
    for name in set(reference) ^ set(candidate):
        present = reference.get(name, candidate.get(name))
        if present.abs().max().item() != 0.0:
            raise KeyError(f"parameter sets differ on a live gradient: {name}")
        reference[name] = present
        candidate[name] = present
    global_scale = max(ref.abs().max().item() for ref in reference.values())
    return {
        name: (ref.float() - candidate[name].float()).abs().max().item() / global_scale
        for name, ref in reference.items()
    }


def report(
    title: str,
    errors: dict[str, float],
    reference: dict[str, torch.Tensor],
    show: int,
    noise: dict[str, float] | None,
    tol: float,
    factor: float,
) -> bool:
    """
    Print the largest discrepancies and decide whether they are significant.

    A fused kernel reduces in a different order than the dense path, so exact
    agreement is unattainable in float32. The verdict therefore compares each
    error against the noise floor measured by running the dense path twice: a
    difference that does not exceed what identical code already produces on its
    own carries no information about correctness.

    Parameters
    ----------
    title : str
        Section heading.
    errors : dict[str, float]
        Per-parameter error relative to the global gradient scale.
    reference : dict[str, torch.Tensor]
        Reference gradients, used to report each parameter's magnitude.
    show : int
        Number of worst entries to print.
    noise : dict[str, float] or None
        Per-parameter noise floor, or None when this report *is* the floor.
    tol : float
        Absolute tolerance applied when the noise floor is negligible.
    factor : float
        Multiple of the noise floor still considered insignificant.

    Returns
    -------
    bool
        Whether every parameter stayed within the accepted bound.
    """
    print(f"\n=== {title} ===")
    ranked = sorted(errors.items(), key=lambda kv: -kv[1])
    header = f"    {'vs global':>10}{'|grad|max':>12}"
    if noise is not None:
        header += f"{'noise':>11}{'ratio':>8}"
    print(header + "  name")
    failures = []
    for name, error in ranked[:show]:
        scale = reference[name].abs().max().item()
        line = f"    {error:10.3e}{scale:12.3e}"
        if noise is not None:
            floor = noise.get(name, 0.0)
            bound = max(tol, factor * floor)
            line += f"{floor:11.3e}{error / (floor + 1e-30):8.2f}"
            if error > bound:
                line += "  <-- FAIL"
        print(line + f"  {name}")
    if noise is not None:
        failures = [
            name
            for name, error in errors.items()
            if error > max(tol, factor * noise.get(name, 0.0))
        ]
        if failures:
            print(f"  !! {len(failures)} parameters exceed {factor}x the noise floor")
    return not failures


def main() -> None:
    """Parse the command line and run the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="training script")
    parser.add_argument(
        "--backend",
        choices=("pt", "pt_expt"),
        default="pt",
        help="trainer backend; both expose the same dispatch hooks",
    )
    parser.add_argument(
        "--path",
        choices=("triton", "cuda"),
        default="cuda",
        help="training path under test: the Triton operator composition or "
        "the fused CUDA value path",
    )
    parser.add_argument("--amp", action="store_true", help="keep bfloat16 autocast")
    parser.add_argument("--compile", action="store_true", help="keep torch.compile")
    parser.add_argument(
        "--tol", type=float, default=1e-6, help="absolute bound vs the global scale"
    )
    parser.add_argument(
        "--factor", type=float, default=4.0, help="accepted multiple of the noise floor"
    )
    parser.add_argument("--show", type=int, default=12)
    parser.add_argument(
        "--inductor",
        default="",
        help="JSON overrides for the Inductor options, as in bench.py",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    if args.inductor:
        from bench import (
            override_inductor_options,
        )

        print(f"inductor overrides: {override_inductor_options(args.inductor)}")
    config = json.loads(Path(args.input).read_text())
    if not args.amp:
        config["model"]["descriptor"]["use_amp"] = False
    if not args.compile:
        config["model"]["use_compile"] = False
    config = normalize(update_deepmd_input(config, warning=False, dump=None))

    if args.backend == "pt_expt":
        from deepmd.pt_expt.entrypoints.main import (
            get_trainer,
        )

        trainer = get_trainer(config)
        input_dict, label_dict = trainer.get_data(is_train=True)
    else:
        from deepmd.pt.entrypoints.main import (
            get_trainer,
        )

        trainer = get_trainer(config)
        input_dict, label_dict, _ = trainer.get_data(is_train=True)
    lr = float(trainer.lr_schedule.value(0))
    batch = (input_dict, label_dict)

    touched = set_train_path(trainer.wrapper, 0, cuda=False)
    print(f"dispatch-level modules: {touched}")
    dense_loss, dense_grads = run_step(trainer, batch, lr)

    # The dense path is run a second time to measure its own non-determinism:
    # the scatter reductions use atomics, so repeated runs of identical code
    # already disagree at the level that a different reduction order produces.
    # That figure is the floor below which a fused-vs-dense difference carries
    # no information.
    _, repeat_grads = run_step(trainer, batch, lr)
    noise = discrepancy(dense_grads, repeat_grads)
    report(
        "dense vs dense (reduction-order noise floor)",
        noise,
        dense_grads,
        args.show,
        None,
        args.tol,
        args.factor,
    )

    if args.path == "cuda":
        set_train_path(trainer.wrapper, 0, cuda=True)
    else:
        set_train_path(trainer.wrapper, 1, cuda=False)
    fused_loss, fused_grads = run_step(trainer, batch, lr)
    loss_error = abs(dense_loss - fused_loss) / (abs(dense_loss) + 1e-30)
    ok = report(
        f"dense vs fused ({args.path})",
        discrepancy(dense_grads, fused_grads),
        dense_grads,
        args.show,
        noise,
        args.tol,
        args.factor,
    )
    print(
        f"loss dense={dense_loss:.8f} fused={fused_loss:.8f} rel-err={loss_error:.3e}"
    )
    ok = ok and loss_error <= max(args.tol, 1e-6)
    print(f"[{'PASS' if ok else 'FAIL'}] path {args.path}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

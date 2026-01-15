# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201, TID253
"""Verify experimental value scaling through force-loss gradients and a CUDA PT2 artifact."""

from __future__ import (
    annotations,
)

import argparse
import json
import sys
from pathlib import (
    Path,
)

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "diagnose"))
from fitting_normalization import configure as configure_fitting
from fitting_normalization import install as install_fitting
from message_envelope import (
    configure,
    install,
)
from repro_spike import (
    load_model,
)


def inputs(type_map: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct a pair batch spanning the short range, tail, and cutoff."""
    distances = torch.tensor(
        [0.7, 1.2, 2.0, 3.0, 4.0, 5.7, 6.01, 7.0], dtype=torch.float64, device="cuda"
    )
    coord = torch.zeros((len(distances), 2, 3), dtype=torch.float64, device="cuda")
    coord[:, 1, 0] = distances
    atype = torch.full(
        (len(distances), 2), type_map.index("H"), dtype=torch.long, device="cuda"
    )
    box = (
        (20 * torch.eye(3, dtype=torch.float64, device="cuda"))
        .reshape(1, 9)
        .repeat(len(distances), 1)
    )
    return coord, atype, box


def gradients(
    model: torch.nn.Module, arguments: tuple[torch.Tensor, ...]
) -> dict[str, torch.Tensor | None]:
    """Differentiate an energy, force, and virial objective through the training path."""
    model.train()
    model.atomic_model.descriptor.use_amp = False
    model.zero_grad(set_to_none=True)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    out = model(*arguments)
    loss = (
        out["energy"].square().sum() * 0.1
        + out["force"].square().sum()
        + out["virial"].square().sum() * 0.01
    )
    loss.backward()
    return {
        name: parameter.grad.detach().clone() if parameter.grad is not None else None
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def training_gradients(
    model: torch.nn.Module, use_amp: bool = False
) -> dict[str, torch.Tensor | None]:
    """Exercise the configured energy/force/virial loss on a non-collinear validation frame."""
    from deepmd.pt.loss.ener import (
        EnergyStdLoss,
    )

    directory = Path(__file__).resolve().parent
    config = json.loads(
        (directory / "runs/h_1gpu_all4_gauss_1env/input.json").read_text()
    )
    dataset = Path(config["training"]["validation_data"]["systems"]) / "set.000"
    values = {
        key: np.load(dataset / f"{key}.npy")[0:1]
        for key in ("coord", "box", "energy", "force", "virial")
    }
    atoms = values["coord"].size // 3
    inputs = {
        key: torch.tensor(values[key], dtype=torch.float64, device="cuda")
        for key in ("coord", "box")
    }
    inputs["atype"] = torch.zeros((1, atoms), dtype=torch.long, device="cuda")
    labels = {
        key: torch.tensor(values[key], dtype=torch.float64, device="cuda")
        for key in ("energy", "force", "virial")
    }
    inputs["coord"] = inputs["coord"].reshape(1, atoms, 3)
    labels["force"] = labels["force"].reshape(1, atoms, 3)
    labels["energy"] = labels["energy"].reshape(1, 1)
    labels["virial"] = labels["virial"].reshape(1, 9)
    labels.update(find_energy=1.0, find_force=1.0, find_virial=1.0)
    params = dict(config["loss"])
    params.pop("type")
    params.pop("safeguard", None)
    params["starter_learning_rate"] = config["learning_rate"]["start_lr"]
    loss_function = EnergyStdLoss(**params)
    model.train()
    model.atomic_model.descriptor.use_amp = use_amp
    model.zero_grad(set_to_none=True)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    _, loss, _ = loss_function(
        inputs, model, labels, atoms, params["starter_learning_rate"]
    )
    loss.backward()
    return {
        name: parameter.grad.detach().clone() if parameter.grad is not None else None
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--fitting-rmsnorm", action="store_true")
    parser.add_argument("--no-output-envelope", action="store_true")
    parser.add_argument("--separate-attention-mass", action="store_true")
    parser.add_argument("--edge-type-neighbors", action="store_true")
    parser.add_argument("--seed-radial-gate", action="store_true")
    parser.add_argument("--outer-radial-gate", type=float, default=0.0)
    parser.add_argument("--pair-fade", type=float, nargs=2, default=(0.0, 0.0))
    args = parser.parse_args()
    from attention_normalization import configure as configure_attention
    from attention_normalization import install as install_attention
    from edge_type_neighbors import configure as configure_edge_type
    from edge_type_neighbors import install as install_edge_type
    from outer_radial_gate import configure as configure_outer_gate
    from outer_radial_gate import install as install_outer_gate
    from pair_fade import install as install_pair_fade
    from seed_radial_gate import configure as configure_seed
    from seed_radial_gate import install as install_seed

    install_pair_fade(*args.pair_fade)
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fused, type_map = load_model(args.checkpoint, "cuda")
    configure(fused.atomic_model.descriptor, not args.no_output_envelope)
    configure_attention(fused.atomic_model.descriptor, args.separate_attention_mass)
    configure_edge_type(fused.atomic_model.descriptor, args.edge_type_neighbors)
    configure_seed(fused.atomic_model.descriptor, args.seed_radial_gate)
    configure_outer_gate(fused.atomic_model.descriptor, args.outer_radial_gate)
    configure_fitting(fused.atomic_model.fitting_net, args.fitting_rmsnorm)
    reference, _ = load_model(args.checkpoint, "cuda")
    configure(reference.atomic_model.descriptor, not args.no_output_envelope)
    configure_attention(reference.atomic_model.descriptor, args.separate_attention_mass)
    configure_edge_type(reference.atomic_model.descriptor, args.edge_type_neighbors)
    configure_seed(reference.atomic_model.descriptor, args.seed_radial_gate)
    configure_outer_gate(reference.atomic_model.descriptor, args.outer_radial_gate)
    configure_fitting(reference.atomic_model.fitting_net, args.fitting_rmsnorm)
    for block in reference.atomic_model.descriptor.blocks:
        block.so2_conv._cuda_value_train = None
    # Keep the GIE/Wigner construction identical while comparing the value kernel.
    # The packed-descriptor route otherwise changes the GIE contraction as well.
    fused.atomic_model.descriptor._packed_wigner_train = False
    reference.atomic_model.descriptor._packed_wigner_train = False
    assert all(
        block.so2_conv._cuda_value_train is not None
        for block in fused.atomic_model.descriptor.blocks
    )
    arguments = inputs(type_map)
    actual, expected = training_gradients(fused), training_gradients(reference)
    compared = 0
    for name in actual:
        a, b = actual[name], expected[name]
        if a is None or b is None:
            assert a is b, f"gradient participation differs for {name}"
            continue
        torch.testing.assert_close(
            a, b, rtol=2e-3, atol=2e-5, msg=lambda text: f"{name}: {text}"
        )
        compared += 1
    print(
        f"CUDA FORCE-LOSS GRADIENT PARITY PASSED: {compared} parameter tensors",
        flush=True,
    )
    args.out.with_suffix(".gradient.json").write_text(
        json.dumps({"parameter_tensors": compared, "passed": True}) + "\n"
    )
    del reference, actual, expected
    fused.eval()
    eager = fused(*arguments)
    baseline = {
        name: eager[name].detach().cpu().numpy()
        for name in ("energy", "force", "virial")
    }
    with args.out.with_suffix(".reference.npz").open("xb") as stream:
        np.savez_compressed(
            stream,
            coord=arguments[0].detach().cpu().numpy(),
            atype=arguments[1].detach().cpu().numpy(),
            box=arguments[2].detach().cpu().numpy(),
            **baseline,
        )
    data = {"model": fused.serialize()}
    install(not args.no_output_envelope)
    install_attention(args.separate_attention_mass)
    install_edge_type(args.edge_type_neighbors)
    install_seed(args.seed_radial_gate)
    install_outer_gate(args.outer_radial_gate)
    install_fitting(args.fitting_rmsnorm)
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file,
    )

    deserialize_to_file(str(args.out), data, lower_kind="auto")
    from deepmd.infer import (
        DeepPot,
    )

    evaluator = DeepPot(str(args.out))
    coord, atype, box = (value.detach().cpu().numpy() for value in arguments)
    energy, force, virial = evaluator.eval(coord, box, atype[0])
    errors = {}
    for name, value in zip(
        ("energy", "force", "virial"), (energy, force, virial), strict=True
    ):
        target = baseline[name].reshape(value.shape)
        np.testing.assert_allclose(value, target, rtol=5e-4, atol=5e-5, err_msg=name)
        errors[name] = float(np.abs(value - target).max())
    args.out.with_suffix(".verification.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "artifact": str(args.out.resolve()),
                "device": "cuda",
                "gradient_parameter_tensors": compared,
                "max_absolute_errors": errors,
            },
            indent=2,
        )
        + "\n"
    )
    print("CUDA PT2 FREEZE/LOAD PARITY PASSED", errors, flush=True)


if __name__ == "__main__":
    main()

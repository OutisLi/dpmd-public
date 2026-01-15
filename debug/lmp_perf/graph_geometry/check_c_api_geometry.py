# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check geometry switching, CUDA replay and charge updates through the C API."""

from __future__ import (
    annotations,
)

import argparse
import ctypes as C
import json
import os
import sys
from pathlib import (
    Path,
)
from typing import (
    Any,
)

BENCH = Path(__file__).resolve().parents[2] / "cuda_bench"
sys.path.insert(0, str(BENCH))

from canonical_geometry import (
    Geometry,
    IndexedGeometry,
)


def bind(library: C.CDLL, name: str, arguments: list[Any], result: Any = None) -> Any:
    function = getattr(library, name)
    function.argtypes = arguments
    function.restype = result
    return function


def run_case(
    library: C.CDLL, path: Path, has_spin: bool, charge_state: bool, halo: bool
) -> list[dict]:
    import torch

    pointer = C.c_void_p
    kind = "DeepSpin" if has_spin else "DeepPot"
    create = bind(
        library,
        "DP_New" + kind + "WithParam2",
        [C.c_char_p, C.c_int, pointer, C.c_int],
        pointer,
    )
    destroy = bind(library, "DP_Delete" + kind, [pointer])
    check = bind(library, "DP_" + kind + "CheckOK", [pointer], pointer)
    free_string = bind(library, "DP_DeleteChar", [pointer])
    supports = bind(
        library,
        "DP_" + kind + "CanonicalGraphSupportsIndexedGeometry",
        [pointer],
        C.c_bool,
    )
    evaluate = bind(
        library,
        "DP_DPA4" + ("Spin" if has_spin else "") + "ComputeCanonicalGraphGPU",
        [pointer] * 12 + [C.c_int64, C.c_int64, C.c_int64, pointer],
    )
    set_charge = bind(
        library, "DP_" + kind + "SetChargeSpin", [pointer, pointer, C.c_int]
    )

    def checked(model: int) -> None:
        message = check(model)
        value = C.string_at(message)
        free_string(message)
        assert not value, value.decode()

    model = create(os.fsencode(path), 0, None, 0)
    checked(model)
    assert supports(model)
    nodes = 5
    owned = nodes - 1 if halo else nodes
    scale = torch.tensor([1 / 32], device="cuda")
    positions = torch.tensor(
        [[0, 0, 0], [48, 8, 0], [0, 48, 16], [8, 0, 48], [32, 32, 24]],
        dtype=torch.int64,
        device="cuda",
    )
    permutation = torch.tensor([3, 0, 4, 1, 2], device="cuda")
    center = permutation.argsort()
    packed = torch.cat(
        (
            (positions[permutation] + (1 << 31) - 17).to(torch.int32),
            permutation[:, None],
        ),
        dim=1,
    ).to(torch.int64)
    ghost = packed[center[1].long()].clone()
    ghost[1] += 4
    packed = torch.cat((packed, ghost[None]))
    pairs = torch.cartesian_prod(
        torch.arange(nodes, device="cuda"), torch.arange(nodes, device="cuda")
    )
    pairs = pairs[(pairs[:, 0] != pairs[:, 1]) & (pairs[:, 0] < owned)]
    atoms = center[pairs[:, 1]].long()
    atoms[(pairs[:, 0] == 0) & (pairs[:, 1] == 1)] = nodes
    delta = (
        packed[atoms, :3].long() - packed[center[pairs[:, 0]].long(), :3].long()
    ).to(torch.int32)
    order = delta.float().square().sum(1).reshape(owned, nodes - 1).argsort(1)
    order = (
        order + (torch.arange(owned, device="cuda") * (nodes - 1))[:, None]
    ).flatten()
    atoms = atoms[order]
    destination = pairs[order, 0]
    edges = owned * (nodes - 1)
    storage = edges + 7
    source = torch.zeros(storage, dtype=torch.int64, device="cuda")
    source[:edges] = packed[atoms, 3].to(torch.int64)
    source_rows = torch.cat(
        (
            torch.zeros(1, dtype=torch.int64, device="cuda"),
            torch.bincount(source[:edges].long(), minlength=nodes).cumsum(0),
        )
    )
    source_order = torch.zeros(storage, dtype=torch.int64, device="cuda")
    source_order[:edges] = source[:edges].long().argsort(stable=True).to(torch.int64)
    atom_virial = torch.empty(nodes, 9, dtype=torch.float64, device="cuda")
    raw = torch.zeros_like(source)
    raw[:edges] = atoms.to(torch.int64)
    vectors = torch.zeros(storage, 3, device="cuda")
    rows = torch.arange(nodes + 1, device="cuda").clamp_max(owned) * (nodes - 1)
    types = torch.arange(nodes, device="cuda") % 2
    moments = (
        torch.arange(nodes * 3, device="cuda").reshape(nodes, 3).float() / 20 - 0.25
    ).contiguous()
    outputs = [
        torch.empty(owned, dtype=torch.float64, device="cuda"),
        torch.empty(nodes, 3, dtype=torch.float64, device="cuda"),
        torch.empty(9, dtype=torch.float64, device="cuda"),
    ]
    if has_spin:
        outputs.append(torch.empty(nodes, 3, dtype=torch.float64, device="cuda"))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    records = []

    def refresh() -> None:
        vectors[:edges] = (
            packed[atoms, :3].long() - packed[center[destination].long(), :3].long()
        ).to(torch.int32).float() * scale

    def call(indexed: bool, instance: int = model, atomic: bool = False) -> list[Any]:
        geometry = Geometry()
        geometry.kind = int(indexed)
        geometry.index_width = source.element_size()
        geometry.edge_index = (raw if indexed else source).data_ptr()
        if indexed:
            geometry.data.indexed = IndexedGeometry(
                packed.data_ptr(), center.data_ptr(), scale.data_ptr(), packed.shape[0]
            )
        else:
            geometry.data.edge_vec = vectors.data_ptr()
        for value in outputs:
            value.fill_(float("nan"))
        atom_virial.fill_(float("nan"))
        av = atom_virial.data_ptr() if atomic else None
        sr = source_rows.data_ptr() if atomic else None
        so = source_order.data_ptr() if atomic else None
        energy, force, virial = outputs[:3]
        if has_spin:
            values = [
                instance,
                energy.data_ptr(),
                force.data_ptr(),
                outputs[3].data_ptr(),
                virial.data_ptr(),
                av,
                types.data_ptr(),
                C.byref(geometry),
                rows.data_ptr(),
                sr,
                so,
                moments.data_ptr(),
            ]
        else:
            values = [
                instance,
                energy.data_ptr(),
                force.data_ptr(),
                virial.data_ptr(),
                av,
                types.data_ptr(),
                C.byref(geometry),
                rows.data_ptr(),
                sr,
                so,
                None,
                None,
            ]
        evaluate(*values, owned, nodes, storage, stream.cuda_stream)
        checked(instance)
        result = [value.clone() for value in outputs]
        if atomic:
            result.append(atom_virial.clone())
        assert all(torch.isfinite(value).all() for value in result)
        return result

    def equal(actual: list[Any], expected: list[Any], label: str) -> None:
        bit_exact = all(
            torch.equal(x, y) for x, y in zip(actual, expected, strict=True)
        )
        errors = []
        for value, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, reference, atol=8e-6, rtol=1e-4)
            delta = value - reference
            errors.append(
                {
                    "max_abs": float(delta.abs().max()),
                    "rms_abs": float(delta.square().mean().sqrt()),
                }
            )
        if label.startswith("source_csr_replay_"):
            assert bit_exact, (
                kind,
                label,
                "identical source-CSR replay must be deterministic",
            )
        records.append(
            {
                "model": kind,
                "case": label,
                "outputs": len(actual),
                "bit_exact": bit_exact,
                "errors": errors,
            }
        )

    with torch.cuda.stream(stream):
        refresh()
        reference = call(False)
        for index, indexed in enumerate(
            (False, False, True, True, True, False, True, False)
        ):
            equal(call(indexed), reference, f"format_{index}_{indexed}")
        scale.mul_(0.99)
        moments.mul_(1.1)
        packed[:nodes, 2].add_(1)
        refresh()
        changed = call(False)
        assert not torch.equal(changed[1], reference[1])
        for index in range(3):
            equal(call(True), changed, f"changed_inputs_{index}")
        packed = torch.cat(
            (packed, torch.zeros(19, 4, dtype=torch.int64, device="cuda"))
        )
        for index in range(3):
            equal(call(True), changed, f"packed_capacity_{index}")
        equal(call(False), changed, "vector_after_growth")
        atomic_reference = call(False, atomic=True)
        equal(atomic_reference[:-1], changed, "source_csr_outputs")
        equal(call(True), changed, "indexed_after_source_csr")
        for index in range(3):
            equal(
                call(False, atomic=True), atomic_reference, f"source_csr_replay_{index}"
            )
        if charge_state:
            charge = (C.c_double * 2)(1.0, 2.0)
            set_charge(model, C.addressof(charge), 2)
            checked(model)
            fresh = create(os.fsencode(path), 0, None, 0)
            checked(fresh)
            set_charge(fresh, C.addressof(charge), 2)
            checked(fresh)
            other = call(False, fresh)
            assert not torch.equal(other[0], changed[0])
            for index, indexed in enumerate((False, True, False, True, True)):
                equal(call(indexed), other, f"charge_update_{index}_{indexed}")
            destroy(fresh)
    stream.synchronize()
    destroy(model)
    print(kind, "passed")  # noqa: T201
    return records


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--charge", action="store_true")
    parser.add_argument("--halo", action="store_true")
    args = parser.parse_args()
    torch.ops.load_library(str(args.artifacts / "integrated/libdeepmd_op_pt.so"))
    library = C.CDLL(str(args.artifacts / "integrated/libdeepmd_c.so"))
    records = []
    for has_spin in (False, True):
        name = "spin_nano.pt2" if has_spin else "nano.pt2"
        records.extend(
            run_case(library, args.models / name, has_spin, args.charge, args.halo)
        )
    args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()

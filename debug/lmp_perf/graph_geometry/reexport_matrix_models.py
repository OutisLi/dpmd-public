# SPDX-License-Identifier: LGPL-3.0-or-later
"""Re-export a matrix with identical serialized weights and measured fitting routes."""

from __future__ import (
    annotations,
)

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import (
    Path,
)


def model_contents(path: Path) -> tuple[str, dict, tuple[int, ...]]:
    """Fingerprint all serialized model data and read the baked fitting table."""
    with zipfile.ZipFile(path) as archive:
        model = json.loads(archive.read("model/extra/model.json"))["model"]
        metadata = json.loads(archive.read("model/extra/metadata.json"))
        routes = set()
        for name in archive.namelist():
            if not name.endswith(".wrapper.json"):
                continue
            wrapper = json.loads(archive.read(name))
            for entry in wrapper["nodes"]:
                node = entry["node"]
                if node["target"] != "deepmd::dpa4c_canonical_compress_energy_force":
                    continue
                values = next(
                    item["arg"]["as_ints"]
                    for item in node["inputs"]
                    if item["name"] == "fitting_routes"
                )
                routes.add(tuple(values))
    assert len(routes) == 1, (path, routes)
    payload = json.dumps(model, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest(), metadata, routes.pop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--names", help="Comma-separated model stems; all by default")
    args = parser.parse_args()
    import torch

    from deepmd.pt_expt.kernels import (
        fitting_routes,
    )
    from deepmd.pt_expt.model.model import (
        BaseModel,
    )
    from deepmd.pt_expt.utils import (
        env,
    )
    from deepmd.pt_expt.utils.serialization import (
        _cpu_reference_descriptors,
        _fold_case_embeddings,
        deserialize_to_file,
        serialize_from_file,
    )

    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = (
        [source / f"{name}.pt2" for name in args.names.split(",")]
        if args.names
        else sorted(source.glob("*.pt2"))
    )
    assert paths
    fingerprint = fitting_routes._fingerprint(env.DEVICE)
    for origin in paths:
        target = output / origin.name
        pending = output / f"{origin.stem}.building.pt2"
        if target.exists() or pending.exists():
            raise FileExistsError(target if target.exists() else pending)
        expected_hash, previous_metadata, routes = model_contents(origin)
        assert previous_metadata["canonical_index_dtype"] == "uint32", origin
        assert routes[1 : 1 + len(fingerprint)] == fingerprint, (
            origin,
            routes,
            fingerprint,
        )
        data = serialize_from_file(str(origin))
        model = BaseModel.deserialize(data["model"])
        # JSON encodes an empty array as [], losing its trailing dimensions.
        # A coupling record has eight columns even when there are no triples.
        learned, _ = model.atomic_model.fused_decomposition()
        descriptor = learned.descriptor
        descriptor.compress_coupling_meta = descriptor.compress_coupling_meta.reshape(
            -1, 8
        )
        data["model"] = model.serialize()
        _fold_case_embeddings(model)
        model.eval().to("cpu")
        with torch.no_grad(), _cpu_reference_descriptors(model):
            model.fold_vacuum_reference()
        learned, _ = model.atomic_model.fused_decomposition()
        key = fitting_routes.network_key(learned.fitting_net)
        assert key is not None, origin
        fitting_routes._ROUTE_TABLES[(fingerprint, key)] = routes
        del descriptor, learned, model
        deserialize_to_file(
            str(pending), data, lower_kind="dpa4c_canonical", do_atomic_virial=True
        )
        actual_hash, metadata, actual_routes = model_contents(pending)
        assert actual_hash == expected_hash, (origin, "serialized model data changed")
        assert actual_routes == routes, (origin, "fitting routes changed")
        assert metadata["canonical_index_dtype"] == "auto", metadata
        assert metadata["canonical_index_storage"] == "byte_words", metadata
        assert metadata["canonical_indexed_geometry"], metadata
        os.rename(pending, target)
        record = json.loads(origin.with_suffix(".json").read_text())
        record.update(
            path=str(target),
            sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            metadata=metadata,
            source_archive=str(origin),
            source_sha256=hashlib.sha256(origin.read_bytes()).hexdigest(),
            serialized_model_sha256=actual_hash,
            fitting_routes=list(actual_routes),
            fitting_network_key=key,
            matching_model_data=True,
            matching_fitting_routes=True,
        )
        target.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
        print(  # noqa: T201
            origin.stem,
            "auto-width export preserves model data and fitting routes",
            flush=True,
        )


if __name__ == "__main__":
    main()

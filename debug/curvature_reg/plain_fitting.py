# SPDX-License-Identifier: LGPL-3.0-or-later
"""Use ordinary dense MLP layers in the SeZM energy fitting network.

The hidden widths and output projection follow the SeZM fitting configuration.
Hidden layers have no residual bypass. The existing energy-fitting format records
the actual layers, so exported artifacts do not require an experimental registry.
"""

from __future__ import (
    annotations,
)

from functools import (
    wraps,
)
from typing import (
    Any,
)

from deepmd.dpmodel.utils.seed import (
    child_seed,
)

_enabled = False
_installed = False


def _install_builder(
    fitting: type, collection: type, network: type, attribute: str
) -> None:
    original = fitting._build_glu_fitting_layers

    @wraps(original)
    def build(self: Any) -> None:
        if not _enabled:
            original(self)
            return
        if self.case_film_embd:
            raise ValueError(
                "The plain fitting experiment requires concatenated case inputs"
            )
        in_dim = (
            self.dim_descrpt
            + self.numb_fparam
            + (0 if self.use_aparam_as_mask else self.numb_aparam)
            + self.dim_case_embd
        )
        networks = []
        for index in range(1 if self.mixed_types else self.ntypes):
            net = network(
                in_dim,
                self._net_out_dim(),
                self.neuron,
                activation_function=self.activation_function,
                resnet_dt=False,
                precision=self.precision,
                bias_out=self.bias_out,
                seed=child_seed(self.seed, index),
                trainable=self.trainable,
            )
            for layer in net.layers:
                layer.resnet = False
            networks.append(net)
        setattr(
            self,
            attribute,
            collection(
                0 if self.mixed_types else 1,
                self.ntypes,
                network_type="fitting_network",
                networks=networks,
            ),
        )

    fitting._build_glu_fitting_layers = build


def _install_serializer(fitting: type) -> None:
    original = fitting.serialize

    @wraps(original)
    def serialize(self: Any) -> dict[str, Any]:
        data = original(self)
        if data["nets"]["network_type"] != "fitting_network":
            return data
        if getattr(self, "_experiment_fitting_rmsnorm", False):
            raise ValueError(
                "Portable plain fitting requires fitting-input RMS normalization to be disabled"
            )
        data["type"] = "ener"
        data.pop("bias_out", None)
        data.pop("case_film_embd", None)
        return data

    fitting.serialize = serialize


def install(enabled: bool = False) -> None:
    """Select ordinary MLP construction and register its existing network format."""
    global _enabled, _installed
    _enabled = bool(enabled)
    if _installed:
        return
    from deepmd.dpmodel.fitting.dpa4_ener import SeZMEnergyFittingNet as NativeFitting
    from deepmd.dpmodel.fitting.dpa4_ener import (
        SeZMNetworkCollection as NativeCollection,
    )
    from deepmd.dpmodel.utils.network import FittingNet as NativeNetwork
    from deepmd.pt.model.network.mlp import FittingNet as PTNetwork
    from deepmd.pt.model.task.sezm_ener import SeZMEnergyFittingNet as PTFitting
    from deepmd.pt.model.task.sezm_ener import SeZMNetworkCollection as PTCollection
    from deepmd.pt_expt.fitting.dpa4_ener import (
        SeZMNetworkCollection as ExportCollection,
    )
    from deepmd.pt_expt.utils.network import FittingNet as ExportNetwork

    for collection, network in (
        (PTCollection, PTNetwork),
        (NativeCollection, NativeNetwork),
        (ExportCollection, ExportNetwork),
    ):
        collection.NETWORK_TYPE_MAP["fitting_network"] = network
    _install_builder(PTFitting, PTCollection, PTNetwork, "filter_layers")
    _install_builder(NativeFitting, NativeCollection, NativeNetwork, "nets")
    _install_serializer(PTFitting)
    _install_serializer(NativeFitting)
    _installed = True

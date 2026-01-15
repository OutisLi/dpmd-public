# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201
"""
The stay-or-step form of the SO(2) depth-attention residual (ledger E14, arm c).

The package's ``so2_attn_res`` replaces every mixing layer's input by a
softmax-weighted convex combination of the layer history (the initial state
and every earlier increment), which forbids the residual stream from
accumulating across depth. Here the same attention modules choose, at every
layer, between the current state and the current state plus the layer's own
increment: the stream may accumulate, but each step is gated by an
input-dependent weight in (0, 1) — a bounded rate of accumulation instead of
no accumulation. Requires ``so2_attn_res: dependent`` (or ``independent``) in
the descriptor so that the attention modules exist; the eager path only.
"""

from __future__ import (
    annotations,
)

import logging
import sys
from typing import (
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )
    from types import (
        ModuleType,
    )

    import torch

    from deepmd.pt.model.descriptor.sezm_nn.edge_cache import (
        EdgeFeatureCache,
    )
    from deepmd.pt.model.descriptor.sezm_nn.so2 import (
        SO2Convolution,
    )

log = logging.getLogger("deepmd.sezm_attnres")


def announce(message: str) -> None:
    log.info(message)
    print(f"[sezm_attnres] {message}", file=sys.stderr, flush=True)


def install(post_add_norm: bool = False) -> None:
    """Install the stay-or-step residual, or (``post_add_norm``) the normalized stream instead."""
    import torch

    from deepmd.pt.model.descriptor.sezm_nn import so2 as so2_mod

    conv = so2_mod.SO2Convolution
    if post_add_norm:
        install_post_add_norm(conv, torch)
        return

    def so2_mixing_layers_stay_or_step(
        self: SO2Convolution,
        x_local: torch.Tensor,
        rad_feat_l0_focus: torch.Tensor,
        focus_gate_src: torch.Tensor,
        edge_cache: EdgeFeatureCache,
    ) -> torch.Tensor:
        if not self.use_so2_attn_res:
            raise RuntimeError("the stay-or-step residual needs so2_attn_res != 'none'")
        if self.layer_scale:
            raise NotImplementedError(
                "layer scale is not combined with the stay-or-step residual"
            )

        def so2_l0_extractor(v: torch.Tensor) -> torch.Tensor:
            return v[:, :, 0, :].reshape(v.shape[0], self.hidden_channels)

        if not getattr(self, "_stay_or_step_logged", False):
            self._stay_or_step_logged = True
            announce(
                f"SO(2) stay-or-step residual active on {len(self.so2_linears)} mixing layers."
            )
        for layer_idx, (so2_linear, inter_norm, non_linear) in enumerate(
            zip(
                self.so2_linears,
                self.so2_inter_norms,
                self.non_linearities,
                strict=True,
            )
        ):
            residual = x_local
            x_local = inter_norm(x_local)
            x_local = so2_linear(x_local)
            if layer_idx == 0 and so2_linear.bias0 is not None:
                if so2_linear.out_channels == self.so2_focus_dim:
                    radial_factor = rad_feat_l0_focus
                else:
                    radial_factor = torch.cat(
                        [rad_feat_l0_focus, rad_feat_l0_focus], dim=-1
                    )
                bias0 = so2_linear.bias0.view(self.n_focus, so2_linear.out_channels)
                correction = bias0.unsqueeze(1) * (
                    radial_factor.transpose(0, 1)
                    * edge_cache.edge_env.reshape(1, -1, 1)
                    - 1.0
                )
                x_local[:, :, 0, :].add_(correction)
            x_local = non_linear(x_local)
            stepped = residual + x_local
            # Attention over two sources, stay (the current state) and step (the
            # state plus this layer's increment), queried by the stepped state;
            # the depth-attention module batches on axis 0, hence the edge-major views.
            x_edge = self.so2_layer_attn_res[layer_idx](
                sources=[residual.transpose(0, 1), stepped.transpose(0, 1)],
                scalar_extractor=so2_l0_extractor,
                current_x=stepped.transpose(0, 1),
            )
            x_local = x_edge.transpose(0, 1)
        if self.focus_compete and self.n_focus > 1:
            alpha = self._focus_alpha(focus_gate_src.transpose(0, 1))
            x_local = x_local * alpha.transpose(0, 1).to(dtype=x_local.dtype).unsqueeze(
                -1
            ).unsqueeze(-1)
        return x_local.permute(1, 0, 2, 3)

    conv._so2_mixing_layers = so2_mixing_layers_stay_or_step
    announce("SeZM stay-or-step residual installed.")


def install_post_add_norm(conv: type[SO2Convolution], torch: ModuleType) -> None:
    """Arm (d): the mixing chain's equivariant RMS norms are moved from the layer
    input to the residual stream after every add, so the stream's magnitude is
    bounded at every depth (the post-norm residual chain). Requires
    ``so2_norm: true`` so that the norms exist; the last layer, which the package
    leaves without a norm, adds its output projection unnormalized.
    """

    def so2_mixing_layers_post_norm(
        self: SO2Convolution,
        x_local: torch.Tensor,
        rad_feat_l0_focus: torch.Tensor,
        focus_gate_src: torch.Tensor,
        edge_cache: EdgeFeatureCache,
    ) -> torch.Tensor:
        if self.use_so2_attn_res or self.layer_scale:
            raise NotImplementedError(
                "the post-add norm is implemented for the plain residual chain only"
            )
        if not self.so2_norm:
            raise RuntimeError("the post-add norm needs so2_norm: true")
        if not getattr(self, "_post_add_norm_logged", False):
            self._post_add_norm_logged = True
            announce(
                f"SO(2) post-add normalization active on {len(self.so2_linears)} mixing layers."
            )
        for layer_idx, (so2_linear, inter_norm, non_linear) in enumerate(
            zip(
                self.so2_linears,
                self.so2_inter_norms,
                self.non_linearities,
                strict=True,
            )
        ):
            residual = x_local
            x_local = so2_linear(x_local)
            if layer_idx == 0 and so2_linear.bias0 is not None:
                if so2_linear.out_channels == self.so2_focus_dim:
                    radial_factor = rad_feat_l0_focus
                else:
                    radial_factor = torch.cat(
                        [rad_feat_l0_focus, rad_feat_l0_focus], dim=-1
                    )
                bias0 = so2_linear.bias0.view(self.n_focus, so2_linear.out_channels)
                correction = bias0.unsqueeze(1) * (
                    radial_factor.transpose(0, 1)
                    * edge_cache.edge_env.reshape(1, -1, 1)
                    - 1.0
                )
                x_local[:, :, 0, :].add_(correction)
            x_local = non_linear(x_local)
            x_local = inter_norm(residual + x_local)
        if self.focus_compete and self.n_focus > 1:
            alpha = self._focus_alpha(focus_gate_src.transpose(0, 1))
            x_local = x_local * alpha.transpose(0, 1).to(dtype=x_local.dtype).unsqueeze(
                -1
            ).unsqueeze(-1)
        return x_local.permute(1, 0, 2, 3)

    conv._so2_mixing_layers = so2_mixing_layers_post_norm
    announce("SeZM post-add normalization installed.")


def install_variant(
    key_mode: str = "l0", weight_mode: str = "shared", value_norm: bool = False
) -> None:
    """
    Replace the SO(2)-level depth-attention modules by ``DegreeAttnRes`` (ledger E14, arms h, i, j).

    The descriptor must be built with ``so2_attn_res: dependent`` or ``independent``; the query
    mode is taken from that option. ``key_mode="l0"``, ``weight_mode="shared"`` and
    ``value_norm=False`` reproduce the package's ``DepthAttnRes`` exactly, so that each arm
    differs from it by one factor.
    """
    import torch
    import torch.nn as nn

    from deepmd.pt.model.descriptor.sezm_nn import so2 as so2_mod
    from deepmd.pt.model.descriptor.sezm_nn.norm import (
        ScalarRMSNorm,
    )
    from deepmd.pt.model.descriptor.sezm_nn.so3 import (
        ChannelLinear,
    )
    from deepmd.pt.utils import (
        env,
    )

    if key_mode not in ("l0", "degrees") or weight_mode not in ("shared", "per_degree"):
        raise ValueError(
            f"unknown attention-residual variant: keys {key_mode!r}, weights {weight_mode!r}"
        )

    class DegreeAttnRes(nn.Module):
        """
        Depth attention over the mixing-layer history with SO(2)-invariant keys from every
        degree, one softmax per degree, or per-degree RMS-normalized values.

        Sources carry the edge-major layout (E, F, D_m, Cf) of the mixing chain, whose D_m axis
        is m-major; ``degree_index`` gives the degree l of every coefficient. Under the local
        SO(2) (rotations about the edge) the m = 0 coefficients are invariant and every
        (l, +m), (l, -m) pair rotates as a 2-vector, so its norm is invariant; the per-degree
        norm sqrt(sum_m x_{l,m}^2) over the coefficients present is therefore invariant as
        well. The keys are the signed (l = 0, m = 0) scalars and, with ``key_mode="degrees"``,
        the per-degree norms of l = 1..lmax, so that the logits are invariants and the weights
        scalars. With ``weight_mode="per_degree"`` each degree has its own softmax over the
        sources, which preserves equivariance because both members of every (l, ±m) pair are
        scaled by the same invariant weight. With ``value_norm`` every source's degree blocks
        are RMS-normalized over their coefficients and channels (per edge and focus stream) and
        rescaled by a learned per-degree gain before the combination, which bounds the state
        absolutely. Zero-initialized queries give a uniform average at initialization, as in
        the package.
        """

        def __init__(
            self,
            *,
            lmax: int,
            degree_index: torch.Tensor,
            n_focus: int,
            focus_dim: int,
            input_dependent: bool,
            eps: float,
            bias: bool,
            dtype: torch.dtype,
            trainable: bool,
        ) -> None:
            super().__init__()
            self.n_deg = int(lmax) + 1
            self.n_focus, self.focus_dim = int(n_focus), int(focus_dim)
            self.n_logits = self.n_deg if weight_mode == "per_degree" else 1
            self.key_dim = (
                self.n_focus
                * self.focus_dim
                * (self.n_deg if key_mode == "degrees" else 1)
            )
            self.input_dependent, self.eps, self.dtype = (
                bool(input_dependent),
                float(eps),
                dtype,
            )
            self.register_buffer("degree_index", degree_index.clone(), persistent=False)
            self.register_buffer(
                "degree_count",
                torch.bincount(degree_index, minlength=self.n_deg).to(dtype=dtype),
                persistent=False,
            )
            self.key_norm = ScalarRMSNorm(
                channels=self.key_dim,
                n_focus=1,
                eps=self.eps,
                dtype=dtype,
                trainable=trainable,
            )
            if self.input_dependent:
                self.query_proj = ChannelLinear(
                    in_channels=self.key_dim,
                    out_channels=self.key_dim * self.n_logits,
                    dtype=dtype,
                    bias=bias,
                    trainable=trainable,
                    init_std=0.0,
                )
            else:
                self.adamw_pseudo_query = nn.Parameter(
                    torch.zeros(
                        self.n_logits, self.key_dim, dtype=dtype, device=env.DEVICE
                    ),
                    requires_grad=trainable,
                )
            if value_norm:
                self.adam_value_gain = nn.Parameter(
                    torch.ones(self.n_deg, dtype=dtype, device=env.DEVICE),
                    requires_grad=trainable,
                )

        def degree_sum_of_squares(self, v: torch.Tensor) -> torch.Tensor:
            """Sum of squares of every degree's coefficients: (..., D_m, Cf) -> (..., n_deg, Cf)."""
            shape = (*v.shape[:-2], self.n_deg, v.shape[-1])
            out = torch.zeros(shape, dtype=v.dtype, device=v.device)
            return out.index_add_(v.ndim - 2, self.degree_index, v.square())

        def invariants(self, v: torch.Tensor) -> torch.Tensor:
            """SO(2)-invariant key features of one source: (E, F, D_m, Cf) -> (E, key_dim)."""
            v = v.to(dtype=self.dtype)
            l0 = v[:, :, 0, :]  # (E, F, Cf), the signed (l = 0, m = 0) scalars
            if key_mode == "l0":
                return l0.reshape(v.shape[0], -1)
            norms = (
                self.degree_sum_of_squares(v)[:, :, 1:, :] + 1e-12
            ).sqrt()  # (E, F, lmax, Cf)
            return torch.cat([l0.unsqueeze(2), norms], dim=2).reshape(v.shape[0], -1)

        def forward(
            self,
            *,
            sources: list,
            scalar_extractor: Callable[[torch.Tensor], torch.Tensor] | None = None,
            current_x: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if len(sources) == 1:
                return sources[0]
            n_edge, value_dtype = sources[0].shape[0], sources[0].dtype
            keys = self.key_norm(
                torch.stack([self.invariants(s) for s in sources], dim=1)
            )  # (E, S, K)
            if self.input_dependent:
                query = self.query_proj(self.invariants(current_x)).reshape(
                    n_edge, self.n_logits, self.key_dim
                )
            else:
                query = self.adamw_pseudo_query.unsqueeze(0).expand(n_edge, -1, -1)
            alpha = torch.softmax(
                torch.einsum("elk,esk->esl", query, keys), dim=1
            )  # (E, S, L)
            values = torch.stack(
                [s.to(dtype=self.dtype) for s in sources], dim=1
            )  # (E, S, F, D_m, Cf)
            if value_norm:
                # Mean square of every degree block over its coefficients and channels, per edge, source and focus.
                mean_sq = self.degree_sum_of_squares(values).sum(dim=-1) / (
                    self.degree_count * values.shape[-1]
                )
                scale = self.adam_value_gain * torch.rsqrt(
                    mean_sq + self.eps
                )  # (E, S, F, n_deg)
                values = values * scale[..., self.degree_index].unsqueeze(-1)
            if self.n_logits == 1:
                weight = alpha[:, :, 0].reshape(n_edge, -1, 1, 1, 1)
            else:
                weight = alpha[:, :, self.degree_index].reshape(
                    n_edge, -1, 1, values.shape[3], 1
                )
            return (weight * values).sum(dim=1).to(dtype=value_dtype)

    conv = so2_mod.SO2Convolution
    original_init = conv.__init__

    def patched_init(self, *args, **kwargs) -> None:  # noqa: ANN001, ANN002, ANN003
        original_init(self, *args, **kwargs)
        if not self.use_so2_attn_res:
            raise RuntimeError(
                "the degree attention residual needs so2_attn_res != 'none'"
            )
        trainable = all(p.requires_grad for p in self.so2_layer_attn_res.parameters())
        self.so2_layer_attn_res = nn.ModuleList(
            [
                DegreeAttnRes(
                    lmax=self.lmax,
                    degree_index=self.degree_index_m,
                    n_focus=self.n_focus,
                    focus_dim=self.so2_focus_dim,
                    input_dependent=self.so2_attn_res_mode == "dependent",
                    eps=self.eps,
                    bias=self.mlp_bias,
                    dtype=self.compute_dtype,
                    trainable=trainable,
                )
                for _ in range(self.mixing_layers)
            ]
        )
        announce(
            f"SO(2) depth attention with keys={key_mode}, weights={weight_mode}, value_norm={value_norm} "
            f"on {self.mixing_layers} mixing layers ({self.so2_attn_res_mode} query)."
        )

    conv.__init__ = patched_init

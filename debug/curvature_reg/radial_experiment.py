# SPDX-License-Identifier: LGPL-3.0-or-later
"""Radial-basis experiments shared by training, evaluation, and export.

Options apply to subsequently constructed models and are stored on each basis.
Changing the options for a second model does not alter the first model's forward
function. The PT and array-API constructors implement the same equation; an
export therefore retains the experimental radial function in its traced graph.
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

import numpy as np

_options = (0.0, False, False, 0.0, None, 0.0)
_installed = False


def install(
    *,
    gauss_inner: float = 0.0,
    single_envelope: bool = False,
    fixed_radial: bool = False,
    gauss_width: float = 0.0,
    gauss_scales: tuple[float, float] | None = None,
    gauss_floor: float = 0.0,
) -> None:
    """Set radial options for subsequently constructed PT and PT-expt models.

    Parameters
    ----------
    gauss_inner : float
        Upper endpoint of the initial Gaussian centres, in Angstrom. Zero uses
        the cutoff radius. Unless ``gauss_width`` is specified, the width is the
        spacing between initial centres. Centres remain trainable unless
        ``fixed_radial`` is true.
    single_envelope : bool
        Return the bare basis and retain the downstream edge envelope.
    fixed_radial : bool
        Keep basis centres or frequencies fixed throughout training.
    gauss_width : float
        Gaussian width in Angstrom. Zero retains the initial centre spacing.
        An explicit width separates centre-position and width experiments.
    gauss_scales : tuple[float, float], optional
        Minimum and maximum widths of zero-centred Gaussian functions, in
        Angstrom. Widths are geometrically spaced and stored in ``adam_freqs``.
        This fixed scale family replaces the translated-centre family and
        cannot combine with ``gauss_inner`` or ``gauss_width``.
    gauss_floor : float
        Lower endpoint of the initial Gaussian centres, in Angstrom, so that the
        centres lie on ``linspace(gauss_floor, gauss_inner, n_radial)``. A basis
        function centred where the data have no pair distances is a parameter
        the data never constrain; the floor removes such functions instead of
        gating their output. Requires ``gauss_inner`` and lies below it.
    """
    global _options, _installed
    if gauss_inner < 0.0 or gauss_width < 0.0:
        raise ValueError(
            f"Gaussian endpoint and width must be nonnegative, got {gauss_inner}, {gauss_width}"
        )
    if gauss_scales is not None:
        gauss_scales = tuple(float(value) for value in gauss_scales)
        if (
            len(gauss_scales) != 2
            or not np.isfinite(gauss_scales).all()
            or not 0 < gauss_scales[0] < gauss_scales[1]
        ):
            raise ValueError(
                f"Gaussian scales require 0 < minimum < maximum, got {gauss_scales}"
            )
        if gauss_inner or gauss_width or not fixed_radial:
            raise ValueError(
                "Gaussian scales require fixed_radial and no translated-centre or width option"
            )
    if (
        gauss_floor < 0.0
        or (gauss_floor and not gauss_inner)
        or gauss_floor >= gauss_inner > 0.0
    ):
        raise ValueError(
            f"Gaussian floor requires 0 <= floor < inner endpoint, got floor={gauss_floor}, inner={gauss_inner}"
        )
    _options = (
        float(gauss_inner),
        bool(single_envelope),
        bool(fixed_radial),
        float(gauss_width),
        gauss_scales,
        float(gauss_floor),
    )
    if _installed:
        return

    import torch

    from deepmd.dpmodel.descriptor.dpa4_nn.radial import RadialBasis as NativeBasis
    from deepmd.pt.model.descriptor.sezm import (
        DescrptSeZM,
    )
    from deepmd.pt.model.descriptor.sezm_nn.radial import (
        RadialBasis,
    )
    from deepmd.pt_expt.descriptor.dpa4 import (
        DescrptDPA4,
    )

    pt_init = RadialBasis.__init__
    pt_forward = RadialBasis.forward
    native_init = NativeBasis.__init__
    native_call = NativeBasis.call

    @wraps(pt_init)
    def init_pt(self: RadialBasis, *args: Any, **kwargs: Any) -> None:
        pt_init(self, *args, **kwargs)
        inner, single, fixed, width, scales, floor = _options
        self._experiment_single_envelope = single
        self._experiment_gaussian_scales = scales is not None
        if single:
            self.exponent = 0
            self.envelope = None
        if inner:
            _validate_inner(self, inner)
            with torch.no_grad():
                self.adam_freqs.copy_(
                    torch.linspace(
                        floor,
                        inner,
                        self.n_radial,
                        dtype=self.adam_freqs.dtype,
                        device=self.adam_freqs.device,
                    ).reshape_as(self.adam_freqs)
                )
                self.gaussian_coeff.fill_(
                    -0.5 / ((inner - floor) / max(self.n_radial - 1, 1)) ** 2
                )
        if width:
            _validate_width(self)
            self.gaussian_coeff.fill_(-0.5 / width**2)
        if scales is not None:
            _validate_width(self)
            with torch.no_grad():
                self.adam_freqs.copy_(
                    torch.as_tensor(
                        np.geomspace(*scales, self.n_radial),
                        dtype=self.adam_freqs.dtype,
                        device=self.adam_freqs.device,
                    ).reshape_as(self.adam_freqs)
                )
        if fixed:
            self.trainable = False
            self.adam_freqs.requires_grad_(False)

    def forward_pt(self: RadialBasis, r: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_experiment_gaussian_scales", False):
            raw = torch.exp(-0.5 * (r / self.adam_freqs).square())
            return raw * self.envelope(r) if self.envelope is not None else raw
        return pt_forward(self, r)

    @wraps(native_init)
    def init_native(self: NativeBasis, *args: Any, **kwargs: Any) -> None:
        native_init(self, *args, **kwargs)
        inner, single, fixed, width, scales, floor = _options
        self._experiment_single_envelope = single
        self._experiment_gaussian_scales = scales is not None
        if single:
            self.exponent = 0
            self.envelope = None
        if inner:
            _validate_inner(self, inner)
            frequencies = self.adam_freqs
            if isinstance(frequencies, torch.Tensor):
                self.adam_freqs = torch.linspace(
                    floor,
                    inner,
                    self.n_radial,
                    dtype=frequencies.dtype,
                    device=frequencies.device,
                ).reshape(1, self.n_radial)
            else:
                self.adam_freqs = np.linspace(
                    floor, inner, self.n_radial, dtype=frequencies.dtype
                ).reshape(1, self.n_radial)
            self.gaussian_coeff = (
                -0.5 / ((inner - floor) / max(self.n_radial - 1, 1)) ** 2
            )
        if width:
            _validate_width(self)
            self.gaussian_coeff = -0.5 / width**2
        if scales is not None:
            _validate_width(self)
            frequencies = self.adam_freqs
            values = np.geomspace(*scales, self.n_radial).reshape(1, self.n_radial)
            self.adam_freqs = (
                torch.as_tensor(
                    values, dtype=frequencies.dtype, device=frequencies.device
                )
                if isinstance(frequencies, torch.Tensor)
                else values.astype(frequencies.dtype)
            )
        if fixed:
            self.trainable = False

    def call_native(self: NativeBasis, r: Any) -> Any:
        if not getattr(self, "_experiment_gaussian_scales", False):
            return native_call(self, r)
        import array_api_compat

        from deepmd.dpmodel.array_api import (
            xp_asarray_nodetach,
        )

        xp = array_api_compat.array_namespace(r)
        widths = xp_asarray_nodetach(
            xp, self.adam_freqs[...], device=array_api_compat.device(r)
        )
        ratio = r / widths
        raw = xp.exp(-0.5 * ratio * ratio)
        return raw * self.envelope(r) if self.envelope is not None else raw

    RadialBasis.__init__ = init_pt
    RadialBasis.forward = forward_pt
    NativeBasis.__init__ = init_native
    NativeBasis.call = call_native
    for descriptor in (DescrptSeZM, DescrptDPA4):
        _install_descriptor(descriptor)
    _installed = True


def _validate_inner(basis: Any, inner: float) -> None:
    """Validate the initial Gaussian interval against the basis definition."""
    if basis.basis_type != "gaussian" or inner > basis.rcut:
        raise ValueError(
            f"Inner centres require a Gaussian basis and radius <= {basis.rcut}, "
            f"got basis={basis.basis_type!r}, radius={inner}"
        )


def _validate_width(basis: Any) -> None:
    if basis.basis_type != "gaussian":
        raise ValueError(
            f"Gaussian width requires a Gaussian basis, got {basis.basis_type!r}"
        )


def _install_descriptor(descriptor: type) -> None:
    """Prevent the fused radial operator from restoring the basis envelope."""
    original = descriptor.__init__

    @wraps(original)
    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        if getattr(self.radial_basis, "_experiment_single_envelope", False) or getattr(
            self.radial_basis, "_experiment_gaussian_scales", False
        ):
            self._cuda_radial_fn = None

    descriptor.__init__ = init

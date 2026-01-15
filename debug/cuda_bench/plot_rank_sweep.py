# SPDX-License-Identifier: LGPL-3.0-or-later
"""Plot per-channel DPA4C radial-mode-rank throughput and capacity sweeps."""

from __future__ import (
    annotations,
)

import sweep_figures


def main() -> None:
    """Generate one two-panel figure per channel width."""
    sweep_figures.draw_sweep(
        sweep_figures.load_summary(),
        column="radial_modes",
        values=(0, 2, 4, 8),
        fixed={"lmax": 2},
        label="Radial mode rank $R$",
        title="DPA4C C{channels}, $l_{{max}}=2$: radial-mode-rank scaling",
        stem="dpa4c_rank_sweep_c{channels}.png",
    )


if __name__ == "__main__":
    main()

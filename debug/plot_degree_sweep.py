# SPDX-License-Identifier: LGPL-3.0-or-later
"""Plot per-channel DPA4C angular-degree throughput and capacity sweeps."""

from __future__ import (
    annotations,
)

import sweep_figures


def main() -> None:
    """Generate one two-panel figure per channel width."""
    sweep_figures.draw_sweep(
        sweep_figures.load_summary(),
        column="lmax",
        values=(2, 3, 4),
        fixed={"radial_modes": 0},
        label="Maximum angular degree $l_{max}$",
        title="DPA4C C{channels}, $R=0$: angular-degree scaling",
        stem="dpa4c_degree_sweep_c{channels}.png",
    )


if __name__ == "__main__":
    main()
